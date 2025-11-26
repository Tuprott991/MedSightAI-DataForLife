import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import MedicalConceptModel
from src.dataset import VinDrCXRDataset, get_default_transforms, collate_fn_with_boxes


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
        # Initialize distributed training
        self.is_distributed = config.get('distributed', False)
        if self.is_distributed:
            self.setup_distributed()
        
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        
        if self.is_main_process:
            print(f"[Trainer] Using device: {self.device}")
            if self.is_distributed:
                print(f"[Trainer] Distributed training on {self.world_size} GPUs")
        
        # Create output directories (only on main process)
        if self.is_main_process:
            self.output_dir = os.path.join('outputs', config['exp_name'])
            self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
            self.log_dir = os.path.join(self.output_dir, 'logs')
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Save config
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            # TensorBoard writer
            self.writer = SummaryWriter(self.log_dir)
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        # Initialize datasets
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Initialize optimizers
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Loss weights
        self.class_loss_weight = config['class_loss_weight']
        self.contrastive_loss_weight = config['contrastive_loss_weight']
        
        # Best metrics tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        
        if self.is_main_process:
            print(f"[Trainer] Initialized successfully")
            print(f"[Trainer] Output directory: {self.output_dir}")
    
    def setup_distributed(self):
        """Setup distributed training environment."""
        # Get rank and world size from environment variables set by torchrun
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.global_rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # Use NCCL for GPU
            init_method='env://',
        )
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        
        self.is_main_process = (self.global_rank == 0)
    
    @property
    def is_main_process(self):
        """Check if current process is main process."""
        if not self.is_distributed:
            return True
        return self._is_main_process
    
    @is_main_process.setter
    def is_main_process(self, value):
        self._is_main_process = value
    
    def _build_model(self):
        """Build the medical concept model."""
        model = MedicalConceptModel(
            model_name=self.config['backbone_name'],
            num_concepts=self.config['num_concepts'],
            num_classes=self.config['num_classes'],
            projection_dim=self.config['projection_dim'],
            prototypes_per_concept=self.config['prototypes_per_concept'],
            freeze_backbone=self.config['freeze_backbone'],
            contrastive_lambda=self.config['contrastive_lambda'],
            contrastive_gamma=self.config['contrastive_gamma'],
            contrastive_delta=self.config['contrastive_delta'],
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if self.is_main_process:
            print(f"[Model] Total parameters: {num_params:,}")
            print(f"[Model] Trainable parameters: {num_trainable:,}")
        
        return model
    
    def _build_dataloaders(self):
        """Build train and validation dataloaders."""
        train_transform = get_default_transforms(
            image_size=self.config['image_size'],
            is_training=True
        )
        val_transform = get_default_transforms(
            image_size=self.config['image_size'],
            is_training=False
        )
        
        train_dataset = VinDrCXRDataset(
            root_dir=self.config['data_root'],
            split='train',
            transform=train_transform,
            return_boxes=False,  # Not using boxes for training
            radiologist_id=self.config.get('radiologist_id', None),
            use_multi_rater=self.config.get('use_multi_rater', True),
        )
        
        # Use a portion of train as validation if no separate val split
        # Or use test split as validation
        val_dataset = VinDrCXRDataset(
            root_dir=self.config['data_root'],
            split='test',  # Using test as validation
            transform=val_transform,
            return_boxes=False,
            radiologist_id=self.config.get('radiologist_id', None),
            use_multi_rater=self.config.get('use_multi_rater', True),
        )
        
        # Store concept and disease names
        self.concept_names = train_dataset.concept_names
        self.disease_names = train_dataset.disease_names
        self.num_concepts = train_dataset.num_concepts
        self.num_diseases = train_dataset.num_diseases
        
        # Create samplers for distributed training
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
                drop_last=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False,
                drop_last=False
            )
        else:
            train_sampler = None
            val_sampler = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn_with_boxes,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn_with_boxes,
        )
        
        if self.is_main_process:
            print(f"[Data] Train samples: {len(train_dataset)}")
            print(f"[Data] Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _build_optimizer(self):
        """Build optimizer."""
        # Separate parameters: backbone vs rest
        backbone_params = []
        other_params = []
        
        # Get model without DDP wrapper
        model = self.model.module if self.is_distributed else self.model
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr': self.config['lr']},
        ]
        
        if len(backbone_params) > 0:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config['lr'] * self.config.get('backbone_lr_multiplier', 0.1)
            })
        
        optimizer = optim.AdamW(
            param_groups,
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['epochs'] // 3,
            T_mult=2,
            eta_min=self.config['lr'] * 0.01
        )
        return scheduler
    
    def compute_loss(self, outputs, disease_labels, concept_labels):
        """
        Compute total loss = classification loss + contrastive loss.
        
        Args:
            outputs: Model outputs dict
            disease_labels: (B, num_diseases) - ground truth disease labels
            concept_labels: (B, num_concepts) - ground truth concept presence
        """
        # 1. Classification loss (multi-label BCE)
        class_logits = outputs['class_logits']  # (B, num_classes)
        
        if self.config['num_classes'] == 1:
            # Binary classification - use any disease present as target
            # Or you can use a specific disease column
            # Here we'll use "No finding" negated or first disease
            target = disease_labels[:, 0:1]  # (B, 1) - first disease column
        else:
            # Multi-class - disease_labels already has shape (B, num_classes)
            target = disease_labels[:, :self.config['num_classes']]
        
        class_loss_fn = nn.BCEWithLogitsLoss()
        class_loss = class_loss_fn(class_logits, target)
        
        # 2. Contrastive loss for concepts
        # Only compute for images where concepts are present
        v_local = outputs['v_local']  # (B, K, C)
        
        # Create mask for positive concepts
        concept_mask = concept_labels > 0.5  # (B, K)
        
        if concept_mask.sum() > 0:
            # Only compute contrastive loss for positive concepts
            # Extract positive concept vectors
            # Get model without DDP wrapper for contrastive_loss
            model = self.model.module if self.is_distributed else self.model
            contrastive_loss = model.contrastive_loss(v_local)
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = (
            self.class_loss_weight * class_loss +
            self.contrastive_loss_weight * contrastive_loss
        )
        
        return {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'contrastive_loss': contrastive_loss,
        }
    
    def compute_metrics(self, all_preds, all_labels, all_probs, prefix=''):
        """
        Compute evaluation metrics for multi-label classification.
        
        Args:
            all_preds: (N, C) binary predictions
            all_labels: (N, C) ground truth labels
            all_probs: (N, C) predicted probabilities
            prefix: Metric name prefix (e.g., 'train_', 'val_')
        """
        metrics = {}
        
        # Overall metrics
        metrics[f'{prefix}accuracy'] = accuracy_score(all_labels.flatten(), all_preds.flatten())
        
        # Per-class metrics (macro average)
        metrics[f'{prefix}precision_macro'] = precision_score(
            all_labels, all_preds, average='macro', zero_division=0
        )
        metrics[f'{prefix}recall_macro'] = recall_score(
            all_labels, all_preds, average='macro', zero_division=0
        )
        metrics[f'{prefix}f1_macro'] = f1_score(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # AUC metrics
        try:
            metrics[f'{prefix}auc_macro'] = roc_auc_score(
                all_labels, all_probs, average='macro', multi_class='ovr'
            )
            metrics[f'{prefix}auc_weighted'] = roc_auc_score(
                all_labels, all_probs, average='weighted', multi_class='ovr'
            )
        except ValueError:
            # Not enough classes
            metrics[f'{prefix}auc_macro'] = 0.0
            metrics[f'{prefix}auc_weighted'] = 0.0
        
        # AP (Average Precision)
        try:
            metrics[f'{prefix}ap_macro'] = average_precision_score(
                all_labels, all_probs, average='macro'
            )
        except ValueError:
            metrics[f'{prefix}ap_macro'] = 0.0
        
        # Per-disease metrics
        for i, disease_name in enumerate(self.disease_names[:all_labels.shape[1]]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                metrics[f'{prefix}auc_{disease_name}'] = auc
            except ValueError:
                pass
        
        return metrics
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        # Set epoch for distributed sampler
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_contrastive_loss = 0.0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Only show progress bar on main process
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            disease_labels = batch['disease_labels'].to(self.device)
            concept_labels = batch['concept_labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, return_all=True)
            
            # Compute loss
            losses = self.compute_loss(outputs, disease_labels, concept_labels)
            loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_class_loss += losses['class_loss'].item()
            total_contrastive_loss += losses['contrastive_loss'].item()
            
            # Accumulate predictions
            with torch.no_grad():
                class_logits = outputs['class_logits']
                probs = torch.sigmoid(class_logits)
                preds = (probs > 0.5).float()
                
                # Get target
                if self.config['num_classes'] == 1:
                    target = disease_labels[:, 0:1]
                else:
                    target = disease_labels[:, :self.config['num_classes']]
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(target.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
            
            # Update progress bar (only on main process)
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls': f'{losses["class_loss"].item():.4f}',
                    'con': f'{losses["contrastive_loss"].item():.4f}',
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = total_class_loss / len(self.train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.train_loader)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        metrics = self.compute_metrics(all_preds, all_labels, all_probs, prefix='train_')
        metrics['train_loss'] = avg_loss
        metrics['train_class_loss'] = avg_class_loss
        metrics['train_contrastive_loss'] = avg_contrastive_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_contrastive_loss = 0.0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Only show progress bar on main process
        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc='Validation')
        else:
            pbar = self.val_loader
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            disease_labels = batch['disease_labels'].to(self.device)
            concept_labels = batch['concept_labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, return_all=True)
            
            # Compute loss
            losses = self.compute_loss(outputs, disease_labels, concept_labels)
            
            total_loss += losses['total_loss'].item()
            total_class_loss += losses['class_loss'].item()
            total_contrastive_loss += losses['contrastive_loss'].item()
            
            # Accumulate predictions
            class_logits = outputs['class_logits']
            probs = torch.sigmoid(class_logits)
            preds = (probs > 0.5).float()
            
            if self.config['num_classes'] == 1:
                target = disease_labels[:, 0:1]
            else:
                target = disease_labels[:, :self.config['num_classes']]
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_class_loss = total_class_loss / len(self.val_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.val_loader)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        metrics = self.compute_metrics(all_preds, all_labels, all_probs, prefix='val_')
        metrics['val_loss'] = avg_loss
        metrics['val_class_loss'] = avg_class_loss
        metrics['val_contrastive_loss'] = avg_contrastive_loss
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return
        
        # Get model state dict without DDP wrapper
        model_state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'concept_names': self.concept_names,
            'disease_names': self.disease_names,
        }
        
        # Save latest
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Saved best model (AUC: {metrics['val_auc_macro']:.4f})")
        
        # Save periodic
        if epoch % self.config.get('save_freq', 10) == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop."""
        if self.is_main_process:
            print("\n" + "=" * 80)
            print("Starting Training")
            print("=" * 80)
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics (only on main process)
            if self.is_main_process:
                all_metrics = {**train_metrics, **val_metrics}
                for key, value in all_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(key, value, epoch)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('lr', current_lr, epoch)
                
                # Print summary
                print(f"\nEpoch {epoch}/{self.config['epochs']}:")
                print(f"  Train - Loss: {train_metrics['train_loss']:.4f}, "
                      f"AUC: {train_metrics.get('train_auc_macro', 0):.4f}, "
                      f"F1: {train_metrics.get('train_f1_macro', 0):.4f}")
                print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, "
                      f"AUC: {val_metrics.get('val_auc_macro', 0):.4f}, "
                      f"F1: {val_metrics.get('val_f1_macro', 0):.4f}")
                print(f"  LR: {current_lr:.6f}")
                
                # Save checkpoint
                is_best = val_metrics.get('val_auc_macro', 0) > self.best_val_auc
                if is_best:
                    self.best_val_auc = val_metrics.get('val_auc_macro', 0)
                    self.best_epoch = epoch
                
                self.save_checkpoint(epoch, all_metrics, is_best=is_best)
        
        if self.is_main_process:
            print("\n" + "=" * 80)
            print("Training Completed!")
            print(f"Best Val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
            print("=" * 80)
            
            self.writer.close()
        
        # Cleanup distributed
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train Medical Concept Model')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of VinDr-CXR dataset')
    parser.add_argument('--image_size', type=int, default=448,
                        help='Input image size')
    
    # Model
    parser.add_argument('--backbone_name', type=str,
                        default='aysangh/medsiglip-448-vindr-bin',
                        help='Backbone model name')
    parser.add_argument('--num_concepts', type=int, default=12,
                        help='Number of medical concepts')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of disease classes')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection dimension')
    parser.add_argument('--prototypes_per_concept', type=int, default=4,
                        help='Number of prototypes per concept')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights')
    
    # Contrastive loss
    parser.add_argument('--contrastive_lambda', type=float, default=10.0,
                        help='Contrastive loss lambda')
    parser.add_argument('--contrastive_gamma', type=float, default=10.0,
                        help='Contrastive loss gamma')
    parser.add_argument('--contrastive_delta', type=float, default=0.1,
                        help='Contrastive loss delta')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--backbone_lr_multiplier', type=float, default=0.1,
                        help='Backbone LR multiplier')
    
    # Loss weights
    parser.add_argument('--class_loss_weight', type=float, default=1.0,
                        help='Classification loss weight')
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.5,
                        help='Contrastive loss weight')
    
    # Data options
    parser.add_argument('--radiologist_id', type=str, default=None,
                        help='Filter by radiologist ID (e.g., R3)')
    parser.add_argument('--use_multi_rater', action='store_true', default=True,
                        help='Use multi-rater aggregation')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Distributed training (automatically set by torchrun)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (set by torchrun)')
    
    args = parser.parse_args()
    
    # Check if running with torchrun (distributed)
    args.distributed = (int(os.environ.get('WORLD_SIZE', 1)) > 1)
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'exp_{timestamp}'
    
    # Convert args to config dict
    config = vars(args)
    
    # Print config (only on main process)
    if not config['distributed'] or int(os.environ.get('RANK', 0)) == 0:
        print("\n" + "=" * 80)
        print("Configuration:")
        print("=" * 80)
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")
    
    # Create trainer and start training
    trainer = ModelTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
