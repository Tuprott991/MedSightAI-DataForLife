"""
Training Stage 1 with Bounding Box Supervision.

This significantly improves CAM localization quality by teaching the model
exactly where each concept is located, not just that it exists.

Usage:
    # Single GPU
    python train_stage1_bbox.py --bbox_csv data/annotations.csv
    
    # Multi-GPU DDP
    torchrun --nproc_per_node=2 train_stage1_bbox.py --bbox_csv data/annotations.csv
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from src.model import CSR
from src.dataloader_bbox import get_dataloaders_with_bbox
from utils_bbox import BBoxGuidedConceptLoss


def setup_ddp():
    """Initialize DDP training environment."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch_bbox(model, loader, optimizer, criterion, scaler, device, rank):
    """Train one epoch with bbox supervision."""
    model.train()
    total_loss = 0
    
    if rank == 0:
        loop = tqdm(loader, desc="Training Stage 1 (BBox)")
    else:
        loop = loader
    
    for batch in loop:
        images = batch['image'].to(device)
        concepts_gt = batch['concepts'].to(device)
        bboxes = batch['bboxes']  # List of bbox lists (not moved to GPU)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(images)
            cams = outputs['cams']
            
            # Loss with bbox guidance
            loss = criterion(cams, concepts_gt, bboxes)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        if rank == 0:
            loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)


def validate_bbox(model, loader, criterion, device, rank):
    """Validate with bbox-aware criterion."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(loader, desc="Validating")
        else:
            pbar = loader
        
        for batch in pbar:
            images = batch['image'].to(device)
            concepts = batch['concepts'].to(device)
            bboxes = batch['bboxes']
            
            outputs = model(images)
            cams = outputs['cams']
            
            # Loss
            loss = criterion(cams, concepts, bboxes)
            total_loss += loss.item()
            
            # AUC computation
            concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)
            all_preds.append(torch.sigmoid(concept_logits).cpu())
            all_targets.append(concepts.cpu())
    
    # Compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        
        valid_classes = (targets.sum(axis=0) > 0) & (targets.sum(axis=0) < len(targets))
        if valid_classes.sum() > 0:
            auc = roc_auc_score(targets[:, valid_classes], preds[:, valid_classes], average='macro')
        else:
            auc = 0.0
        
        return total_loss / len(loader), auc
    except Exception as e:
        if rank == 0:
            print(f"Warning: Could not compute AUC - {e}")
        return total_loss / len(loader), 0.0


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--train_csv', type=str, default='data/train.csv')
    parser.add_argument('--test_csv', type=str, default='data/test.csv')
    parser.add_argument('--bbox_csv', type=str, required=True, help='CSV with bbox annotations')
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--test_dir', type=str, default='data/test')
    
    # Model
    parser.add_argument('--backbone', type=str, default='medmae')
    parser.add_argument('--medmae_weights', type=str, default='weights/pre_trained_medmae.pth')
    parser.add_argument('--num_concepts', type=int, default=22)
    parser.add_argument('--num_classes', type=int, default=6)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--beta', type=float, default=0.5, help='Localization loss weight')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs_bbox')
    
    args = parser.parse_args()
    
    # Setup DDP
    rank = setup_ddp() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print("=" * 60)
        print("STAGE 1: CONCEPT LEARNING WITH BBOX SUPERVISION")
        print("=" * 60)
        print(f"🎯 BBox annotations: {args.bbox_csv}")
        print(f"📦 Batch size: {args.batch_size} per GPU")
        print(f"🎓 Learning rate: {args.lr}")
        print(f"⚖️  Loss weights: α={args.alpha} (classification), β={args.beta} (localization)")
    
    # Load data
    train_loader, val_loader, test_loader, num_concepts, num_classes, train_sampler = \
        get_dataloaders_with_bbox(
            args.train_csv, args.test_csv, args.bbox_csv,
            args.train_dir, args.test_dir,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            val_split=0.1
        )
    
    if rank == 0:
        print(f"Data loaded: {num_concepts} Concepts, {num_classes} Diseases")
    
    # Create model
    model = CSR(
        num_concepts=num_concepts,
        num_classes=num_classes,
        num_prototypes_per_concept=1,
        backbone_type=args.backbone,
        model_name=args.medmae_weights
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Optimizer
    actual_model = model.module if hasattr(model, 'module') else model
    optimizer = optim.AdamW([
        {'params': actual_model.backbone.parameters(), 'lr': args.lr * 0.01},
        {'params': actual_model.concept_head.parameters(), 'lr': args.lr * 0.1}
    ])
    
    # Criterion with bbox supervision
    criterion = BBoxGuidedConceptLoss(alpha=args.alpha, beta=args.beta)
    
    # Mixed precision
    scaler = GradScaler('cuda')
    
    # Training
    best_auc = 0.0
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_one_epoch_bbox(model, train_loader, optimizer, criterion, 
                                         scaler, device, rank)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss {train_loss:.4f}")
        
        # Validate
        if rank == 0:
            actual_model = model.module if hasattr(model, 'module') else model
            val_loss, val_auc = validate_bbox(actual_model, val_loader, criterion, device, rank)
            print(f"  Val Loss {val_loss:.4f}, Concept AUC {val_auc:.4f}")
            
            # Save best
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(actual_model.state_dict(), output_dir / 'best_model_stage1_bbox.pth')
                print(f"  ✅ Saved best model (AUC: {best_auc:.4f})")
    
    if rank == 0:
        print(f"\n🎉 Stage 1 Complete! Best Concept AUC: {best_auc:.4f}")
        print(f"Model saved to: {output_dir / 'best_model_stage1_bbox.pth'}")
        print(f"\n💡 Next: Visualize CAMs to verify localization quality:")
        print(f"  python visualize_cams.py --checkpoint {output_dir / 'best_model_stage1_bbox.pth'} --image test.png")
    
    cleanup_ddp()


if __name__ == '__main__':
    main()
