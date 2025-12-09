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

# Import model và utils của bạn
from src.model import CSR 
from src.dataloader import get_dataloaders
from utils import PrototypeContrastiveLoss
from utils_bbox import BBoxGuidedConceptLoss
import pandas as pd
import numpy as np

def setup_ddp():
    """Initialize DDP training environment."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Cleanup DDP training environment."""
    dist.destroy_process_group()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='CSR Model Training with DDP')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, default='labels_train.csv', help='Path to training CSV')
    parser.add_argument('--test_csv', type=str, default='labels_test.csv', help='Path to test CSV')
    parser.add_argument('--train_bbox_csv', type=str, default=None, help='Path to train bbox annotations CSV (optional for Stage 1)')
    parser.add_argument('--test_bbox_csv', type=str, default=None, help='Path to test bbox annotations CSV (optional for Stage 1)')
    parser.add_argument('--train_resize_factor_csv', type=str, default=None, help='Path to train resize factor CSV (required when using bbox)')
    parser.add_argument('--test_resize_factor_csv', type=str, default=None, help='Path to test resize factor CSV (required when using bbox)')
    parser.add_argument('--train_dir', type=str, default='train/', help='Path to training images')
    parser.add_argument('--test_dir', type=str, default='test/', help='Path to test images')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs_stage1', type=int, default=10, help='Epochs for Stage 1')
    parser.add_argument('--epochs_stage2', type=int, default=10, help='Epochs for Stage 2')
    parser.add_argument('--epochs_stage3', type=int, default=10, help='Epochs for Stage 3')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for classification loss (Stage 1 with bbox)')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for localization loss (Stage 1 with bbox)')
    
    # Model configuration
    parser.add_argument('--backbone_type', type=str, default='medmae', choices=['medmae', 'resnet50', 'vit'], help='Backbone type')
    parser.add_argument('--model_name', type=str, default='facebook/vit-mae-base', help='Pretrained model name or path')
    parser.add_argument('--num_prototypes', type=int, default=5, help='Number of prototypes per concept (M)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--exp_name', type=str, default='csr_exp', help='Experiment name')
    
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, criterion, stage, scaler, device, rank):
    model.train()
    total_loss = 0
    
    # Progress bar (only show on rank 0)
    if rank == 0:
        loop = tqdm(loader, desc=f"Stage {stage}")
    else:
        loop = loader
    
    for batch in loop:
        images = batch['image'].to(device)
        concepts_gt = batch['concepts'].to(device) # Nhãn concept (Ground Truth)
        targets_gt = batch['targets'].to(device)   # Nhãn bệnh (Ground Truth)
        
        optimizer.zero_grad()
        
        # Dùng Mixed Precision để train nhanh và nhẹ hơn
        with autocast('cuda'):
            outputs = model(images)
            loss = 0
            
            # --- GIAI ĐOẠN 1: Train Concept Model ---
            if stage == 1:
                # CAMs từ concept_head: (B, K, H, W)
                cams = outputs['cams']  # (B, K, H, W)
                
                # Check if using BBoxGuidedConceptLoss (needs spatial CAMs + bboxes)
                if hasattr(criterion, 'alpha') and hasattr(criterion, 'beta'):
                    # BBoxGuidedConceptLoss: Pass raw CAMs (does max pooling internally)
                    # Also need to pass bboxes if available
                    bboxes = batch.get('bboxes', None)
                    loss = criterion(cams, concepts_gt, bboxes)
                else:
                    # Standard BCEWithLogitsLoss: Need to max pool first
                    concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)  # (B, K)
                    loss = criterion(concept_logits, concepts_gt)
            
            # --- GIAI ĐOẠN 2: Train Prototypes (Contrastive) ---
            elif stage == 2:
                # DDP wrap: Phải dùng .module để truy cập methods của model gốc
                actual_model = model.module if hasattr(model, 'module') else model
                
                # Use projected features from forward pass: (B, 128, H, W)
                f_proj = outputs['projected_features']
                
                # Get local concept vectors in PROJECTED space
                # Apply CAM-weighted pooling to projected features
                local_vectors = actual_model.get_local_concept_vectors(f_proj, outputs['cams'])
                # (B, K, 128) - Already in projection space
                
                # Tính Loss Contrastive
                # Chỉ tính loss cho những concept có trong ảnh (concepts_gt == 1)
                loss = criterion(local_vectors, actual_model.prototypes, concepts_gt, 
                                 num_prototypes_per_concept=actual_model.M)
                
            # --- GIAI ĐOẠN 3: Train Task Head (Disease Prediction) ---
            elif stage == 3:
                # Logits cuối cùng dự đoán bệnh
                preds = outputs['logits']
                # Target là multi-label thì dùng BCE, single-label dùng CrossEntropy
                # Ở đây giả sử target cũng là multi-label (vì 1 người có thể vừa viêm phổi vừa COPD)
                loss = criterion(preds, targets_gt)
        
        # Backward & Optimizer
        scaler.scale(loss).backward()
        
        # Gradient clipping để tránh exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        if rank == 0:
            loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def compute_metrics(preds, targets, threshold=0.5):
    """Compute comprehensive metrics for multi-label classification."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    
    metrics = {}
    
    # Valid classes (have both positive and negative samples)
    valid_classes = (targets.sum(axis=0) > 0) & (targets.sum(axis=0) < len(targets))
    
    if valid_classes.sum() > 0:
        # AUC-ROC
        try:
            auc = roc_auc_score(targets[:, valid_classes], preds[:, valid_classes], average='macro')
            metrics['auc'] = auc
        except:
            metrics['auc'] = 0.0
        
        # mAP (mean Average Precision)
        try:
            map_score = average_precision_score(targets[:, valid_classes], preds[:, valid_classes], average='macro')
            metrics['mAP'] = map_score
        except:
            metrics['mAP'] = 0.0
        
        # F1 Score (with threshold)
        try:
            preds_binary = (preds[:, valid_classes] > threshold).astype(int)
            f1_macro = f1_score(targets[:, valid_classes], preds_binary, average='macro', zero_division=0)
            f1_micro = f1_score(targets[:, valid_classes], preds_binary, average='micro', zero_division=0)
            metrics['f1_macro'] = f1_macro
            metrics['f1_micro'] = f1_micro
        except:
            metrics['f1_macro'] = 0.0
            metrics['f1_micro'] = 0.0
    else:
        metrics = {'auc': 0.0, 'mAP': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0}
    
    return metrics

def compute_iou_score(cams, bboxes, threshold=0.5):
    """Compute IoU between CAM heatmaps and ground truth bounding boxes.
    
    Args:
        cams: (B, K, H, W) concept activation maps
        bboxes: List of bbox lists for each sample
        threshold: Threshold to binarize CAMs
    
    Returns:
        mean_iou: Average IoU across all concepts with bboxes
    """
    if not bboxes or len(bboxes) == 0:
        return 0.0
    
    ious = []
    B, K, H, W = cams.shape
    
    # Apply sigmoid and threshold to get binary masks
    cam_masks = (torch.sigmoid(cams) > threshold).float()
    
    for batch_idx, sample_bboxes in enumerate(bboxes):
        if sample_bboxes is None or len(sample_bboxes) == 0:
            continue
        
        for bbox_info in sample_bboxes:
            concept_idx = bbox_info['concept_idx']
            bbox = bbox_info['bbox']  # [x_min, y_min, x_max, y_max] normalized [0, 1]
            
            # Get CAM mask for this concept
            cam_mask = cam_masks[batch_idx, concept_idx]  # (H, W)
            
            # Create bbox mask
            x_min, y_min, x_max, y_max = bbox
            x_min_px = int(x_min * W)
            x_max_px = int(x_max * W)
            y_min_px = int(y_min * H)
            y_max_px = int(y_max * H)
            
            bbox_mask = torch.zeros_like(cam_mask)
            bbox_mask[y_min_px:y_max_px, x_min_px:x_max_px] = 1.0
            
            # Compute IoU
            intersection = (cam_mask * bbox_mask).sum()
            union = ((cam_mask + bbox_mask) > 0).float().sum()
            
            if union > 0:
                iou = (intersection / union).item()
                ious.append(iou)
    
    return np.mean(ious) if len(ious) > 0 else 0.0

def validate(model, loader, device, rank=0, stage=3, criterion=None, compute_iou=False):
    """Validation function with comprehensive metrics.
    
    Args:
        stage: 1 = Concept Learning, 3 = Task Learning
        criterion: Loss function
        compute_iou: Whether to compute IoU with bboxes (Stage 1 only)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_cams = []
    all_bboxes = []
    
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(loader, desc="Validating")
        else:
            pbar = loader
        
        for batch in pbar:
            images = batch['image'].to(device)
            concepts = batch['concepts'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(images)
            
            # Stage 1: Validate concept prediction
            if stage == 1:
                cams = outputs['cams']
                concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)
                
                # Check if using BBoxGuidedConceptLoss
                if hasattr(criterion, 'alpha') and hasattr(criterion, 'beta'):
                    # BBoxGuidedConceptLoss: Pass raw CAMs
                    bboxes = batch.get('bboxes', None)
                    loss = criterion(cams, concepts, bboxes)
                else:
                    # Standard BCE: Use max-pooled logits
                    loss = criterion(concept_logits, concepts)
                    bboxes = batch.get('bboxes', [])
                
                all_preds.append(torch.sigmoid(concept_logits).cpu())
                all_targets.append(concepts.cpu())
                
                if compute_iou and 'bboxes' in batch:
                    all_cams.append(cams.cpu())
                    all_bboxes.extend(batch['bboxes'])
            
            # Stage 3: Validate disease prediction
            else:
                loss = criterion(outputs['logits'], targets)
                all_preds.append(torch.sigmoid(outputs['logits']).cpu())
                all_targets.append(targets.cpu())
            
            total_loss += loss.item()
    
    # Compute metrics
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    
    metrics = compute_metrics(preds, targets)
    metrics['loss'] = total_loss / len(loader)
    
    # Compute IoU if requested
    if compute_iou and len(all_cams) > 0 and stage == 1:
        all_cams_tensor = torch.cat(all_cams)
        iou = compute_iou_score(all_cams_tensor, all_bboxes)
        metrics['iou'] = iou
    
    return metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup DDP
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create output directory
    if rank == 0:
        output_dir = Path(args.output_dir) / args.exp_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # 1. Prepare Data
    # Use bbox dataloader if bbox_csv is provided
    if args.train_bbox_csv and args.test_bbox_csv:
        from src.dataloader_bbox import get_dataloaders_with_bbox
        
        # Validate resize_factor_csv files are provided
        if not args.train_resize_factor_csv or not args.test_resize_factor_csv:
            raise ValueError("Both --train_resize_factor_csv and --test_resize_factor_csv are required when using bbox")
        
        if rank == 0:
            print(f"Loading data WITH bounding boxes:")
            print(f"  - Train bbox: {args.train_bbox_csv}")
            print(f"  - Test bbox: {args.test_bbox_csv}")
            print(f"  - Train resize factors: {args.train_resize_factor_csv}")
            print(f"  - Test resize factors: {args.test_resize_factor_csv}")
        
        train_loader, val_loader, test_loader, num_concepts, num_classes, train_sampler = get_dataloaders_with_bbox(
            args.train_csv, args.test_csv, args.train_bbox_csv, args.test_bbox_csv,
            args.train_resize_factor_csv, args.test_resize_factor_csv, args.train_dir, args.test_dir,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            val_split=0.1  # 10% của training set làm validation
        )
    elif args.train_bbox_csv or args.test_bbox_csv:
        raise ValueError("Both --train_bbox_csv and --test_bbox_csv must be provided together")
    else:
        if rank == 0:
            print("Loading data WITHOUT bounding boxes (standard dataloader)")
        train_loader, val_loader, test_loader, num_concepts, num_classes, train_sampler = get_dataloaders(
            args.train_csv, args.test_csv, args.train_dir, args.test_dir, 
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            val_split=0.1  # 10% của training set làm validation
        )
    
    if rank == 0:
        print(f"Data Loaded: {num_concepts} Concepts, {num_classes} Diseases")
        print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")
        print(f"World Size: {world_size}, Batch Size per GPU: {args.batch_size}")
        if world_size > 1:
            print(f"✅ DDP enabled: Each GPU trains on {len(train_loader.dataset) // world_size} samples")

    # 2. Model Init (Dùng MedMAE backbone)
    model = CSR(
        num_concepts=num_concepts, 
        num_classes=num_classes,
        num_prototypes_per_concept=args.num_prototypes,
        backbone_type=args.backbone_type,
        model_name=args.model_name
    ).to(device)
    
    # Wrap with DDP
    # find_unused_parameters=True vì các stage khác nhau dùng các parameters khác nhau
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # PyTorch 2.6: torch.amp.GradScaler thay vì torch.cuda.amp.GradScaler
    from torch.amp import GradScaler
    scaler = GradScaler('cuda') # Cho AMP
    
    # Best model tracking
    best_auc = 0.0
    best_stage = None

    # ====================================================
    # GIAI ĐOẠN 1: Concept Learning
    # Mục tiêu: Học CAMs chuẩn. Train Backbone + Concept Head
    # ====================================================
    if rank == 0:
        print("\n--- START STAGE 1: Concept Learning ---")
    
    # Tính pos_weight để xử lý class imbalance
    # Tính trực tiếp từ CSV thay vì load ảnh (nhanh hơn 100x)
    if rank == 0:
        print("Computing pos_weight for balanced BCE loss...")
        import pandas as pd
        df_train = pd.read_csv(args.train_csv)
        
        # Lấy concept columns (giống logic trong dataloader)
        meta_cols = ['image_id', 'rad_id']
        target_keywords = ['COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'No finding']
        target_cols = []
        for col in df_train.columns:
            if col in meta_cols:
                continue
            if 'other' in col.lower():
                if 'disease' in col.lower():
                    target_cols.append(col)
                continue
            if any(keyword.lower() in col.lower() for keyword in target_keywords):
                target_cols.append(col)
        concept_cols = [c for c in df_train.columns if c not in target_cols + meta_cols]
        
        # Aggregate như trong dataset
        df_agg = df_train.groupby('image_id')[concept_cols + target_cols].max().reset_index()
        
        # Tính pos_weight từ concept columns
        concept_values = torch.tensor(df_agg[concept_cols].values, dtype=torch.float32)
        pos_weight = (concept_values == 0).sum(dim=0) / (concept_values == 1).sum(dim=0).clamp(min=1)
        # Clip pos_weight để tránh loss quá extreme (max 10x)
        pos_weight = pos_weight.clamp(max=10.0)
        print(f"Pos weights range: {pos_weight.min():.2f} - {pos_weight.max():.2f} (clipped to max 10)")
    else:
        # Các rank khác dùng uniform weights (sẽ được broadcast từ rank 0)
        pos_weight = torch.ones(num_concepts)
    
    # Stage 1: Learning rate thấp hơn để ổn định, nhất là khi dùng pos_weight
    optimizer = optim.AdamW([
        {'params': model.module.backbone.parameters(), 'lr': args.lr * 0.01}, # Backbone học rất chậm
        {'params': model.module.concept_head.parameters(), 'lr': args.lr * 0.1}  # Concept head cũng thận trọng
    ])
    
    # Use BBoxGuidedConceptLoss if bboxes available, else standard BCE
    if args.train_bbox_csv and args.test_bbox_csv:
        if rank == 0:
            print(f"Using BBoxGuidedConceptLoss (alpha={args.alpha}, beta={args.beta}) with pos_weight")
        criterion_s1 = BBoxGuidedConceptLoss(
            alpha=args.alpha, 
            beta=args.beta, 
            pos_weight=pos_weight.to(device)
        )
    else:
        if rank == 0:
            print(f"Using BCEWithLogitsLoss with pos_weight")
        criterion_s1 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    for epoch in range(args.epochs_stage1):
        # Set epoch for DistributedSampler to shuffle differently each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        loss = train_one_epoch(model, train_loader, optimizer, criterion_s1, stage=1, scaler=scaler, device=device, rank=rank)
        if rank == 0:
            print(f"Epoch {epoch+1}: Train Loss {loss:.4f}")
        
        # Validate (only on rank 0)
        if rank == 0:
            metrics = validate(model.module, val_loader, device, rank, stage=1, criterion=criterion_s1)
            # Show comprehensive metrics for Stage 1
            print(f"Epoch {epoch+1}: Val Loss {metrics['loss']:.4f}, AUC {metrics['auc']:.4f}, "
                  f"mAP {metrics['mAP']:.4f}, F1-macro {metrics['f1_macro']:.4f}, F1-micro {metrics['f1_micro']:.4f}", end="")
            if 'iou' in metrics:
                print(f", IoU {metrics['iou']:.4f}")
            else:
                print()  # newline
            
            # Save best model (based on AUC)
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_stage = 'stage1'
                torch.save(model.module.state_dict(), output_dir / 'best_model_stage1.pth')
                print(f"✅ Saved best Stage 1 model (Concept AUC: {best_auc:.4f}, mAP: {metrics['mAP']:.4f})")

    # ====================================================
    # GIAI ĐOẠN 2: Prototype Learning
    # Mục tiêu: Học Prototypes chuẩn. Freeze Backbone.
    # ====================================================
    if rank == 0:
        print("\n--- START STAGE 2: Prototype Learning ---")
    # Freeze Backbone & Concept Head
    for param in model.module.backbone.parameters(): param.requires_grad = False
    for param in model.module.concept_head.parameters(): param.requires_grad = False
    
    # Chỉ train Projector và Prototypes
    optimizer = optim.AdamW([
        {'params': model.module.projector.parameters(), 'lr': args.lr},
        {'params': model.module.prototypes, 'lr': args.lr}
    ])
    criterion_s2 = PrototypeContrastiveLoss(temperature=0.1) # Custom Loss
    
    for epoch in range(args.epochs_stage2):
        # Set epoch for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        loss = train_one_epoch(model, train_loader, optimizer, criterion_s2, stage=2, scaler=scaler, device=device, rank=rank)
        if rank == 0:
            print(f"Epoch {epoch+1}: Loss {loss:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == args.epochs_stage2 - 1:
                torch.save(model.module.state_dict(), output_dir / f'model_stage2_epoch{epoch+1}.pth')
        
    # ====================================================
    # GIAI ĐOẠN 3: Task Learning
    # Mục tiêu: Dự đoán bệnh. Freeze tất cả trừ Task Head.
    # ====================================================
    if rank == 0:
        print("\n--- START STAGE 3: Task Learning ---")
    
    # Freeze ALL previous layers (Backbone, Concept Head, Projector, Prototypes)
    # Only Task Head should be trainable
    for param in model.module.backbone.parameters(): param.requires_grad = False
    for param in model.module.concept_head.parameters(): param.requires_grad = False
    for param in model.module.projector.parameters(): param.requires_grad = False
    model.module.prototypes.requires_grad = False
    
    # Chỉ train Task Head
    optimizer = optim.AdamW(model.module.task_head.parameters(), lr=args.lr)
    criterion_s3 = torch.nn.BCEWithLogitsLoss() # Nếu target là multi-label
    
    for epoch in range(args.epochs_stage3):
        # Set epoch for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        loss = train_one_epoch(model, train_loader, optimizer, criterion_s3, stage=3, scaler=scaler, device=device, rank=rank)
        if rank == 0:
            print(f"Epoch {epoch+1}: Train Loss {loss:.4f}")
        
        # Validate (only on rank 0)
        if rank == 0:
            metrics = validate(model.module, val_loader, device, rank, stage=3, criterion=criterion_s3)
            # Show comprehensive metrics for Stage 3
            print(f"Epoch {epoch+1}: Val Loss {metrics['loss']:.4f}, Disease AUC {metrics['auc']:.4f}, "
                  f"mAP {metrics['mAP']:.4f}, F1-macro {metrics['f1_macro']:.4f}, F1-micro {metrics['f1_micro']:.4f}")
            
            # Save best model (based on AUC)
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_stage = 'stage3'
                torch.save(model.module.state_dict(), output_dir / 'best_model_stage3.pth')
                print(f"✅ Saved best Stage 3 model (Disease AUC: {best_auc:.4f}, mAP: {metrics['mAP']:.4f})")
    
    # ====================================================
    # FINAL TEST: Đánh giá trên test set sau khi train xong hết
    # ====================================================
    if rank == 0:
        print("\n" + "="*60)
        print("FINAL EVALUATION ON TEST SET")
        print("="*60)
        
        # Load best model
        best_model_path = output_dir / 'best_model_stage3.pth'
        if best_model_path.exists():
            print(f"Loading best model from {best_model_path}")
            model.module.load_state_dict(torch.load(best_model_path))
        
        # Test (dùng BCE loss không pos_weight cho fair comparison)
        test_criterion = torch.nn.BCEWithLogitsLoss()
        test_metrics = validate(model.module, test_loader, device, rank, stage=3, criterion=test_criterion)
        print(f"\n🏆 Final Test Results (Disease Prediction):")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test AUC (macro): {test_metrics['auc']:.4f}")
        print(f"  Test mAP: {test_metrics['mAP']:.4f}")
        print(f"  Test F1-macro: {test_metrics['f1_macro']:.4f}")
        print(f"  Test F1-micro: {test_metrics['f1_micro']:.4f}")
        print(f"  {'Excellent!' if test_metrics['auc'] >= 0.85 else 'Very Good!' if test_metrics['auc'] >= 0.80 else 'Good!' if test_metrics['auc'] >= 0.75 else 'Fair' if test_metrics['auc'] >= 0.70 else 'Needs Improvement'}")
        print(f"\n🎉 Training Complete!")
        print(f"Best Val AUC: {best_auc:.4f} from {best_stage}")
        print(f"Final Test Disease AUC: {test_metrics['auc']:.4f}, mAP: {test_metrics['mAP']:.4f}")
        print(f"\nBenchmark: VinDr-CXR ResNet-50 ~0.78 | DenseNet-121 ~0.82 | SOTA ~0.87")
        print(f"All checkpoints saved to: {output_dir}")
    
    # Save Final Model
    if rank == 0:
        torch.save(model.module.state_dict(), output_dir / "final_model.pth")
    
    # Cleanup DDP
    cleanup_ddp()

if __name__ == "__main__":
    main()