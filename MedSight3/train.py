import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import os
from pathlib import Path

# Import model và utils của bạn
from src.model import CSR 
from src.dataloader import get_dataloaders
from utils import PrototypeContrastiveLoss

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
    parser.add_argument('--train_dir', type=str, default='train/', help='Path to training images')
    parser.add_argument('--test_dir', type=str, default='test/', help='Path to test images')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs_stage1', type=int, default=10, help='Epochs for Stage 1')
    parser.add_argument('--epochs_stage2', type=int, default=10, help='Epochs for Stage 2')
    parser.add_argument('--epochs_stage3', type=int, default=10, help='Epochs for Stage 3')
    
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
        with autocast():
            outputs = model(images)
            loss = 0
            
            # --- GIAI ĐOẠN 1: Train Concept Model ---
            if stage == 1:
                # Tính Global Average Pooling trên CAMs để so sánh với nhãn concept
                # cams shape: (B, K, H, W) -> mean(dim=(2,3)) -> (B, K)
                cams = outputs['cams']
                concept_logits = cams.mean(dim=(2, 3)) 
                loss = criterion(concept_logits, concepts_gt)
            
            # --- GIAI ĐOẠN 2: Train Prototypes (Contrastive) ---
            elif stage == 2:
                # Lấy vector cục bộ: (B, K, Dim)
                local_vectors = model.get_local_concept_vectors(outputs['features'], outputs['cams'])
                # Chiếu qua Projector: (B, K, Projection_Dim)
                projected = model.projector(local_vectors.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
                
                # Tính Loss Contrastive
                # Chỉ tính loss cho những concept có trong ảnh (concepts_gt == 1)
                loss = criterion(projected, model.prototypes, concepts_gt, 
                                 num_prototypes_per_concept=model.M)
                
            # --- GIAI ĐOẠN 3: Train Task Head (Disease Prediction) ---
            elif stage == 3:
                # Logits cuối cùng dự đoán bệnh
                preds = outputs['logits']
                # Target là multi-label thì dùng BCE, single-label dùng CrossEntropy
                # Ở đây giả sử target cũng là multi-label (vì 1 người có thể vừa viêm phổi vừa COPD)
                loss = criterion(preds, targets_gt)
        
        # Backward & Optimizer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        if rank == 0:
            loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def validate(model, loader, device, rank=0):
    """Validation function để đánh giá model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(loader, desc="Validating")
        else:
            pbar = loader
        
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs['logits'], targets)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs['logits']).cpu())
            all_targets.append(targets.cpu())
    
    # Tính AUC
    try:
        from sklearn.metrics import roc_auc_score
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        auc = roc_auc_score(targets, preds, average='macro')
        return total_loss / len(loader), auc
    except:
        return total_loss / len(loader), 0.0

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
    train_loader, test_loader, num_concepts, num_classes = get_dataloaders(
        args.train_csv, args.test_csv, args.train_dir, args.test_dir, 
        batch_size=args.batch_size
    )
    if rank == 0:
        print(f"Data Loaded: {num_concepts} Concepts, {num_classes} Diseases")
        print(f"World Size: {world_size}, Batch Size per GPU: {args.batch_size}")

    # 2. Model Init (Dùng MedMAE backbone)
    model = CSR(
        num_concepts=num_concepts, 
        num_classes=num_classes,
        num_prototypes_per_concept=args.num_prototypes,
        backbone_type=args.backbone_type,
        model_name=args.model_name
    ).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    scaler = GradScaler() # Cho AMP
    
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
    if rank == 0:
        print("Computing pos_weight for balanced BCE loss...")
    train_concepts = []
    for i, batch in enumerate(train_loader):
        train_concepts.append(batch['concepts'])
        if i >= 100:  # Sample 100 batches
            break
    train_concepts = torch.cat(train_concepts, dim=0)
    pos_weight = (train_concepts == 0).sum(dim=0) / (train_concepts == 1).sum(dim=0).clamp(min=1)
    if rank == 0:
        print(f"Pos weights range: {pos_weight.min():.2f} - {pos_weight.max():.2f}")
    
    optimizer = optim.AdamW([
        {'params': model.module.backbone.parameters(), 'lr': args.lr * 0.1}, # Backbone học chậm
        {'params': model.module.concept_head.parameters(), 'lr': args.lr}
    ])
    criterion_s1 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    for epoch in range(args.epochs_stage1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion_s1, stage=1, scaler=scaler, device=device, rank=rank)
        if rank == 0:
            print(f"Epoch {epoch+1}: Train Loss {loss:.4f}")
        
        # Validate (only on rank 0)
        if rank == 0:
            val_loss, val_auc = validate(model.module, test_loader, device, rank)
            print(f"Epoch {epoch+1}: Val Loss {val_loss:.4f}, AUC {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                best_stage = 'stage1'
                torch.save(model.module.state_dict(), output_dir / 'best_model_stage1.pth')
                print(f"✅ Saved best Stage 1 model (AUC: {best_auc:.4f})")

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
    # Freeze Projector & Prototypes
    for param in model.module.projector.parameters(): param.requires_grad = False
    model.module.prototypes.requires_grad = False
    
    # Chỉ train Task Head
    optimizer = optim.AdamW(model.module.task_head.parameters(), lr=args.lr)
    criterion_s3 = torch.nn.BCEWithLogitsLoss() # Nếu target là multi-label
    
    for epoch in range(args.epochs_stage3):
        loss = train_one_epoch(model, train_loader, optimizer, criterion_s3, stage=3, scaler=scaler, device=device, rank=rank)
        if rank == 0:
            print(f"Epoch {epoch+1}: Train Loss {loss:.4f}")
        
        # Validate (only on rank 0)
        if rank == 0:
            val_loss, val_auc = validate(model.module, test_loader, device, rank)
            print(f"Epoch {epoch+1}: Val Loss {val_loss:.4f}, AUC {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                best_stage = 'stage3'
                torch.save(model.module.state_dict(), output_dir / 'best_model_stage3.pth')
                print(f"✅ Saved best Stage 3 model (AUC: {best_auc:.4f})")
    
    # Save Final Model
    if rank == 0:
        torch.save(model.module.state_dict(), output_dir / "final_model.pth")
        print(f"\n🎉 Training Complete!")
        print(f"Best AUC: {best_auc:.4f} from {best_stage}")
        print(f"All checkpoints saved to: {output_dir}")
    
    # Cleanup DDP
    cleanup_ddp()

if __name__ == "__main__":
    main()