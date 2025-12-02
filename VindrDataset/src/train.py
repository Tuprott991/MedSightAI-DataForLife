import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

# --- IMPORTS TỪ SOURCE CỦA BẠN ---
# Đảm bảo model.py đã được cập nhật Class CSRModel như hướng dẫn trước
from src.dataset import VINDRCXRDataset
from src.model import CSRModel 
from src.loss import CSRContrastiveLoss

# ==========================================
# 1. CÁC HÀM TIỆN ÍCH (UTILS)
# ==========================================

def collate_fn_ignore_none(batch):
    """Lọc bỏ các mẫu bị None do lỗi đọc ảnh trước khi tạo batch"""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def compute_contrastive_loss(projected_vecs, prototypes, labels, temperature=0.1):
    """
    Tính Contrastive Loss cho Phase 2 (Simplified InfoNCE)
    - projected_vecs: [Batch, K, Emb_Dim] (v')
    - prototypes: [K, Num_Proto, Emb_Dim] (p)
    - labels: [Batch, K] (Multi-label 0/1)
    """
    B, K, D = projected_vecs.shape
    M = prototypes.shape[1] # Số prototypes per class
    
    total_loss = torch.tensor(0.0, device=projected_vecs.device)
    
    # Chuẩn hóa L2 để tính Cosine Similarity
    prototypes = F.normalize(prototypes, p=2, dim=-1) # [K, M, D]
    projected_vecs = F.normalize(projected_vecs, p=2, dim=-1) # [B, K, D]

    # Duyệt qua từng Concept K (vì Contrastive Loss tính theo từng concept)
    for k in range(K):
        # Lấy các mẫu dương tính với concept k trong batch này
        # labels[:, k] -> [B]
        pos_indices = torch.where(labels[:, k] == 1)[0]
        
        if len(pos_indices) == 0:
            continue # Không có mẫu dương tính nào cho concept này trong batch
            
        # Lấy vector v' của các mẫu dương tính: [N_pos, D]
        anchors = projected_vecs[pos_indices, k, :] 
        
        # Lấy prototypes của concept k (Positive Keys): [M, D]
        pos_protos = prototypes[k, :, :]
        
        # Lấy prototypes của các concept KHÁC k (Negative Keys): [ (K-1)*M, D ]
        # Tạo mask để loại bỏ concept k
        other_protos = torch.cat([prototypes[i] for i in range(K) if i != k], dim=0)
        
        # Tính Similarity: Anchor vs All Prototypes (Pos + Neg)
        # [N_pos, D] x [M, D].T -> [N_pos, M] (Sim với Positive Protos)
        sim_pos = torch.matmul(anchors, pos_protos.T) / temperature
        
        # [N_pos, D] x [All_Neg, D].T -> [N_pos, All_Neg] (Sim với Negative Protos)
        sim_neg = torch.matmul(anchors, other_protos.T) / temperature
        
        # --- InfoNCE Loss ---
        # Tử số: exp(sim_pos). Mẫu số: sum(exp(sim_pos)) + sum(exp(sim_neg))
        # Vì 1 concept có nhiều prototype (M), ta lấy max hoặc mean của sim_pos
        # Ở đây lấy max (prototype gần nhất) để tối ưu
        max_sim_pos, _ = torch.max(sim_pos, dim=1, keepdim=True) # [N_pos, 1]
        
        # LogSumExp trick cho mẫu số
        # Ghép tất cả logits lại: [N_pos, M + All_Neg]
        all_logits = torch.cat([sim_pos, sim_neg], dim=1)
        
        # Loss = -log ( exp(pos) / sum(exp(all)) )
        #      = -pos + log(sum(exp(all)))
        loss_k = -max_sim_pos + torch.logsumexp(all_logits, dim=1, keepdim=True)
        
        total_loss += loss_k.mean()

    return total_loss / K

# ==========================================
# 2. CÁC PHASE TRAIN
# ==========================================

def train_phase_1(model, loader, optimizer, device):
    """
    PHASE 1: CONCEPT LEARNING
    Train: Backbone (F) + Concept Head (C)
    Loss: BCE trên Concept Prediction
    """
    model.train()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    pbar = tqdm(loader, desc="Phase 1 (Concept)")
    for batch in pbar:
        if batch is None: continue
        images, cls_labels, _, _ = batch
        images, cls_labels = images.to(device), cls_labels.to(device)
        
        optimizer.zero_grad()
        
        # Chỉ chạy phần lấy features và CAM
        _, attn_logits = model.get_features_and_cam(images)
        
        # Global Average Pooling lên CAM để ra logits phân loại: [B, K, H, W] -> [B, K]
        concept_preds = F.adaptive_avg_pool2d(attn_logits, (1, 1)).view(images.size(0), -1)
        
        loss = criterion(concept_preds, cls_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    return running_loss / len(loader)

def train_phase_2(model, loader, optimizer, device):
    """
    PHASE 2: PROTOTYPE LEARNING
    Train: Projector (P) + Prototypes (p)
    Freeze: F, C
    Loss: Contrastive Loss
    """
    model.train()
    # Đảm bảo F và C ở chế độ eval (để BatchNorm không cập nhật sai)
    model.backbone.eval()
    model.concept_head.eval()
    criterion = CSRContrastiveLoss().to(device)
    
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Phase 2 (Proto)")
    for batch in pbar:
        if batch is None: continue
        images, cls_labels, _, _ = batch
        images, cls_labels = images.to(device), cls_labels.to(device)
        
        optimizer.zero_grad()
        
        # 1. Lấy features (No Grad cho Backbone)
        with torch.no_grad():
            features, attn_logits = model.get_features_and_cam(images)
            
        # 2. Project features (Có Grad cho Projector)
        projected_vecs = model.get_projected_vectors(features, attn_logits)
        
        # 3. Tính Loss
        loss = criterion(projected_vecs, model.prototypes, cls_labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"ConLoss": f"{loss.item():.4f}"})
        
    return running_loss / len(loader)

def train_phase_3(model, loader, optimizer, device):
    """
    PHASE 3: TASK LEARNING
    Train: Task Head (H)
    Freeze: F, C, P, p
    Loss: BCE trên Final Prediction
    """
    model.train()
    # Freeze logic đã xử lý ở main, chỉ cần gọi forward
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    pbar = tqdm(loader, desc="Phase 3 (Task)")
    for batch in pbar:
        if batch is None: continue
        images, cls_labels, _, _ = batch
        images, cls_labels = images.to(device), cls_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward đầy đủ
        outputs = model(images)
        logits = outputs["logits"] # [B, K]
        
        loss = criterion(logits, cls_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"TaskLoss": f"{loss.item():.4f}"})
        
    return running_loss / len(loader)

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description="CSR Training Script")
    parser.add_argument("--csv_path", type=str, default="./train.csv")
    parser.add_argument("--image_path", type=str, default="./train_images")
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Epochs cho từng phase
    parser.add_argument("--epochs_p1", type=int, default=5, help="Epochs for Concept Phase")
    parser.add_argument("--epochs_p2", type=int, default=5, help="Epochs for Prototype Phase")
    parser.add_argument("--epochs_p3", type=int, default=5, help="Epochs for Task Phase")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_prototypes", type=int, default=5, help="Số prototypes mỗi class")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--num_classes", type=int, default=14)
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Using device: {device}")

    # --- 1. DATA SETUP ---
    full_dataset = VINDRCXRDataset(
        csv_file=args.csv_path,
        image_dir=args.image_path,
        num_classes=args.num_classes,
        target_size=args.img_size,
        map_size=args.img_size // 32, # Ví dụ 384/32 = 12
    )
    
    if args.max_samples:
        indices = np.random.choice(len(full_dataset), args.max_samples, replace=False)
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn_ignore_none # QUAN TRỌNG
    )
    
    # --- 2. MODEL SETUP ---
    print("-> Initializing CSRModel...")
    # Khởi tạo model với số prototypes
    model = CSRModel(
        num_classes=args.num_classes, 
        num_prototypes=args.num_prototypes, 
        model_name="resnet50"
    )
    model.to(device)

    # ==========================================
    # PHASE 1: TRAIN CONCEPTS
    # ==========================================
    print("\n[PHASE 1] Starting Concept Learning...")
    
    # Unfreeze Backbone & Head
    for param in model.parameters(): param.requires_grad = True # Mở hết (hoặc đóng backbone tùy ý)
    
    optimizer_p1 = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1}, # Backbone học chậm
        {'params': model.concept_head.parameters(), 'lr': args.lr}
    ])
    
    for epoch in range(args.epochs_p1):
        loss = train_phase_1(model, train_loader, optimizer_p1, device)
        print(f"Phase 1 - Epoch {epoch+1}/{args.epochs_p1} - Loss: {loss:.4f}")
    
    torch.save(model.state_dict(), os.path.join(args.save_path, "csr_phase1.pth"))

    # ==========================================
    # PHASE 2: TRAIN PROTOTYPES
    # ==========================================
    print("\n[PHASE 2] Starting Prototype Learning...")
    
    # Freeze F & C
    for param in model.backbone.parameters(): param.requires_grad = False
    for param in model.concept_head.parameters(): param.requires_grad = False
    
    # Unfreeze P & Prototypes
    for param in model.projector.parameters(): param.requires_grad = True
    model.prototypes.requires_grad = True
    
    optimizer_p2 = torch.optim.AdamW([
        {'params': model.projector.parameters()},
        {'params': model.prototypes}
    ], lr=args.lr)
    
    for epoch in range(args.epochs_p2):
        loss = train_phase_2(model, train_loader, optimizer_p2, device)
        print(f"Phase 2 - Epoch {epoch+1}/{args.epochs_p2} - Loss: {loss:.4f}")
        
    torch.save(model.state_dict(), os.path.join(args.save_path, "csr_phase2.pth"))

    # ==========================================
    # PHASE 3: TRAIN TASK HEAD
    # ==========================================
    print("\n[PHASE 3] Starting Task Learning...")
    
    # Freeze F, C, P, Prototypes
    for param in model.parameters(): param.requires_grad = False
    # Unfreeze Task Head
    for param in model.task_head.parameters(): param.requires_grad = True
    
    optimizer_p3 = torch.optim.AdamW(model.task_head.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs_p3):
        loss = train_phase_3(model, train_loader, optimizer_p3, device)
        print(f"Phase 3 - Epoch {epoch+1}/{args.epochs_p3} - Loss: {loss:.4f}")
        
    # Lưu Model Final
    torch.save(model.state_dict(), os.path.join(args.save_path, "csr_final_model.pth"))
    print("\n-> Training Complete! Final model saved.")

if __name__ == "__main__":
    main()