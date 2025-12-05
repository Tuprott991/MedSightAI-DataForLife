import torch
import torch.nn as nn
import torch.nn.functional as F

class CSRContrastiveLoss(nn.Module):
    """
    Hàm loss cho Phase 2: Prototype Learning.
    Dựa trên công thức Contrastive Loss (InfoNCE) giữa Projected Vectors và Prototypes.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected_vecs, prototypes, labels):
        """
        Args:
            projected_vecs: [Batch, K, Emb_Dim] - Vector đặc trưng concept đã project (v')
            prototypes:     [K, M, Emb_Dim]     - Các vector prototype học được (p)
            labels:         [Batch, K]          - Nhãn one-hot của concepts (0 hoặc 1)
        Returns:
            Scalar Loss
        """
        B, K, D = projected_vecs.shape
        M = prototypes.shape[1]  # Số prototypes cho mỗi concept

        total_loss = torch.tensor(0.0, device=projected_vecs.device)
        
        # 1. Chuẩn hóa L2 (quan trọng để tính Cosine Similarity)
        # Prototypes: [K, M, D]
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        # Projected Vectors: [Batch, K, D]
        projected_vecs = F.normalize(projected_vecs, p=2, dim=-1)

        # 2. Tính Loss cho từng Concept k
        for k in range(K):
            # Lấy các mẫu trong batch thực sự có chứa concept k (Positive samples)
            # labels[:, k] có shape [Batch]
            pos_indices = torch.where(labels[:, k] == 1)[0]

            # Nếu trong batch không có ảnh nào chứa bệnh k, bỏ qua concept này
            if len(pos_indices) == 0:
                continue

            # --- A. Lấy Anchors (Mẫu dương tính) ---
            # anchors shape: [N_pos, D]
            anchors = projected_vecs[pos_indices, k, :]

            # --- B. Lấy Positive Keys (Prototypes của concept k) ---
            # pos_protos shape: [M, D]
            pos_protos = prototypes[k, :, :]

            # --- C. Lấy Negative Keys (Prototypes của tất cả concept KHÁC k) ---
            # Tạo danh sách indices khác k
            other_indices = [i for i in range(K) if i != k]
            if not other_indices: # Trường hợp chỉ có 1 class (hiếm)
                continue
                
            # Gom tất cả prototype của các class khác lại
            # neg_protos shape: [(K-1)*M, D]
            neg_protos = prototypes[other_indices, :, :].view(-1, D)

            # --- D. Tính Similarity (Cosine / Temperature) ---
            # 1. Sim với Positive Prototypes: [N_pos, M]
            sim_pos = torch.matmul(anchors, pos_protos.T) / self.temperature

            # 2. Sim với Negative Prototypes: [N_pos, (K-1)*M]
            sim_neg = torch.matmul(anchors, neg_protos.T) / self.temperature

            # --- E. Tính InfoNCE Loss ---
            # Mục tiêu: Kéo Anchor lại gần Prototype gần nhất của nó
            # Lấy max similarity trong các positive prototypes (để tìm 'best match prototype')
            # max_sim_pos shape: [N_pos, 1]
            max_sim_pos, _ = torch.max(sim_pos, dim=1, keepdim=True)

            # Mẫu số (Denominator): exp(sim_pos) + sum(exp(sim_neg))
            # Để tính toán ổn định số học (LogSumExp trick), ta gộp tất cả logits lại
            # all_logits shape: [N_pos, M + Negative_Count]
            all_logits = torch.cat([sim_pos, sim_neg], dim=1)

            # Loss = -log( exp(pos) / sum(exp(all)) )
            #      = -pos + log(sum(exp(all)))
            # Ở đây dùng max_sim_pos làm tử số (positive term)
            loss_k = -max_sim_pos + torch.logsumexp(all_logits, dim=1, keepdim=True)

            # Mean loss trên các mẫu dương tính của concept k
            total_loss += loss_k.mean()

        # Trung bình loss trên số lượng concepts (K)
        return total_loss / K