import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CSRModel(nn.Module):
    def __init__(self, num_classes=14, num_prototypes=5, model_name="resnet50", pretrained=True):
        super().__init__()
        
        # --- PHẦN 1: CONCEPT MODEL (Giai đoạn 1) ---
        # F: Feature Extractor
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(4,)
        )
        feature_info = self.backbone.feature_info.get_dicts()[-1]
        self.feature_dim = feature_info["num_chs"]

        # C: Concept Head (Tạo CAMs)
        self.concept_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)

        # --- PHẦN 2: PROTOTYPES (Giai đoạn 2) ---
        self.embedding_dim = 128
        # P: Projector (Chiếu feature về không gian contrastive)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_dim)
        )
        
        # Learnable Prototypes: [Num_Classes, Num_Prototypes_Per_Class, Emb_Dim]
        # Bài báo gọi là p^{k_m} [cite: 123]
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, self.embedding_dim))
        
        # --- PHẦN 3: TASK HEAD (Giai đoạn 3) ---
        # H: Task Head (Dự đoán bệnh từ điểm tương đồng)
        # Input là vector similarity score có kích thước [Num_Classes * Num_Prototypes]
        self.task_head = nn.Linear(num_classes * num_prototypes, num_classes)

    def get_features_and_cam(self, x):
        """Dùng cho Giai đoạn 1"""
        if x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        features = self.backbone(x)[0]       # f
        attn_logits = self.concept_head(features) # cam
        return features, attn_logits

    def get_projected_vectors(self, features, attn_logits):
        """Dùng cho Giai đoạn 2: Lấy Local Concept Vectors v^k [cite: 145]"""
        B, C, H, W = features.shape
        K = attn_logits.shape[1]
        
        # 1. Normalize CAM (Spatial Softmax)
        attn_weights = F.softmax(attn_logits.view(B, K, -1), dim=-1).view(B, K, H, W)
        
        # 2. Weighted Sum để lấy vector đại diện cho từng concept
        # features: [B, C, H*W] -> [B, H*W, C]
        features_flat = features.view(B, C, -1).permute(0, 2, 1)
        # v = weights * features -> [B, K, C]
        local_concept_vectors = torch.bmm(attn_weights.view(B, K, -1), features_flat)
        
        # 3. Project sang không gian embedding -> v' [cite: 192]
        projected_vectors = self.projector(local_concept_vectors) # [B, K, Emb_Dim]
        return F.normalize(projected_vectors, p=2, dim=-1)

    def forward(self, x):
        """Luồng chạy Full (Dùng cho Giai đoạn 3 & Inference)"""
        # 1. Trích xuất đặc trưng & CAM
        features, attn_logits = self.get_features_and_cam(x)
        
        # 2. Tính Similarity Map S [cite: 149]
        # features: [B, C, H, W] -> project từng pixel -> [B, Emb_Dim, H, W]
        # Đoạn này tính toán nặng, trong thực tế ta tính similarity trên projected vectors v'
        
        # Để đơn giản hóa theo luồng Inference của bài báo[cite: 126]:
        # Ta lấy similarity giữa Prototypes và Feature Map đã project
        # Nhưng để code chạy nhanh, ta dùng công thức (10) trong bài báo:
        # s^{k_m} = max <p, P(f(h,w))>
        
        projected_vectors = self.get_projected_vectors(features, attn_logits) # [B, K, Emb_Dim]
        
        # Tính Similarity Score s [cite: 126]
        # Prototypes: [K, M, Emb]
        # Vectors: [B, K, Emb]
        # Similarity: [B, K, M]
        # (Lưu ý: đây là phiên bản đơn giản hóa, bản chuẩn phải tính trên từng patch HxW)
        
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # Tính Cosine Similarity
        # Kết quả: [B, K, M] (Batch, Class, Num_Prototypes)
        sim_scores = torch.einsum('bkc,kmc->bkm', projected_vectors, prototypes_norm)
        
        # Flatten thành vector s [B, K*M]
        s_vector = sim_scores.view(x.size(0), -1)
        
        # 3. Predict y từ s [cite: 128]
        logits = self.task_head(s_vector)
        
        return {
            "logits": logits,
            "attn_maps": attn_logits,
            "projected_vectors": projected_vectors,
            "sim_scores": sim_scores
        }