import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from medmae import MedMAEBackbone

        # ... (các phần khác giữ nguyên)

class CSR(nn.Module):
    def __init__(self, num_concepts, num_classes, num_prototypes_per_concept=1, backbone_type='medmae', model_name=None):
        """
        Concept-based Similarity Reasoning (CSR) Network.
        
        Args:
            num_concepts (K): Số lượng khái niệm y khoa (concepts).
            num_classes: Số lượng bệnh cần dự đoán (target classes).
            num_prototypes_per_concept (M): Số lượng mẫu (prototypes) cho mỗi concept.
            backbone_type: Loại backbone (ví dụ: 'medmae', 'resnet18', 'resnet50').
            model_name: Tên hoặc đường dẫn tới model MedMAE nếu dùng backbone_type='medmae'.
        """
        super(CSR, self).__init__()
        
        self.K = num_concepts
        self.M = num_prototypes_per_concept
        
        # ---------------------------------------------------------
        # 1. Feature Extractor (F) [cite: 125]
        # ---------------------------------------------------------
        # Sử dụng ResNet bỏ lớp FC cuối cùng

        if backbone_type == 'medmae':
            # Kiểm tra nếu model_name là path đến .pth file
            pretrained_weights = None
            hf_model = 'facebook/vit-mae-base'
            
            if model_name and model_name.endswith('.pth'):
                pretrained_weights = model_name
            elif model_name:
                hf_model = model_name
            
            self.backbone = MedMAEBackbone(
                model_name=hf_model,
                pretrained_weights=pretrained_weights
            )
            feature_dim = self.backbone.out_channels # 768
        else:
            base_model = getattr(models, backbone_type)(pretrained=True)
            self.backbone = nn.Sequential(*list(base_model.children())[:-2]) 
            feature_dim = base_model.fc.in_features 
        
        # Lấy số channel đầu ra của backbone (ví dụ: ResNet18 là 512)
       
        
        # ---------------------------------------------------------
        # 2. Concept Head (C) [cite: 141]
        # ---------------------------------------------------------
        # 1x1 Conv mapping features (f) -> Concept Activation Maps (CAMs)
        self.concept_head = nn.Conv2d(feature_dim, num_concepts, kernel_size=1)
        
        # ---------------------------------------------------------
        # 3. Projector (P) [cite: 94, 192]
        # ---------------------------------------------------------
        # Chiếu feature vectors sang không gian embedding gọn gàng hơn
        # để tính Similarity. Output dimension thường nhỏ hơn (ví dụ 128).
        self.projection_dim = 128
        self.projector = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1), # MLP dạng conv 1x1
            nn.ReLU(),
            nn.Conv2d(feature_dim, self.projection_dim, kernel_size=1)
        )
        
        # ---------------------------------------------------------
        # 4. Concept Prototypes (p) [cite: 123]
        # ---------------------------------------------------------
        # Learnable parameters: K concepts * M prototypes * Dim
        # Shape: (K*M, projection_dim, 1, 1) để dùng như filters trong Conv2d
        self.prototypes = nn.Parameter(
            torch.randn(num_concepts * num_prototypes_per_concept, self.projection_dim, 1, 1)
        )
        
        # ---------------------------------------------------------
        # 5. Task Head (H) [cite: 128]
        # ---------------------------------------------------------
        # Dự đoán bệnh từ điểm tương đồng (similarity scores)
        # Input size là tổng số prototypes (K * M)
        self.task_head = nn.Linear(num_concepts * num_prototypes_per_concept, num_classes)

    def get_local_concept_vectors(self, f, cams):
        """
        Tính toán Local Concept Vectors (v) dùng cho quá trình huấn luyện (Eq. 2).
        Đây là bước quan trọng để train contrastive learning.
        
        Args:
            f: Feature map (B, C_feat, H, W)
            cams: Concept Activation Maps (B, K, H, W)
        Returns:
            vectors: (B, K, C_feat)
        """
        # Spatial Softmax trên CAMs [cite: 144]
        B, K, H, W = cams.size()
        cams_flat = cams.view(B, K, -1)
        attn_weights = F.softmax(cams_flat, dim=-1).view(B, K, H, W)
        
        # Weighted Sum: Tổng hợp feature f dựa trên trọng số của CAM
        # f: (B, C_feat, H, W) -> mở rộng thành (B, 1, C_feat, H, W)
        # attn: (B, K, H, W) -> mở rộng thành (B, K, 1, H, W)
        f_expanded = f.unsqueeze(1) 
        attn_expanded = attn_weights.unsqueeze(2)
        
        # Sum over spatial dimensions (H, W) -> (B, K, C_feat)
        local_vectors = (f_expanded * attn_expanded).sum(dim=(-1, -2))
        return local_vectors

    def forward(self, x, importance_map=None):
        """
        Luồng xử lý chính của CSR.
        
        Args:
            x: Ảnh đầu vào (B, 3, H, W)
            importance_map (Optional): Map tương tác từ bác sĩ (A) [cite: 290]
                                       Shape: (B, 1, H, W) hoặc (B, H, W)
        """
        # 1. Trích xuất đặc trưng
        f = self.backbone(x)  # (B, 512, H/32, W/32)
        
        # 2. Tính Concept Activation Maps (CAMs) - [Eq. 2 context]
        cams = self.concept_head(f) # (B, K, H, W)
        
        # 3. Chiếu đặc trưng sang không gian Prototype (Projected Features)
        f_proj = self.projector(f)  # (B, 128, H, W)
        
        # Chuẩn hóa L2 cho feature và prototypes để tính Cosine Similarity [cite: 150]
        f_proj_norm = F.normalize(f_proj, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # 4. Tính Similarity Maps (S) 
        # Dùng Conv2d để tính dot product giữa mọi patch của f_proj và mọi prototype
        # Output: (B, K*M, H, W)
        similarity_maps = F.conv2d(f_proj_norm, prototypes_norm)
        
        # --- Spatial Interaction (Bác sĩ can thiệp) [cite: 290, 295] ---
        if importance_map is not None:
            # Importance map A nhân element-wise vào Similarity map S
            # Đảm bảo dimension khớp nhau
            if importance_map.dim() == 3: importance_map = importance_map.unsqueeze(1)
            # Resize importance map về kích thước feature map nếu cần
            if importance_map.shape[-2:] != similarity_maps.shape[-2:]:
                importance_map = F.interpolate(importance_map, size=similarity_maps.shape[-2:], mode='nearest')
            
            # Eq. 13: [S_hat] = A * [S]
            similarity_maps = similarity_maps * importance_map
            
            # Clip negative values (theo mô tả bài báo để đảm bảo monotonic) [cite: 300]
            similarity_maps = torch.clamp(similarity_maps, min=0)

        # 5. Tính Similarity Scores (s) [cite: 126, 247]
        # Max pooling trên không gian (H, W) -> Lấy điểm giống nhất
        # Output: (B, K*M)
        similarity_scores = F.max_pool2d(similarity_maps, kernel_size=similarity_maps.shape[-2:])
        similarity_scores = similarity_scores.view(similarity_scores.size(0), -1)
        
        # 6. Dự đoán cuối cùng (y) [cite: 128]
        logits = self.task_head(similarity_scores)
        
        # Trả về đầy đủ để tính loss hoặc giải thích (Explainability)
        return {
            "logits": logits,                   # Dự đoán bệnh (y)
            "similarity_scores": similarity_scores, # Điểm số tương đồng (s)
            "similarity_maps": similarity_maps, # Bản đồ tương đồng (S) - Để visualize
            "cams": cams,                       # Concept maps (để train Concept Loss)
            "features": f,                      # Feature gốc
            "projected_features": f_proj        # Feature đã chiếu
        }

# =========================================================================
# Ví dụ cách khởi tạo và chạy thử (Dummy Pass)
# =========================================================================
if __name__ == "__main__":
    # Giả lập tham số
    BATCH_SIZE = 2
    NUM_CONCEPTS = 14        # K: Ví dụ: Tràn dịch, Bóng tim to, Mờ phổi...
    NUM_CLASSES = 15         # Target: Bình thường, Viêm phổi, Suy tim
    NUM_PROTOTYPES = 5      # M: 5 mẫu cho mỗi concept
    
    # Khởi tạo model
    model = CSR(num_concepts=NUM_CONCEPTS, 
                num_classes=NUM_CLASSES, 
                num_prototypes_per_concept=NUM_PROTOTYPES,
                backbone_type='medmae',
                model_name='weights/pre_trained_medmae.pth')  # Đường dẫn tương đối từ MedSight3/
    
    # Dummy Input
    input_img = torch.randn(BATCH_SIZE, 3, 224, 224)
    
    # 1. Forward thông thường
    output = model(input_img)
    print("Logits shape:", output['logits'].shape)              # (2, 3)
    print("Sim Maps shape:", output['similarity_maps'].shape)   # (2, 10, 7, 7) - 10 = 5 concepts * 2 prototypes
    
    # 2. Forward với Interaction (Bác sĩ vẽ box)
    # Giả sử bác sĩ muốn focus vào vùng trung tâm (importance = 1), vùng khác (importance = 0.2)
    imp_map = torch.ones(BATCH_SIZE, 1, 7, 7) * 0.2
    imp_map[:, :, 3:5, 3:5] = 1.0 # Focus vùng giữa
    
    output_interactive = model(input_img, importance_map=imp_map)
    print("Interactive Logits:", output_interactive['logits'])
    
    # 3. Lấy Local Concept Vectors để tính Contrastive Loss (Lúc training)
    local_vectors = model.get_local_concept_vectors(output['features'], output['cams'])
    print("Local Vectors shape (for training):", local_vectors.shape) # (2, 5, 512)