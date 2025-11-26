import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class MedicalConceptModel(nn.Module):
    def __init__(
        self,
        model_name="chiphuyen/med-siglip",  # Hoặc tên model bạn dùng trên HF
        num_concepts=12 ,  # VinDR CXR có 14 loại, để 13 vì bỏ loại "No Finding"
        num_class= 0, # Không dùng classifier trong retrieval
        feature_dim=768,  # Hidden size của ViT Base
        projection_dim=128,
    ):  # Kích thước vector để retrieval
        super().__init__()

        print(f"Initializing Backbone: {model_name}...")x   

        # 1. Backbone (MedSigLIP Image Encoder)
        # Load model pre-trained từ HuggingFace
        self.backbone = AutoModel.from_pretrained(model_name)

        # Đóng băng backbone (Frozen Strategy)
        # Giúp train nhanh hơn và giữ lại kiến thức y tế đã học
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Concept Layer (The Localization Block)
        # Nhiệm vụ: Biến Feature Map thành Heatmap (Mask)
        # Input: (Batch, 768, H, W) -> Output: (Batch, 1, H, W)
        self.concept_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_concepts, kernel_size=1),
            nn.Sigmoid(),  # Ép giá trị về [0, 1] để làm Mask
        )
        # 3. Projector (The Retrieval Block)
        # Chiếu vector đặc trưng sang không gian nhỏ hơn để so sánh cosine
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
        )

        # 4. Classifier (Optional but Recommended)
        # Dùng vector đặc trưng để đưa ra kết luận cuối cùng (Có bệnh/Không bệnh)
        self.classifier = nn.Linear(feature_dim, num_class)

    def forward(self, x):
        """
        x: Input Image Tensor (Batch, 3, 384, 384)
        """
        # --- A. BACKBONE PASS ---
        outputs = self.backbone.vision_model(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state  # Shape: (B, L, D)

        # --- B. RESHAPE (FIXED FOR SIGLIP) ---
        B, L, D = last_hidden_state.shape

        # Kiểm tra xem L có phải là số chính phương không (ví dụ 576 = 24^2)
        H_grid = int(L**0.5)

        if H_grid * H_grid == L:
            # Trường hợp SigLIP: Không có CLS token, toàn bộ là spatial tokens
            patch_tokens = last_hidden_state
            W_grid = H_grid
        else:
            # Trường hợp ViT/CLIP thường: Có 1 CLS token ở đầu -> Cắt bỏ
            patch_tokens = last_hidden_state[:, 1:, :]
            # Tính lại kích thước lưới sau khi cắt
            L_new = patch_tokens.shape[1]
            H_grid = int(L_new**0.5)
            W_grid = H_grid

        # Reshape về dạng ảnh 2D: (B, D, H, W)
        feature_map = patch_tokens.permute(0, 2, 1).view(B, D, H_grid, W_grid)

        # --- C. CONCEPT LOCALIZATION (Segmentation) ---
        mask_pred = self.concept_head(feature_map)

        # --- D. FEATURE AGGREGATION ---
        # Normalize mask
        mask_weights = mask_pred / (mask_pred.sum(dim=(2, 3), keepdim=True) + 1e-6)

        # Weighted Average Pooling
        concept_vector = (feature_map * mask_weights).sum(dim=(2, 3))

        # --- E. PROJECTION & CLASSIFICATION ---
        retrieval_vector = self.projector(concept_vector)
        class_logits = self.classifier(concept_vector)

        return {
            "mask_pred": mask_pred,
            "retrieval_vector": retrieval_vector,
            "class_logits": class_logits,
            "feature_map": feature_map,
        }


# --- Code Test kích thước (Unit Test) ---
if __name__ == "__main__":
    # Giả lập input
    dummy_img = torch.randn(2, 3, 384, 384)  # Batch size 2

    # Init model (Lưu ý: máy cần có internet để tải weight lần đầu)
    # Nếu không tải được 'chiphuyen/med-siglip', thử 'google/siglip-base-patch16-384'
    try:
        model = MedicalConceptModel(model_name="google/siglip-base-patch16-384")
        output = model(dummy_img)

        print("Model Output Shapes:")
        print(f"Mask Pred: {output['mask_pred'].shape}")  # Kỳ vọng: (2, 1, 24, 24)
        print(f"Vector:    {output['retrieval_vector'].shape}")  # Kỳ vọng: (2, 128)
        print(f"Logits:    {output['class_logits'].shape}")  # Kỳ vọng: (2, 1)
        print("Test thành công!")
    except Exception as e:
        print(f"Lỗi khởi tạo (có thể do mạng hoặc tên model): {e}")
