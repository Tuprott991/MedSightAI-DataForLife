import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Import CSRModel thay vì model cũ
from src.model import CSRModel

# --- CẤU HÌNH ---
CLASS_NAMES = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis'
]

# --- THIẾT LẬP BIẾN ---
# Đường dẫn đến ảnh và checkpoint
image_path = "/kaggle/input/vindr-image-convert/train_png_384/9a5094b2563a1ef3ff50dc5c7ff71345.png"  # Thay bằng đường dẫn ảnh của bạn
checkpoint_path = "/kaggle/input/vin-csr-training/checkpoints/csr_final_model.pth"  # Thay bằng đường dẫn checkpoint của bạn

# Cấu hình mô hình
num_classes = 14
num_prototypes = 10
model_name = "resnet50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn lưu kết quả
save_path = "./results/output.png"  # Thay bằng đường dẫn bạn muốn lưu ảnh kết quả

# --- HÀM TIỀN XỬ LÝ ---
def preprocess_image(image_path, target_size=384):
    """Đọc và tiền xử lý ảnh"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
        
    # Đọc ảnh grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot read image")
        
    # Resize & Normalize
    image = cv2.resize(image, (target_size, target_size))
    img_norm = image.astype(np.float32) / 255.0
    
    # Tensor: [1, 1, H, W] (Batch=1, Channel=1)
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    return img_tensor, image

def compute_attention_difference(attn_logits):
    """
    Tính toán sự khác biệt giữa các lớp dựa trên attention weights.
    Args:
        attn_logits: Tensor [Batch, Num_Classes, H, W] - Attention logits từ concept head.
    Returns:
        diff_matrix: Tensor [Num_Classes, Num_Classes] - Ma trận sự khác biệt giữa các lớp.
    """
    B, K, H, W = attn_logits.shape
    
    # 1. Normalize CAM (Spatial Softmax)
    attn_weights = F.softmax(attn_logits.view(B, K, -1), dim=-1).view(B, K, H, W)  # [B, K, H, W]
    
    # 2. Tính giá trị trung bình của attention weights cho mỗi lớp
    class_means = attn_weights.mean(dim=(0, 2, 3))  # [K] - Giá trị trung bình cho mỗi lớp
    
    # 3. Tính sự khác biệt giữa các lớp
    diff_matrix = torch.abs(class_means.unsqueeze(0) - class_means.unsqueeze(1))  # [K, K]
    
    return diff_matrix

# --- HÀM HIỂN THỊ KẾT QUẢ ---
def visualize_result(original_img, probs, similarities, attn_maps, top_k=3, save_path=None):
    """Vẽ ảnh gốc, kết quả dự đoán và heatmap của top-k bệnh, đồng thời lưu ảnh nếu cần"""
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    plt.figure(figsize=(15, 6))
    
    # 1. Ảnh gốc + Text
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Input X-Ray")
    plt.axis('off')
    
    info_text = "PREDICTIONS:\n"
    for idx in top_indices:
        name = CLASS_NAMES[idx]
        sim_score = similarities[0, idx, :].max().item() 
        prob = probs[idx]
        info_text += f"{name}: {prob*100:.1f}% (Sim: {sim_score:.2f})\n"
        
    plt.xlabel(info_text, fontsize=12, loc='left')

    # 2. Heatmap các bệnh Top K
    for i, idx in enumerate(top_indices):
        name = CLASS_NAMES[idx]
        
        # Lấy attention map (CAM)
        cam = attn_maps[0, idx].cpu().numpy()
        
        # Resize CAM lên size ảnh
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))

        # Normalize CAM
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

        # 🔥 Tạo mask — chỉ highlight vùng cam > threshold
        threshold = 0.8
        mask = (cam_norm > threshold).astype(np.float32)

        # Colormap chỉ áp dụng trên vùng mask
        heatmap = cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Chỉ giữ màu trong vùng mask
        heatmap_masked = heatmap * mask[..., None]

        # Chuẩn hóa ảnh gốc (H,W → RGB)
        img_rgb = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

        # 🔥 Overlay CHỈ ở vùng mask
        alpha = 0.5
        overlay = img_rgb * (1 - alpha * mask[..., None]) + heatmap_masked * (alpha * mask[..., None])

        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(overlay)
        plt.title(f"{name}\n{probs[idx]*100:.1f}%")
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    
    plt.show()

# --- CHẠY INFERENCE ---
print("-> Loading CSRModel...")
model = CSRModel(num_classes=num_classes, num_prototypes=num_prototypes, model_name=model_name)

# Load checkpoint
ckpt = torch.load(checkpoint_path, map_location=device)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)

model.to(device)
model.eval()

# Tiền xử lý ảnh
img_tensor, original_img = preprocess_image(image_path)
img_tensor = img_tensor.to(device)

# Dự đoán
print(f"-> Predicting: {image_path}")
with torch.no_grad():
    outputs = model(img_tensor)
    
    # Outputs từ CSRModel bao gồm: logits, sim_scores, attn_maps, ...
    logits = outputs['logits'][0]           # [Num_Classes]
    sim_scores = outputs['sim_scores']      # [Batch, Num_Classes, Num_Proto]
    attn_maps = outputs['attn_maps']        # [Batch, Num_Classes, H, W]

    # Tính sự khác biệt giữa các lớp
    # diff_matrix = compute_attention_difference(attn_maps)
    # print("Attention Difference Matrix:", diff_matrix)
        
    probs = torch.sigmoid(logits).cpu().numpy()

# Hiển thị kết quả
visualize_result(original_img, probs, sim_scores, attn_maps, save_path=save_path)