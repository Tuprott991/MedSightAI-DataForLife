import argparse
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
        # Similarity lấy max hoặc mean của các prototypes thuộc class đó
        sim_score = similarities[0, idx, :].max().item() 
        prob = probs[idx]
        info_text += f"{name}: {prob*100:.1f}% (Sim: {sim_score:.2f})\n"
        
    plt.xlabel(info_text, fontsize=12, loc='left')
    
    # 2. Heatmap các bệnh Top K
    for i, idx in enumerate(top_indices):
        name = CLASS_NAMES[idx]
        
        # Lấy attention map (CAM)
        cam = attn_maps[0, idx].cpu().numpy() # [H_map, W_map]
        
        # Resize CAM lên kích thước ảnh gốc
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # Normalize heatmap để hiển thị màu đẹp
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        img_rgb = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
        
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(overlay)
        plt.title(f"{name}\n{probs[idx]*100:.1f}%")
        plt.axis('off')
        
    plt.tight_layout()
    
    # Lưu ảnh nếu `save_path` được cung cấp
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    
    plt.show()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the result image")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print("-> Loading CSRModel...")
    model = CSRModel(num_classes=14, num_prototypes=10, model_name="resnet50")
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device)
    model.eval()
    
    # 2. Preprocess Image
    img_tensor, original_img = preprocess_image(args.image_path)
    img_tensor = img_tensor.to(device)
    
    # 3. Inference
    print(f"-> Predicting: {args.image_path}")
    with torch.no_grad():
        outputs = model(img_tensor)
        
        # Outputs từ CSRModel bao gồm: logits, sim_scores, attn_maps, ...
        logits = outputs['logits'][0]           # [Num_Classes]
        sim_scores = outputs['sim_scores']      # [Batch, Num_Classes, Num_Proto]
        attn_maps = outputs['attn_maps']        # [Batch, Num_Classes, H, W]
        
        probs = torch.sigmoid(logits).cpu().numpy()
        
    # 4. Visualize
    visualize_result(original_img, probs, sim_scores, attn_maps, save_path=args.save_path)
    
if __name__ == "__main__":
    main()