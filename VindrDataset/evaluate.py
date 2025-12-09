import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from tqdm import tqdm
import os

# Import module của bạn
from src.dataset import VINDRCXRDataset
from src.model import CSRModel

# ==========================================
# 1. HÀM ARGUMENTS
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for CSR Model")

    # --- Path Parameters ---
    parser.add_argument("--test_csv", type=str, default="test_split.csv", 
                        help="Đường dẫn đến file CSV tập test (test_split.csv)")
    parser.add_argument("--image_path", type=str, default="./train_images", 
                        help="Thư mục chứa ảnh gốc (PNG/DICOM)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/csr_final_model.pth", 
                        help="Đường dẫn đến file weight model (.pth)")

    # --- Model Parameters ---
    parser.add_argument("--num_classes", type=int, default=14, help="Số lượng lớp bệnh")
    parser.add_argument("--num_prototypes", type=int, default=5, 
                        help="Số prototypes mỗi class (PHẢI GIỐNG LÚC TRAIN)")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Backbone model")
    
    # --- Eval Parameters ---
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch khi eval")
    parser.add_argument("--img_size", type=int, default=384, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị chạy (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Ngưỡng quyết định (cắt) cho F1-Score và báo cáo chi tiết. (Mặc định: 0.5)") # <--- THAM SỐ MỚI

    return parser.parse_args()

# ==========================================
# 2. HÀM HỖ TRỢ (COLLATE)
# ==========================================
def collate_fn_ignore_none(batch):
    """Lọc bỏ các mẫu bị None do lỗi đọc ảnh"""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# ==========================================
# 3. HÀM ĐÁNH GIÁ CHÍNH
# ==========================================
def evaluate_model(model, dataloader, device, class_names, threshold):
    """
    Đánh giá mô hình CSRModel trên tập test.
    Args:
        model: Mô hình CSRModel đã được load weight.
        dataloader: DataLoader của tập test.
        device: Thiết bị (cuda hoặc cpu).
        class_names: Danh sách tên các lớp bệnh.
        threshold: Ngưỡng để phân loại nhị phân (mặc định: 0.5).
    Returns:
        metrics: Dictionary chứa các chỉ số đánh giá.
    """
    model.eval()
    
    y_true = []
    y_probs = []
    
    print(f"-> Đang đánh giá trên {len(dataloader.dataset)} mẫu...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None: continue
            
            # Unpack data
            images, labels, _, _ = batch 
            
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Forward qua model CSR
            outputs = model(images)
            logits = outputs['logits']  # [B, Num_Classes]
            
            # Chuyển logits -> Probability
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, Num_Classes]
            
            y_true.append(labels)
            y_probs.append(probs)
            
    # Gộp kết quả
    if len(y_true) == 0:
        print("(!) Không có dữ liệu để đánh giá.")
        return {}

    y_true = np.concatenate(y_true, axis=0)  # [Total_Samples, Num_Classes]
    y_probs = np.concatenate(y_probs, axis=0)  # [Total_Samples, Num_Classes]
    
    metrics = {}
    
    # --- A. AUC-ROC (Không phụ thuộc vào ngưỡng) ---
    try:
        auc_score = roc_auc_score(y_true, y_probs, average='macro')
        metrics['AUC_Macro'] = auc_score
        
        auc_per_class = roc_auc_score(y_true, y_probs, average=None)
        # Handle case where auc_per_class might be a scalar float
        if isinstance(auc_per_class, (int, float)):
            auc_per_class = [auc_per_class]
        
        for idx, cls in enumerate(class_names):
            if idx < len(auc_per_class):
                metrics[f'AUC_{cls}'] = auc_per_class[idx]
    except ValueError:
        print("(!) Cảnh báo: Lỗi tính AUC (có thể do thiếu class trong tập test).")
        metrics['AUC_Macro'] = 0.0

    # --- B. F1-Score và Báo cáo chi tiết (Phụ thuộc vào ngưỡng) ---
    y_pred_binary = (y_probs > threshold).astype(int)  # Áp dụng ngưỡng để phân loại nhị phân
    metrics['F1_Macro'] = f1_score(y_true, y_pred_binary, average='macro')
    
    # --- C. Báo cáo chi tiết ---
    print("\n" + "="*40)
    print(f"KẾT QUẢ ĐÁNH GIÁ (Threshold={threshold})")
    print("="*40)
    print(classification_report(y_true, y_pred_binary, target_names=class_names, zero_division=0))
    
    print("-" * 30)
    print(f"AUC Macro: {metrics.get('AUC_Macro', 0):.4f}")
    print(f"F1 Macro:  {metrics['F1_Macro']:.4f}")
    print("="*40)
    
    return metrics

# ==========================================
# 4. MAIN
# ==========================================
def main():
    args = get_args()
    
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"-> Using device: {device}")

    # Danh sách tên bệnh (VinDr-CXR)
    CLASS_NAMES = [
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
        'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening',
        'Pneumothorax', 'Pulmonary fibrosis'
    ]
    
    # 1. Load Data
    # ... (phần load data giữ nguyên) ...
    test_dataset = VINDRCXRDataset(
        csv_file=args.test_csv,
        image_dir=args.image_path,
        num_classes=args.num_classes,
        target_size=args.img_size,
        map_size=args.img_size // 32 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        collate_fn=collate_fn_ignore_none
    )
    
    # 2. Load Model
    # ... (phần load model giữ nguyên) ...
    model = CSRModel(
        num_classes=args.num_classes, 
        num_prototypes=args.num_prototypes, 
        model_name=args.model_name
    )
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("-> Load weight thành công!")
    else:
        print(f"(!) Lỗi: Không tìm thấy file checkpoint tại {args.checkpoint}")
        return

    model.to(device)
    
    # 3. Run Eval
    # Truyền tham số ngưỡng vào hàm đánh giá
    evaluate_model(model, test_loader, device, CLASS_NAMES, args.threshold)

if __name__ == "__main__":
    main()