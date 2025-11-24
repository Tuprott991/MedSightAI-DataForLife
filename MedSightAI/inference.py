import argparse
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# Import kiến trúc model (đảm bảo file src/model.py tồn tại)
from src.model import MedicalConceptModel


def get_args():
    parser = argparse.ArgumentParser(
        description="Chạy inference chẩn đoán bệnh TB qua X-quang"
    )

    # --- Tham số bắt buộc ---
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn tới file ảnh đầu vào (VD: data/test.png)",
    )

    # --- Tham số tùy chọn (có giá trị mặc định) ---
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pth",
        help="Đường dẫn tới file trọng số model đã train",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result.png",
        help="Đường dẫn lưu file ảnh kết quả (kèm heatmap)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị chạy (cuda hoặc cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng xác suất để quyết định có bệnh hay không (mặc định 0.5)",
    )

    return parser.parse_args()


def preprocess_image(img_path, img_size=384):
    """Đọc và xử lý ảnh theo chuẩn training"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {img_path}")

    image = Image.open(img_path).convert("RGB")

    # Transform chuẩn (giống hệt lúc train)
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img_tensor = transform(image).unsqueeze(0)  # Thêm batch dim -> (1, 3, 384, 384)
    return image, img_tensor


def visualize_result(original_img, mask_pred, score, threshold, save_path):
    """Vẽ và lưu kết quả"""
    # Resize mask dự đoán về kích thước ảnh gốc
    mask_pred = F.interpolate(
        mask_pred, size=original_img.size[::-1], mode="bilinear", align_corners=False
    )
    mask_np = mask_pred.squeeze().detach().cpu().numpy()

    # Thiết lập vẽ
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Ảnh 1: Ảnh gốc
    ax[0].imshow(original_img)
    ax[0].set_title("Original X-Ray")
    ax[0].axis("off")

    # Ảnh 2: Heatmap
    ax[1].imshow(original_img)
    # Heatmap: vmin=0, vmax=1 để cố định thang màu
    im = ax[1].imshow(mask_np, cmap="jet", alpha=0.5, vmin=0, vmax=1.0)

    # Tiêu đề kết quả
    is_sick = score > threshold
    status = "SICK (TB)" if is_sick else "HEALTHY"
    color = "red" if is_sick else "green"

    ax[1].set_title(
        f"Prediction: {status}\nScore: {score:.4f}",
        color=color,
        fontsize=14,
        fontweight="bold",
    )
    ax[1].axis("off")

    # Thanh màu
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Đã lưu ảnh kết quả tại: {save_path}")
    # plt.show() # Bỏ comment nếu chạy trên Jupyter Notebook


def main():
    args = get_args()

    print(f"--- BẮT ĐẦU INFERENCE ---")
    print(f"Input: {args.input}")
    print(f"Device: {args.device}")

    # 1. Load Model
    print("Loading model...")
    # Lưu ý: model_name phải khớp với loại model bạn đã dùng trong train.py
    model = MedicalConceptModel(model_name="google/siglip-base-patch16-384")

    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file trọng số tại {args.checkpoint}")
        return
    except Exception as e:
        print(f"❌ Lỗi khi load weights: {e}")
        return

    model.to(args.device)
    model.eval()

    # 2. Xử lý ảnh
    try:
        original_img, img_tensor = preprocess_image(args.input)
        img_tensor = img_tensor.to(args.device)
    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh: {e}")
        return

    # 3. Dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)

        # Lấy Logits và chuyển thành xác suất (Probability)
        logits = outputs["class_logits"]
        prob = torch.sigmoid(logits).item()

        # Lấy Heatmap
        mask = outputs["mask_pred"]

    # 4. Hiển thị & Lưu
    print(f"-> Xác suất bệnh (TB Probability): {prob:.4f}")

    visualize_result(
        original_img=original_img,
        mask_pred=mask,
        score=prob,
        threshold=args.threshold,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
