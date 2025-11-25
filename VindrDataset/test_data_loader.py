import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

"""
Example output tensor:
tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])

Label: Consolidaton, Infiltration, Lung Opacity, Pleural effusion, Pleural thickening
"""

# Import class Dataset từ file dataset.py của bạn
from dataset import VinDrClassifierDataset, TARGET_CLASSES

# --- CẤU HÌNH ---
CSV_PATH = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
IMG_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/"
BATCH_SIZE = 4  # Thử lấy 4 ảnh một lúc
IMG_SIZE = 512  # Resize về 512x512

# 1. Định nghĩa Transform (BẮT BUỘC phải có Resize và ToTensor)
# Lý do: DataLoader không thể gộp các ảnh khác kích thước, và Model cần Tensor
test_transform = A.Compose(
    [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # Chuyển từ HWC (numpy) -> CHW (Tensor)
    ]
)

# 2. Khởi tạo Dataset & DataLoader
dataset = VinDrClassifierDataset(
    csv_file=CSV_PATH, image_dir=IMG_DIR, transform=test_transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # Tráo bài để xem ngẫu nhiên
    num_workers=2,  # Tăng tốc độ đọc
)


# 3. Hàm hỗ trợ hiển thị ảnh (Un-normalize)
def show_batch(images, targets, class_names):
    """
    images: Tensor (B, C, H, W)
    targets: Tensor (B, N_Classes)
    """
    # Chuyển về CPU và Numpy
    images = images.numpy()

    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    if len(images) == 1:
        axes = [axes]  # Xử lý trường hợp batch=1

    # Mean và Std dùng lúc Normalize (để đảo ngược lại cho dễ nhìn)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, ax in enumerate(axes):
        # Chuyển từ (C, H, W) -> (H, W, C) để hiển thị bằng Matplotlib
        img = images[i].transpose((1, 2, 0))

        # Un-normalize: nhân std + cộng mean
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Đảm bảo giá trị nằm trong [0, 1]

        ax.imshow(img)

        # Lấy tên các bệnh có nhãn = 1
        active_labels = np.where(targets[i] == 1)[0]
        label_text = "\n".join([class_names[idx] for idx in active_labels])

        if not label_text:
            label_text = "No Finding"

        ax.set_title(label_text, fontsize=10, color="red")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# 4. CHẠY TEST
print(f"--- Đang test DataLoader ---")
# Lấy 1 batch
images, targets = next(iter(loader))

print(f"Batch Image Shape: {images.shape}")
# Mong đợi: torch.Size([4, 3, 512, 512]) -> (Batch, Channel, Height, Width)

print(f"Batch Target Shape: {targets.shape}")
# Mong đợi: torch.Size([4, 26]) -> (Batch, Num_Classes)

print("\nĐang hiển thị ảnh minh họa...")
show_batch(images, targets, TARGET_CLASSES)

print("=" * 30)
print("🔍 PHÂN TÍCH CHI TIẾT 1 BATCH")
print("=" * 30)

# 1. KIỂM TRA INPUT (IMAGES)
print(f"\n📸 [IMAGES TENSOR]")
print(f"1. Kích thước (Shape): {images.shape}")
print(f"   -> Ý nghĩa: [Batch_Size, Channels, Height, Width]")
print(f"   -> Mong đợi: [4, 3, 512, 512] (với batch=4, size=512)")

print(f"2. Kiểu dữ liệu (Dtype): {images.dtype}")
print(f"   -> Mong đợi: torch.float32 (Bắt buộc cho Model train)")

print(f"3. Miền giá trị (Min/Max): Min={images.min():.3f}, Max={images.max():.3f}")
print(f"   -> Lưu ý: Giá trị sẽ KHÔNG còn là 0-255 hay 0-1 nữa.")
print(f"   -> Vì đã qua Normalize ((x - mean) / std), giá trị sẽ nằm khoảng -2 đến +2.")

print(f"4. Mẫu dữ liệu (Pixel 5x5 góc trái trên của ảnh đầu tiên):")
print(images[0, 0, :5, :5])


# 2. KIỂM TRA OUTPUT (TARGETS)
print(f"\n🎯 [TARGETS TENSOR]")
print(f"1. Kích thước (Shape): {targets.shape}")
print(f"   -> Ý nghĩa: [Batch_Size, Num_Classes]")
print(f"   -> Mong đợi: [4, 26] (Số lượng bệnh trong danh sách)")

print(f"2. Kiểu dữ liệu (Dtype): {targets.dtype}")
print(f"   -> Mong đợi: torch.float32 (Bắt buộc để dùng BCEWithLogitsLoss)")

print(f"3. Mẫu vector nhãn của ảnh đầu tiên:")
print(targets[0])
print(f"   -> Ý nghĩa: 1.0 là có bệnh, 0.0 là không có.")
