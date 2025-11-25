import os
import cv2
import torch
import pydicom
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# 1. Danh sách bệnh
TARGET_CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "Interstitial lung disease",
    "Infiltration",
    "Lung Opacity",
    "Lung Cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
    "COPD",
    "Lung Tumor",
    "Pneumonia",
    "Tuberculosis",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


class VinDrClassifierDataset(Dataset):
    def __init__(
        self, csv_file, image_dir, transform=None, class_map=CLASS_TO_IDX, img_size=384
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.class_map = class_map
        self.img_size = img_size  # Cần biết kích thước đích để scale bbox

        # Đọc CSV
        df = pd.read_csv(csv_file)

        # Lọc bỏ class "No finding" khỏi danh sách bbox cần vẽ (nếu muốn)
        # Nhưng vẫn giữ dòng đó để biết ảnh đó là ảnh bình thường

        # Group dữ liệu: Gom cả labels và bboxes
        # Ta tạo dict để lưu: image_id -> list các annotations
        self.annotations = (
            df.groupby("image_id")
            .apply(
                lambda x: x[["class_name", "x_min", "y_min", "x_max", "y_max"]].to_dict(
                    "records"
                )
            )
            .to_dict()
        )

        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # 1. Đọc ảnh (Dùng hàm read_dicom hoặc cv2 như cũ)
        # Giả sử ta có biến 'image' (H_orig, W_orig, 3)
        img_id = self.image_ids[idx]

        # --- SỬA ĐỔI ĐƯỜNG DẪN ---
        # Kiểm tra file có đuôi .dicom hay không có đuôi
        dicom_path = os.path.join(self.image_dir, f"{img_id}.dicom")
        if not os.path.exists(dicom_path):
            # Thử trường hợp file không có đuôi (một số dataset Kaggle như vậy)
            dicom_path = os.path.join(self.image_dir, f"{img_id}")

        # Đọc ảnh bằng hàm custom
        image = self.read_dicom(dicom_path)

        h_orig, w_orig = image.shape[:2]

        # 2. Tạo Mask và Label Vector
        target_label = np.zeros(len(self.class_map), dtype=np.float32)
        target_mask = np.zeros(
            (len(self.class_map), self.img_size, self.img_size), dtype=np.float32
        )

        anns = self.annotations[img_id]

        for ann in anns:
            class_name = ann["class_name"]
            if class_name in self.class_map:
                class_idx = self.class_map[class_name]

                # a. Cập nhật Label Classification
                target_label[class_idx] = 1.0

                # b. Cập nhật Mask (Nếu có bbox hợp lệ)
                # Kiểm tra xem có bbox không (vì VinDr có giá trị NaN cho class No finding)
                if not np.isnan(ann["x_min"]):
                    # Scale bbox từ kích thước gốc về kích thước 384x384
                    scale_x = self.img_size / w_orig
                    scale_y = self.img_size / h_orig

                    x1 = int(ann["x_min"] * scale_x)
                    y1 = int(ann["y_min"] * scale_y)
                    x2 = int(ann["x_max"] * scale_x)
                    y2 = int(ann["y_max"] * scale_y)

                    # Vẽ hình chữ nhật màu trắng (1.0) lên mask của class đó
                    # Đảm bảo toạ độ không vượt quá img_size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.img_size, x2), min(self.img_size, y2)

                    target_mask[class_idx, y1:y2, x1:x2] = 1.0

        # 3. Transform (Resize ảnh)
        # Lưu ý: Vì ta đã tự vẽ mask theo kích thước img_size,
        # nên transform chỉ cần Resize ảnh về img_size là khớp.
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(target_label), torch.tensor(target_mask)
