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


# 2. Dataset Class hỗ trợ DICOM
class VinDrClassifierDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, class_map=CLASS_TO_IDX):
        self.image_dir = image_dir
        self.transform = transform
        self.class_map = class_map

        # Đọc CSV
        df = pd.read_csv(csv_file)
        df = df[df["class_name"].isin(self.class_map.keys())]

        # Gộp nhãn theo ảnh
        self.data = df.groupby("image_id")["class_name"].apply(set).reset_index()
        self.data.columns = ["image_id", "labels"]  # Đặt tên cột rõ ràng

        self.image_ids = self.data["image_id"].values
        self.labels = self.data["labels"].values

    def __len__(self):
        return len(self.image_ids)

    def read_dicom(self, path):
        """Hàm đọc và xử lý ảnh DICOM"""
        try:
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array

            # Xử lý Photometric Interpretation (Đảo màu nếu cần)
            # Một số ảnh DICOM lưu dạng MONOCHROME1 (đen là trắng), cần đảo lại
            if hasattr(dcm, "PhotometricInterpretation"):
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    image = np.amax(image) - image

            # Chuẩn hóa về khoảng 0-255 (uint8)
            # Vì DICOM thường là 12-bit hoặc 14-bit
            image = image - np.min(image)
            image = image / np.max(image)
            image = (image * 255).astype(np.uint8)

            # Chuyển từ 1 kênh (Grayscale) sang 3 kênh (RGB) để khớp với input của Model
            image = np.stack([image] * 3, axis=-1)

            return image
        except Exception as e:
            print(f"Lỗi đọc DICOM {path}: {e}")
            # Trả về ảnh đen nếu lỗi
            return np.zeros((512, 512, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # --- SỬA ĐỔI ĐƯỜNG DẪN ---
        # Kiểm tra file có đuôi .dicom hay không có đuôi
        dicom_path = os.path.join(self.image_dir, f"{img_id}.dicom")
        if not os.path.exists(dicom_path):
            # Thử trường hợp file không có đuôi (một số dataset Kaggle như vậy)
            dicom_path = os.path.join(self.image_dir, f"{img_id}")

        # Đọc ảnh bằng hàm custom
        image = self.read_dicom(dicom_path)

        # Tạo vector nhãn
        current_labels = self.labels[idx]
        target = np.zeros(len(self.class_map), dtype=np.float32)

        for label_name in current_labels:
            if label_name in self.class_map:
                target[self.class_map[label_name]] = 1.0

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(target)
