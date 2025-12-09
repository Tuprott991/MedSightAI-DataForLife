import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2  # Thay pydicom bằng opencv
from PIL import Image


class VINDRCXRDataset(Dataset):
    def __init__(self, csv_file, image_dir, num_classes, target_size, map_size):
        """
        Khởi tạo Dataset cho file PNG.
        """
        # Đọc CSV
        self.data_df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.target_size = target_size
        self.map_size = map_size

        # 1. Chuẩn bị mapping Class ID
        self.image_ids = self.data_df["image_id"].unique().tolist()
        class_mapping_data = self.data_df[["class_name", "class_id"]].drop_duplicates()

        self.class_to_id = {}
        pathology_class_count = 0
        for _, row in class_mapping_data.iterrows():
            name = row["class_name"]
            original_id = row["class_id"]
            # Chỉ lấy 14 bệnh lý (ID 0-13)
            if name.lower() != "no finding" and 0 <= original_id <= 13:
                self.class_to_id[name] = int(original_id)
                pathology_class_count += 1

        self.num_classes = pathology_class_count

        # 2. Tối ưu hóa việc tra cứu kích thước gốc (Nếu có trong CSV)
        # Tạo dictionary để tra cứu nhanh width/height nếu CSV có cột này
        self.has_dim_info = (
            "width" in self.data_df.columns and "height" in self.data_df.columns
        )

        if self.has_dim_info:
            # BƯỚC QUAN TRỌNG:
            # 1. Chỉ lấy 3 cột cần thiết
            # 2. Loại bỏ các dòng trùng image_id (để mỗi ảnh chỉ còn 1 dòng duy nhất chứa width/height)
            unique_dims = self.data_df[["image_id", "width", "height"]].drop_duplicates(
                subset=["image_id"]
            )

            # 3. Giờ thì set_index sẽ an toàn và to_dict sẽ nhanh hơn nhiều
            self.dim_lookup = unique_dims.set_index("image_id").to_dict("index")
        else:
            self.dim_lookup = {}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Lấy các dòng annotation của ảnh này
        img_annotations = self.data_df[self.data_df["image_id"] == image_id]

        # --- 1. Load Ảnh PNG (Thay thế phần DICOM) ---
        img_path = os.path.join(self.image_dir, f"{image_id}.png")

        # Đọc ảnh grayscale (để khớp với logic cũ là 1 channel)
        # Nếu muốn dùng 3 kênh màu RGB thì bỏ flag cv2.IMREAD_GRAYSCALE
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:  # Xử lý lỗi nếu không đọc được ảnh
            return None, None, None, image_id

        # --- 2. Xác định Kích thước Gốc (QUAN TRỌNG ĐỂ TÍNH BBOX) ---
        # Nếu CSV có thông tin width/height gốc, ta ưu tiên dùng nó
        # (Vì nếu ảnh PNG đã bị resize trước đó, image.shape sẽ sai lệch với bbox gốc)
        if self.has_dim_info:
            dims = self.dim_lookup[image_id]
            original_w = dims["width"]
            original_h = dims["height"]
        else:
            # Nếu CSV không có, đành phải dùng kích thước thật của ảnh vừa đọc
            # Lưu ý: Nếu ảnh này đã bị resize (vd 512x512) thì bbox sẽ bị tính sai!
            original_h, original_w = image.shape[:2]

        if original_w <= 0 or original_h <= 0:
            return None, None, None, image_id

        # --- 3. Resize & Normalize ---
        # Resize về target_size (ví dụ 384x384)
        image = cv2.resize(image, (self.target_size, self.target_size))

        # Normalize 0-255 -> 0.0-1.0
        image = image.astype(np.float32) / 255.0

        # Chuyển sang Tensor [1, H, W]
        image = torch.from_numpy(image).unsqueeze(0)

        # --- 4. Tạo Label & Attention Map (Logic giữ nguyên) ---
        cls_label = torch.zeros(self.num_classes, dtype=torch.float32)
        present_classes = img_annotations["class_name"].unique()

        for cls_name in present_classes:
            cls_id = self.class_to_id.get(cls_name)
            if cls_id is not None:
                cls_label[cls_id] = 1.0

        # Tạo Map
        attn_maps = torch.zeros(
            (self.num_classes, self.map_size, self.map_size), dtype=torch.float32
        )

        for _, row in img_annotations.iterrows():
            cls_name = row["class_name"]
            cls_id = self.class_to_id.get(cls_name)

            if cls_id is None:
                continue
            if pd.isna(row[["x_min", "y_min", "x_max", "y_max"]]).any():
                continue

            # Tính tọa độ trên Map 12x12 dựa trên tỷ lệ với kích thước gốc
            x_min = int(row["x_min"] * (self.map_size / original_w))
            y_min = int(row["y_min"] * (self.map_size / original_h))
            x_max = int(row["x_max"] * (self.map_size / original_w))
            y_max = int(row["y_max"] * (self.map_size / original_h))

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.map_size, x_max)
            y_max = min(self.map_size, y_max)

            if x_max > x_min and y_max > y_min:
                attn_maps[cls_id, y_min:y_max, x_min:x_max] = 1.0

        return image, cls_label, attn_maps, image_id