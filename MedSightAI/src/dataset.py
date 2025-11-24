import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F


class TBX11KDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=384, processor=None):
        """
        Args:
            root_dir (str): Đường dẫn gốc tới folder dataset (vd: /kaggle/input/tbx-11/TBX11K)
            split (str): 'train', 'val', hoặc 'test'.
            img_size (int): Kích thước ảnh đầu vào.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.processor = processor

        # 1. Xác định file JSON tương ứng
        if split == "train":
            json_file = "TBX11K_train.json"
        elif split == "val":
            json_file = "TBX11K_val.json"
        elif split == "test":
            json_file = "all_test.json"
        else:
            raise ValueError(f"Split {split} không hợp lệ!")

        json_path = os.path.join(root_dir, "annotations", "json", json_file)
        self.img_dir = os.path.join(root_dir, "imgs")

        # 2. Load và Parse JSON (COCO Format)
        print(f"Loading annotations from {json_path}...")
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        # Tạo mapping để tra cứu nhanh: image_id -> annotation
        self.img_info = []  # List chứa dict thông tin ảnh
        self.img_to_anns = {}  # Dict: image_id -> list of annotations

        # Lấy danh sách ảnh
        if "images" in coco_data:
            self.img_info = coco_data["images"]
        else:
            # Fallback nếu format lạ (ít xảy ra với TBX11K chuẩn)
            self.img_info = coco_data

        # Lấy annotations và gom nhóm theo image_id
        if "annotations" in coco_data:
            for ann in coco_data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

        print(f"Đã load {len(self.img_info)} ảnh cho tập {split}.")

        # 3. Transform cơ bản
        self.normalize = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        # Lấy thông tin ảnh
        img_data = self.img_info[idx]
        img_id = img_data.get("id")

        # Lấy tên file (xử lý cả key 'file_name' và 'fname' đề phòng)
        img_name = img_data.get("file_name") or img_data.get("fname")

        # Tạo đường dẫn ảnh (TBX11K có thể để ảnh trong subfolder, cần check thực tế)
        # Giả sử ảnh nằm phẳng trong thư mục imgs/
        img_path = os.path.join(self.img_dir, img_name)

        # 1. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback: Thử tìm trong các subfolder phổ biến của TBX11K nếu có
            # (TBX11K đôi khi chia folder 'tb', 'healthy' bên trong 'imgs')
            # Đoạn này bạn có thể tùy chỉnh nếu folder imgs có cấu trúc con.
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

        w_orig, h_orig = image.size

        # 2. Lấy Bounding Boxes từ dict đã map
        anns = self.img_to_anns.get(img_id, [])

        # 3. Tạo Mask nền đen
        mask = torch.zeros((h_orig, w_orig), dtype=torch.float32)
        has_disease = 0.0

        for ann in anns:
            bbox = ann.get("bbox", [])  # Format COCO: [x, y, w, h]
            if len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                # Chuyển sang toạ độ (x1, y1, x2, y2) và clip trong khung ảnh
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_orig, x + w)
                y2 = min(h_orig, y + h)

                # Vẽ vùng bệnh lên mask (màu trắng = 1)
                mask[y1:y2, x1:x2] = 1.0
                has_disease = 1.0  # Đánh dấu là có bệnh

        # 4. Resize Image & Mask
        # Ảnh dùng bicubic để đẹp, Mask dùng nearest để giữ biên sắc nét (0 và 1)
        image = F.resize(
            image,
            (self.img_size, self.img_size),
            interpolation=T.InterpolationMode.BICUBIC,
        )
        mask_img = Image.fromarray(mask.numpy())
        mask = F.resize(
            mask_img,
            (self.img_size, self.img_size),
            interpolation=T.InterpolationMode.NEAREST,
        )
        mask_tensor = T.ToTensor()(mask)  # (1, H, W)

        # 5. Normalize Image Input
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            image_tensor = inputs.pixel_values.squeeze(0)
        else:
            image_tensor = self.normalize(image)

        return {
            "pixel_values": image_tensor,  # Input cho Backbone
            "mask": mask_tensor,  # Target cho Segmentation Loss
            "label": torch.tensor(
                has_disease, dtype=torch.float32
            ),  # Target cho Classification Loss
            "name": img_name,  # Dùng để hiển thị/retrieval sau này
        }
