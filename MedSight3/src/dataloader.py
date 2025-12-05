import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import dicom_to_image

class CSRDataset(Dataset):
    def __init__(self, csv_file, root_dir, phase='train', transform=None):
        """
        Args:
            csv_file: Đường dẫn file labels.csv
            root_dir: Folder chứa ảnh ('train/' hoặc 'test/')
            phase: 'train' hoặc 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 1. Load và xử lý CSV
        df = pd.read_csv(csv_file)
        
        # Tách tên cột tự động
        # Meta columns (không phải label)
        meta_cols = ['image_id', 'rad_id']
        
        # Known target diseases (tự động match các variants)
        # Note: 'Other lesion' là concept, chỉ 'Other disease/diseases' mới là target
        target_keywords = ['COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'No finding']
        target_cols = []
        
        for col in df.columns:
            if col in meta_cols:
                continue
            
            # Special handling for 'Other' - chỉ match 'Other disease(s)', không match 'Other lesion'
            if 'other' in col.lower():
                if 'disease' in col.lower():
                    target_cols.append(col)
                # Skip 'Other lesion' - nó là concept
                continue
            
            # Check nếu column name chứa target keyword
            if any(keyword.lower() in col.lower() for keyword in target_keywords):
                target_cols.append(col)
        
        # Concept columns = tất cả còn lại (không phải meta và target)
        concept_cols = [c for c in df.columns if c not in target_cols + meta_cols]
        
        print(f"📊 Dataset '{phase}' loaded:")
        print(f"  - Concepts: {len(concept_cols)} columns: {concept_cols[:5]}...")
        print(f"  - Targets: {len(target_cols)} columns: {target_cols}")
        
        self.concept_names = concept_cols
        self.target_names = target_cols
        
        # 2. Aggregation (Gộp nhãn từ nhiều bác sĩ)
        # MAX (Logic OR): Nếu bất kỳ bác sĩ nào thấy bệnh → đánh dấu có bệnh
        # Nguyên tắc y khoa: Nhầm còn hơn bỏ sót (High Sensitivity > High Specificity)
        self.data = df.groupby('image_id')[concept_cols + target_cols].max().reset_index()
        
        # Fill NaN bằng 0 (nếu có)
        self.data = self.data.fillna(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        
        # Tạo đường dẫn ảnh (giả sử file dicom có đuôi .dicom hoặc không đuôi)
        # Cần check thực tế file trong folder tên là gì (VD: id.dicom)
        img_path = os.path.join(self.root_dir, f"{image_id}.dicom") 
        if not os.path.exists(img_path):
             img_path = os.path.join(self.root_dir, image_id) # Nếu ko có đuôi
             
        # Xử lý ảnh
        image = dicom_to_image(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        # Lấy nhãn
        concepts = torch.tensor(row[self.concept_names].values.astype('float32'))
        targets = torch.tensor(row[self.target_names].values.astype('float32')) # Multi-label targets (nếu 1 ảnh bị nhiều bệnh)
        # Hoặc dùng argmax nếu là Single-label classification
        # targets = torch.tensor(np.argmax(row[self.target_names].values), dtype=torch.long)
        
        return {
            'image': image,
            'concepts': concepts,
            'targets': targets,
            'image_id': image_id
        }

def get_dataloaders(train_csv, test_csv, train_dir, test_dir, batch_size=32):
    # Augmentation cho MedMAE (Input 224x224)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Contrast rất quan trọng với X-ray
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CSRDataset(train_csv, train_dir, phase='train', transform=train_transform)
    test_dataset = CSRDataset(test_csv, test_dir, phase='test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)  # Giữ workers alive giữa các epochs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    return train_loader, test_loader, len(train_dataset.concept_names), len(train_dataset.target_names)