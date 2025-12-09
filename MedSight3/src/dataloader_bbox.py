"""
Extended dataloader with bounding box support.
Use this for Stage 1 training when bbox annotations are available.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import dicom_to_image


class CSRDatasetWithBBox(Dataset):
    """Dataset with bounding box annotations for improved CAM localization."""
    
    def __init__(self, csv_file, bbox_csv_file, root_dir, phase='train', transform=None):
        """
        Args:
            csv_file: Path to labels CSV (image-level annotations)
            bbox_csv_file: Path to bbox CSV with columns:
                          [image_id, rad_id, class_name, x_min, y_min, x_max, y_max]
            root_dir: Folder containing images
            phase: 'train' or 'test'
            transform: Image transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        
        # 1. Load image-level labels
        df = pd.read_csv(csv_file)
        
        # Parse columns
        meta_cols = ['image_id', 'rad_id']
        target_keywords = ['COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'No finding']
        target_cols = []
        
        for col in df.columns:
            if col in meta_cols:
                continue
            if 'other' in col.lower():
                if 'disease' in col.lower():
                    target_cols.append(col)
                continue
            if any(keyword.lower() in col.lower() for keyword in target_keywords):
                target_cols.append(col)
        
        concept_cols = [c for c in df.columns if c not in target_cols + meta_cols]
        
        # Aggregate by image_id (max across radiologists)
        self.data = df.groupby('image_id')[concept_cols + target_cols].max().reset_index()
        
        self.concept_cols = concept_cols
        self.target_cols = target_cols
        
        # Create concept name to index mapping
        self.concept_name_to_idx = {name: idx for idx, name in enumerate(concept_cols)}
        
        # 2. Load bounding box annotations
        self.bbox_df = pd.read_csv(bbox_csv_file)
        
        print(f"📊 Dataset '{phase}' with BBox loaded:")
        print(f"  - Images: {len(self.data)}")
        print(f"  - Concepts: {len(concept_cols)}")
        print(f"  - Targets: {len(target_cols)}")
        print(f"  - BBox annotations: {len(self.bbox_df)} rows")
        
        # Check how many images have bboxes
        images_with_bbox = self.bbox_df[
            (self.bbox_df['class_name'] != 'No finding') & 
            (~self.bbox_df['x_min'].isna())
        ]['image_id'].nunique()
        print(f"  - Images with bbox: {images_with_bbox} / {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def _parse_bboxes(self, image_id, img_width, img_height):
        """Parse bounding boxes for a single image."""
        sample_df = self.bbox_df[self.bbox_df['image_id'] == image_id]
        
        if len(sample_df) == 0:
            return []
        
        bboxes = []
        
        for _, row in sample_df.iterrows():
            class_name = row['class_name']
            
            # Skip "No finding" or missing
            if class_name == 'No finding' or pd.isna(class_name):
                continue
            
            # Skip if no bbox coords
            if pd.isna(row['x_min']) or pd.isna(row['y_min']):
                continue
            
            # Get concept index
            if class_name not in self.concept_name_to_idx:
                continue  # Unknown concept or it's a disease class
            
            concept_idx = self.concept_name_to_idx[class_name]
            
            # Normalize bbox to [0, 1]
            x_min = float(row['x_min']) / img_width
            y_min = float(row['y_min']) / img_height
            x_max = float(row['x_max']) / img_width
            y_max = float(row['y_max']) / img_height
            
            # Clamp to [0, 1]
            x_min = max(0.0, min(x_min, 1.0))
            x_max = max(0.0, min(x_max, 1.0))
            y_min = max(0.0, min(y_min, 1.0))
            y_max = max(0.0, min(y_max, 1.0))
            
            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue
            
            bboxes.append({
                'concept_idx': concept_idx,
                'bbox': [x_min, y_min, x_max, y_max]
            })
        
        return bboxes
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        
        # Load image (try PNG first, then JPG, then DICOM)
        img_path_png = os.path.join(self.root_dir, f"{image_id}.png")
        img_path_jpg = os.path.join(self.root_dir, f"{image_id}.jpg")
        img_path_dicom = os.path.join(self.root_dir, f"{image_id}.dicom")
        
        if os.path.exists(img_path_png):
            image = Image.open(img_path_png).convert('RGB')
        elif os.path.exists(img_path_jpg):
            image = Image.open(img_path_jpg).convert('RGB')
        elif os.path.exists(img_path_dicom):
            image = dicom_to_image(img_path_dicom)
        else:
            raise FileNotFoundError(f"Image not found: {image_id}")
        
        # Get original image dimensions for bbox normalization
        img_width, img_height = image.size
        
        # Parse bboxes BEFORE transforming image
        bboxes = self._parse_bboxes(image_id, img_width, img_height)
        
        # Transform image
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        concepts = torch.tensor(row[self.concept_cols].values, dtype=torch.float32)
        targets = torch.tensor(row[self.target_cols].values, dtype=torch.float32)
        
        return {
            'image': image,
            'concepts': concepts,
            'targets': targets,
            'bboxes': bboxes,  # List of bbox dicts
            'image_id': image_id
        }


def get_dataloaders_with_bbox(train_csv, test_csv, bbox_csv, train_dir, test_dir,
                               batch_size=16, rank=-1, world_size=1, val_split=0.1):
    """
    Create dataloaders with bounding box support.
    
    Args:
        bbox_csv: Path to CSV with bbox annotations
        Other args: Same as get_dataloaders()
    
    Returns:
        train_loader, val_loader, test_loader, num_concepts, num_classes, train_sampler
    """
    from torch.utils.data import random_split
    from torch.utils.data.distributed import DistributedSampler
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    full_train_dataset = CSRDatasetWithBBox(train_csv, bbox_csv, train_dir, 
                                            phase='train', transform=train_transform)
    test_dataset = CSRDatasetWithBBox(test_csv, bbox_csv, test_dir,
                                      phase='test', transform=val_transform)
    
    # Split train into train + val
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Get metadata
    num_concepts = len(full_train_dataset.concept_cols)
    num_classes = len(full_train_dataset.target_cols)
    
    # Create dataloaders
    if world_size > 1:
        # DDP mode
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                          rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                        rank=rank, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               sampler=val_sampler, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 sampler=DistributedSampler(test_dataset, num_replicas=world_size,
                                                           rank=rank, shuffle=False),
                                 num_workers=4)
    else:
        # Single GPU
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, num_concepts, num_classes, train_sampler
