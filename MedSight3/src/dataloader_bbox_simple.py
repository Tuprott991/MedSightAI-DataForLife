"""
Simple dataloader for bounding boxes that are ALREADY in 224x224 coordinates.
Use this when your images are already preprocessed to 224x224 and bbox annotations 
have already been adjusted to match the resized images.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import dicom_to_image


class CSRDatasetWithBBoxSimple(Dataset):
    """Dataset with bounding box annotations that are already in 224x224 space."""
    
    def __init__(self, csv_file, bbox_csv_file, root_dir, phase='train', transform=None):
        """
        Args:
            csv_file: Path to labels CSV (image-level annotations)
            bbox_csv_file: Path to bbox CSV with columns:
                          [image_id, class_name, x_min, y_min, x_max, y_max]
                          WHERE bbox coordinates are ALREADY in 224x224 space
            root_dir: Folder containing images (already 224x224)
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
        
        # 2. Load bounding box annotations (ALREADY in 224x224 space)
        self.bbox_df = pd.read_csv(bbox_csv_file)
        
        print(f"📊 Dataset '{phase}' with BBox (Simple) loaded:")
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
        print(f"  ⚠️  Assuming bbox coordinates are ALREADY in 224x224 space!")
    
    def __len__(self):
        return len(self.data)
    
    def _parse_bboxes(self, image_id):
        """Parse bounding boxes for a single image (already in 224x224 space)."""
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
            
            # Bbox coordinates are ALREADY in 224x224 pixel space
            # Just normalize to [0, 1]
            x_min = float(row['x_min']) / 224.0
            y_min = float(row['y_min']) / 224.0
            x_max = float(row['x_max']) / 224.0
            y_max = float(row['y_max']) / 224.0
            
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
        
        # Parse bboxes (already in 224x224, just normalize to [0, 1])
        bboxes = self._parse_bboxes(image_id)
        
        # Transform image
        if self.transform:
            image = self.transform(image)
        
        # Get labels - ensure numeric conversion
        concepts = torch.tensor(row[self.concept_cols].values.astype(float), dtype=torch.float32)
        targets = torch.tensor(row[self.target_cols].values.astype(float), dtype=torch.float32)
        
        return {
            'image': image,
            'concepts': concepts,
            'targets': targets,
            'bboxes': bboxes,  # List of bbox dicts with normalized coords [0, 1]
            'image_id': image_id
        }


def get_dataloaders_with_bbox_simple(train_csv, test_csv, train_bbox_csv, test_bbox_csv, 
                                      train_dir, test_dir, batch_size=16, rank=-1, world_size=1, val_split=0.1):
    """
    Create dataloaders with bounding box support (simple version).
    Use this when bbox annotations are ALREADY in 224x224 space.
    
    Args:
        train_csv: Path to training labels CSV
        test_csv: Path to test labels CSV
        train_bbox_csv: Path to CSV with bbox annotations for train set (in 224x224 space)
        test_bbox_csv: Path to CSV with bbox annotations for test set (in 224x224 space)
        train_dir: Directory with training images (already 224x224)
        test_dir: Directory with test images (already 224x224)
        batch_size: Batch size per GPU
        rank: GPU rank for DDP
        world_size: Total number of GPUs
        val_split: Validation split ratio
    
    Returns:
        train_loader, val_loader, test_loader, num_concepts, num_classes, train_sampler
    """
    from torch.utils.data import random_split
    from torch.utils.data.distributed import DistributedSampler
    
    # Transforms - images are already 224x224, so only normalize
    # Still apply augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure 224x224 (should be no-op if already sized)
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
    full_train_dataset = CSRDatasetWithBBoxSimple(train_csv, train_bbox_csv, train_dir, 
                                                   phase='train', transform=train_transform)
    test_dataset = CSRDatasetWithBBoxSimple(test_csv, test_bbox_csv, test_dir,
                                            phase='test', transform=val_transform)
    
    # Split train into train + val
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Get metadata
    num_concepts = len(full_train_dataset.concept_cols)
    num_classes = len(full_train_dataset.target_cols)
    
    # DDP: Create DistributedSampler for training
    if rank >= 0 and world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_concepts, num_classes, train_sampler
