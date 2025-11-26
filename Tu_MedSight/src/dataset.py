import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pydicom
import cv2


class VinDrCXRDataset(Dataset):
    """
    Dataset for VinDr-CXR (Chest X-ray) data.
    
    Structure:
    - annotations.csv: Contains bounding boxes and concept labels (class_name)
    - image_labels.csv: Contains disease classifications (multi-label)
    - Images are organized in train/ and test/ folders
    
    Args:
        root_dir: Root directory containing train/test folders
        split: 'train' or 'test'
        annotations_csv: Path to annotations CSV (bounding boxes + concepts)
        image_labels_csv: Path to image labels CSV (disease classifications)
        transform: Optional transforms to apply to images
        return_boxes: If True, return bounding boxes and concept labels
        radiologist_id: Filter by specific radiologist (e.g., 'R3', 'R6'), None for all
        use_multi_rater: If True, aggregate labels from multiple radiologists
    """
    
    def __init__(
        self,
        root_dir,
        split='train',
        annotations_csv=None,
        image_labels_csv=None,
        transform=None,
        return_boxes=False,
        radiologist_id=None,
        use_multi_rater=True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split, split)
        self.transform = transform
        self.return_boxes = return_boxes
        self.radiologist_id = radiologist_id
        self.use_multi_rater = use_multi_rater
        
        # Load annotations (concepts/findings with bounding boxes)
        if annotations_csv is None:
            annotations_csv = os.path.join(root_dir, f'annotations_{split}.csv')
        self.annotations_df = pd.read_csv(annotations_csv)
        
        # Load image labels (disease classifications)
        if image_labels_csv is None:
            image_labels_csv = os.path.join(root_dir, f'image_labels_{split}.csv')
        self.image_labels_df = pd.read_csv(image_labels_csv)
        
        # Filter by radiologist if specified
        if radiologist_id is not None:
            self.annotations_df = self.annotations_df[
                self.annotations_df['rad_ID'] == radiologist_id
            ]
            self.image_labels_df = self.image_labels_df[
                self.image_labels_df['rad_ID'] == radiologist_id
            ]
        
        # Get concept names (class_name column in annotations)
        self.concept_names = sorted(self.annotations_df['class_name'].unique().tolist())
        self.num_concepts = len(self.concept_names)
        self.concept_to_idx = {name: idx for idx, name in enumerate(self.concept_names)}
        
        # Get disease names (columns after rad_ID in image_labels)
        # Exclude 'image_id' and 'rad_ID' columns
        disease_cols = [col for col in self.image_labels_df.columns 
                       if col not in ['image_id', 'rad_ID']]
        self.disease_names = disease_cols
        self.num_diseases = len(self.disease_names)
        
        # Aggregate labels across radiologists if needed
        if use_multi_rater:
            # For image labels: average across radiologists
            self.image_labels_agg = self.image_labels_df.groupby('image_id')[disease_cols].mean().reset_index()
            
            # For annotations: keep all boxes from all radiologists
            # Or you can choose majority voting / consensus strategy
            self.annotations_agg = self.annotations_df.copy()
        else:
            self.image_labels_agg = self.image_labels_df.copy()
            self.annotations_agg = self.annotations_df.copy()
        
        # Get unique image IDs
        self.image_ids = sorted(self.image_labels_agg['image_id'].unique().tolist())
        
        print(f"[Dataset] Loaded {split} split:")
        print(f"  - Total images: {len(self.image_ids)}")
        print(f"  - Concepts: {self.num_concepts} ({', '.join(self.concept_names[:5])}...)")
        print(f"  - Diseases: {self.num_diseases} ({', '.join(self.disease_names)})")
        print(f"  - Annotations (boxes): {len(self.annotations_agg)}")
        if radiologist_id:
            print(f"  - Filtered by radiologist: {radiologist_id}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def read_dicom(self, path):
        """Read and process DICOM image."""
        try:
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array
            
            # Handle Photometric Interpretation (invert if MONOCHROME1)
            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == 'MONOCHROME1':
                    image = np.amax(image) - image
            
            # Normalize to 0-255 (uint8)
            image = image - np.min(image)
            image = image / (np.max(image) + 1e-8)
            image = (image * 255).astype(np.uint8)
            
            # Convert grayscale to RGB
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # Convert to PIL Image
            return Image.fromarray(image)
        except Exception as e:
            print(f"Error reading DICOM {path}: {e}")
            # Return black image if error
            return Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    
    def __getitem__(self, idx):
        # Get image ID
        image_id = self.image_ids[idx]
        
        # Load DICOM image
        # Try multiple file extensions
        image_path = os.path.join(self.image_dir, f'{image_id}.dicom')
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f'{image_id}.dcm')
        if not os.path.exists(image_path):
            # Try without extension (some datasets store DICOM without extension)
            image_path = os.path.join(self.image_dir, f'{image_id}')
        
        image = self.read_dicom(image_path)
        original_size = image.size  # (width, height)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get disease labels (multi-label classification)
        disease_row = self.image_labels_agg[self.image_labels_agg['image_id'] == image_id]
        disease_labels = disease_row[self.disease_names].values[0].astype(np.float32)
        disease_labels = torch.from_numpy(disease_labels)
        
        # Get concept annotations (bounding boxes + class_name)
        concept_annots = self.annotations_agg[self.annotations_agg['image_id'] == image_id]
        
        # Create concept presence vector (binary: concept present or not)
        concept_labels = np.zeros(self.num_concepts, dtype=np.float32)
        boxes = []
        box_concept_ids = []
        
        if len(concept_annots) > 0:
            for _, row in concept_annots.iterrows():
                concept_name = row['class_name']
                concept_idx = self.concept_to_idx.get(concept_name)
                
                if concept_idx is not None:
                    concept_labels[concept_idx] = 1.0
                    
                    # Store bounding box (normalized to [0, 1])
                    if self.return_boxes:
                        x_min = row['x_min'] / original_size[0]
                        y_min = row['y_min'] / original_size[1]
                        x_max = row['x_max'] / original_size[0]
                        y_max = row['y_max'] / original_size[1]
                        
                        boxes.append([x_min, y_min, x_max, y_max])
                        box_concept_ids.append(concept_idx)
        
        concept_labels = torch.from_numpy(concept_labels)
        
        # Prepare return dict
        sample = {
            'image': image,
            'image_id': image_id,
            'disease_labels': disease_labels,  # (num_diseases,) - multi-label for classification
            'concept_labels': concept_labels,   # (num_concepts,) - binary presence of concepts
        }
        
        # Add bounding boxes if requested
        if self.return_boxes:
            if len(boxes) > 0:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                box_concept_ids = torch.tensor(box_concept_ids, dtype=torch.long)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                box_concept_ids = torch.zeros((0,), dtype=torch.long)
            
            sample['boxes'] = boxes  # (N, 4) - normalized [x_min, y_min, x_max, y_max]
            sample['box_concept_ids'] = box_concept_ids  # (N,) - concept index for each box
        
        return sample
    
    def get_concept_name(self, idx):
        """Get concept name by index."""
        return self.concept_names[idx]
    
    def get_disease_name(self, idx):
        """Get disease name by index."""
        return self.disease_names[idx]
    
    def get_image_info(self, idx):
        """Get image information for debugging."""
        image_id = self.image_ids[idx]
        disease_row = self.image_labels_agg[self.image_labels_agg['image_id'] == image_id]
        concept_annots = self.annotations_agg[self.annotations_agg['image_id'] == image_id]
        
        info = {
            'image_id': image_id,
            'num_boxes': len(concept_annots),
            'concepts_present': concept_annots['class_name'].tolist() if len(concept_annots) > 0 else [],
            'diseases': {},
        }
        
        for disease in self.disease_names:
            value = disease_row[disease].values[0]
            if value > 0:
                info['diseases'][disease] = float(value)
        
        return info


def get_default_transforms(image_size=448, is_training=True):
    """
    Get default transforms for VinDr-CXR images.
    
    Args:
        image_size: Target image size (assumed square)
        is_training: If True, apply data augmentation
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats (common for medical imaging)
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return transform


def collate_fn_with_boxes(batch):
    """
    Custom collate function for batching samples with variable number of boxes.
    
    Returns:
        Dictionary with:
            - images: (B, 3, H, W)
            - disease_labels: (B, num_diseases)
            - concept_labels: (B, num_concepts)
            - boxes: List of (N_i, 4) tensors
            - box_concept_ids: List of (N_i,) tensors
            - image_ids: List of strings
    """
    images = torch.stack([item['image'] for item in batch])
    disease_labels = torch.stack([item['disease_labels'] for item in batch])
    concept_labels = torch.stack([item['concept_labels'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    
    result = {
        'images': images,
        'disease_labels': disease_labels,
        'concept_labels': concept_labels,
        'image_ids': image_ids,
    }
    
    # Handle boxes if present
    if 'boxes' in batch[0]:
        boxes = [item['boxes'] for item in batch]
        box_concept_ids = [item['box_concept_ids'] for item in batch]
        result['boxes'] = boxes
        result['box_concept_ids'] = box_concept_ids
    
    return result


# --- Quick test ---
if __name__ == "__main__":
    import sys
    
    # Test dataset loading
    root_dir = r"D:\Github Repos\SoftAI---DataForLife---MedSightAI"  # Adjust path
    
    print("=" * 80)
    print("Testing VinDrCXRDataset")
    print("=" * 80)
    
    # Create dataset
    transform = get_default_transforms(image_size=448, is_training=True)
    
    try:
        dataset = VinDrCXRDataset(
            root_dir=root_dir,
            split='train',
            transform=transform,
            return_boxes=True,
            radiologist_id=None,  # Use all radiologists
            use_multi_rater=True,
        )
        
        print(f"\nDataset created successfully!")
        print(f"Total samples: {len(dataset)}")
        print(f"\nConcept names ({len(dataset.concept_names)}):")
        for i, name in enumerate(dataset.concept_names):
            print(f"  {i}: {name}")
        
        print(f"\nDisease names ({len(dataset.disease_names)}):")
        for i, name in enumerate(dataset.disease_names):
            print(f"  {i}: {name}")
        
        # Test loading a sample
        print("\n" + "=" * 80)
        print("Testing sample loading...")
        print("=" * 80)
        
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image ID: {sample['image_id']}")
        print(f"  Disease labels shape: {sample['disease_labels'].shape}")
        print(f"  Disease labels (non-zero): {sample['disease_labels'].nonzero().squeeze().tolist()}")
        print(f"  Concept labels shape: {sample['concept_labels'].shape}")
        print(f"  Concept labels (non-zero): {sample['concept_labels'].nonzero().squeeze().tolist()}")
        
        if 'boxes' in sample:
            print(f"  Boxes shape: {sample['boxes'].shape}")
            print(f"  Box concept IDs: {sample['box_concept_ids'].tolist()}")
        
        # Get detailed info
        info = dataset.get_image_info(0)
        print(f"\nDetailed info:")
        print(f"  Concepts present: {info['concepts_present']}")
        print(f"  Diseases: {info['diseases']}")
        
        # Test DataLoader with custom collate
        from torch.utils.data import DataLoader
        
        print("\n" + "=" * 80)
        print("Testing DataLoader...")
        print("=" * 80)
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_with_boxes,
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Disease labels shape: {batch['disease_labels'].shape}")
        print(f"  Concept labels shape: {batch['concept_labels'].shape}")
        print(f"  Number of samples with boxes: {len(batch['boxes'])}")
        for i, boxes in enumerate(batch['boxes']):
            print(f"    Sample {i}: {boxes.shape[0]} boxes")
        
        print("\n✓ All tests passed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease adjust the root_dir path to point to your VinDr-CXR dataset.")
        print("Expected structure:")
        print("  root_dir/")
        print("    ├── train/")
        print("    │   └── train/  (images)")
        print("    ├── test/")
        print("    │   └── test/   (images)")
        print("    ├── annotations_train.csv")
        print("    ├── annotations_test.csv")
        print("    ├── image_labels_train.csv")
        print("    └── image_labels_test.csv")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
