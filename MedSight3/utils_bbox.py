"""
Utilities for training with bounding box supervision.
Improves CAM localization quality significantly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class BBoxGuidedConceptLoss(nn.Module):
    """
    Concept loss with bounding box supervision for better localization.
    
    Combines:
    1. Global classification loss (BCE on max-pooled CAMs)
    2. Spatial localization loss (CAMs should be high inside bbox, low outside)
    """
    
    def __init__(self, alpha=1.0, beta=0.5):
        """
        Args:
            alpha: Weight for classification loss
            beta: Weight for localization loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, cams, concepts_gt, bboxes=None):
        """
        Args:
            cams: Concept Activation Maps (B, K, H, W)
            concepts_gt: Binary labels (B, K) - 1 if concept present
            bboxes: List of bounding boxes for each sample, each is dict:
                    {
                        'concept_idx': int,  # which concept (0 to K-1)
                        'bbox': [x_min, y_min, x_max, y_max],  # normalized [0, 1]
                    }
                    If None, falls back to standard BCE loss
        
        Returns:
            loss: Combined classification + localization loss
        """
        B, K, H, W = cams.shape
        device = cams.device
        
        # 1. Classification Loss: Global max pooling
        concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)  # (B, K)
        cls_loss = self.bce_loss(concept_logits, concepts_gt).mean()
        
        # 2. Localization Loss: Only if bboxes provided
        if bboxes is None or len(bboxes) == 0:
            return self.alpha * cls_loss
        
        loc_loss = 0.0
        num_boxes = 0
        
        for batch_idx, sample_bboxes in enumerate(bboxes):
            if sample_bboxes is None or len(sample_bboxes) == 0:
                continue
            
            for bbox_info in sample_bboxes:
                concept_idx = bbox_info['concept_idx']
                bbox = bbox_info['bbox']  # [x_min, y_min, x_max, y_max] in [0, 1]
                
                # Get CAM for this concept
                cam = cams[batch_idx, concept_idx]  # (H, W)
                
                # Create bbox mask (1 inside bbox, 0 outside)
                x_min, y_min, x_max, y_max = bbox
                mask = self._create_bbox_mask(H, W, x_min, y_min, x_max, y_max, device)
                
                # Inside bbox: CAM should be high (positive)
                # Outside bbox: CAM should be low (negative or close to 0)
                inside_mask = mask
                outside_mask = 1 - mask
                
                # Loss: maximize CAM inside, minimize CAM outside
                # Using hinge loss style: want cam > 0 inside, cam < 0 outside
                inside_loss = F.relu(-cam * inside_mask).sum() / (inside_mask.sum() + 1e-6)
                outside_loss = F.relu(cam * outside_mask).sum() / (outside_mask.sum() + 1e-6)
                
                loc_loss += inside_loss + outside_loss
                num_boxes += 1
        
        if num_boxes > 0:
            loc_loss = loc_loss / num_boxes
        
        # Combined loss
        total_loss = self.alpha * cls_loss + self.beta * loc_loss
        
        return total_loss
    
    def _create_bbox_mask(self, H, W, x_min, y_min, x_max, y_max, device):
        """
        Create binary mask for bounding box.
        
        Args:
            H, W: Height and width of feature map
            x_min, y_min, x_max, y_max: Normalized bbox coords [0, 1]
            device: torch device
        
        Returns:
            mask: (H, W) binary mask, 1 inside bbox, 0 outside
        """
        # Convert normalized coords to feature map coords
        x_min_px = int(x_min * W)
        x_max_px = int(x_max * W)
        y_min_px = int(y_min * H)
        y_max_px = int(y_max * H)
        
        # Clamp to valid range
        x_min_px = max(0, min(x_min_px, W - 1))
        x_max_px = max(0, min(x_max_px, W))
        y_min_px = max(0, min(y_min_px, H - 1))
        y_max_px = max(0, min(y_max_px, H))
        
        # Create mask
        mask = torch.zeros((H, W), device=device)
        mask[y_min_px:y_max_px, x_min_px:x_max_px] = 1.0
        
        return mask


def parse_bbox_annotations(df, image_id, img_width, img_height, concept_name_to_idx):
    """
    Parse bounding box annotations from DataFrame for a single image.
    
    Args:
        df: DataFrame with columns [image_id, class_name, x_min, y_min, x_max, y_max]
        image_id: ID of the image
        img_width: Original image width (for normalization)
        img_height: Original image height (for normalization)
        concept_name_to_idx: Dict mapping concept names to indices
    
    Returns:
        bboxes: List of dicts with 'concept_idx' and 'bbox' (normalized)
                Returns empty list if no bboxes or "No finding"
    """
    # Filter for this image
    sample_df = df[df['image_id'] == image_id]
    
    if len(sample_df) == 0:
        return []
    
    bboxes = []
    
    for _, row in sample_df.iterrows():
        class_name = row['class_name']
        
        # Skip "No finding" or missing class names
        if class_name == 'No finding' or pd.isna(class_name):
            continue
        
        # Skip if no bbox coords
        if pd.isna(row['x_min']) or pd.isna(row['y_min']):
            continue
        
        # Get concept index
        if class_name not in concept_name_to_idx:
            continue  # Unknown concept
        
        concept_idx = concept_name_to_idx[class_name]
        
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


# Example usage in dataloader
"""
In your dataset __getitem__:

def __getitem__(self, idx):
    image_id = self.data.iloc[idx]['image_id']
    image = load_image(image_id)  # PIL Image
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Parse bboxes
    bboxes = parse_bbox_annotations(
        self.bbox_df,  # DataFrame with bbox annotations
        image_id,
        img_width,
        img_height,
        self.concept_name_to_idx
    )
    
    # Transform image
    image = self.transform(image)
    
    return {
        'image': image,
        'concepts': concepts,
        'targets': targets,
        'bboxes': bboxes  # Add bboxes to batch
    }

In your training loop:

# Stage 1 with bbox supervision
criterion_s1 = BBoxGuidedConceptLoss(alpha=1.0, beta=0.5)

for batch in train_loader:
    images = batch['image'].to(device)
    concepts = batch['concepts'].to(device)
    bboxes = batch['bboxes']  # List of bbox lists
    
    outputs = model(images)
    cams = outputs['cams']
    
    # Loss with bbox guidance
    loss = criterion_s1(cams, concepts, bboxes)
    
    loss.backward()
    optimizer.step()
"""
