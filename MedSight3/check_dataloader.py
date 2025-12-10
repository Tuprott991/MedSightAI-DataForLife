"""
Comprehensive dataloader verification script.
This will check:
1. Data loading correctness
2. Label distribution and class balance
3. BBox annotations quality
4. Image statistics
5. Model forward pass sanity check
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataloader_bbox_simple import get_dataloaders_with_bbox_simple, CSRDatasetWithBBoxSimple
from src.model import CSR
from utils_bbox import BBoxGuidedConceptLoss


def check_csv_labels(csv_path, bbox_csv_path=None):
    """Check label distribution in CSV files."""
    print(f"\n{'='*80}")
    print(f"📋 CHECKING CSV: {csv_path}")
    print(f"{'='*80}")
    
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    print(f"Unique images: {df['image_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    
    # Identify concept and target columns
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
    
    print(f"\n📊 Concepts ({len(concept_cols)}): {concept_cols}")
    print(f"🎯 Targets ({len(target_cols)}): {target_cols}")
    
    # Aggregate by image_id (like in dataloader)
    df_agg = df.groupby('image_id')[concept_cols + target_cols].max().reset_index()
    
    print(f"\n📈 CONCEPT LABEL DISTRIBUTION (after aggregation):")
    print(f"{'Concept':<40} {'Positive':<10} {'Negative':<10} {'Pos %':<10} {'Pos Weight':<10}")
    print("-" * 80)
    
    concept_stats = []
    for col in concept_cols:
        pos_count = (df_agg[col] == 1).sum()
        neg_count = (df_agg[col] == 0).sum()
        pos_pct = pos_count / len(df_agg) * 100
        pos_weight = neg_count / max(pos_count, 1)
        concept_stats.append({
            'name': col,
            'pos': pos_count,
            'neg': neg_count,
            'pos_pct': pos_pct,
            'pos_weight': pos_weight
        })
        print(f"{col:<40} {pos_count:<10} {neg_count:<10} {pos_pct:<10.2f} {pos_weight:<10.2f}")
    
    print(f"\n📈 TARGET LABEL DISTRIBUTION:")
    print(f"{'Target':<40} {'Positive':<10} {'Negative':<10} {'Pos %':<10}")
    print("-" * 80)
    
    target_stats = []
    for col in target_cols:
        pos_count = (df_agg[col] == 1).sum()
        neg_count = (df_agg[col] == 0).sum()
        pos_pct = pos_count / len(df_agg) * 100
        target_stats.append({
            'name': col,
            'pos': pos_count,
            'neg': neg_count,
            'pos_pct': pos_pct
        })
        print(f"{col:<40} {pos_count:<10} {neg_count:<10} {pos_pct:<10.2f}")
    
    # Check for all-zero or all-one columns (these can't be learned!)
    print(f"\n⚠️  POTENTIAL ISSUES:")
    
    # Concepts with very few positives (< 1%)
    rare_concepts = [s for s in concept_stats if s['pos_pct'] < 1.0]
    if rare_concepts:
        print(f"  ❌ {len(rare_concepts)} concepts have < 1% positive samples:")
        for c in rare_concepts:
            print(f"     - {c['name']}: {c['pos_pct']:.2f}% ({c['pos']} samples)")
    
    # Concepts with very few negatives (> 99%)
    common_concepts = [s for s in concept_stats if s['pos_pct'] > 99.0]
    if common_concepts:
        print(f"  ❌ {len(common_concepts)} concepts have > 99% positive samples (almost always present):")
        for c in common_concepts:
            print(f"     - {c['name']}: {c['pos_pct']:.2f}%")
    
    # Targets with issues
    rare_targets = [s for s in target_stats if s['pos_pct'] < 1.0]
    if rare_targets:
        print(f"  ❌ {len(rare_targets)} targets have < 1% positive samples:")
        for t in rare_targets:
            print(f"     - {t['name']}: {t['pos_pct']:.2f}% ({t['pos']} samples)")
    
    # Check bbox annotations if provided
    if bbox_csv_path:
        print(f"\n📦 CHECKING BBOX ANNOTATIONS: {bbox_csv_path}")
        bbox_df = pd.read_csv(bbox_csv_path)
        print(f"Total bbox rows: {len(bbox_df)}")
        
        # Count valid bboxes (not "No finding" and has coordinates)
        valid_bboxes = bbox_df[
            (bbox_df['class_name'] != 'No finding') & 
            (~bbox_df['x_min'].isna())
        ]
        print(f"Valid bboxes: {len(valid_bboxes)}")
        print(f"Images with bbox: {valid_bboxes['image_id'].nunique()}")
        
        # Check bbox class distribution
        print(f"\n📦 BBox Class Distribution:")
        class_counts = valid_bboxes['class_name'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count} boxes")
        
        # Check if bbox classes match concept names
        bbox_classes = set(valid_bboxes['class_name'].unique())
        concept_names = set(concept_cols)
        
        unmatched_bbox = bbox_classes - concept_names
        if unmatched_bbox:
            print(f"\n  ⚠️  WARNING: {len(unmatched_bbox)} bbox classes NOT in concept columns:")
            for cls in unmatched_bbox:
                print(f"     - {cls}")
        
        # Check coordinate ranges (should be in 224x224 space for simple dataloader)
        print(f"\n📏 BBox Coordinate Statistics (should be in 0-224 range for simple loader):")
        for coord in ['x_min', 'y_min', 'x_max', 'y_max']:
            valid_coords = valid_bboxes[coord].dropna()
            print(f"  {coord}: min={valid_coords.min():.1f}, max={valid_coords.max():.1f}, "
                  f"mean={valid_coords.mean():.1f}")
        
        # Check for invalid boxes
        invalid_boxes = valid_bboxes[
            (valid_bboxes['x_max'] <= valid_bboxes['x_min']) |
            (valid_bboxes['y_max'] <= valid_bboxes['y_min'])
        ]
        if len(invalid_boxes) > 0:
            print(f"\n  ❌ Found {len(invalid_boxes)} invalid bboxes (x_max <= x_min or y_max <= y_min)")
    
    return concept_cols, target_cols, df_agg


def check_dataloader_output(train_loader, val_loader, test_loader):
    """Check actual dataloader output."""
    print(f"\n{'='*80}")
    print(f"🔍 CHECKING DATALOADER OUTPUT")
    print(f"{'='*80}")
    
    # Get one batch from train loader
    batch = next(iter(train_loader))
    
    print(f"\n📦 Batch Contents:")
    print(f"  - image shape: {batch['image'].shape}")
    print(f"  - concepts shape: {batch['concepts'].shape}")
    print(f"  - targets shape: {batch['targets'].shape}")
    print(f"  - bboxes type: {type(batch['bboxes'])}, length: {len(batch['bboxes'])}")
    print(f"  - image_ids: {batch['image_id'][:3]}...")
    
    # Check image statistics
    images = batch['image']
    print(f"\n📊 Image Statistics:")
    print(f"  - dtype: {images.dtype}")
    print(f"  - min: {images.min():.4f}, max: {images.max():.4f}")
    print(f"  - mean: {images.mean():.4f}, std: {images.std():.4f}")
    print(f"  - Expected after normalization: mean~0, std~1")
    
    # Check label statistics
    concepts = batch['concepts']
    targets = batch['targets']
    
    print(f"\n📊 Concept Labels in Batch:")
    print(f"  - dtype: {concepts.dtype}")
    print(f"  - min: {concepts.min():.0f}, max: {concepts.max():.0f}")
    print(f"  - Positive concepts per sample: {concepts.sum(dim=1).tolist()}")
    print(f"  - Total positives in batch: {concepts.sum().item():.0f} / {concepts.numel()} "
          f"({concepts.sum() / concepts.numel() * 100:.1f}%)")
    
    print(f"\n📊 Target Labels in Batch:")
    print(f"  - dtype: {targets.dtype}")
    print(f"  - min: {targets.min():.0f}, max: {targets.max():.0f}")
    print(f"  - Positive targets per sample: {targets.sum(dim=1).tolist()}")
    print(f"  - Total positives in batch: {targets.sum().item():.0f} / {targets.numel()} "
          f"({targets.sum() / targets.numel() * 100:.1f}%)")
    
    # Check bboxes
    print(f"\n📦 BBox Annotations in Batch:")
    for i, sample_bboxes in enumerate(batch['bboxes'][:3]):  # Check first 3 samples
        print(f"  Sample {i} ({batch['image_id'][i]}): {len(sample_bboxes)} bboxes")
        for bbox_info in sample_bboxes:
            print(f"    - Concept {bbox_info['concept_idx']}: bbox {bbox_info['bbox']}")
    
    total_bboxes = sum(len(b) for b in batch['bboxes'])
    print(f"  Total bboxes in batch: {total_bboxes}")
    
    # Check if concepts with bboxes have label=1
    print(f"\n✅ Checking Label-BBox Consistency:")
    inconsistencies = 0
    for i, sample_bboxes in enumerate(batch['bboxes']):
        for bbox_info in sample_bboxes:
            concept_idx = bbox_info['concept_idx']
            label = concepts[i, concept_idx].item()
            if label != 1:
                print(f"  ❌ Sample {i}: Concept {concept_idx} has bbox but label={label}")
                inconsistencies += 1
    
    if inconsistencies == 0:
        print(f"  ✅ All bboxes match their concept labels!")
    else:
        print(f"  ❌ Found {inconsistencies} inconsistencies!")
    
    return batch


def test_model_forward(batch, num_concepts, num_classes, device='cpu'):
    """Test model forward pass and loss computation."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING MODEL FORWARD PASS")
    print(f"{'='*80}")
    
    # Create model
    model = CSR(
        num_concepts=num_concepts,
        num_classes=num_classes,
        num_prototypes_per_concept=5,
        backbone_type='medmae',
        model_name='/kaggle/input/medmae-weight/weights/chestx-medmae_finetune.pth'
    ).to(device)
    
    print(f"✅ Model created successfully")
    print(f"  - Concepts: {num_concepts}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Prototypes per concept: 5")
    
    # Move batch to device
    images = batch['image'].to(device)
    concepts = batch['concepts'].to(device)
    targets = batch['targets'].to(device)
    
    print(f"\n🔄 Running forward pass...")
    
    with torch.no_grad():
        outputs = model(images)
    
    print(f"✅ Forward pass successful!")
    print(f"\n📊 Output Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.shape}")
    
    # Check CAMs
    cams = outputs['cams']
    print(f"\n📊 CAM Statistics:")
    print(f"  - min: {cams.min():.4f}, max: {cams.max():.4f}")
    print(f"  - mean: {cams.mean():.4f}, std: {cams.std():.4f}")
    
    # Test Stage 1 loss
    print(f"\n💡 Testing Stage 1 Loss (Concept Learning):")
    
    # Test with BBoxGuidedConceptLoss
    criterion = BBoxGuidedConceptLoss(alpha=1.0, beta=0.5)
    bboxes = batch.get('bboxes', None)
    
    loss = criterion(cams, concepts, bboxes)
    print(f"  - BBoxGuidedConceptLoss: {loss.item():.4f}")
    
    # Test with standard BCE
    import torch.nn.functional as F
    concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    bce_loss = bce_criterion(concept_logits, concepts)
    print(f"  - Standard BCEWithLogitsLoss: {bce_loss.item():.4f}")
    
    # Check predictions
    concept_preds = torch.sigmoid(concept_logits)
    print(f"\n📊 Concept Predictions (after sigmoid):")
    print(f"  - min: {concept_preds.min():.4f}, max: {concept_preds.max():.4f}")
    print(f"  - mean: {concept_preds.mean():.4f}, std: {concept_preds.std():.4f}")
    
    # Check if model is just predicting all same value (not learning)
    pred_variance = concept_preds.var().item()
    print(f"  - prediction variance: {pred_variance:.6f}")
    if pred_variance < 0.01:
        print(f"  ⚠️  WARNING: Very low prediction variance! Model might be predicting same value for all.")
    
    # Test Stage 3 loss
    print(f"\n💡 Testing Stage 3 Loss (Task Learning):")
    logits = outputs['logits']
    task_criterion = torch.nn.BCEWithLogitsLoss()
    task_loss = task_criterion(logits, targets)
    print(f"  - Task BCEWithLogitsLoss: {task_loss.item():.4f}")
    
    task_preds = torch.sigmoid(logits)
    print(f"\n📊 Task Predictions (after sigmoid):")
    print(f"  - min: {task_preds.min():.4f}, max: {task_preds.max():.4f}")
    print(f"  - mean: {task_preds.mean():.4f}, std: {task_preds.std():.4f}")
    
    return model, outputs


def main():
    """Run all checks."""
    print(f"\n{'='*80}")
    print(f"🔍 DATALOADER VERIFICATION SCRIPT")
    print(f"{'='*80}")
    
    # Configuration (adjust these paths to match your setup)
    train_csv = '/kaggle/input/vindr-cxr-physionet/image_labels_train.csv'
    test_csv = '/kaggle/input/vindr-cxr-physionet/image_labels_test.csv'
    train_bbox_csv = '/kaggle/input/vindr-cxr-physionet/annotations_train_resized.csv'  # Adjust if different
    test_bbox_csv = '/kaggle/input/vindr-cxr-physionet/annotations_test_resized.csv'    # Adjust if different
    train_dir = '/kaggle/input/vindr-cxr-physionet/train_data/train'
    test_dir = '/kaggle/input/vindr-cxr-physionet/test_data/test'
    batch_size = 8
    
    # Step 1: Check CSV labels
    print(f"\n{'='*80}")
    print(f"STEP 1: Checking CSV Files")
    print(f"{'='*80}")
    
    concept_cols, target_cols, train_df = check_csv_labels(train_csv, train_bbox_csv)
    _, _, test_df = check_csv_labels(test_csv, test_bbox_csv)
    
    num_concepts = len(concept_cols)
    num_classes = len(target_cols)
    
    # Step 2: Create dataloaders
    print(f"\n{'='*80}")
    print(f"STEP 2: Creating Dataloaders")
    print(f"{'='*80}")
    
    try:
        train_loader, val_loader, test_loader, _, _, _ = get_dataloaders_with_bbox_simple(
            train_csv, test_csv, train_bbox_csv, test_bbox_csv,
            train_dir, test_dir,
            batch_size=batch_size,
            rank=-1,
            world_size=1,
            val_split=0.1
        )
        print(f"✅ Dataloaders created successfully!")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Check dataloader output
    print(f"\n{'='*80}")
    print(f"STEP 3: Checking Dataloader Output")
    print(f"{'='*80}")
    
    try:
        batch = check_dataloader_output(train_loader, val_loader, test_loader)
    except Exception as e:
        print(f"❌ Error checking dataloader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test model forward pass
    print(f"\n{'='*80}")
    print(f"STEP 4: Testing Model Forward Pass")
    print(f"{'='*80}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model, outputs = test_model_forward(batch, num_concepts, num_classes, device)
    except Exception as e:
        print(f"❌ Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✅ VERIFICATION COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n📋 SUMMARY:")
    print(f"  - Total concepts: {num_concepts}")
    print(f"  - Total classes: {num_classes}")
    print(f"  - Train samples: {len(train_loader.dataset)}")
    print(f"  - Val samples: {len(val_loader.dataset)}")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    print(f"  - Model forward: ✅ Working")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. Check the 'POTENTIAL ISSUES' section above for data quality problems")
    print(f"  2. Verify bbox annotations match concept names")
    print(f"  3. Check if there are enough positive samples for each concept/target")
    print(f"  4. Monitor prediction variance during training (should increase)")


if __name__ == '__main__':
    main()
