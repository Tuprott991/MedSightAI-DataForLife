"""
Debug script to check what's actually happening during training.
Run this to diagnose why the model isn't learning.
"""

import torch
import torch.nn.functional as F
from src.dataloader_bbox_simple import get_dataloaders_with_bbox_simple
from src.model import CSR
from utils_bbox import BBoxGuidedConceptLoss
import numpy as np

def debug_training(train_csv, test_csv, train_dir, test_dir, 
                   train_bbox_csv, test_bbox_csv, pretrained_weights):
    """Debug training to find why model isn't learning."""
    
    print("="*80)
    print("🐛 DEBUGGING TRAINING")
    print("="*80)
    
    # Print paths for debugging
    print("\nPaths:")
    print(f"  train_csv: {train_csv}")
    print(f"  test_csv: {test_csv}")
    print(f"  train_dir: {train_dir}")
    print(f"  test_dir: {test_dir}")
    print(f"  train_bbox_csv: {train_bbox_csv}")
    print(f"  test_bbox_csv: {test_bbox_csv}")
    
    # 1. Load data
    print("\n1️⃣ Loading data...")
    train_loader, val_loader, test_loader, num_concepts, num_classes, _ = \
        get_dataloaders_with_bbox_simple(
            train_csv, test_csv, train_dir, test_dir,
            train_bbox_csv, test_bbox_csv,
            batch_size=4,  # Small batch for debugging
            rank=0, world_size=1, val_split=0.1,
            filter_rare=True, balance_no_finding=True
        )
    
    print(f"✅ Data loaded: {num_concepts} concepts, {num_classes} classes")
    
    # 2. Get a batch
    print("\n2️⃣ Getting a batch...")
    batch = next(iter(train_loader))
    images = batch['image']
    concepts_gt = batch['concepts']
    targets_gt = batch['targets']
    bboxes = batch['bboxes']
    
    print(f"Batch shapes:")
    print(f"  - images: {images.shape}")
    print(f"  - concepts_gt: {concepts_gt.shape}")
    print(f"  - targets_gt: {targets_gt.shape}")
    print(f"  - num_bboxes: {[len(b) for b in bboxes]}")
    
    # Check image statistics
    print(f"\nImage statistics:")
    print(f"  - min: {images.min():.3f}")
    print(f"  - max: {images.max():.3f}")
    print(f"  - mean: {images.mean():.3f}")
    print(f"  - std: {images.std():.3f}")
    print(f"  ⚠️  Expected: min~-2.1, max~2.6, mean~0, std~1")
    
    # Check label statistics
    print(f"\nLabel statistics:")
    print(f"  - concepts_gt positive ratio: {concepts_gt.mean():.3f}")
    print(f"  - targets_gt positive ratio: {targets_gt.mean():.3f}")
    print(f"  - concepts_gt sum per sample: {concepts_gt.sum(dim=1).tolist()}")
    
    # 3. Create model
    print("\n3️⃣ Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSR(
        num_concepts=num_concepts,
        num_classes=num_classes,
        num_prototypes_per_concept=5,
        backbone_type='medmae',
        model_name=pretrained_weights
    ).to(device)
    
    print(f"✅ Model created on {device}")
    
    # 4. Forward pass
    print("\n4️⃣ Forward pass...")
    images = images.to(device)
    concepts_gt = concepts_gt.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        cams = outputs['cams']  # (B, K, H, W)
        
    print(f"CAMs shape: {cams.shape}")
    print(f"CAMs statistics:")
    print(f"  - min: {cams.min():.3f}")
    print(f"  - max: {cams.max():.3f}")
    print(f"  - mean: {cams.mean():.3f}")
    print(f"  - std: {cams.std():.3f}")
    
    # Get predictions
    concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)
    concept_probs = torch.sigmoid(concept_logits)
    
    print(f"\nConcept predictions:")
    print(f"  - logits min: {concept_logits.min():.3f}")
    print(f"  - logits max: {concept_logits.max():.3f}")
    print(f"  - logits mean: {concept_logits.mean():.3f}")
    print(f"  - probs min: {concept_probs.min():.3f}")
    print(f"  - probs max: {concept_probs.max():.3f}")
    print(f"  - probs mean: {concept_probs.mean():.3f}")
    
    # 5. Compute loss
    print("\n5️⃣ Computing loss...")
    
    # Compute pos_weight
    import pandas as pd
    if hasattr(train_loader.dataset, 'dataset'):
        base_dataset = train_loader.dataset.dataset
    else:
        base_dataset = train_loader.dataset
    
    concept_cols = base_dataset.concept_cols
    df_train = pd.read_csv(train_csv)
    df_agg = df_train.groupby('image_id')[concept_cols].max().reset_index()
    concept_values = torch.tensor(df_agg[concept_cols].values, dtype=torch.float32)
    pos_weight = (concept_values == 0).sum(dim=0) / (concept_values == 1).sum(dim=0).clamp(min=1)
    pos_weight = pos_weight.clamp(max=20.0)
    
    print(f"Pos weights: min={pos_weight.min():.2f}, max={pos_weight.max():.2f}, mean={pos_weight.mean():.2f}")
    
    # Compute loss WITHOUT pos_weight
    criterion_no_weight = torch.nn.BCEWithLogitsLoss()
    loss_no_weight = criterion_no_weight(concept_logits, concepts_gt)
    
    # Compute loss WITH pos_weight
    criterion_with_weight = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    loss_with_weight = criterion_with_weight(concept_logits, concepts_gt)
    
    print(f"Loss WITHOUT pos_weight: {loss_no_weight.item():.4f}")
    print(f"Loss WITH pos_weight: {loss_with_weight.item():.4f}")
    print(f"⚠️  Expected random: ~0.69 without pos_weight")
    
    # Compute loss with bbox guidance
    criterion_bbox = BBoxGuidedConceptLoss(alpha=1.0, beta=0.5, pos_weight=pos_weight.to(device))
    loss_bbox = criterion_bbox(cams, concepts_gt, bboxes)
    
    print(f"Loss WITH bbox guidance: {loss_bbox.item():.4f}")
    
    # 6. Check gradients
    print("\n6️⃣ Checking gradients...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    outputs = model(images)
    cams = outputs['cams']
    loss = criterion_bbox(cams, concepts_gt, bboxes)
    loss.backward()
    
    # Check gradient norms
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    print(f"\nGradient norms:")
    print(f"  - backbone: {[v for k,v in grad_norms.items() if 'backbone' in k and v > 0][:3]}")
    print(f"  - concept_head: {[v for k,v in grad_norms.items() if 'concept_head' in k]}")
    
    total_norm = sum(v**2 for v in grad_norms.values()) ** 0.5
    print(f"  - total norm: {total_norm:.2e}")
    
    if total_norm < 1e-7:
        print(f"  ❌ VANISHING GRADIENTS!")
    elif total_norm > 100:
        print(f"  ❌ EXPLODING GRADIENTS!")
    else:
        print(f"  ✅ Gradients look normal")
    
    # 7. Check if optimizer updates parameters
    print("\n7️⃣ Checking optimizer updates...")
    param_before = {name: param.clone() for name, param in model.named_parameters()}
    
    optimizer.step()
    
    param_changes = {}
    for name, param in model.named_parameters():
        change = (param - param_before[name]).abs().max().item()
        if change > 0:
            param_changes[name] = change
    
    print(f"\nParameter changes:")
    print(f"  - backbone: {[v for k,v in param_changes.items() if 'backbone' in k][:3]}")
    print(f"  - concept_head: {[v for k,v in param_changes.items() if 'concept_head' in k]}")
    
    if len(param_changes) == 0:
        print(f"  ❌ NO PARAMETERS UPDATED!")
    else:
        print(f"  ✅ {len(param_changes)} parameters updated")
    
    # 8. Summary
    print("\n" + "="*80)
    print("📊 DIAGNOSIS SUMMARY")
    print("="*80)
    
    issues = []
    if images.mean() > 1:
        issues.append("❌ Images not normalized correctly (mean should be ~0)")
    if loss_bbox.item() > 2.0:
        issues.append("❌ Loss is too high (>2.0), model predicting very wrong")
    if total_norm < 1e-7:
        issues.append("❌ Vanishing gradients - learning rate might be too low")
    if len(param_changes) == 0:
        issues.append("❌ No parameters being updated")
    if concept_probs.mean() < 0.1 or concept_probs.mean() > 0.9:
        issues.append(f"❌ Model outputs are saturated (mean prob: {concept_probs.mean():.3f})")
    
    if issues:
        print("\n🚨 ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No obvious issues found. Model should be learning.")
    
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--train_bbox_csv', type=str, required=True)
    parser.add_argument('--test_bbox_csv', type=str, required=True)
    parser.add_argument('--pretrained_weights', type=str, default='weights/pre_trained_medmae.pth')
    args = parser.parse_args()
    
    debug_training(
        args.train_csv, args.test_csv,
        args.train_dir, args.test_dir,
        args.train_bbox_csv, args.test_bbox_csv,
        args.pretrained_weights
    )
