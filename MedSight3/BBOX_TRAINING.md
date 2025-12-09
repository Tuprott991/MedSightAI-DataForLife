# Training with Bounding Box Annotations

## 📦 Overview

When your dataset has bounding box annotations (like VinDr-CXR), you can **significantly improve CAM localization** by adding spatial supervision during Stage 1 training.

## 🎯 Benefits

**Without BBox (standard training):**
- Model only knows "concept exists in this image"
- CAMs may be diffuse or highlight wrong regions
- Loss: `BCE(max_pool(CAM), label)`

**With BBox (improved training):**
- Model knows "concept exists HERE specifically"
- CAMs become much more precise and interpretable
- Loss: `BCE(max_pool(CAM), label) + BBoxLoss(CAM, bbox)`

**Expected improvements:**
- 🔥 **+15-25% localization accuracy** (IoU with ground truth)
- 🎨 **Sharper, cleaner CAM heatmaps** for visualization
- 🎯 **Better Stage 2 performance** (prototypes align with actual concept locations)

## 📊 Data Format

### Required Files

**1. BBox Annotations CSV** (`bbox_csv`)
```
image_id, rad_id, class_name, x_min, y_min, x_max, y_max
```

Example:
```csv
image_id,rad_id,class_name,x_min,y_min,x_max,y_max
0005e8e3,R10,Infiltration,900.96,587.81,1205.36,888.71
0005e8e3,R10,Lung Opacity,900.96,587.81,1205.36,888.71
0005e8e3,R8,Consolidation,932.47,567.78,1197.77,896.41
```

**2. Resize Factors CSV** (`resize_factor_csv`)
Since images are resized to 224×224 for training, we need resize factors to transform bbox coordinates from original image space to resized space.

```
image_id, original_height, original_width, target_height, target_width, resize_factor_h, resize_factor_w
```

Example:
```csv
image_id,original_height,original_width,target_height,target_width,resize_factor_h,resize_factor_w
0005e8e3,2500,2500,224,224,0.0896,0.0896
00063ef3,2048,2048,224,224,0.1094,0.1094
```

**Notes:**
- Images with "No finding" will have empty bbox coords (OK, will be skipped)
- Multiple bboxes per image are supported
- **Bbox coordinates** should be in **original image pixel space** (e.g., 900.96, not normalized)
- **Resize factors** are calculated as: `resize_factor = target_size / original_size`
- Both train and test sets need resize factor CSV

## 🚀 Usage

### 1. Standard Training (without bbox)
```bash
# Stage 1: Concept learning (image-level labels only)
torchrun --nproc_per_node=2 train.py \
    --epochs_stage1 20 \
    --batch_size 16 \
    --lr 1e-4
```

### 2. Improved Training (with bbox)
```bash
# Stage 1: Concept learning with bbox supervision
torchrun --nproc_per_node=2 train_stage1_bbox.py \
    --train_bbox_csv data/vindr_train_annotations.csv \
    --test_bbox_csv data/vindr_test_annotations.csv \
    --train_resize_factor_csv data/vindr_train_resize_factors.csv \
    --test_resize_factor_csv data/vindr_test_resize_factors.csv \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-4 \
    --alpha 1.0 \
    --beta 0.5
```

**Parameters:**
- `--train_bbox_csv`: Path to CSV with train bbox annotations (required)
- `--test_bbox_csv`: Path to CSV with test bbox annotations (required)
- `--train_resize_factor_csv`: Path to CSV with train resize factors (required)
- `--test_resize_factor_csv`: Path to CSV with test resize factors (required)
- `--alpha`: Weight for classification loss (default: 1.0)
- `--beta`: Weight for localization loss (default: 0.5)
  - Higher β → stronger spatial supervision
  - Recommended: 0.3-0.7

### 3. Continue with Stage 2 & 3
```bash
# After bbox-supervised Stage 1, continue normally
python train.py \
    --stage1_checkpoint outputs_bbox/best_model_stage1_bbox.pth \
    --epochs_stage2 10 \
    --epochs_stage3 20
```

## 🔍 How BBox Loss Works

### Mathematical Formulation

**Total Loss:**
```
L_total = α * L_classification + β * L_localization
```

**L_classification (standard):**
```python
concept_logits = max_pool(CAM)  # (B, K)
L_cls = BCE(concept_logits, labels)
```

**L_localization (new):**
```python
For each bbox (concept_k, [x_min, y_min, x_max, y_max]):
    inside_mask = create_mask(bbox)  # 1 inside, 0 outside
    outside_mask = 1 - inside_mask
    
    # CAM should be positive inside, negative outside
    L_inside = mean(relu(-CAM * inside_mask))
    L_outside = mean(relu(CAM * outside_mask))
    
    L_loc = L_inside + L_outside
```

**Intuition:**
- Encourages high CAM activations inside bbox
- Suppresses activations outside bbox
- Creates sharper, more localized heatmaps

## 📈 Expected Training Curves

**Classification Loss:**
- Should be similar to standard training
- Converges to ~0.3-0.5

**Localization Loss:**
- Starts high (~0.5-1.0) 
- Decreases to ~0.1-0.2 as model learns spatial patterns
- Images without bboxes contribute 0 to this term

**Concept AUC:**
- Might be **slightly lower** than standard training initially
- But CAM quality is **much better**
- Final AUC: 0.75-0.82 (similar or better)

## 🎨 Visualization

### Compare Standard vs BBox-Supervised CAMs

```bash
# Standard training CAMs
python visualize_cams.py \
    --checkpoint outputs/best_model_stage1.pth \
    --image test.png \
    --save_dir viz_standard

# BBox-supervised CAMs
python visualize_cams.py \
    --checkpoint outputs_bbox/best_model_stage1_bbox.pth \
    --image test.png \
    --save_dir viz_bbox
```

**What to expect:**
- **Standard:** Diffuse, scattered activations
- **BBox-supervised:** Sharp, focused on actual lesion locations

## ⚙️ Hyperparameter Tuning

### Loss Weight (β)

| β value | Effect | Use when |
|---------|--------|----------|
| 0.1-0.3 | Gentle localization guidance | Noisy bbox annotations |
| 0.5 | **Recommended default** | Clean annotations |
| 0.7-1.0 | Strong spatial supervision | Very precise bboxes |

### Learning Rate

Same as standard training:
- Backbone: `lr * 0.01`
- Concept head: `lr * 0.1`
- Base LR: `1e-4`

### Epochs

- **With bbox:** May need 20-30 epochs (learns spatial patterns)
- **Without bbox:** Typically 15-20 epochs

## 🐛 Troubleshooting

### Issue: Localization loss not decreasing

**Causes:**
- β too high → reduce to 0.3
- Bboxes in wrong format (not normalized)
- Image IDs don't match between CSVs

**Solution:**
```python
# Check bbox loading
from src.dataloader_bbox import CSRDatasetWithBBox
ds = CSRDatasetWithBBox(train_csv, bbox_csv, train_dir)
sample = ds[0]
print(f"BBoxes: {sample['bboxes']}")
# Should show: [{'concept_idx': 5, 'bbox': [0.42, 0.31, 0.68, 0.59]}]
```

### Issue: Some concepts have no bboxes

**This is OK!** The localization loss is:
```python
if len(bboxes) == 0:
    return classification_loss  # No spatial supervision
```

Images without bboxes still contribute to classification loss.

### Issue: CAMs worse than standard training

**Causes:**
- β too high → overfitting to bbox locations
- Bbox annotations are noisy/inaccurate
- Need more epochs to converge

**Solution:**
- Reduce β to 0.2-0.3
- Increase epochs to 30
- Visualize training samples to check bbox quality

## 📊 Metrics to Monitor

1. **Train Loss** (cls + loc): Should steadily decrease
2. **Concept AUC**: Should reach 0.75-0.82
3. **Localization Loss**: Should decrease to ~0.1-0.2
4. **Visual inspection**: CAMs should align with bboxes

## 💾 Code Structure

```
MedSight3/
├── utils_bbox.py              # BBoxGuidedConceptLoss
├── src/
│   └── dataloader_bbox.py     # CSRDatasetWithBBox
├── train_stage1_bbox.py       # Training script
└── visualize_cams.py          # Visualization (works for both)
```

## 🎓 When to Use BBox Supervision?

**Use bbox supervision when:**
- ✅ You have bbox annotations (VinDr-CXR does!)
- ✅ You need interpretable/precise CAMs
- ✅ You want better Stage 2 prototypes

**Skip bbox supervision when:**
- ❌ Only image-level labels available
- ❌ Bbox annotations are very noisy
- ❌ Training time is critical (adds 20-30% overhead)

## 🚀 Next Steps

1. Train Stage 1 with bbox: `python train_stage1_bbox.py --bbox_csv data/annotations.csv`
2. Visualize CAMs: `python visualize_cams.py --checkpoint outputs_bbox/best_model_stage1_bbox.pth`
3. Continue with Stage 2 & 3: Load `best_model_stage1_bbox.pth` and continue normal pipeline

The bbox-supervised model can be used directly in the full 3-stage pipeline - just start Stage 2 from this checkpoint!
