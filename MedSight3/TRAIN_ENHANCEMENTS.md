# Training Enhancements - BBox Support & Comprehensive Metrics

## Overview
Enhanced `train.py` with:
1. **Bounding box-supervised training** for Stage 1
2. **Comprehensive evaluation metrics**: AUC, mAP, F1-score (macro/micro), IoU

---

## 1. New Command Line Arguments

```bash
# BBox supervision (optional)
--bbox_csv PATH              # Path to bbox CSV (image_id, rad_id, class_name, x_min, y_min, x_max, y_max)
--alpha 1.0                  # Weight for classification loss in BBoxGuidedConceptLoss
--beta 0.5                   # Weight for localization loss in BBoxGuidedConceptLoss
```

**Example usage:**
```bash
# Standard training (no bbox)
python train.py --train_csv data/train.csv --test_csv data/test.csv ...

# With bbox supervision
python train.py --train_csv data/train.csv --test_csv data/test.csv \
    --bbox_csv data/train_bboxes.csv --alpha 1.0 --beta 0.5 ...
```

---

## 2. BBox-Supervised Training (Stage 1)

### What Changes When bbox_csv is Provided?

**Dataloader:**
- Uses `get_dataloaders_with_bbox()` from `src/dataloader_bbox.py`
- Returns `(images, concept_labels, disease_labels, bboxes)` instead of `(images, concept_labels, disease_labels)`
- Bboxes are normalized to [0, 1] and padded to max_boxes_per_image

**Loss Function:**
- **Without bbox:** `BCEWithLogitsLoss(pos_weight)` (standard)
- **With bbox:** `BBoxGuidedConceptLoss(alpha, beta, pos_weight)`
  ```python
  total_loss = α * classification_loss + β * localization_loss
  ```
  - Classification: Standard BCE for concept presence
  - Localization: Spatial supervision using bbox ground truth

**Training Loop:**
```python
# Standard training
logits, _, _ = model(images, stage=1)
loss = criterion(logits, concept_labels)  # BCE loss

# BBox-supervised training  
logits, cams, _ = model(images, stage=1)  
loss = criterion(logits, cams, concept_labels, bboxes)  # α*cls + β*loc
```

---

## 3. Comprehensive Evaluation Metrics

### Added Metrics (computed by `validate()`)

**For All Stages:**
- **AUC-ROC (macro):** Area Under ROC Curve, macro-averaged across classes
- **mAP:** Mean Average Precision (average of per-class AP scores)
- **F1-macro:** Harmonic mean of precision/recall, macro-averaged
- **F1-micro:** Harmonic mean computed globally (good for imbalanced datasets)

**For Stage 1 with BBox:**
- **IoU (Intersection over Union):** Measures CAM-bbox overlap quality
  - IoU = Area of Overlap / Area of Union
  - Computed between predicted CAMs and ground truth bboxes
  - Only available when `bbox_csv` is provided

### Metric Interpretation

| Metric | Good | Very Good | Excellent |
|--------|------|-----------|-----------|
| AUC | ≥0.75 | ≥0.82 | ≥0.87 |
| mAP | ≥0.70 | ≥0.80 | ≥0.85 |
| F1 | ≥0.70 | ≥0.80 | ≥0.85 |
| IoU | ≥0.40 | ≥0.50 | ≥0.60 |

### Output Format

**Stage 1 (with bbox):**
```
Epoch 10: Val Loss 0.2341, AUC 0.8234, mAP 0.7856, F1-macro 0.7623, F1-micro 0.7891, IoU 0.5234
✅ Saved best Stage 1 model (Concept AUC: 0.8234, mAP: 0.7856)
```

**Stage 1 (without bbox):**
```
Epoch 10: Val Loss 0.2341, AUC 0.8234, mAP 0.7856, F1-macro 0.7623, F1-micro 0.7891
✅ Saved best Stage 1 model (Concept AUC: 0.8234, mAP: 0.7856)
```

**Stage 3 (Disease Prediction):**
```
Epoch 20: Val Loss 0.1823, Disease AUC 0.8567, mAP 0.8234, F1-macro 0.8012, F1-micro 0.8156
✅ Saved best Stage 3 model (Disease AUC: 0.8567, mAP: 0.8234)
```

**Final Test:**
```
🏆 Final Test Results (Disease Prediction):
  Test Loss: 0.1756
  Test AUC (macro): 0.8623
  Test mAP: 0.8345
  Test F1-macro: 0.8123
  Test F1-micro: 0.8267
  Excellent!

🎉 Training Complete!
Best Val AUC: 0.8567 from stage3
Final Test Disease AUC: 0.8623, mAP: 0.8345

Benchmark: VinDr-CXR ResNet-50 ~0.78 | DenseNet-121 ~0.82 | SOTA ~0.87
```

---

## 4. Code Changes Summary

### New Imports
```python
from utils_bbox import BBoxGuidedConceptLoss
import pandas as pd
import numpy as np
```

### New Functions
```python
def compute_metrics(y_true, y_pred, y_prob):
    """Compute AUC, mAP, F1-macro, F1-micro"""
    
def compute_iou_score(cams, bboxes):
    """Compute IoU between CAMs and ground truth bboxes"""
    
def validate(model, dataloader, device, rank, stage, criterion):
    """Enhanced validation with comprehensive metrics"""
    # Returns: {'loss': float, 'auc': float, 'mAP': float, 
    #           'f1_macro': float, 'f1_micro': float, 'iou': float (optional)}
```

### Modified Training Flow
1. **Dataloader Selection:** Conditional based on `args.bbox_csv`
2. **Criterion Creation:** `BBoxGuidedConceptLoss` vs `BCEWithLogitsLoss`
3. **Validation Calls:** Return metrics dict instead of `(loss, auc)` tuple
4. **Print Statements:** Show all metrics instead of just loss + AUC

---

## 5. Usage Examples

### Standard Training (No BBox)
```bash
python train.py \
    --train_csv data/train.csv \
    --test_csv data/test.csv \
    --train_dir data/train_images \
    --test_dir data/test_images \
    --epochs_stage1 15 \
    --epochs_stage2 10 \
    --epochs_stage3 15 \
    --batch_size 32 \
    --lr 0.001 \
    --exp_name "baseline_no_bbox"
```

**Expected metrics:**
- AUC, mAP, F1-macro, F1-micro (no IoU)

---

### BBox-Supervised Training
```bash
python train.py \
    --train_csv data/train.csv \
    --test_csv data/test.csv \
    --train_dir data/train_images \
    --test_dir data/test_images \
    --bbox_csv data/train_bboxes.csv \
    --alpha 1.0 \
    --beta 0.5 \
    --epochs_stage1 15 \
    --epochs_stage2 10 \
    --epochs_stage3 15 \
    --batch_size 32 \
    --lr 0.001 \
    --exp_name "bbox_supervised"
```

**Expected metrics:**
- AUC, mAP, F1-macro, F1-micro + **IoU** (for Stage 1 only)

**Expected improvements:**
- Better CAM localization (higher IoU)
- Better concept detection (higher Stage 1 AUC/mAP)
- Potentially better disease prediction (Stage 3) due to better features

---

## 6. Hyperparameter Tuning

### Alpha/Beta Trade-off

| α | β | Effect |
|---|---|--------|
| 1.0 | 0.0 | Pure classification (standard BCE) |
| 1.0 | 0.5 | **Recommended:** Balanced (good for most cases) |
| 1.0 | 1.0 | Equal weight to classification and localization |
| 1.0 | 2.0 | Strong localization emphasis (may hurt classification) |

**Tuning Strategy:**
1. Start with α=1.0, β=0.5
2. If IoU too low: Increase β (e.g., 0.7, 1.0)
3. If AUC/mAP drops: Decrease β (e.g., 0.3, 0.2)
4. Monitor both classification metrics (AUC, mAP) and localization (IoU)

---

## 7. Troubleshooting

### Issue: "BBoxGuidedConceptLoss is not defined"
**Solution:** Make sure `utils_bbox.py` exists in the same directory as `train.py`

### Issue: "get_dataloaders_with_bbox not found"
**Solution:** Make sure `src/dataloader_bbox.py` exists

### Issue: IoU is 0.0 or very low
**Possible causes:**
- β too small (increase to 0.7 or 1.0)
- BBox CSV format wrong (check: `image_id, rad_id, class_name, x_min, y_min, x_max, y_max`)
- BBox coordinates not normalized (should be in pixel coordinates, not [0,1])

### Issue: Training loss NaN with BBoxGuidedConceptLoss
**Possible causes:**
- Learning rate too high (reduce to 0.0001)
- β too large (reduce to 0.3)
- Add gradient clipping: already implemented in `train_one_epoch()`

---

## 8. Performance Benchmarks

### VinDr-CXR Dataset (Expected Results)

**Without BBox Supervision:**
- Stage 1 Concept AUC: ~0.78-0.82
- Stage 3 Disease AUC: ~0.78-0.82
- mAP: ~0.75-0.80

**With BBox Supervision (α=1.0, β=0.5):**
- Stage 1 Concept AUC: ~0.82-0.85 ↑
- Stage 1 IoU: ~0.45-0.55
- Stage 3 Disease AUC: ~0.82-0.87 ↑
- mAP: ~0.80-0.85 ↑

**SOTA Comparison:**
- ResNet-50: ~0.78
- DenseNet-121: ~0.82
- CSR (ours, no bbox): ~0.78-0.82
- CSR (ours, with bbox): ~0.82-0.87 🎯

---

## 9. Monitoring Training

### Key Metrics to Watch

**Stage 1:**
- **Val Loss:** Should decrease steadily
- **Concept AUC:** Target ≥0.82 (very good)
- **mAP:** Target ≥0.78
- **IoU (if bbox):** Target ≥0.50 (indicates good localization)

**Stage 3:**
- **Disease AUC:** Target ≥0.82 (matches DenseNet-121 baseline)
- **mAP:** Target ≥0.80
- **F1-macro:** Should be close to AUC (±0.05)

### Signs of Good Training
✅ Val loss decreases smoothly
✅ AUC improves over epochs
✅ mAP tracks closely with AUC
✅ F1-macro and F1-micro are close (indicates balanced predictions)
✅ IoU > 0.45 (if using bbox)

### Red Flags
❌ Val loss increases or oscillates wildly
❌ Large gap between AUC and mAP (>0.10)
❌ Large gap between F1-macro and F1-micro (>0.15, indicates class imbalance issues)
❌ IoU < 0.30 (if using bbox, indicates poor localization)

---

## 10. Next Steps

### Visualization
Use `visualize_cams.py` to inspect CAM quality:
```bash
python visualize_cams.py --checkpoint best_model_stage1.pth --image_path sample.png
```

### Advanced Tuning
1. Try different α/β combinations
2. Experiment with different pos_weight strategies
3. Tune learning rates per stage
4. Add focal loss for hard examples

### Evaluation
```bash
# Comprehensive evaluation on test set
python evaluate.py \
    --checkpoint best_model_stage3.pth \
    --test_csv data/test.csv \
    --test_dir data/test_images \
    --bbox_csv data/test_bboxes.csv  # optional
```

---

## References
- BBox-guided training: `BBOX_TRAINING.md`
- Stage 1 details: `STAGE1_EXPLAINED.md`
- Stage 2/3/Inference: `STAGE2_STAGE3_INFERENCE.md`
- Full training guide: `TRAIN_GUIDE.md`
