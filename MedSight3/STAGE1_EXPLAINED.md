# Stage 1: Concept Learning - Detailed Explanation

## 🎯 Goal of Stage 1

**Train the model to:**
1. Identify which anatomical concepts are present in an X-ray
2. Localize WHERE each concept appears (via CAMs)

**Output:** A model that can produce interpretable **Concept Activation Maps (CAMs)** showing spatial locations of medical findings.

---

## 📊 Data Flow Diagram

```
INPUT IMAGE                                    LABELS
(224×224 RGB)                                  concepts: [0,1,0,1,...]  (K=22 concepts)
     ↓                                         targets:  [0,0,1,0,...]  (C=6 diseases)
     │
     ├─────────────────────────────────────────────────────────────────┐
     │                                                                  │
     ↓                                                                  │
┌─────────────────┐                                                    │
│  MedMAE Backbone│  (Pre-trained ViT)                                │
│   (FROZEN or    │                                                    │
│   Fine-tuning)  │                                                    │
└─────────────────┘                                                    │
     ↓                                                                  │
  Features f                                                            │
  (B, 768, 7, 7)  ← 768 channels, 7×7 spatial grid                    │
     ↓                                                                  │
     │                                                                  │
     ├────────────────────┬─────────────────┬──────────────────────┐  │
     │                    │                 │                      │  │
     ↓                    ↓                 ↓                      ↓  │
┌──────────┐      ┌────────────┐   ┌─────────────┐      ┌──────────┐│
│ Concept  │      │ Projector  │   │ Prototypes  │      │   Task   ││
│   Head   │      │   (not     │   │   (not      │      │   Head   ││
│          │      │   trained) │   │   trained)  │      │   (not   ││
│ (1×1 Conv)│     └────────────┘   └─────────────┘      │ trained) ││
└──────────┘                                             └──────────┘│
     ↓                                                                 │
   CAMs                                                                │
 (B, 22, 7, 7)  ← One heatmap per concept                             │
     ↓                                                                 │
     │                                                                 │
     ↓                                                                 │
┌──────────────────────┐                                              │
│  Global Max Pooling  │  ← Extract strongest activation per concept │
└──────────────────────┘                                              │
     ↓                                                                 │
Concept Logits                                                         │
  (B, 22)                                                              │
     ↓                                                                 │
     │                                                                 │
     ├─────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────┐
│ BCEWithLogitsLoss      │
│ (with pos_weight)      │
└────────────────────────┘
     ↓
   Loss → Backprop → Update weights
```

---

## 🔄 Step-by-Step Data Flow

### **Step 1: Load Training Batch**

```python
batch = {
    'image': torch.tensor([B, 3, 224, 224]),      # X-ray images
    'concepts': torch.tensor([B, 22]),            # Concept labels (binary)
    'targets': torch.tensor([B, 6]),              # Disease labels (unused in Stage 1)
    'bboxes': [list of bbox dicts]                # Optional: bbox annotations
}
```

**Example:**
- Image: Chest X-ray of a patient
- Concepts: `[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, ...]` 
  - Index 1 (Atelectasis): Present ✓
  - Index 4 (Cardiomegaly): Present ✓
  - Index 10 (Lung Opacity): Present ✓
  - Others: Absent ✗

### **Step 2: Feature Extraction (Backbone)**

```python
f = model.backbone(images)  # (B, 768, 7, 7)
```

**What happens:**
- MedMAE (ViT-Base) processes 224×224 image
- Outputs feature maps: **768 channels × 7×7 spatial grid**
- Each 7×7 location has receptive field of ~32×32 pixels in original image
- Features encode high-level semantic information

**Learning rate:** Very slow (`lr * 0.01`) to preserve pre-trained medical knowledge

### **Step 3: Generate Concept Activation Maps**

```python
cams = model.concept_head(f)  # (B, 22, 7, 7)
```

**What happens:**
- **1×1 Convolution**: `Conv2d(768, 22, kernel_size=1)`
- Transforms 768-dim features → 22 concept channels
- **No activation function** (raw logits for BCEWithLogitsLoss)

**Output interpretation:**
- `cams[batch_idx, concept_idx, y, x]` = activation strength at spatial location (x, y)
- High value → concept likely present at this location
- Low/negative value → concept likely absent

**Example CAM for "Cardiomegaly":**
```
CAM[0, 4] =  [[-0.3, -0.1,  0.2,  0.1, -0.2, -0.3, -0.4],
              [-0.2,  0.3,  0.8,  0.7,  0.2, -0.1, -0.3],
              [ 0.1,  0.6,  1.5,  1.3,  0.5,  0.0, -0.2],
              [ 0.0,  0.7,  1.8,  1.6,  0.6,  0.1, -0.1],
              [-0.1,  0.4,  1.2,  1.0,  0.3, -0.1, -0.2],
              [-0.3,  0.0,  0.4,  0.3, -0.1, -0.3, -0.4],
              [-0.4, -0.2,  0.1,  0.0, -0.3, -0.4, -0.5]]
```
→ Peak at center (3, 3) where heart is located! ❤️

### **Step 4: Pool to Concept Predictions**

```python
concept_logits = F.adaptive_max_pool2d(cams, (1, 1))  # (B, 22)
concept_logits = concept_logits.squeeze(-1).squeeze(-1)
```

**What happens:**
- **Global Max Pooling** across spatial dimensions (7×7 → 1×1)
- Takes the **strongest activation** from each concept's CAM
- Result: One score per concept indicating its presence

**Why max pooling (not average)?**
- Concepts may occupy small regions (nodules, fractures)
- Max captures "exists somewhere in image"
- Average would dilute signal from small findings

**Example:**
```python
cams[0, 4].max() = 1.8  → High confidence Cardiomegaly present
cams[0, 7].max() = -0.9 → Emphysema likely absent
```

### **Step 5: Compute Loss**

#### **5a. Standard Training (Image-level labels only)**

```python
loss = BCEWithLogitsLoss(pos_weight)(concept_logits, concepts_gt)
```

**What happens:**
- Binary cross-entropy between predictions and ground truth
- `pos_weight` handles class imbalance (rare concepts get higher weight)
- Loss encourages:
  - High logits for present concepts (label=1)
  - Low logits for absent concepts (label=0)

**Example:**
```python
concept_logits[0] = [−0.5, 1.8, −1.2, 0.3, 1.5, ...]  # Model predictions
concepts_gt[0]    = [   0,   1,    0,   0,   1, ...]  # Ground truth

# Losses per concept:
BCE([−0.5], [0]) = 0.47  # Correct (low for absent)
BCE([1.8],  [1]) = 0.15  # Correct (high for present)
BCE([−1.2], [0]) = 0.26  # Correct
BCE([0.3],  [0]) = 0.85  # WRONG! Should be negative
BCE([1.5],  [1]) = 0.20  # Correct

# Weighted average → Total loss
```

#### **5b. BBox-Supervised Training (Spatial labels)**

```python
loss = BBoxGuidedConceptLoss(cams, concepts_gt, bboxes)
     = α * classification_loss + β * localization_loss
```

**Classification Loss (same as above):**
```python
classification_loss = BCE(max_pool(cams), concepts_gt)
```

**Localization Loss (NEW):**
```python
For each bbox [concept_idx=4, x_min=0.3, y_min=0.4, x_max=0.7, y_max=0.8]:
    # Create spatial mask
    inside_mask  = [0 0 0 0 0 0 0]
                   [0 0 1 1 1 0 0]  ← 1 inside bbox
                   [0 0 1 1 1 0 0]
                   [0 0 1 1 1 0 0]
                   [0 0 0 0 0 0 0]
    
    outside_mask = 1 - inside_mask
    
    # Loss: CAM should be HIGH inside, LOW outside
    L_inside  = mean(relu(-cams[4] * inside_mask))   # Penalize negative values inside
    L_outside = mean(relu(cams[4] * outside_mask))   # Penalize positive values outside
    
    localization_loss = L_inside + L_outside
```

**Effect:**
- Model learns **exact spatial locations** of concepts
- CAMs become sharper and more interpretable
- Better initialization for Stage 2

### **Step 6: Backpropagation**

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

**What gets updated:**
- ✅ **Concept Head weights** (main learning)
- ✅ **Backbone weights** (fine-tuning, slow LR)
- ❌ Projector (frozen, not used in Stage 1)
- ❌ Prototypes (frozen, not used in Stage 1)
- ❌ Task Head (frozen, not used in Stage 1)

**Gradient flow:**
```
Loss → concept_logits → max_pool → CAMs → concept_head → features → backbone
```

---

## 📈 What the Model Learns

### **Epoch 1-3: Initial Adaptation**
- Backbone: Adapts pre-trained features to chest X-rays
- Concept head: Learns rough associations (e.g., bright region → opacity)
- CAMs: Very noisy, scattered activations

### **Epoch 4-10: Concept Discovery**
- Model identifies discriminative regions
- CAMs start focusing on relevant anatomy
- Common concepts (opacity, cardiomegaly) learned first
- Rare concepts (fractures) still struggling

### **Epoch 11-20: Refinement**
- CAMs become sharper and more localized
- Spatial patterns stabilize
- Model distinguishes similar concepts (infiltration vs consolidation)

### **With BBox Supervision:**
- **Faster convergence** (10-15 epochs vs 20)
- **Precise localization** from early epochs
- **Higher interpretability** throughout training

---

## 🎨 Visual Example: CAM Evolution

**Ground Truth:**
- Image: Chest X-ray with left lower lobe pneumonia
- Labels: Consolidation=1, Lung Opacity=1

**Epoch 1:**
```
CAM[Consolidation]:     CAM[Lung Opacity]:
[░░░░░░░]               [▓▓░░░░░]
[░░░▓░░░]               [▓▓▓░░░░]
[░░▓▓░░░]  ← Noisy      [▓▓▓▓░░░]  ← Diffuse
[░░░▓░░░]               [▓▓▓░░░░]
[░░░░░░░]               [▓▓░░░░░]
```

**Epoch 10:**
```
CAM[Consolidation]:     CAM[Lung Opacity]:
[░░░░░░░]               [░░░░░░░]
[░░░░░░░]               [░░▓▓▓░░]
[░░▓▓▓░░]  ← Focused    [░▓▓▓▓▓░]  ← Covers area
[░░▓▓▓░░]               [░░▓▓▓░░]
[░░░▓░░░]               [░░░░░░░]
```

**Epoch 20 (Final):**
```
CAM[Consolidation]:     CAM[Lung Opacity]:
[░░░░░░░]               [░░░░░░░]
[░░░░░░░]               [░░░▓░░░]
[░░░███░]  ← Sharp!     [░░▓▓▓▓░]  ← Broader
[░░░███░]               [░▓▓▓▓▓░]
[░░░░█░░]               [░░▓▓▓░░]
```

---

## 📊 Metrics During Training

### **Training Loss**
- Starts: ~0.6-0.8 (random initialization)
- Target: ~0.3-0.5 (converged)
- Oscillates if LR too high → reduce to 1e-4

### **Validation Loss**
- Should track training loss closely
- Gap > 0.1 → overfitting (use early stopping)

### **Concept AUC**
- Random: 0.50
- Epoch 5: ~0.65-0.70
- Epoch 10: ~0.72-0.78
- **Target: ≥ 0.75** (good), **≥ 0.80** (excellent)

### **Per-Concept Performance**
```
Easy concepts (AUC > 0.85):
  - Cardiomegaly (large, obvious)
  - Pleural Effusion (distinct pattern)
  
Medium concepts (AUC 0.70-0.85):
  - Lung Opacity (common but varied)
  - Consolidation (overlaps with others)
  
Hard concepts (AUC < 0.70):
  - Rib Fracture (subtle, small)
  - Lung Cyst (rare in training data)
```

---

## 💾 Stage 1 Output

### **Saved Checkpoint: `best_model_stage1.pth`**

Contains:
```python
{
    'backbone': weights,        # Fine-tuned MedMAE
    'concept_head': weights,    # Trained 1×1 Conv
    'projector': weights,       # Untrained (random init)
    'prototypes': weights,      # Untrained (random init)
    'task_head': weights        # Untrained (random init)
}
```

### **What You Can Do:**

1. **Visualize CAMs:**
```bash
python visualize_cams.py --checkpoint best_model_stage1.pth --image test.png
```

2. **Predict Concepts:**
```python
model.load_state_dict(torch.load('best_model_stage1.pth'))
outputs = model(image)
cams = outputs['cams']  # (1, 22, 7, 7) heatmaps
concept_scores = cams.amax(dim=(-1, -2))  # (1, 22) predictions
```

3. **Proceed to Stage 2:**
```python
# Load Stage 1 weights, freeze backbone & concept_head
# Train projector & prototypes with contrastive loss
```

---

## 🎯 Key Takeaways

1. **Stage 1 is about concept localization**, not disease prediction
2. **CAMs are the core output** - they become prototypes in Stage 2
3. **Max pooling connects CAMs to labels** - strongest activation determines presence
4. **BBox supervision significantly improves** CAM quality (optional but recommended)
5. **Slow learning rates preserve** pre-trained knowledge
6. **Target AUC ≥ 0.75** for good Stage 1 performance

**Next:** Stage 2 uses these CAMs to learn prototypical representations of each concept! 🚀
