# Stage 2: Prototype Learning - Detailed Explanation

## 🎯 Goal of Stage 2

**Train the model to:**
1. Learn **prototypical representations** for each concept
2. Map local concept features to a shared embedding space
3. Use contrastive learning to cluster similar concepts together

**Output:** Learned prototypes (P) that capture canonical patterns of each medical concept.

---

## 📊 Data Flow Diagram

```
INPUT IMAGE                                    LABELS
(224×224 RGB)                                  concepts: [0,1,0,1,...]  (K=22)
     ↓                                         (only used for masking)
     │
┌─────────────────┐
│  MedMAE Backbone│  *** FROZEN ***
│   (from Stage 1)│
└─────────────────┘
     ↓
  Features f
  (B, 768, 7, 7)
     ↓
     │
     ├────────────────────┬─────────────────┐
     │                    │                 │
     ↓                    ↓                 ↓
┌──────────┐      ┌────────────┐   ┌─────────────┐
│ Concept  │      │ Projector  │   │ Prototypes  │
│   Head   │      │   (TRAIN)  │   │   (TRAIN)   │
│*** FROZEN│      │            │   │  Learnable  │
│   ***    │      │ 768→768→128│   │  Parameters │
└──────────┘      └────────────┘   └─────────────┘
     ↓                    ↓                 ↓
   CAMs              f_projected       prototypes
 (B, 22, 7, 7)      (B, 128, 7, 7)   (22M, 128, 1, 1)
     ↓                    │                 │
     │                    │                 │
     ↓                    ↓                 ↓
┌────────────────────────────────────────────────┐
│  get_local_concept_vectors()                   │
│  • Applies spatial softmax on CAMs             │
│  • Weighted sum of projected features          │
└────────────────────────────────────────────────┘
     ↓
Local Concept Vectors
  (B, 22, 768)  ← One vector per concept per image
     ↓
     │
     ↓
┌────────────────────────────────────────────────┐
│  Projector                                      │
│  • Projects to 128-dim embedding space          │
└────────────────────────────────────────────────┘
     ↓
Projected Vectors
  (B, 22, 128)
     ↓
     │
     ├─────────────────────────────────────────────┐
     ↓                                              ↓
┌─────────────────────────┐              ┌──────────────┐
│ Prototypes (learnable)  │              │ Concept GT   │
│  (22M, 128)             │              │  (B, 22)     │
│  M prototypes/concept   │              │  Mask        │
└─────────────────────────┘              └──────────────┘
     ↓                                              ↓
     │                                              │
     └──────────────────┬───────────────────────────┘
                        ↓
            ┌─────────────────────────┐
            │ PrototypeContrastiveLoss│
            │  (InfoNCE)               │
            │  • Pull to same concept  │
            │  • Push from others      │
            └─────────────────────────┘
                        ↓
                  Contrastive Loss
                        ↓
            Backprop → Update projector + prototypes
```

---

## 🔄 Step-by-Step Data Flow

### **Step 1: Load Training Batch**

```python
batch = {
    'image': torch.tensor([B, 3, 224, 224]),
    'concepts': torch.tensor([B, 22]),  # Used as MASK, not targets
    'targets': torch.tensor([B, 6])     # Unused in Stage 2
}
```

**Key difference from Stage 1:**
- Concepts are used to **mask which prototypes to train**
- Only concepts present in image contribute to loss

### **Step 2: Forward Pass (Frozen Backbone & Concept Head)**

```python
# Backbone (frozen)
f = model.backbone(images)  # (B, 768, 7, 7)

# Concept Head (frozen, already trained in Stage 1)
cams = model.concept_head(f)  # (B, 22, 7, 7)
```

**CAMs are now reliable** because they were trained in Stage 1!
- Show WHERE each concept is located
- Used as attention weights for feature extraction

### **Step 3: Extract Local Concept Vectors**

```python
local_vectors = model.get_local_concept_vectors(f, cams)  # (B, 22, 768)
```

**What happens:**

```python
def get_local_concept_vectors(self, f, cams):
    B, K, H, W = cams.size()  # (B, 22, 7, 7)
    C_feat = f.size(1)         # 768
    
    # Step 3a: Spatial Softmax on CAMs
    # Normalize CAMs to create attention weights
    cams_flat = cams.view(B, K, -1)  # (B, 22, 49)
    attn_weights = F.softmax(cams_flat, dim=-1)  # Sum to 1 across spatial locations
    
    # Step 3b: Weighted sum of features
    # For each concept k, extract features from regions where CAM[k] is high
    f_flat = f.view(B, C_feat, -1)  # (B, 768, 49)
    
    # Einstein summation: weighted average over spatial locations
    # local_vectors[b, k, c] = Σᵢ f[b, c, i] × attn_weights[b, k, i]
    local_vectors = torch.einsum('bci,bki->bkc', f_flat, attn_flat)
    
    return local_vectors  # (B, 22, 768)
```

**Intuition:**
- For "Cardiomegaly" concept:
  - CAM highlights heart region
  - Attention focuses on those spatial locations
  - Extract feature vector describing "what heart looks like in this image"
  
**Visual example:**
```
CAM[Cardiomegaly]:          Attention Weights:         Features:
[0.1, 0.2, 0.1]            [0.02, 0.04, 0.02]        [768 values]
[0.3, 0.9, 0.4]    →       [0.06, 0.18, 0.08]   ×    [768 values]  →  Local Vector
[0.2, 0.6, 0.3]            [0.04, 0.12, 0.06]        [768 values]     (768 values)
                           (normalized)
```

Result: One 768-dim vector per concept that captures its appearance in this specific image.

### **Step 4: Project to Embedding Space**

```python
# Projector transforms vectors: 768 → 128 dimensions
projected = model.projector(local_vectors.permute(0, 2, 1).unsqueeze(-1))
projected = projected.squeeze(-1).permute(0, 2, 1)  # (B, 22, 128)
```

**Why projection?**
- **Dimensionality reduction**: 768 → 128 (computational efficiency)
- **Non-linear mapping**: Learns better embedding space for similarity
- **Shared space**: All concepts embedded in same 128-dim space

**Projector architecture:**
```python
nn.Sequential(
    nn.Conv2d(768, 768, kernel_size=1),  # Linear transform
    nn.ReLU(),                            # Non-linearity
    nn.Conv2d(768, 128, kernel_size=1)   # Project to 128-dim
)
```

### **Step 5: Normalize Vectors**

```python
projected_norm = F.normalize(projected, p=2, dim=-1)  # (B, 22, 128)
prototypes_norm = F.normalize(prototypes, p=2, dim=1)  # (22M, 128)
```

**L2 Normalization:**
- All vectors have unit length (||v|| = 1)
- Similarity becomes **cosine similarity**
- Range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite

### **Step 6: Compute Contrastive Loss (InfoNCE)**

```python
loss = PrototypeContrastiveLoss(projected_norm, prototypes_norm, concepts_gt, M=5)
```

**InfoNCE Loss Formula:**
```
For each concept k present in image (concepts_gt[k] = 1):
    
    Positives: M prototypes of concept k
    Negatives: All prototypes of other concepts (21 × M)
    
    For each positive prototype p_k:
        similarity_pos = dot(local_vector_k, p_k)
        
        For each negative prototype p_j (j ≠ k):
            similarity_neg = dot(local_vector_k, p_j)
        
        # Softmax across all prototypes
        exp_pos = exp(similarity_pos / τ)
        exp_all = exp_pos + Σ_j exp(similarity_neg / τ)
        
        loss_k = -log(exp_pos / exp_all)
    
    Total loss = average over all present concepts
```

**Intuition:**
- **Pull:** Local vector should be similar to its own concept's prototypes
- **Push:** Local vector should be different from other concepts' prototypes
- Temperature τ = 0.1 controls hardness (lower = harder negatives)

**Example:**
```python
Image has: Cardiomegaly + Lung Opacity

Local vector for Cardiomegaly:
  Similarity to Cardiomegaly prototypes: [0.85, 0.82, 0.79, 0.88, 0.81]  ← HIGH
  Similarity to Lung Opacity prototypes:  [0.32, 0.28, 0.35, 0.30, 0.29]  ← MEDIUM
  Similarity to other prototypes:         [0.15, -0.05, 0.08, ...]        ← LOW

Loss encourages:
  ✅ Increase 0.85 → 0.95 (pull to own prototypes)
  ⚠️  Decrease 0.32 → 0.15 (push from similar concepts)
  ✅ Keep 0.15 low (already different enough)
```

### **Step 7: Backpropagation**

```python
loss.backward()
optimizer.step()  # Updates projector + prototypes only
```

**What gets updated:**
- ✅ **Projector weights** (learns embedding space)
- ✅ **Prototype parameters** (learns canonical patterns)
- ❌ Backbone (frozen)
- ❌ Concept head (frozen)
- ❌ Task head (not used yet)

**Gradient flow:**
```
Loss → similarities → projected_vectors → projector → local_vectors (stops here)
                   → prototypes (direct update)
```

---

## 📈 What the Model Learns

### **Epoch 1-3: Initial Clustering**
- Prototypes randomly initialized
- Start moving toward concept clusters
- High inter-concept confusion

### **Epoch 4-7: Cluster Formation**
- Prototypes form distinct groups per concept
- Intra-concept similarity increases
- Inter-concept separation improves

### **Epoch 8-10: Refinement**
- Prototypes capture variations within concept
  - Example: 5 Cardiomegaly prototypes capture:
    - Mild enlargement
    - Severe enlargement
    - With pleural effusion
    - With pulmonary edema
    - Rotated heart position
- Embedding space becomes well-structured

**No validation in Stage 2** - contrastive loss is self-supervised!

---

## 💾 Stage 2 Output

### **Saved Checkpoint: `model_stage2_epoch10.pth`**

```python
{
    'backbone': weights,        # Frozen (from Stage 1)
    'concept_head': weights,    # Frozen (from Stage 1)
    'projector': weights,       # ✅ TRAINED
    'prototypes': weights,      # ✅ TRAINED
    'task_head': weights        # Untrained (random init)
}
```

### **Learned Prototypes Visualization:**

```python
# Extract prototypes
prototypes = model.prototypes  # (110, 128, 1, 1) for K=22, M=5

# Prototypes for Cardiomegaly (concept_idx=3):
proto_cardio = prototypes[15:20]  # 5 prototypes

# They should cluster together in embedding space:
# similarity(proto_cardio[0], proto_cardio[1]) ≈ 0.7-0.9  (same concept)
# similarity(proto_cardio[0], proto_opacity[0]) ≈ 0.2-0.4  (different concepts)
```

---

## 🎯 Key Takeaways

1. **Stage 2 is unsupervised** - no ground truth targets needed
2. **Learns embedding space** where similar concepts cluster
3. **Prototypes capture variations** - multiple patterns per concept
4. **Typical duration**: 10 epochs (faster than Stage 1)
5. **No validation** - monitor training loss convergence

**Next:** Stage 3 uses similarity to these prototypes for disease classification! 🚀

---

# Stage 3: Task Learning - Detailed Explanation

## 🎯 Goal of Stage 3

**Train the model to:**
1. Predict diseases (final classification task)
2. Use similarity scores to learned prototypes as features
3. Maintain interpretability through prototype-based reasoning

**Output:** End-to-end disease classification model with full interpretability.

---

## 📊 Data Flow Diagram

```
INPUT IMAGE                                    LABELS
(224×224 RGB)                                  targets: [0,0,1,0,...]  (C=6 diseases)
     ↓
┌─────────────────┐
│  MedMAE Backbone│  *** FROZEN ***
│   (from Stage 1)│
└─────────────────┘
     ↓
  Features f
  (B, 768, 7, 7)
     ↓
     ├────────────────────┬─────────────────┐
     │                    │                 │
     ↓                    ↓                 ↓
┌──────────┐      ┌────────────┐   ┌─────────────┐
│ Concept  │      │ Projector  │   │ Prototypes  │
│   Head   │      │*** FROZEN  │   │*** FROZEN   │
│*** FROZEN│      │    ***     │   │    ***      │
└──────────┘      └────────────┘   └─────────────┘
     ↓                    ↓                 ↓
   CAMs              f_projected       prototypes
 (B, 22, 7, 7)      (B, 128, 7, 7)   (110, 128, 1, 1)
     │                    │                 │
     └────────────────────┴─────────────────┘
                          ↓
            ┌──────────────────────────────┐
            │ Compute Similarity Maps       │
            │ F.conv2d(f_proj, prototypes) │
            │ Cosine similarity at each     │
            │ spatial location              │
            └──────────────────────────────┘
                          ↓
                 Similarity Maps
                 (B, 110, 7, 7)
            ← One map per prototype
                          ↓
            ┌──────────────────────────────┐
            │ Global Max Pooling            │
            │ Extract strongest similarity  │
            └──────────────────────────────┘
                          ↓
                 Similarity Scores
                    (B, 110)
            ← Max similarity to each prototype
                          ↓
            ┌──────────────────────────────┐
            │    Task Head (Linear)         │
            │    (TRAIN THIS)               │
            │    110 → 6 diseases           │
            └──────────────────────────────┘
                          ↓
                   Disease Logits
                      (B, 6)
            [Pneumonia, TB, Lung Tumor, COPD, ...]
                          ↓
            ┌──────────────────────────────┐
            │  BCEWithLogitsLoss            │
            │  Compare with disease labels  │
            └──────────────────────────────┘
                          ↓
                        Loss
                          ↓
            Backprop → Update task_head ONLY
```

---

## 🔄 Step-by-Step Data Flow

### **Step 1: Load Training Batch**

```python
batch = {
    'image': torch.tensor([B, 3, 224, 224]),
    'concepts': torch.tensor([B, 22]),  # Unused in Stage 3
    'targets': torch.tensor([B, 6])     # ✅ USED HERE
}
```

**Example:**
```python
targets[0] = [0, 0, 1, 0, 0, 0]  # Pneumonia present
```

### **Step 2: Forward Pass Through Frozen Modules**

```python
# All frozen from previous stages
f = model.backbone(images)           # (B, 768, 7, 7)
f_proj = model.projector(f)          # (B, 128, 7, 7)
```

### **Step 3: Compute Similarity Maps**

```python
# Normalize for cosine similarity
f_proj_norm = F.normalize(f_proj, p=2, dim=1)      # (B, 128, 7, 7)
prototypes_norm = F.normalize(model.prototypes, p=2, dim=1)  # (110, 128, 1, 1)

# Compute similarity at each spatial location
similarity_maps = F.conv2d(f_proj_norm, prototypes_norm)  # (B, 110, 7, 7)
```

**What happens:**
- For each spatial location (x, y) in the 7×7 grid:
  - Feature vector: `f_proj[b, :, y, x]` (128-dim)
  - Compare with ALL 110 prototypes
  - Result: 110 similarity values

**Intuition:**
```
Image has pneumonia in lower left lung

Similarity map for "Consolidation Prototype #2":
[0.2, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1]
[0.4, 0.6, 0.5, 0.2, 0.1, 0.0, 0.1]
[0.5, 0.8, 0.9, 0.4, 0.2, 0.1, 0.0]  ← High in affected region
[0.3, 0.7, 0.8, 0.5, 0.3, 0.1, 0.1]
[0.2, 0.4, 0.5, 0.3, 0.2, 0.1, 0.1]

Max = 0.9 → Strong match!
```

### **Step 4: Pool to Similarity Scores**

```python
similarity_scores = F.max_pool2d(similarity_maps, kernel_size=(7, 7))  # (B, 110)
similarity_scores = similarity_scores.view(B, -1)
```

**Result:**
- One score per prototype
- Represents "How similar is ANY region of this image to this prototype?"

**Example scores:**
```python
Prototypes for Cardiomegaly (5 total):     [0.85, 0.82, 0.68, 0.91, 0.75]
Prototypes for Lung Opacity (5 total):     [0.92, 0.88, 0.85, 0.90, 0.87]  ← HIGH!
Prototypes for Consolidation (5 total):    [0.78, 0.85, 0.91, 0.83, 0.79]  ← HIGH!
Prototypes for Pneumothorax (5 total):     [0.25, 0.18, 0.32, 0.28, 0.21]  ← LOW
...
```

### **Step 5: Disease Prediction**

```python
disease_logits = model.task_head(similarity_scores)  # (B, 6)
```

**Task Head:**
```python
nn.Linear(110, 6)  # 110 prototype similarities → 6 disease predictions
```

**Learned weights interpretation:**
```python
# Pneumonia is associated with:
#   - High similarity to Consolidation prototypes: +weight
#   - High similarity to Lung Opacity prototypes: +weight
#   - Low similarity to Normal prototypes: -weight

Weight matrix (6 diseases × 110 prototypes):
              [Cardio_1, Cardio_2, ..., Opacity_1, Opacity_2, ..., Consol_1, ...]
Pneumonia:    [  0.12,     0.08,   ...,   0.85,      0.78,    ...,   0.92,   ...]
COPD:         [  0.45,     0.52,   ...,   0.34,      0.28,    ...,   0.15,   ...]
Lung Tumor:   [ -0.10,    -0.05,   ...,   0.68,      0.72,    ...,   0.55,   ...]
...
```

**Reasoning example:**
```python
similarity_scores = [0.85(Cardio), 0.92(Opacity), 0.91(Consol), ...]

disease_logits[Pneumonia] = 0.12×0.85 + ... + 0.85×0.92 + 0.78×... + 0.92×0.91 + ...
                           = 0.102 + ... + 0.782 + ... + 0.837 + ...
                           = 2.45  → High confidence!

disease_logits[COPD]      = 0.45×0.85 + ... + 0.34×0.92 + ... + 0.15×0.91 + ...
                           = -0.35  → Low confidence
```

### **Step 6: Compute Loss**

```python
loss = BCEWithLogitsLoss()(disease_logits, targets)
```

**Standard multi-label classification:**
- Each disease independently predicted
- Loss encourages high logits for present diseases, low for absent

### **Step 7: Backpropagation**

```python
loss.backward()
optimizer.step()  # Updates task_head ONLY
```

**What gets updated:**
- ✅ **Task Head weights** (110 → 6 linear layer)
- ❌ Everything else frozen

**Gradient flow:**
```
Loss → disease_logits → task_head → similarity_scores (stops here, no grad)
```

---

## 📈 What the Model Learns

### **Epoch 1-5: Initial Mapping**
- Task head learns which prototypes correlate with which diseases
- Example: "High similarity to Consolidation prototypes → Pneumonia"

### **Epoch 6-15: Pattern Discovery**
- Learns combinations: "Consolidation + Pleural Effusion → Severe Pneumonia"
- Distinguishes similar diseases: "Lung Opacity vs Lung Tumor"

### **Epoch 16-20: Refinement**
- Fine-tunes weights for edge cases
- Handles confusing patterns (e.g., COPD + Pneumonia overlap)

---

## 📊 Validation Metrics

### **Disease AUC (Primary Metric)**
- **Target:** ≥ 0.82 (good), ≥ 0.85 (excellent)
- Computed for each disease, then macro-averaged

### **Per-Disease Performance:**
```
Common diseases (AUC > 0.85):
  - Pneumonia (clear consolidation patterns)
  - No Finding (exclusionary diagnosis)

Medium diseases (AUC 0.75-0.85):
  - COPD (subtle emphysema signs)
  - Lung Tumor (varies in size/shape)

Hard diseases (AUC < 0.75):
  - Tuberculosis (overlaps with pneumonia)
  - Other diseases (heterogeneous category)
```

### **Validation Loss:**
- Should decrease from ~0.5 to ~0.2-0.3
- Gap with training loss should be small (<0.05)

---

## 💾 Stage 3 Output

### **Final Model: `best_model_stage3.pth`**

```python
{
    'backbone': weights,        # Frozen
    'concept_head': weights,    # Frozen
    'projector': weights,       # Frozen
    'prototypes': weights,      # Frozen
    'task_head': weights        # ✅ TRAINED
}
```

**This is your production model!** 🎉

---

## 🎯 Key Takeaways

1. **Stage 3 is standard classification** using prototype similarities as features
2. **Learns disease-prototype associations** (interpretable!)
3. **Fast training:** 15-20 epochs, only 6×110 parameters
4. **Target AUC ≥ 0.82** for good performance
5. **Fully interpretable:** Can trace prediction back to prototypes → CAMs → image regions

---

# Inference: Using the Trained Model

## 🔮 Inference Pipeline

```
                    INPUT: Chest X-ray
                            ↓
        ┌───────────────────────────────────────┐
        │  1. Preprocess Image                   │
        │     • Resize to 224×224                │
        │     • Normalize                        │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  2. Extract Features                   │
        │     f = backbone(image)                │
        │     (B, 768, 7, 7)                     │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  3. Generate CAMs                      │
        │     cams = concept_head(f)             │
        │     (B, 22, 7, 7)                      │
        │     → Shows WHERE concepts are         │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  4. Project Features                   │
        │     f_proj = projector(f)              │
        │     (B, 128, 7, 7)                     │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  5. Compute Prototype Similarities     │
        │     similarity_maps = conv2d(f_proj, P)│
        │     (B, 110, 7, 7)                     │
        │     similarity_scores = max_pool(maps) │
        │     (B, 110)                           │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  6. Predict Diseases                   │
        │     logits = task_head(scores)         │
        │     probs = sigmoid(logits)            │
        │     (B, 6)                             │
        └───────────────────────────────────────┘
                            ↓
        OUTPUT: Disease predictions + Explanations
```

---

## 💻 Inference Code

### **Basic Inference:**

```python
import torch
from src.model import CSR
from PIL import Image
from torchvision import transforms

# Load model
model = CSR(num_concepts=22, num_classes=6, 
            num_prototypes_per_concept=5,
            backbone_type='medmae',
            model_name='weights/pre_trained_medmae.pth')
model.load_state_dict(torch.load('outputs/best_model_stage3.pth'))
model.eval()
model = model.cuda()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load image
image = Image.open('patient_xray.png').convert('RGB')
image_tensor = transform(image).unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    outputs = model(image_tensor)
    
    disease_probs = torch.sigmoid(outputs['logits']).cpu().numpy()[0]

# Results
disease_names = ['No finding', 'Pneumonia', 'Tuberculosis', 
                 'Lung tumor', 'COPD', 'Other diseases']

print("Disease Predictions:")
for name, prob in zip(disease_names, disease_probs):
    print(f"  {name}: {prob:.1%}")
```

**Output:**
```
Disease Predictions:
  No finding: 2.3%
  Pneumonia: 87.5%      ← HIGH!
  Tuberculosis: 15.2%
  Lung tumor: 8.1%
  COPD: 23.4%
  Other diseases: 5.6%
```

### **Interpretable Inference:**

```python
# Get full outputs
with torch.no_grad():
    outputs = model(image_tensor)

# 1. Concept detection
cams = outputs['cams']  # (1, 22, 7, 7)
concept_scores = cams.amax(dim=(-1, -2))[0]  # (22,) max per concept

concept_names = ['Aortic enlargement', 'Atelectasis', 'Calcification',
                 'Cardiomegaly', 'Consolidation', 'Edema', ...]

print("\nDetected Concepts:")
for name, score in zip(concept_names, concept_scores):
    if torch.sigmoid(score) > 0.5:
        print(f"  ✓ {name}: {torch.sigmoid(score):.1%}")

# Output:
# Detected Concepts:
#   ✓ Lung Opacity: 92.3%
#   ✓ Consolidation: 88.7%
#   ✓ Pleural effusion: 65.4%

# 2. Prototype activation
similarity_scores = outputs['similarity_scores'][0]  # (110,)

# Find top activated prototypes
top_k = 5
top_indices = similarity_scores.topk(top_k).indices

print(f"\nTop {top_k} Activated Prototypes:")
for idx in top_indices:
    concept_idx = idx // 5  # M=5 prototypes per concept
    proto_num = idx % 5
    similarity = similarity_scores[idx].item()
    print(f"  Prototype {idx}: {concept_names[concept_idx]} #{proto_num+1} "
          f"(similarity: {similarity:.3f})")

# Output:
# Top 5 Activated Prototypes:
#   Prototype 55: Lung Opacity #1 (similarity: 0.923)
#   Prototype 25: Consolidation #1 (similarity: 0.887)
#   Prototype 58: Lung Opacity #4 (similarity: 0.865)
#   Prototype 80: Pleural effusion #1 (similarity: 0.789)
#   Prototype 27: Consolidation #3 (similarity: 0.754)

# 3. Reasoning trace
task_head_weights = model.task_head.weight  # (6, 110)
pneumonia_weights = task_head_weights[1]  # Weights for Pneumonia

print("\nPneumonia Prediction Reasoning:")
print(f"  Disease probability: {disease_probs[1]:.1%}")
print(f"\n  Contributing factors:")

# Top contributing prototypes
contributions = pneumonia_weights * similarity_scores
top_contrib = contributions.topk(3)

for score, idx in zip(top_contrib.values, top_contrib.indices):
    concept_idx = idx // 5
    similarity = similarity_scores[idx].item()
    weight = pneumonia_weights[idx].item()
    print(f"    • {concept_names[concept_idx]}: "
          f"weight={weight:.2f} × similarity={similarity:.3f} = {score:.3f}")

# Output:
# Pneumonia Prediction Reasoning:
#   Disease probability: 87.5%
#
#   Contributing factors:
#     • Consolidation: weight=0.92 × similarity=0.887 = 0.816
#     • Lung Opacity: weight=0.85 × similarity=0.923 = 0.785
#     • Pleural effusion: weight=0.43 × similarity=0.789 = 0.339
```

### **Generate Explanation Heatmaps:**

```python
# Visualize CAMs
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Show original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Input X-ray")
axes[0, 0].axis('off')

# Show top 5 concept CAMs
for i, idx in enumerate([11, 5, 16, 1, 6][:5]):  # Top concepts
    ax = axes.flatten()[i+1]
    
    # Get CAM and upsample
    cam = cams[0, idx].cpu()
    cam = torch.sigmoid(cam)  # Normalize to [0, 1]
    cam_upsampled = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode='bilinear'
    )[0, 0]
    
    # Overlay
    ax.imshow(image, alpha=0.7)
    ax.imshow(cam_upsampled, cmap='jet', alpha=0.5)
    ax.set_title(f"{concept_names[idx]}\n{torch.sigmoid(concept_scores[idx]):.1%}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('explanation.png', dpi=150, bbox_inches='tight')
```

---

## 🎯 Inference Modes

### **1. Fast Inference (Disease prediction only)**
```python
outputs = model(image)
probs = torch.sigmoid(outputs['logits'])
# ~5-10ms on GPU
```

### **2. Interpretable Inference (With explanations)**
```python
outputs = model(image)
cams = outputs['cams']
similarity_scores = outputs['similarity_scores']
# ~10-15ms on GPU (includes heatmap generation)
```

### **3. Interactive Inference (With doctor input)**
```python
# Doctor draws ROI on image → importance_map
importance_map = create_importance_map(doctor_roi)  # (1, 1, 224, 224)

outputs = model(image, importance_map=importance_map)
# Model focuses on specified region
# ~15-20ms on GPU
```

---

## 📊 Inference Output Structure

```python
outputs = {
    'logits': (B, 6),              # Disease predictions (raw scores)
    'similarity_scores': (B, 110),  # Similarity to each prototype
    'similarity_maps': (B, 110, 7, 7),  # Spatial similarity maps
    'cams': (B, 22, 7, 7),         # Concept activation maps
    'features': (B, 768, 7, 7),    # Backbone features
    'projected_features': (B, 128, 7, 7)  # Projected features
}
```

**Use cases:**
- `logits` → Final diagnosis
- `cams` → Show concept locations to doctor
- `similarity_scores` → Explain which prototypes fired
- `similarity_maps` → Show exact regions matching prototypes

---

## 🚀 Deployment

### **REST API Example:**

```python
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load model once at startup
model = CSR(...)
model.load_state_dict(torch.load('best_model_stage3.pth'))
model.eval().cuda()

@app.route('/predict', methods=['POST'])
def predict():
    # Receive image
    image_bytes = request.files['image'].read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0).cuda()
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
    
    # Format response
    result = {
        'diseases': {
            name: float(prob) 
            for name, prob in zip(disease_names, probs)
        },
        'threshold': 0.5
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Usage:**
```bash
curl -X POST -F "image=@xray.png" http://localhost:5000/predict

{
  "diseases": {
    "No finding": 0.023,
    "Pneumonia": 0.875,
    "Tuberculosis": 0.152,
    ...
  },
  "threshold": 0.5
}
```

---

## 🎯 Key Takeaways

1. **Inference is fast:** 5-15ms per image on GPU
2. **Fully interpretable:** Every prediction can be explained
3. **Multiple outputs:** Diseases + Concepts + Prototypes + Heatmaps
4. **Production-ready:** Easy to deploy as API or batch processor
5. **Doctor-in-the-loop:** Supports interactive ROI focusing

The complete 3-stage pipeline creates an interpretable, high-performance medical AI! 🏥✨
