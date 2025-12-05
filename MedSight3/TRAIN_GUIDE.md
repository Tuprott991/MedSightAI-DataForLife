# CSR Model Training - DDP Guide

## 🚀 Quick Start

### Training với Multi-GPU (DDP)

**Windows PowerShell:**
```powershell
.\train_ddp.ps1
```

**Linux/Mac:**
```bash
bash train_ddp.sh
```

## 📋 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_csv` | str | `labels_train.csv` | Path to training CSV |
| `--test_csv` | str | `labels_test.csv` | Path to test CSV |
| `--train_dir` | str | `train/` | Training images directory |
| `--test_dir` | str | `test/` | Test images directory |
| `--batch_size` | int | `16` | Batch size **per GPU** |
| `--lr` | float | `1e-4` | Learning rate |
| `--epochs_stage1` | int | `10` | Epochs for Stage 1 (Concept Learning) |
| `--epochs_stage2` | int | `10` | Epochs for Stage 2 (Prototype Learning) |
| `--epochs_stage3` | int | `10` | Epochs for Stage 3 (Task Learning) |
| `--backbone_type` | str | `medmae` | Backbone type: `medmae`, `resnet50`, `vit` |
| `--model_name` | str | `facebook/vit-mae-base` | Pretrained model name/path |
| `--num_prototypes` | int | `5` | Number of prototypes per concept (M) |
| `--output_dir` | str | `checkpoints` | Output directory for checkpoints |
| `--exp_name` | str | `csr_exp` | Experiment name |

## 💡 Example Usage

### 1. Train với 4 GPUs, batch size 32
```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --batch_size 32 \
    --lr 2e-4 \
    --exp_name "csr_4gpu_bs32"
```

### 2. Train với custom dataset paths
```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --train_csv "/data/vindr/train.csv" \
    --test_csv "/data/vindr/test.csv" \
    --train_dir "/data/vindr/images/train/" \
    --test_dir "/data/vindr/images/test/" \
    --exp_name "vindr_exp"
```

### 3. Train với ResNet50 backbone
```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --backbone_type "resnet50" \
    --model_name "resnet50" \
    --num_prototypes 10 \
    --exp_name "csr_resnet50"
```

### 4. Fine-tune với learning rate thấp hơn
```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --lr 5e-5 \
    --epochs_stage1 5 \
    --epochs_stage2 5 \
    --epochs_stage3 20 \
    --exp_name "csr_finetune"
```

## 📂 Output Structure

```
checkpoints/
└── csr_medmae_exp1/
    ├── best_model_stage1.pth    # Best model from Stage 1 (highest val AUC)
    ├── best_model_stage3.pth    # Best model from Stage 3 (highest val AUC)
    ├── model_stage2_epoch5.pth  # Stage 2 checkpoints (every 5 epochs)
    ├── model_stage2_epoch10.pth
    └── final_model.pth          # Final model after all stages
```

## 🔧 Troubleshooting

### Out of Memory (OOM)
- Giảm `--batch_size` (ví dụ: 8 hoặc 4)
- Dùng ít GPUs hơn
- Dùng backbone nhẹ hơn: `resnet50` thay vì `medmae`

### Duplicate gradient error
- Code đã được fix để tránh lỗi này với DDP
- Đảm bảo dùng `model.module.xxx` khi access parameters trong DDP

### NCCL initialization error
- Kiểm tra môi trường: `echo $CUDA_VISIBLE_DEVICES`
- Chỉ định GPUs: `CUDA_VISIBLE_DEVICES=0,1 torchrun ...`

## 📊 Monitoring Training

Training sẽ hiển thị:
```
--- START STAGE 1: Concept Learning ---
Computing pos_weight for balanced BCE loss...
Pos weights range: 1.23 - 8.45
Stage 1: 100%|████████████| 500/500 [05:23<00:00, 1.55it/s, loss=0.423]
Epoch 1: Train Loss 0.4234
Validating: 100%|████████| 100/100 [00:45<00:00, 2.21it/s]
Epoch 1: Val Loss 0.3821, AUC 0.7234
✅ Saved best Stage 1 model (AUC: 0.7234)
```

## ⚡ Performance Tips

1. **Effective Batch Size = batch_size × num_gpus**
   - 2 GPUs × 16 = 32 effective batch size
   - Có thể tăng `--lr` tương ứng

2. **Stage 2 không cần validate** (chỉ học prototypes)
   - Chỉ save checkpoints định kỳ

3. **Best model tracking**
   - Code tự động save model tốt nhất dựa trên Val AUC
   - Sử dụng `best_model_stage3.pth` cho inference

## 🎯 Next Steps

Sau khi train xong, dùng model inference:
```python
from src.model import CSR

model = CSR(num_concepts=14, num_classes=6, num_prototypes_per_concept=5)
model.load_state_dict(torch.load('checkpoints/csr_exp/best_model_stage3.pth'))
model.eval()
```
