# Kaggle Notebook Cell - Run this in a Kaggle notebook with 2 GPUs enabled

# Install dependencies if needed
# !pip install -q pydicom transformers tensorboard gradio

# Navigate to code directory
import os
os.chdir('/kaggle/working/SoftAI---DataForLife---MedSightAI/Tu_MedSight')

# Run distributed training with torchrun
!torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --data_root "/kaggle/input/vindr-cxr" \
    --exp_name "ddp_experiment" \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --num_concepts 37 \
    --num_classes 15 \
    --projection_dim 128 \
    --prototypes_per_concept 4 \
    --freeze_backbone \
    --contrastive_loss_weight 0.5 \
    --num_workers 2 \
    --save_freq 5

# Note: 
# - Make sure to enable "GPU x2" accelerator in Kaggle notebook settings
# - batch_size=8 is per GPU, so effective batch_size = 8 * 2 = 16
# - Adjust --data_root to match your Kaggle dataset path
