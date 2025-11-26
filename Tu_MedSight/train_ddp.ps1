# Run distributed training with torchrun on Kaggle (2 GPUs)
# For Windows PowerShell

# Training script with torchrun for 2 GPUs
torchrun `
    --standalone `
    --nnodes=1 `
    --nproc_per_node=2 `
    train.py `
    --data_root "D:\Github Repos\SoftAI---DataForLife---MedSightAI" `
    --exp_name "ddp_experiment" `
    --epochs 50 `
    --batch_size 8 `
    --lr 1e-4 `
    --num_concepts 37 `
    --num_classes 15 `
    --projection_dim 128 `
    --prototypes_per_concept 4 `
    --freeze_backbone `
    --contrastive_loss_weight 0.5 `
    --num_workers 2 `
    --save_freq 5

# Note: batch_size is per GPU, so effective batch size = 8 * 2 = 16
