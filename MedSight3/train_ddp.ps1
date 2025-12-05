# Script để train CSR model với DDP (Multi-GPU) trên Windows PowerShell

# Số GPUs sử dụng
$NUM_GPUS = 2

# Run training với torchrun
torchrun `
    --standalone `
    --nproc_per_node=$NUM_GPUS `
    train.py `
    --train_csv "labels_train.csv" `
    --test_csv "labels_test.csv" `
    --train_dir "train/" `
    --test_dir "test/" `
    --batch_size 16 `
    --lr 1e-4 `
    --epochs_stage1 10 `
    --epochs_stage2 10 `
    --epochs_stage3 10 `
    --backbone_type "medmae" `
    --model_name "weights/pre_trained_medmae.pth" `
    --num_prototypes 5 `
    --output_dir "checkpoints" `
    --exp_name "csr_medmae_exp1"
