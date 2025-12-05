import torch

ckpt = torch.load("../weights/chestx-medmae_finetune.pth", map_location="cpu")
print(ckpt.keys())