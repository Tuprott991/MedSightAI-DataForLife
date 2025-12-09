import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class MedMAEBackbone(nn.Module):
    def __init__(self, model_name='facebook/vit-mae-base', pretrained_weights=None, img_size=224):
        """
        Args:
            model_name: HuggingFace model name
            pretrained_weights: Path to .pth file
            img_size: Target image size (224 hoặc 384)
        """
        super(MedMAEBackbone, self).__init__()
        
        self.img_size = img_size
        self.patch_size = 16  # ViT-Base mặc định
        
        if pretrained_weights and pretrained_weights.endswith('.pth'):
            print(f"📦 Loading MedMAE weights from {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Detect config từ weights
            print(f"🔍 Detecting architecture from weights...")
            
            embed_key = None
            for k in state_dict.keys():
                if 'embed' in k.lower() and 'weight' in k.lower() and 'patch' not in k.lower():
                    embed_key = k
                    break
            
            if embed_key:
                hidden_size = state_dict[embed_key].shape[-1]
            else:
                hidden_size = 768
            
            num_layers = 0
            for k in state_dict.keys():
                if 'layer' in k.lower() or 'block' in k.lower():
                    parts = k.split('.')
                    for p in parts:
                        if p.isdigit():
                            num_layers = max(num_layers, int(p) + 1)
            
            if num_layers == 0:
                num_layers = 12
                
            print(f"   Detected: hidden_size={hidden_size}, num_layers={num_layers}")
            
            # Tạo config với image_size mới
            config = ViTConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=hidden_size // 64,
                intermediate_size=hidden_size * 4,
                image_size=img_size,  # ⚠️ Quan trọng: set image size mới
                patch_size=self.patch_size,
            )
            
            self.vit = ViTModel(config)
            
            # ===== INTERPOLATE POSITION EMBEDDINGS =====
            if img_size != 224:
                state_dict = self._interpolate_pos_embed(state_dict, img_size)
            
            # Load weights
            missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
            print(f"   Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
        else:
            print(f"📦 Loading from HuggingFace: {model_name}")
            self.vit = ViTModel.from_pretrained(model_name)
            
            # Nếu img_size != 224, cần resize position embeddings
            if img_size != 224:
                self._resize_pos_embed_for_new_size(img_size)
        
        self.embed_dim = self.vit.config.hidden_size
        
    def _interpolate_pos_embed(self, state_dict, new_img_size):
        """
        Interpolate position embeddings từ 224 lên size mới (vd: 384)
        """
        # Tìm key của position embeddings
        pos_embed_key = None
        for k in state_dict.keys():
            if 'position_embed' in k.lower() or 'pos_embed' in k.lower():
                pos_embed_key = k
                break
        
        if pos_embed_key is None:
            print("   ⚠️ Position embeddings not found, skipping interpolation")
            return state_dict
        
        pos_embed = state_dict[pos_embed_key]  # Shape: (1, num_patches + 1, hidden_dim)
        
        # Tính số patches cũ và mới
        old_num_patches = pos_embed.shape[1] - 1  # Trừ CLS token
        old_grid_size = int(old_num_patches ** 0.5)  # 14 cho 224/16
        new_grid_size = new_img_size // self.patch_size  # 24 cho 384/16
        
        if old_grid_size == new_grid_size:
            return state_dict
        
        print(f"   🔄 Interpolating pos_embed: {old_grid_size}x{old_grid_size} -> {new_grid_size}x{new_grid_size}")
        
        # Tách CLS token và patch embeddings
        cls_token = pos_embed[:, :1, :]  # (1, 1, hidden_dim)
        patch_pos_embed = pos_embed[:, 1:, :]  # (1, old_num_patches, hidden_dim)
        
        # Reshape về 2D grid
        hidden_dim = patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, hidden_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, hidden_dim, H, W)
        
        # Interpolate
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_grid_size, new_grid_size),
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape lại
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, H, W, hidden_dim)
        patch_pos_embed = patch_pos_embed.reshape(1, new_grid_size * new_grid_size, hidden_dim)
        
        # Ghép lại với CLS token
        new_pos_embed = torch.cat([cls_token, patch_pos_embed], dim=1)
        state_dict[pos_embed_key] = new_pos_embed
        
        print(f"   ✅ New pos_embed shape: {new_pos_embed.shape}")
        
        return state_dict
    
    def _resize_pos_embed_for_new_size(self, new_img_size):
        """
        Resize position embeddings của model đã load từ HuggingFace
        """
        pos_embed = self.vit.embeddings.position_embeddings  # (1, num_patches+1, hidden_dim)
        
        old_num_patches = pos_embed.shape[1] - 1
        old_grid_size = int(old_num_patches ** 0.5)
        new_grid_size = new_img_size // self.patch_size
        
        if old_grid_size == new_grid_size:
            return
        
        print(f"   🔄 Resizing pos_embed: {old_grid_size}x{old_grid_size} -> {new_grid_size}x{new_grid_size}")
        
        cls_token = pos_embed[:, :1, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        
        hidden_dim = patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, hidden_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_grid_size, new_grid_size),
            mode='bicubic',
            align_corners=False
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        patch_pos_embed = patch_pos_embed.reshape(1, new_grid_size * new_grid_size, hidden_dim)
        
        new_pos_embed = torch.cat([cls_token, patch_pos_embed], dim=1)
        
        # Update model
        self.vit.embeddings.position_embeddings = nn.Parameter(new_pos_embed)
        self.vit.config.image_size = new_img_size
        
    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W)
        Returns:
            feature_map: (B, Embed_Dim, Grid_H, Grid_W)
        """
        outputs = self.vit(x)
        last_hidden_state = outputs.last_hidden_state
        
        # Bỏ CLS token
        patch_tokens = last_hidden_state[:, 1:, :]
        
        # Reshape về 2D
        B, N, C = patch_tokens.shape
        H_grid = int(N ** 0.5)
        W_grid = H_grid
        
        patch_tokens = patch_tokens.permute(0, 2, 1)
        feature_map = patch_tokens.view(B, C, H_grid, W_grid)
        
        return feature_map

    @property
    def out_channels(self):
        return self.embed_dim