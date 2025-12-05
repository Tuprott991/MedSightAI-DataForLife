import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class MedMAEBackbone(nn.Module):
    def __init__(self, model_name='facebook/vit-mae-base', pretrained_weights=None):
        super(MedMAEBackbone, self).__init__()
        # Load pre-trained ViT/MAE
        # Nếu model_name là HuggingFace repo → load từ HF
        # Nếu pretrained_weights được cung cấp → load local weights
        
        if pretrained_weights and pretrained_weights.endswith('.pth'):
            print(f"📦 Loading MedMAE weights from {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location='cpu')
            
            # Extract model weights nếu checkpoint có structure {'model': ..., 'optimizer': ...}
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                print(f"✓ Detected checkpoint format (with 'model' key)")
                state_dict = checkpoint['model']
                if 'epoch' in checkpoint:
                    print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
            else:
                state_dict = checkpoint
            
            # Tự động detect config từ weights
            print(f"🔍 Detecting architecture from weights...")
            
            # Phân tích keys để suy ra config
            sample_keys = list(state_dict.keys())[:10]
            print(f"Sample keys: {sample_keys[:3]}")
            
            # Detect hidden_size từ embedding weights
            embed_key = None
            for k in state_dict.keys():
                if 'embeddings.patch_embeddings.projection.weight' in k or 'patch_embed.proj.weight' in k:
                    embed_key = k
                    break
            
            if embed_key:
                hidden_size = state_dict[embed_key].shape[0]
                print(f"✓ Detected hidden_size: {hidden_size}")
            else:
                hidden_size = 768  # Default ViT-Base
                print(f"⚠️  Could not detect hidden_size, using default: {hidden_size}")
            
            # Detect num_layers từ số lượng attention layers
            num_layers = 0
            for k in state_dict.keys():
                if 'encoder.layer' in k or 'blocks' in k:
                    # Extract layer index
                    if 'encoder.layer.' in k:
                        layer_idx = int(k.split('encoder.layer.')[1].split('.')[0])
                    elif 'blocks.' in k:
                        layer_idx = int(k.split('blocks.')[1].split('.')[0])
                    else:
                        continue
                    num_layers = max(num_layers, layer_idx + 1)
            
            if num_layers == 0:
                num_layers = 12  # Default ViT-Base
                print(f"⚠️  Could not detect num_layers, using default: {num_layers}")
            else:
                print(f"✓ Detected num_layers: {num_layers}")
            
            # Tạo config dựa trên detected values
            config = ViTConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=hidden_size // 64,  # Thường là hidden_size / 64
                intermediate_size=hidden_size * 4,      # Thường là 4x hidden_size
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                image_size=224,
                patch_size=16,
                num_channels=3
            )
            
            print(f"🔧 Creating ViT with config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, heads={config.num_attention_heads}")
            
            # Khởi tạo model từ config (không download)
            self.vit = ViTModel(config)
            
            # MedMAE weights có thể có prefix 'encoder.' hoặc 'vit.' hoặc 'model.'
            # Cần xử lý để match với ViTModel
            new_state_dict = {}
            for k, v in state_dict.items():
                # Loại bỏ prefix nếu có
                new_key = k.replace('encoder.', '').replace('vit.', '').replace('model.', '')
                new_state_dict[new_key] = v
            
            # Load weights (strict=False để bỏ qua các keys không khớp như decoder)
            missing_keys, unexpected_keys = self.vit.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Loaded MedMAE weights successfully")
            if len(missing_keys) > 0:
                print(f"⚠️  Missing keys: {len(missing_keys)} (this is normal if MedMAE has different head)")
            if len(unexpected_keys) > 0:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)} (decoder weights will be ignored)")
        else:
            # Load trực tiếp từ HuggingFace
            print(f"📥 Loading model from HuggingFace: {model_name}")
            self.vit = ViTModel.from_pretrained(model_name)
        
        # Lấy hidden dimension (ví dụ: ViT-Base là 768)
        self.embed_dim = self.vit.config.hidden_size
        
        # Patch size (thường là 16)
        self.patch_size = self.vit.config.patch_size
        
    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W). Ví dụ: 224x224
        Returns:
            feature_map: (B, Embed_Dim, Grid_H, Grid_W). Ví dụ: (B, 768, 14, 14)
        """
        # 1. Forward qua ViT
        # outputs.last_hidden_state shape: (B, Sequence_Length, Hidden_Dim)
        # Sequence_Length = 1 (CLS token) + (H*W)/(P*P) patches
        outputs = self.vit(x)
        last_hidden_state = outputs.last_hidden_state
        
        # 2. Loại bỏ CLS token (token đầu tiên dùng để phân loại chung)
        # Chúng ta cần features của từng vùng ảnh cho CSR
        patch_tokens = last_hidden_state[:, 1:, :] # (B, 196, 768) với ảnh 224, patch 16
        
        # 3. Reshape từ Sequence về 2D Grid
        # Tính kích thước lưới grid: H_grid = H_img // patch_size
        B, N, C = patch_tokens.shape
        H_grid = int(N**0.5) # Căn bậc 2 của số patch (ví dụ: căn(196) = 14)
        W_grid = H_grid
        
        # Permute để đưa Channel lên trước: (B, H*W, C) -> (B, C, H*W)
        patch_tokens = patch_tokens.permute(0, 2, 1)
        
        # View lại thành 2D: (B, C, H_grid, W_grid)
        feature_map = patch_tokens.view(B, C, H_grid, W_grid)
        
        return feature_map

    @property
    def out_channels(self):
        # Property để CSR biết dimension đầu ra (thay vì fc.in_features của ResNet)
        return self.embed_dim