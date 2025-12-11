import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import cv2

NUM_CLASSES = 14
NUM_PROTOTYPES = 15
MODEL_NAME = "densenet121"
IMG_SIZE = 384


class CSRModel(nn.Module):
    def __init__(self, num_classes=14, num_prototypes=5, model_name="resnet50", pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(4,)
        )
        feature_info = self.backbone.feature_info.get_dicts()[-1]
        self.feature_dim = feature_info["num_chs"]
        self.concept_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)
        
        self.embedding_dim = 128
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_dim)
        )
        
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, self.embedding_dim))
        self.task_head = nn.Linear(num_classes * num_prototypes, num_classes)
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

    def get_features_and_cam(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.backbone(x)[0]
        attn_logits = self.concept_head(features)
        return features, attn_logits
    
    def forward_phase1(self, x):
        """Forward cho Phase 1 - dùng GAP trên CAM"""
        _, attn_logits = self.get_features_and_cam(x)
        logits = F.adaptive_avg_pool2d(attn_logits, (1, 1)).view(x.size(0), -1)
        return {"logits": logits, "attn_maps": attn_logits}


def load_csr_model(checkpoint_path, device):
    model = CSRModel(
        num_classes=NUM_CLASSES,
        num_prototypes=NUM_PROTOTYPES,
        model_name=MODEL_NAME,
        pretrained=False
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, target_size=IMG_SIZE):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.resize(image, (target_size, target_size))
    img_norm = image.astype("float32") / 255.0
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    return img_tensor

def infer_cams(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model.forward_phase1(image_tensor)
        logits = outputs['logits'][0]
        cams = outputs['attn_maps'][0]  # [K, H, W]
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs, cams.cpu().numpy()