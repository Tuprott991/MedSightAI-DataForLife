import pydicom
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def dicom_to_image(dicom_path, size=(224, 224)):
    """
    Đọc DICOM, áp dụng Lung Windowing chuẩn và resize.
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(float)

        # Rescale Slope/Intercept (quan trọng để lấy đúng Hounsfield Units)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        pixel_array = pixel_array * slope + intercept

        # Apply Lung Windowing (Level: -600, Width: 1500)
        # Giúp làm rõ cấu trúc phổi
        window_center = -600
        window_width = 1500
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        
        pixel_array = np.clip(pixel_array, img_min, img_max)

        # Normalize về 0-255
        pixel_array = ((pixel_array - img_min) / (img_max - img_min)) * 255.0
        pixel_array = pixel_array.astype(np.uint8)

        # Nếu ảnh là Monochrome1 (nền trắng), đảo ngược lại
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel_array = 255 - pixel_array

        img = Image.fromarray(pixel_array).convert("RGB") # MedMAE cần 3 kênh
        img = img.resize(size, Image.Resampling.BICUBIC)
        return img
        
    except Exception as e:
        print(f"Error reading DICOM {dicom_path}: {e}")
        # Trả về ảnh đen nếu lỗi để code không crash
        return Image.new('RGB', size, (0, 0, 0))

class PrototypeContrastiveLoss(nn.Module):
    """
    Contrastive Loss cho Prototype Learning (Stage 2).
    Pull: Kéo local vector của concept k về gần các prototypes của concept k.
    Push: Đẩy xa các prototypes của concepts khác.
    
    Tương ứng Eq. 5 & 9 trong paper.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, local_vectors, prototypes, concepts_gt, num_prototypes_per_concept=1):
        """
        Args:
            local_vectors: (B, K, Dim) - Local concept vectors từ get_local_concept_vectors()
            prototypes: (K*M, Dim, 1, 1) - Learnable prototypes
            concepts_gt: (B, K) - Binary labels (1 nếu concept xuất hiện trong ảnh)
            num_prototypes_per_concept: M - Số prototypes mỗi concept
        """
        B, K, D = local_vectors.shape
        M = num_prototypes_per_concept
        
        # Squeeze prototypes: (K*M, Dim, 1, 1) -> (K*M, Dim)
        protos = prototypes.squeeze(-1).squeeze(-1)  # (K*M, Dim)
        
        # Normalize để tính Cosine Similarity
        local_norm = F.normalize(local_vectors, p=2, dim=-1)  # (B, K, Dim)
        protos_norm = F.normalize(protos, p=2, dim=-1)        # (K*M, Dim)
        
        total_loss = 0
        count = 0
        
        # Duyệt qua từng sample trong batch
        for b in range(B):
            # Lấy các concept thực sự có trong ảnh này
            present_concepts = torch.where(concepts_gt[b] == 1)[0]
            
            if len(present_concepts) == 0:
                continue
            
            # Với mỗi concept k có trong ảnh
            for k in present_concepts:
                # Local vector của concept k: (Dim,)
                query = local_norm[b, k]
                
                # Tính similarity với TẤT CẢ prototypes: (K*M,)
                similarities = torch.matmul(protos_norm, query) / self.temperature  # (K*M,)
                
                # Prototypes của concept k nằm ở indices [k*M : (k+1)*M]
                positive_indices = list(range(k * M, (k + 1) * M))
                
                # InfoNCE Loss cho từng positive prototype
                for pos_idx in positive_indices:
                    # Log-sum-exp trick cho numerical stability
                    # loss = -log( exp(sim_positive) / sum(exp(sim_all)) )
                    #      = -sim_positive + log(sum(exp(sim_all)))
                    pos_sim = similarities[pos_idx]
                    log_sum_exp = torch.logsumexp(similarities, dim=0)
                    
                    total_loss += -pos_sim + log_sum_exp
                    count += 1
        
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=local_vectors.device, requires_grad=True)