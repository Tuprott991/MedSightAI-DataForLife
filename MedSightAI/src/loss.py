import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Hàm loss chuyên dụng cho Segmentation (tốt hơn BCE khi vùng bệnh nhỏ so với nền).
    Đo lường độ chồng lắp giữa Mask dự đoán và Mask thật.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (Batch, 1, H, W) - sau khi qua Sigmoid
        # target: (Batch, 1, H, W)

        # Làm phẳng (flatten) để tính toán
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()

        # Công thức Dice: (2 * giao) / (tổng diện tích pred + tổng diện tích target)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice


class SupervisedContrastiveLoss(nn.Module):
    """
    Hàm loss để gom cụm các vector cùng loại bệnh (Positive) và đẩy xa loại khác (Negative).
    Sử dụng biến thể đơn giản dựa trên Cosine Similarity.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels):
        """
        features: (Batch_Size, Embed_Dim) - Vector concept đầu ra
        labels: (Batch_Size) - Nhãn bệnh (0 hoặc 1)
        """
        # Chuẩn hóa vector về độ dài 1 (Unit sphere)
        features = F.normalize(features, dim=1)

        # Tính ma trận tương đồng (Batch x Batch)
        # similarity_matrix[i, j] là độ giống nhau giữa ảnh i và ảnh j
        similarity_matrix = torch.matmul(features, features.T)

        # Tạo mask xác định các cặp cùng nhãn (Positive pairs)
        # labels: [0, 1, 1, 0] -> labels.unsqueeze(0) == labels.unsqueeze(1)
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float()

        # Mask loại bỏ chính nó (đường chéo của ma trận)
        logits_mask = torch.scatter(
            torch.ones_like(mask_positive),
            1,
            torch.arange(mask_positive.shape[0]).view(-1, 1).to(mask_positive.device),
            0,
        )
        mask_positive = mask_positive * logits_mask

        # Tính Loss (Dạng đơn giản hóa của NT-Xent)
        # Mục tiêu: Tăng similarity của positive, giảm similarity của phần còn lại

        # Lấy các cặp dương tính
        sim_pos = (similarity_matrix * mask_positive).sum(1) / (
            mask_positive.sum(1) + 1e-6
        )

        # Lấy trung bình các cặp âm tính (đơn giản hoá để train ổn định hơn)
        sim_neg = (similarity_matrix * (1 - mask_positive) * logits_mask).sum(1) / (
            (1 - mask_positive).sum(1) + 1e-6
        )

        # Margin loss: Pos phải lớn hơn Neg một khoảng margin
        loss = F.relu(sim_neg - sim_pos + 0.5).mean()

        return loss


class MedicalLoss(nn.Module):
    def __init__(self, seg_weight=5.0, contra_weight=1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.contra_weight = contra_weight

        # 1. Classification Loss (Binary Cross Entropy)
        self.cls_loss_fn = nn.BCEWithLogitsLoss()

        # 2. Segmentation Loss (Dice)
        self.seg_loss_fn = DiceLoss()

        # 3. Contrastive Loss
        self.contra_loss_fn = SupervisedContrastiveLoss()

    def forward(self, outputs, targets):
        """
        outputs: Dict trả về từ Model (class_logits, mask_pred, retrieval_vector)
        targets: Dict trả về từ Dataset (label, mask)
        """

        # --- A. Classification Loss ---
        # targets['label'] shape (Batch,) -> cần unsqueeze thành (Batch, 1)
        labels = targets["label"].unsqueeze(1).to(outputs["class_logits"].device)
        loss_cls = self.cls_loss_fn(outputs["class_logits"], labels)

        # --- B. Segmentation Loss ---
        # outputs['mask_pred'] shape: (B, 1, 24, 24)
        # targets['mask'] shape: (B, 1, 384, 384)
        # -> Cần resize Mask dự đoán lên 384x384 để so khớp
        mask_pred_up = F.interpolate(
            outputs["mask_pred"],
            size=targets["mask"].shape[-2:],  # (384, 384)
            mode="bilinear",
            align_corners=False,
        )

        target_mask = targets["mask"].to(mask_pred_up.device)
        loss_seg = self.seg_loss_fn(mask_pred_up, target_mask)

        # --- C. Contrastive Loss ---
        # Chỉ tính contrastive loss nếu trong batch có ít nhất 2 class khác nhau
        # để tránh lỗi NaN
        loss_contra = torch.tensor(0.0, device=labels.device)
        if labels.sum() > 0 and (1 - labels).sum() > 0:
            loss_contra = self.contra_loss_fn(
                outputs["retrieval_vector"], targets["label"].to(labels.device)
            )

        # --- D. Tổng hợp ---
        total_loss = (
            loss_cls + (self.seg_weight * loss_seg) + (self.contra_weight * loss_contra)
        )

        return {
            "total_loss": total_loss,
            "loss_cls": loss_cls.item(),
            "loss_seg": loss_seg.item(),
            "loss_contra": loss_contra.item(),
        }
