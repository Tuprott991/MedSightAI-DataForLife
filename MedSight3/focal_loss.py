"""
Focal Loss for handling severe class imbalance in disease classification.
Especially useful when you have 70% "No finding" vs 30% diseases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification with severe class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the model's estimated probability for the correct class
    - alpha: balancing factor (default 0.25)
    - gamma: focusing parameter (default 2.0)
    
    When gamma > 0, it down-weights easy examples and focuses on hard negatives.
    This is perfect for medical datasets where "No finding" dominates.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Balancing factor in [0, 1]. Higher alpha gives more weight to rare classes.
            gamma: Focusing parameter >= 0. Higher gamma focuses more on hard examples.
                   gamma=0 is equivalent to BCE loss.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw predictions (before sigmoid)
            targets: (B, C) binary targets
        
        Returns:
            loss: scalar or (B, C) depending on reduction
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # For numerical stability, clip probabilities
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        
        # Compute p_t (probability of the true class)
        # If target=1, p_t = p, if target=0, p_t = 1-p
        p_t = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute BCE loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        # Apply alpha balancing
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification with imbalanced data.
    
    Different from Focal Loss, this loss applies different margins and focusing
    for positive vs negative samples. Useful when you want to:
    - Reduce false positives (e.g., predicting disease when there's none)
    - Increase true positives (e.g., catching rare diseases)
    
    Reference: https://arxiv.org/abs/2009.14119
    """
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, reduction='mean'):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples (default 4)
            gamma_pos: Focusing parameter for positive samples (default 1)
            clip: Clipping value for negative probabilities (default 0.05)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw predictions
            targets: (B, C) binary targets
        """
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # For positive targets
        pos_loss = targets * torch.log(probs.clamp(min=1e-7))
        pos_loss = pos_loss * (1 - probs) ** self.gamma_pos
        
        # For negative targets - with probability clipping
        neg_probs = (probs - self.clip).clamp(min=0)
        neg_loss = (1 - targets) * torch.log((1 - neg_probs).clamp(min=1e-7))
        neg_loss = neg_loss * neg_probs ** self.gamma_neg
        
        loss = -(pos_loss + neg_loss)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss using Effective Number of Samples.
    
    Reweights loss based on how many samples of each class exist.
    Good for datasets where some diseases are extremely rare.
    
    Reference: https://arxiv.org/abs/1901.05555
    """
    
    def __init__(self, samples_per_class, beta=0.9999, loss_type='focal', gamma=2.0):
        """
        Args:
            samples_per_class: List or tensor of positive sample counts per class
            beta: Hyperparameter in [0, 1). Higher beta = more aggressive reweighting
            loss_type: 'focal', 'sigmoid', or 'softmax'
            gamma: Focal loss gamma if loss_type='focal'
        """
        super().__init__()
        
        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        self.register_buffer('weights', weights)
        self.loss_type = loss_type
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=None, gamma=gamma, reduction='none')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw predictions
            targets: (B, C) binary targets
        """
        # Compute base loss
        loss = self.criterion(logits, targets)
        
        # Apply class weights
        weighted_loss = loss * self.weights.unsqueeze(0)
        
        return weighted_loss.mean()


# Example usage
if __name__ == '__main__':
    # Simulate imbalanced data
    batch_size = 16
    num_classes = 6
    
    # Simulate predictions and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.zeros(batch_size, num_classes)
    
    # Make "No finding" (class 5) very common (70%)
    targets[:, 5] = (torch.rand(batch_size) > 0.3).float()
    
    # Make rare diseases (classes 0-4) rare (2-5% each)
    for i in range(5):
        targets[:, i] = (torch.rand(batch_size) > 0.95).float()
    
    print("Target distribution:")
    print(targets.sum(dim=0) / batch_size)
    
    # Compare losses
    print("\n" + "="*60)
    print("LOSS COMPARISON")
    print("="*60)
    
    # Standard BCE
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"Standard BCE Loss: {bce_loss.item():.4f}")
    
    # Focal Loss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal(logits, targets)
    print(f"Focal Loss (alpha=0.25, gamma=2): {focal_loss.item():.4f}")
    
    # Asymmetric Loss
    asym = AsymmetricLoss(gamma_neg=4, gamma_pos=1)
    asym_loss = asym(logits, targets)
    print(f"Asymmetric Loss: {asym_loss.item():.4f}")
    
    # Class-Balanced Loss
    samples_per_class = torch.tensor([32, 48, 95, 120, 70, 10600])  # Simulated class distribution
    cb_loss_fn = ClassBalancedLoss(samples_per_class, beta=0.9999, loss_type='focal')
    cb_loss = cb_loss_fn(logits, targets)
    print(f"Class-Balanced Focal Loss: {cb_loss.item():.4f}")
