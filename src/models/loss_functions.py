"""Module cài đặt các hàm mất mát cân bằng lớp (Class-Balanced CE & Focal Loss)."""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_effective_num_weights(
    class_counts: List[int], beta: float = 0.999
) -> torch.Tensor:
    """Tính trọng số dựa trên 'Effective Number of Samples' (Cui et al. CVPR 2019)."""
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(class_counts)
    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedCrossEntropyLoss(nn.Module):
    """Class-Balanced Cross-Entropy Loss (Cui et al., CVPR 2019) cho bài toán đuôi dài."""

    def __init__(
        self,
        class_counts: Optional[List[int]] = None,
        beta: float = 0.999,
        custom_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if custom_weights is not None:
            self.weights = custom_weights
        elif class_counts is not None:
            self.weights = compute_effective_num_weights(class_counts, beta=beta)
        else:
            self.weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        weight = self.weights.to(device) if self.weights is not None else None
        return F.cross_entropy(logits, targets, weight=weight)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., ICCV 2017) tập trung vào các mẫu khó (Hard Examples).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính toán Focal Loss đa lớp.

        Args:
            logits: Tensor dự đoán [Batch, Num_Classes].
            targets: Nhãn thực tế [Batch].
        """
        device = logits.device
        num_classes = logits.size(1)

        # 1. Tính Softmax probabilities và Log-Softmax
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # 2. Tạo One-hot tensor cho nhãn mục tiêu
        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # 3. Lấy xác suất p_t của đúng lớp thực tế
        p_t = (p * target_one_hot).sum(dim=-1)
        log_p_t = (log_p * target_one_hot).sum(dim=-1)

        # 4. Tính Modulating Factor: (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        loss = -modulating_factor * log_p_t

        # 5. Áp dụng hệ số cân bằng alpha nếu có
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                loss = self.alpha * loss
            elif isinstance(self.alpha, torch.Tensor):
                alpha_device = self.alpha.to(device)
                alpha_t = (alpha_device * target_one_hot).sum(dim=-1)
                loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
