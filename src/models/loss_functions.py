"""Module cài đặt các hàm mất mát cân bằng lớp và điều chuẩn (Class-Balanced, Focal, và Label Smoothing Loss)."""

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
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        if custom_weights is not None:
            self.weights = custom_weights
        elif class_counts is not None:
            self.weights = compute_effective_num_weights(class_counts, beta=beta)
        else:
            self.weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        weight = self.weights.to(device) if self.weights is not None else None
        return F.cross_entropy(
            logits, targets, weight=weight, label_smoothing=self.label_smoothing
        )


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., ICCV 2017) tập trung vào các mẫu khó (Hard Examples)."""

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
        device = logits.device
        num_classes = logits.size(1)

        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        p_t = (p * target_one_hot).sum(dim=-1)
        log_p_t = (log_p * target_one_hot).sum(dim=-1)

        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = -modulating_factor * log_p_t

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


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (CB-Focal Loss) kết hợp Cui et al. (CVPR 2019) và Lin et al. (ICCV 2017)."""

    def __init__(
        self,
        class_counts: List[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.class_weights = compute_effective_num_weights(class_counts, beta=beta)
        self.focal_loss = FocalLoss(
            gamma=gamma, alpha=self.class_weights, reduction=reduction
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(logits, targets)


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Label Smoothing Cross-Entropy Loss hỗ trợ Class Weights chuẩn y khoa."""

    def __init__(
        self,
        epsilon: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        w = self.weight.to(device) if self.weight is not None else None
        return F.cross_entropy(
            logits,
            targets,
            weight=w,
            label_smoothing=self.epsilon,
            reduction=self.reduction,
        )
