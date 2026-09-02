"""Module cài đặt các hàm mất mát cân bằng lớp (Class-Weighted & Class-Balanced Cross-Entropy Loss)."""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_effective_num_weights(
    class_counts: List[int], beta: float = 0.999
) -> torch.Tensor:
    """Tính trọng số dựa trên 'Effective Number of Samples' (Cui et al.

    CVPR 2019).

    Args:
        class_counts: Danh sách số lượng mẫu của từng lớp [N_0, N_1, ..., N_22].
        beta: Tham số siêu phẳng làm mịn (Mặc định 0.999 cho y tế).
    """
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)

    # Chuẩn hóa để tổng trọng số bằng số lớp C
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
        """Tính toán giá trị Loss.

        Args:
            logits: Đầu ra dự đoán chưa qua softmax [B, C].
            targets: Nhãn thực tế [B].
        """
        device = logits.device
        weight = self.weights.to(device) if self.weights is not None else None
        return F.cross_entropy(logits, targets, weight=weight)
