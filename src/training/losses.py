# -*- coding: utf-8 -*-
"""Module định nghĩa các hàm mất mát tiên tiến cho bài toán mất cân bằng lớp cực đoan (Task #127 / Run 2).

Bao gồm:
1. Class-Balanced Focal Loss (CB-Focal):
   - Trọng số alpha_c tính theo tần suất nghịch đảo lũy thừa: alpha_c = 1 / (N_c)^beta (beta in [0.5, 0.9999]).
   - Hoặc theo số mẫu hiệu dụng (Effective Number of Samples - Cui et al., CVPR 2019):
     alpha_c = (1 - beta) / (1 - beta^{N_c}).
   - Modulating factor (1 - p_t)^gamma triệt tiêu gradient từ mẫu dễ (Lin et al., ICCV 2017).
   - Tích hợp Label Smoothing điều hòa phân bố nhãn.

2. Asymmetric Loss (ASL - Ridnik et al., ICCV 2021):
   - Phân tách tham số tập trung: gamma_pos (mẫu dương) và gamma_neg (mẫu âm, gamma_neg >> gamma_pos).
   - Asymmetric Probability Margin Shifting (clip m): Triệt tiêu triệt để gradient từ mẫu âm dễ (p_neg < m -> grad = 0),
     loại bỏ hoàn toàn hiện tượng 22 lớp âm áp đảo 1 lớp dương trong ảnh y khoa.
   - Hỗ trợ nhân vector trọng số lớp alpha_c để tăng cường độ nhạy cho các lớp hiếm (hemorrhoids, ileum, barretts).
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    class_counts: Dict[str, int],
    class_to_idx: Dict[str, int],
    mode: str = "power",
    beta: float = 0.75,
) -> torch.Tensor:
    """Tính toán vector trọng số lớp alpha_c (shape: C) đồng bộ 100% với class_to_idx.

    Args:
        class_counts: Dict ánh xạ tên lớp sang số lượng mẫu (ví dụ: {'polyps': 720, ...}).
        class_to_idx: Dict ánh xạ tên lớp sang index số nguyên (ví dụ: {'barretts': 0, ...}).
        mode:
            - 'power': alpha_c = 1.0 / (N_c ** beta). Với beta=0.5 (căn bậc hai), beta=0.75 (khuyến nghị),
              hoặc beta=1.0 (nghịch đảo hoàn toàn).
            - 'effective_num': alpha_c = (1.0 - beta) / (1.0 - beta ** N_c) (Cui et al., CVPR 2019).
            - 'equal': alpha_c = 1.0 cho tất cả các lớp.
        beta: Tham số lũy thừa hoặc dung sai mẫu hiệu dụng.

    Returns:
        torch.Tensor shape (C,) với mean = 1.0 (bảo toàn thang đo gradient tổng thể).
    """
    num_classes = len(class_to_idx)
    weights = np.zeros(num_classes, dtype=np.float64)

    for cname, idx in class_to_idx.items():
        count = float(class_counts.get(cname, 1))
        count = max(1.0, count)

        if mode == "power":
            # alpha_c = 1 / (N_c)^beta
            w = 1.0 / (count**beta)
        elif mode == "effective_num":
            # Cui et al., CVPR 2019: E_n = (1 - beta^n) / (1 - beta)
            eff_beta = min(0.99999, max(0.9, beta))
            effective_num = 1.0 - np.power(eff_beta, count)
            w = (1.0 - eff_beta) / max(effective_num, 1e-8)
        elif mode == "equal":
            w = 1.0
        else:
            # Mặc định an toàn: inverse_sqrt (power với beta=0.5)
            w = 1.0 / np.sqrt(count)

        weights[idx] = w

    # Chuẩn hóa để trung bình trọng số bằng 1.0 (giữ nguyên quy mô learning rate)
    mean_w = np.mean(weights)
    if mean_w > 0:
        weights = weights / mean_w

    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    """Hàm tổn thất Class-Balanced Focal Loss với hỗ trợ Label Smoothing.

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.clone().detach().float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính toán Class-Balanced Focal Loss.

        logits: Tensor (B, C) - Raw output từ mô hình.
        targets: Tensor (B,) - Nhãn đích dạng chỉ số nguyên [0, C-1].
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce_loss)
        modulating_factor = (1.0 - pt) ** self.gamma
        focal_loss = modulating_factor * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class AsymmetricLoss(nn.Module):
    """Hàm tổn thất Asymmetric Loss (ASL) chuyên biệt cho dữ liệu mất cân bằng cực đoan.

    Tham khảo: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    Được điều chỉnh cho bài toán phân loại đa lớp y sinh học HyperKvasir (23 lớp).

    Tính năng cốt lõi:
    - gamma_pos (mặc định 1.0): Tập trung vào mẫu dương mà không làm suy yếu gradient.
    - gamma_neg (mặc định 4.0): Suy giảm lũy thừa mạnh đối với các mẫu âm.
    - clip (mặc định 0.05): Dịch chuyển ngưỡng xác suất p_neg -> max(p_neg - clip, 0).
      Khi p_neg <= clip, tổn thất và gradient của mẫu âm triệt tiêu về 0 hoàn toàn.
    - alpha (tùy chọn): Vector trọng số lớp alpha_c giúp khuếch đại lớp hiếm.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.clone().detach().float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính toán Asymmetric Loss.

        logits: Tensor (B, C)
        targets: Tensor (B,)
        """
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Tính xác suất per-class qua hàm Sigmoid
        probs = torch.sigmoid(logits)

        # 1. Tổn thất trên lớp dương (Positive): y_c = 1
        p_pos = probs
        loss_pos = (
            targets_one_hot
            * ((1.0 - p_pos) ** self.gamma_pos)
            * torch.log(p_pos.clamp(min=self.eps))
        )

        # 2. Tổn thất trên các lớp âm (Negative): y_k = 0 (22 lớp âm)
        p_neg = probs
        if self.clip > 0:
            p_neg = (p_neg - self.clip).clamp(min=0.0)

        loss_neg = (
            (1.0 - targets_one_hot)
            * (p_neg**self.gamma_neg)
            * torch.log((1.0 - p_neg).clamp(min=self.eps))
        )

        # Tổng hợp per-class loss
        loss = -(loss_pos + loss_neg)

        # Áp dụng trọng số lớp alpha nếu có
        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)

        # Cộng dồn qua 23 lớp cho mỗi mẫu: (B,)
        sample_loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return sample_loss.mean()
        elif self.reduction == "sum":
            return sample_loss.sum()
        return sample_loss


def build_loss_criterion(
    loss_type: str = "cb_focal",
    class_counts: Optional[Dict[str, int]] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
    weight_mode: str = "power",
    cb_beta: float = 0.75,
    gamma: float = 2.0,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 1.0,
    asl_clip: float = 0.05,
    label_smoothing: float = 0.05,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Optional[torch.Tensor]]:
    """Hàm khởi tạo linh hoạt bộ loss function dựa trên yêu cầu người dùng.

    Returns:
        (criterion, alpha_tensor)
    """
    alpha = None
    if class_counts is not None and class_to_idx is not None:
        alpha = compute_class_weights(
            class_counts=class_counts,
            class_to_idx=class_to_idx,
            mode=weight_mode,
            beta=cb_beta,
        )
        if device is not None:
            alpha = alpha.to(device)

    loss_key = loss_type.lower().strip()

    if loss_key in ["asl", "asymmetric", "asymmetric_loss"]:
        criterion = AsymmetricLoss(
            gamma_neg=asl_gamma_neg,
            gamma_pos=asl_gamma_pos,
            clip=asl_clip,
            alpha=alpha,
        )
    elif loss_key in ["cb_focal", "class_balanced_focal", "focal"]:
        criterion = ClassBalancedFocalLoss(
            gamma=gamma,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )
    else:
        # Cross entropy chuẩn có class weights
        criterion = nn.CrossEntropyLoss(
            weight=alpha,
            label_smoothing=label_smoothing,
        )

    if device is not None:
        criterion = criterion.to(device)

    return criterion, alpha
