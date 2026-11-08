# -*- coding: utf-8 -*-
"""Hàm mất mát phân tầng: Class-Balanced Focal Loss kết hợp Ordinal/Cost Matrix (Task D49 / Run 7).

Đặc tả kỹ thuật:
1. Class-Balanced Focal Loss (Cui et al., CVPR 2019 + Lin et al., ICCV 2017):
   - alpha_c = (1 - beta) / (1 - beta^N_c) với beta = 0.9999.
   - Modulating factor (1 - p_t)^gamma với gamma = 2.0.
   - Label smoothing = 0.08 chống overconfidence.
2. Ordinal Cost Matrix C(i, j) cho 23 phân lớp HyperKvasir:
   - Khoảng cách thứ bậc cấp độ viêm loét Mayo: C(i, j) = 0.1 * |rank(i) - rank(j)|.
   - Phạt nặng rủi ro an toàn lâm sàng (False Negative): C(i, j) = 2.0 khi bỏ sót tổn thương
     (Mayo, Barrett, Polyp, Esophagitis, Hemorrhoids) và phân loại nhầm thành mốc giải phẫu
     lành tính (Cecum, Pylorus, Z-line, Retroflex Stomach/Rectum, Ileum).
   - L_total = L_CB_Focal + lambda_ord * sum_j C(y_true, j) * P(y_hat = j) (lambda_ord = 0.35).
"""

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAYO_RANKS = {
    "ulcerative-colitis-grade-0-1": 0.5,
    "ulcerative-colitis-grade-1": 1.0,
    "ulcerative-colitis-grade-1-2": 1.5,
    "ulcerative-colitis-grade-2": 2.0,
    "ulcerative-colitis-grade-2-3": 2.5,
    "ulcerative-colitis-grade-3": 3.0,
}

LESION_CLASSES = {
    "barretts",
    "barretts-short-segment",
    "esophagitis-a",
    "esophagitis-b-d",
    "polyps",
    "dyed-lifted-polyps",
    "hemorrhoids",
    "ulcerative-colitis-grade-0-1",
    "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-1-2",
    "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-2-3",
    "ulcerative-colitis-grade-3",
}

BENIGN_LANDMARKS = {
    "cecum",
    "pylorus",
    "z-line",
    "retroflex-stomach",
    "retroflex-rectum",
    "ileum",
}


def build_medical_cost_matrix(
    class_names: List[str],
    mayo_penalty_scale: float = 0.1,
    lesion_miss_penalty: float = 2.0,
    benign_false_alarm_penalty: float = 0.5,
) -> torch.Tensor:
    """Xây dựng ma trận phạt chi phí sai số lâm sàng C (shape: [num_classes, num_classes]).

    C[i, j]: Chi phí phạt khi nhãn thực là i nhưng mô hình dự đoán là j.
    """
    num_classes = len(class_names)
    cost_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    for i, c_true in enumerate(class_names):
        for j, c_pred in enumerate(class_names):
            if i == j:
                cost_matrix[i, j] = 0.0
                continue

            # 1. Hai lớp cùng thuộc thang đo thứ bậc Mayo Grade
            if c_true in MAYO_RANKS and c_pred in MAYO_RANKS:
                dist = abs(MAYO_RANKS[c_true] - MAYO_RANKS[c_pred])
                cost_matrix[i, j] = mayo_penalty_scale * dist

            # 2. Nguy cơ y khoa nghiêm trọng: Tổn thương bệnh lý bị chẩn đoán nhầm thành cơ quan lành tính
            elif c_true in LESION_CLASSES and c_pred in BENIGN_LANDMARKS:
                cost_matrix[i, j] = lesion_miss_penalty

            # 3. Chẩn đoán dương tính giả từ mốc lành tính sang tổn thương
            elif c_true in BENIGN_LANDMARKS and c_pred in LESION_CLASSES:
                cost_matrix[i, j] = benign_false_alarm_penalty

            # 4. Các sai số khác (cùng nhóm lành tính hoặc can thiệp lâm sàng)
            else:
                cost_matrix[i, j] = 0.0

    return torch.tensor(cost_matrix, dtype=torch.float32)


def compute_effective_class_weights(
    samples_per_class: List[int],
    beta: float = 0.9999,
) -> torch.Tensor:
    """Tính toán vector trọng số lớp alpha_c theo số mẫu hiệu dụng (Cui et al., CVPR 2019)."""
    samples = np.array(samples_per_class, dtype=np.float64)
    # E_n = (1 - beta^n) / (1 - beta)
    effective_num = 1.0 - np.power(beta, samples)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    # Chuẩn hóa để mean = 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedOrdinalFocalLoss(nn.Module):
    """Hàm tổn thất kết hợp Class-Balanced Focal Loss và Ma trận Chi phí Thứ bậc Y sinh."""

    def __init__(
        self,
        class_names: List[str],
        samples_per_class: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        label_smoothing: float = 0.08,
        lambda_ord: float = 0.35,
        mayo_penalty_scale: float = 0.1,
        lesion_miss_penalty: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.lambda_ord = float(lambda_ord)
        self.reduction = reduction

        # Trọng số Class-Balanced
        alpha = compute_effective_class_weights(samples_per_class, beta=beta)
        self.register_buffer("alpha", alpha)

        # Ma trận chi phí lâm sàng C(i, j)
        cost_matrix = build_medical_cost_matrix(
            class_names=class_names,
            mayo_penalty_scale=mayo_penalty_scale,
            lesion_miss_penalty=lesion_miss_penalty,
        )
        self.register_buffer("cost_matrix", cost_matrix)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Tính toán tổn thất tổng hợp trên mini-batch.

        Args:
            logits: Tensor [B, num_classes] (logits thô chưa qua softmax).
            targets: Tensor [B] (chỉ số nhãn thực [0, num_classes - 1]).

        Returns:
            Scalar Loss tensor.
        """
        # 1. Tính Softmax và Log-Softmax
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        probs = torch.exp(log_probs)  # [B, C]

        # 2. Label Smoothing targets: y_smooth = (1 - eps)*y + eps/C
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs, self.label_smoothing / self.num_classes
            )
            smooth_targets.scatter_(
                -1,
                targets.unsqueeze(-1),
                1.0 - self.label_smoothing + (self.label_smoothing / self.num_classes),
            )

        # 3. Focal Modulating Factor: (1 - p)^gamma
        focal_weight = torch.pow(1.0 - probs, self.gamma)  # [B, C]

        # 4. Trọng số Class-Balanced alpha: [1, C]
        alpha = self.alpha.unsqueeze(0)  # [1, C]

        # Thành phần CB-Focal: - sum_c [ alpha_c * (1 - p_c)^gamma * y_smooth_c * log(p_c) ]
        focal_per_sample = -torch.sum(
            alpha * focal_weight * smooth_targets * log_probs, dim=-1
        )  # [B]

        # 5. Thành phần Ordinal Cost Penalty: sum_j [ C(y_true, j) * P(j) ]
        # Lấy hàng chi phí tương ứng với y_true cho từng mẫu trong batch
        batch_costs = self.cost_matrix[targets]  # [B, C]
        ordinal_penalty_per_sample = torch.sum(batch_costs * probs, dim=-1)  # [B]

        # 6. Tổng hợp tổn thất
        total_loss_per_sample = (
            focal_per_sample + self.lambda_ord * ordinal_penalty_per_sample
        )

        if self.reduction == "mean":
            return total_loss_per_sample.mean()
        elif self.reduction == "sum":
            return total_loss_per_sample.sum()
        else:
            return total_loss_per_sample
