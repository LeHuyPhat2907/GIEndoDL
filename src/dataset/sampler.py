# -*- coding: utf-8 -*-
"""Module tính toán trọng số lớp và xây dựng WeightedRandomSampler cho PyTorch DataLoader."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_and_sample_weights(
    train_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    mode: str = "power",
    beta: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Tính toán trọng số lớp và trọng số mẫu cho PyTorch DataLoader.

    Args:
        train_df: Dataframe tập train (chứa cột 'class_name').
        class_to_idx: Bảng ánh xạ nhãn chuỗi sang index số nguyên.
        mode:
            - 'power': w_c = 1 / (N_c ** beta) (khuyến nghị beta=0.75 cho cân bằng tối ưu).
            - 'inverse_sqrt': w_c = 1 / sqrt(N_c) (tương đương power với beta=0.5).
            - 'inverse': w_c = 1 / N_c (tương đương power với beta=1.0, cân bằng tuyệt đối xác suất lớp).
            - 'effective_num': w_c = (1 - beta) / (1 - beta ** N_c) (Cui et al., CVPR 2019).
        beta: Hệ số lũy thừa hoặc dung sai mẫu hiệu dụng.
    """
    class_counts = train_df["class_name"].value_counts().to_dict()
    num_classes = len(class_to_idx)

    # 1. Tính trọng số cho từng lớp
    class_weights_dict = {}
    class_weights_tensor = torch.zeros(num_classes, dtype=torch.float32)

    for cls_name, idx in class_to_idx.items():
        count = float(class_counts.get(cls_name, 1))
        count = max(1.0, count)

        if mode == "inverse":
            weight = 1.0 / count
        elif mode == "inverse_sqrt":
            weight = 1.0 / np.sqrt(count)
        elif mode == "effective_num":
            eff_beta = min(0.99999, max(0.9, beta))
            effective_num = 1.0 - np.power(eff_beta, count)
            weight = (1.0 - eff_beta) / max(effective_num, 1e-8)
        else:  # 'power'
            weight = 1.0 / (count**beta)

        class_weights_dict[cls_name] = float(weight)
        class_weights_tensor[idx] = float(weight)

    # 2. Gán trọng số cho từng bức ảnh trong tập train
    sample_weights = []
    for _, row in train_df.iterrows():
        cls_name = row["class_name"]
        sample_weights.append(class_weights_dict.get(cls_name, 1.0))

    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

    return sample_weights_tensor, class_weights_tensor, class_weights_dict


def get_weighted_sampler(
    train_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    mode: str = "power",
    beta: float = 0.75,
) -> WeightedRandomSampler:
    """Tạo PyTorch WeightedRandomSampler tối ưu cho DataLoader."""
    sample_weights, _, _ = compute_class_and_sample_weights(
        train_df, class_to_idx, mode=mode, beta=beta
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True,
    )
    return sampler
