"""Module tính toán trọng số lớp và xây dựng WeightedRandomSampler cho PyTorch DataLoader."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_and_sample_weights(
    train_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    mode: str = "inverse_sqrt",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Tính toán trọng số lớp và trọng số mẫu.

    Args:
        train_df: Dataframe tập train (chứa cột 'class_name').
        class_to_idx: Bảng ánh xạ nhãn chuỗi sang index số nguyên.
        mode: 'inverse' (1/N) hoặc 'inverse_sqrt' (1/sqrt(N) - khuyến nghị làm mịn).
    """
    class_counts = train_df["class_name"].value_counts().to_dict()
    num_classes = len(class_to_idx)

    # 1. Tính trọng số cho từng lớp
    class_weights_dict = {}
    class_weights_tensor = torch.zeros(num_classes, dtype=torch.float32)

    for cls_name, idx in class_to_idx.items():
        count = class_counts.get(cls_name, 1)
        if mode == "inverse":
            weight = 1.0 / count
        else:  # inverse_sqrt (cân bằng mịn, tránh oversample quá đà)
            weight = 1.0 / np.sqrt(count)

        class_weights_dict[cls_name] = float(weight)
        class_weights_tensor[idx] = float(weight)

    # 2. Gán trọng số cho từng bức ảnh trong tập train
    sample_weights = []
    for _, row in train_df.iterrows():
        cls_name = row["class_name"]
        sample_weights.append(class_weights_dict[cls_name])

    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

    return sample_weights_tensor, class_weights_tensor, class_weights_dict


def get_weighted_sampler(
    train_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    mode: str = "inverse_sqrt",
) -> WeightedRandomSampler:
    """Tạo PyTorch WeightedRandomSampler tối ưu cho DataLoader."""
    sample_weights, _, _ = compute_class_and_sample_weights(
        train_df, class_to_idx, mode=mode
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True,
    )
    return sampler
