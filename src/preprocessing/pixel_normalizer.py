"""Module tính toán thống kê kênh màu (Mean/Std) và chuẩn hóa Tensor PyTorch."""

import json
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import torchvision.transforms as transforms


class PixelNormalizer:
    """Bộ chuẩn hóa điểm ảnh Tensor hỗ trợ cả ImageNet và HyperKvasir Custom Stats."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, custom_stats_path: str = None):
        self.custom_mean = None
        self.custom_std = None
        if custom_stats_path and Path(custom_stats_path).exists():
            with open(custom_stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
                self.custom_mean = stats["mean_rgb"]
                self.custom_std = stats["std_rgb"]

    @staticmethod
    def compute_dataset_stats(
        image_paths: List[Path],
    ) -> Tuple[List[float], List[float]]:
        """Tính toán chính xác Mean và Std trên toàn bộ tập ảnh trong không gian [0.0, 1.0]."""
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        total_pixels = 0

        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            h, w = img_rgb.shape[:2]
            pixels = h * w
            total_pixels += pixels

            channel_sum += np.sum(img_rgb, axis=(0, 1))
            channel_sq_sum += np.sum(img_rgb**2, axis=(0, 1))

        mean = channel_sum / total_pixels
        std = np.sqrt((channel_sq_sum / total_pixels) - (mean**2))

        return [round(float(m), 4) for m in mean], [round(float(s), 4) for s in std]

    def get_transform(
        self,
        img_size: Tuple[int, int] = (224, 224),
        use_custom_stats: bool = True,
    ) -> transforms.Compose:
        """Tạo pipeline transform PyTorch chuẩn."""
        mean = (
            self.custom_mean
            if (use_custom_stats and self.custom_mean)
            else self.IMAGENET_MEAN
        )
        std = (
            self.custom_std
            if (use_custom_stats and self.custom_std)
            else self.IMAGENET_STD
        )

        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
