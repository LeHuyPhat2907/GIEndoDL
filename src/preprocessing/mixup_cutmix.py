"""Module cài đặt thuật toán CutMix và MixUp cấp độ Batch và Tensor cho PyTorch."""

from typing import Tuple
import numpy as np
import torch


class CutMixMixUpAugmenter:
    """Bộ tạo tăng cường CutMix và MixUp tối ưu cho huấn luyện mạng nơ-ron."""

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob_mixup: float = 0.5,
        prob_cutmix: float = 0.5,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob_mixup = prob_mixup
        self.prob_cutmix = prob_cutmix

    @staticmethod
    def rand_bbox(w: int, h: int, lam: float) -> Tuple[int, int, int, int, float]:
        """Tạo tọa độ bounding box ngẫu nhiên cho CutMix dựa trên tỷ lệ lambda."""
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Tâm bounding box ngẫu nhiên
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Tính lại lambda thực tế dựa trên diện tích vùng cắt
        adjusted_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        return bbx1, bby1, bbx2, bby2, adjusted_lam

    def apply_mixup_pair(
        self, img1: np.ndarray, img2: np.ndarray, lam: float = None
    ) -> Tuple[np.ndarray, float]:
        """Trộn 2 ảnh cấp độ pixel (MixUp)."""
        if lam is None:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_img = (
            lam * img1.astype(np.float32) + (1.0 - lam) * img2.astype(np.float32)
        ).astype(np.uint8)
        return mixed_img, float(lam)

    def apply_cutmix_pair(
        self, img1: np.ndarray, img2: np.ndarray, lam: float = None
    ) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
        """Cắt dán patch giữa 2 ảnh (CutMix)."""
        h, w = img1.shape[:2]
        if lam is None:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        bbx1, bby1, bbx2, bby2, actual_lam = self.rand_bbox(w, h, lam)
        cut_img = img1.copy()
        cut_img[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]

        return cut_img, actual_lam, (bbx1, bby1, bbx2, bby2)

    def apply_batch(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Áp dụng CutMix hoặc MixUp trên một batch Tensor trong quá trình Training Loop."""
        batch_size = images.size(0)
        rand_index = torch.randperm(batch_size)

        targets_a = targets
        targets_b = targets[rand_index]

        r = np.random.rand()
        if r < self.prob_cutmix:
            # Áp dụng CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            h, w = images.size(2), images.size(3)
            bbx1, bby1, bbx2, bby2, actual_lam = self.rand_bbox(w, h, lam)
            images[:, :, bby1:bby2, bbx1:bbx2] = images[
                rand_index, :, bby1:bby2, bbx1:bbx2
            ]
            return images, targets_a, targets_b, actual_lam
        else:
            # Áp dụng MixUp
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            images = lam * images + (1 - lam) * images[rand_index]
            return images, targets_a, targets_b, lam
