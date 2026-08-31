"""Module xây dựng Two-View Augmentation Pipeline cho Supervised Contrastive Learning (SupCon)."""

from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch


class TwoCropTransform:
    """Wrapper biến đổi một ảnh đầu vào thành 2 Views độc lập (v1, v2) cho Contrastive Loss."""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img_rgb) -> List[torch.Tensor]:
        return [
            self.transform(image=img_rgb)["image"],
            self.transform(image=img_rgb)["image"],
        ]


class SupConAugmenter:
    """Bộ tạo pipeline tăng cường cường độ cao (Strong Augmentations) cho Contrastive Learning."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.5729, 0.3557, 0.2515),
        std: Tuple[float, float, float] = (0.3105, 0.2116, 0.1834),
    ):
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def get_contrastive_pipeline(self) -> A.Compose:
        """Tạo chuỗi biến đổi mạnh cho từng góc nhìn của SupCon."""
        return A.Compose(
            [
                # 1. Cắt tỉa ngẫu nhiên tỷ lệ rộng (Scale 50% - 100%)
                A.RandomResizedCrop(
                    size=self.img_size,
                    scale=(0.50, 1.0),
                    ratio=(0.85, 1.15),
                    interpolation=cv2.INTER_CUBIC,
                    p=1.0,
                ),
                # 2. Biến đổi hình học 3D
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # 3. Biến dạng cơ sinh học mô mềm
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=1.0,
                            sigma=40,
                            interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_REFLECT,
                            p=1.0,
                        ),
                        A.OpticalDistortion(
                            distort_limit=0.15,
                            shift_limit=0.05,
                            interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_REFLECT,
                            p=1.0,
                        ),
                    ],
                    p=0.4,
                ),
                # 4. Color Jitter mạnh hơn có kiểm soát
                A.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.20,
                    hue=0.04,
                    p=0.8,
                ),
                # 5. Chuyển đổi xám ngẫu nhiên (Ép mạng học hình thái học khi mất màu đỏ)
                A.ToGray(p=0.20),
                # 6. Làm mờ Gaussian (Chống bẫy nhiễu tần số cao)
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                # 7. Chuẩn hóa Tensor
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

    def get_two_view_transform(self) -> TwoCropTransform:
        """Trả về callable TwoCropTransform để tích hợp vào PyTorch Dataset."""
        return TwoCropTransform(self.get_contrastive_pipeline())
