"""Module xây dựng các pipeline Data Augmentation Hình học & Màu sắc chuẩn y tế bằng Albumentations."""

from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class MedicalDataAugmenter:
    """Bộ tạo pipeline Data Augmentation chuyên biệt cho nội soi đường tiêu hóa."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.5729, 0.3557, 0.2515),
        std: Tuple[float, float, float] = (0.3105, 0.2116, 0.1834),
    ):
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def get_color_transforms(self) -> A.Compose:
        """Pipeline tăng cường màu sắc đã hiệu chuẩn an toàn y khoa."""
        return A.Compose(
            [
                # 1. Hiệu chỉnh sáng/tương phản/bão hòa vừa phải
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.04,  # Giới hạn góc lệch Hue <= 8 độ để bảo toàn sắc tố hồng niêm mạc
                    p=0.6,
                ),
                # 2. Điều chỉnh Gamma phi tuyến tính
                A.RandomGamma(gamma_limit=(85, 115), p=0.4),
            ]
        )

    def get_full_training_pipeline(self) -> A.Compose:
        """Pipeline tăng cường toàn diện: Hình học + Màu sắc + Chuẩn hóa Tensor."""
        return A.Compose(
            [
                # Biến đổi hình học
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.06,
                    scale_limit=0.10,
                    rotate_limit=30,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                A.RandomResizedCrop(
                    size=self.img_size,
                    scale=(0.80, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=cv2.INTER_CUBIC,
                    p=0.5,
                ),
                # Biến đổi màu sắc an toàn
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.04,
                    p=0.5,
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=0.3),
                # Chuẩn hóa Tensor cuối cùng
                A.Resize(
                    height=self.img_size[1],
                    width=self.img_size[0],
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )
