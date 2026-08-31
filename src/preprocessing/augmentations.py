"""Module xây dựng các pipeline Data Augmentation Hình học, Màu sắc & Biến dạng Mô mềm chuyên biệt Y tế."""

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

    def get_deformable_transforms(self) -> A.Compose:
        """Pipeline biến dạng cơ sinh học mô mềm và thấu kính quang học y tế."""
        return A.Compose(
            [
                # 1. Biến dạng đàn hồi (Mô phỏng sóng nhu động ruột co bóp)
                A.ElasticTransform(
                    alpha=1.0,
                    sigma=50,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.4,
                ),
                # 2. Biến dạng lưới (Mô phỏng áp lực bơm khí CO2 làm giãn thành ruột)
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.15,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.4,
                ),
                # 3. Biến dạng quang học (Mô phỏng thấu kính góc rộng mắt cá 140-170 độ)
                A.OpticalDistortion(
                    distort_limit=0.15,
                    shift_limit=0.05,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.4,
                ),
            ]
        )

    def get_full_training_pipeline(self) -> A.Compose:
        """Pipeline tăng cường toàn diện đỉnh cao: Hình học + Biến dạng Mô mềm + Màu sắc + Tensor."""
        return A.Compose(
            [
                # 1. Biến đổi hình học đa hướng
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
                # 2. Biến dạng cơ sinh học mô mềm y tế
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=1.0,
                            sigma=40,
                            interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_REFLECT,
                            p=1.0,
                        ),
                        A.GridDistortion(
                            num_steps=5,
                            distort_limit=0.12,
                            interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_REFLECT,
                            p=1.0,
                        ),
                        A.OpticalDistortion(
                            distort_limit=0.12,
                            shift_limit=0.04,
                            interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_REFLECT,
                            p=1.0,
                        ),
                    ],
                    p=0.4,
                ),
                # 3. Hiệu chuẩn màu sắc an toàn
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.04,
                    p=0.5,
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=0.3),
                # 4. Chuẩn hóa kích thước và Tensor
                A.Resize(
                    height=self.img_size[1],
                    width=self.img_size[0],
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )
