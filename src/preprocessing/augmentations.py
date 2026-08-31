"""Module xây dựng các pipeline Data Augmentation chuẩn y tế bằng thư viện Albumentations."""

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

    def get_train_transforms(self) -> A.Compose:
        """Pipeline tăng cường cho tập Huấn luyện (Train Split)."""
        return A.Compose(
            [
                # 1. Bảo toàn hình học tự do trong không gian ruột
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
                # 2. Random Crop & Resize để tập trung vào các vùng rãnh tổn thương
                A.RandomResizedCrop(
                    size=self.img_size,
                    scale=(0.80, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=cv2.INTER_CUBIC,
                    p=0.5,
                ),
                # Đảm bảo kích thước chuẩn cuối cùng
                A.Resize(
                    height=self.img_size[1],
                    width=self.img_size[0],
                    interpolation=cv2.INTER_CUBIC,
                ),
                # 3. Chuẩn hóa Tensor
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

    def get_val_test_transforms(self) -> A.Compose:
        """Pipeline chuẩn hóa cố định cho tập Validation & Test (Không biến dạng ngẫu nhiên)."""
        return A.Compose(
            [
                A.Resize(
                    height=self.img_size[1],
                    width=self.img_size[0],
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )
