"""Module định nghĩa lớp HyperKvasirDataset kế thừa torch.utils.data.Dataset cho PyTorch."""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class HyperKvasirDataset(Dataset):
    """Custom PyTorch Dataset cho dữ liệu nội soi tiêu hóa HyperKvasir (23 lớp)."""

    def __init__(
        self,
        csv_file: Union[str, Path],
        raw_images_dir: Union[str, Path],
        split: str = "train",
        img_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        mean: Tuple[float, float, float] = (0.5729, 0.3557, 0.2515),
        std: Tuple[float, float, float] = (0.3105, 0.2116, 0.1834),
    ):
        """Khởi tạo dataset.

        Args:
            csv_file: Đường dẫn tới file split CSV (train_split.csv, val_split.csv, test_split.csv).
            raw_images_dir: Thư mục chứa ảnh gốc (data/raw/labeled-images).
            split: Chế độ dữ liệu ('train', 'val', 'test').
            img_size: Kích thước đích (W, H).
            transform: Pipeline biến đổi tùy chỉnh (nếu có).
            class_to_idx: Bảng ánh xạ nhãn tên lớp sang số nguyên (0-22).
            mean: Bộ thông số mean chuẩn hóa RGB.
            std: Bộ thông số std chuẩn hóa RGB.
        """
        self.df = pd.read_csv(csv_file)
        self.raw_images_dir = Path(raw_images_dir)
        self.split = split.lower()
        self.img_size = img_size
        self.mean = mean
        self.std = std

        # Thiết lập class_to_idx
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_classes = sorted(self.df["class_name"].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Thiết lập transform mặc định nếu người dùng không truyền vào
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()

    def _get_default_transform(self) -> A.Compose:
        """Tạo pipeline biến đổi theo từng chế độ split."""
        if self.split == "train":
            # Chế độ huấn luyện: Đầy đủ Augmentation an toàn y tế
            return A.Compose(
                [
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
                    A.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.15,
                        hue=0.04,
                        p=0.5,
                    ),
                    A.Resize(
                        height=self.img_size[1],
                        width=self.img_size[0],
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.Normalize(mean=self.mean, std=self.std),
                    ToTensorV2(),
                ]
            )
        else:
            # Chế độ Val/Test: Cố định, chỉ Resize và Normalize
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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Lấy một mẫu dữ liệu: Trả về (image_tensor, label_idx, filename)."""
        row = self.df.iloc[idx]
        img_rel_path = row["relative_path"]
        img_full_path = self.raw_images_dir / img_rel_path

        # Đọc ảnh bằng OpenCV và chuyển sang RGB
        img_bgr = cv2.imread(str(img_full_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Không thể đọc file ảnh tại: {img_full_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Áp dụng Albumentations
        transformed = self.transform(image=img_rgb)
        image_tensor = transformed["image"]

        # Lấy nhãn số nguyên
        class_name = row["class_name"]
        label_idx = self.class_to_idx[class_name]

        return image_tensor, label_idx, row["filename"]
