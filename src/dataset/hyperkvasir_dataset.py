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
        preload_ram: bool = False,
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
            preload_ram: Nạp trước toàn bộ ảnh vào RAM ở độ phân giải 256x256 để xóa sạch 100% độ trễ I/O ổ đĩa.
        """
        self.df = pd.read_csv(csv_file)
        self.raw_images_dir = Path(raw_images_dir)
        self.split = split.lower()
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.preload_ram = preload_ram

        # Thiết lập class_to_idx
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_classes = sorted(self.df["class_name"].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Trích xuất sẵn danh sách dạng Python List để tránh gọi DataFrame.iloc gây chậm trong vòng lặp nạp ảnh
        self.relative_paths = self.df["relative_path"].tolist()
        self.class_names = self.df["class_name"].tolist()
        self.filenames = (
            self.df["filename"].tolist()
            if "filename" in self.df.columns
            else [Path(p).name for p in self.relative_paths]
        )
        self.labels = [self.class_to_idx[c] for c in self.class_names]

        # Kiểm tra tính tồn tại của ảnh, tự động dò tìm trong /kaggle/input nếu cần
        if len(self.relative_paths) > 0:
            sample_p = self.raw_images_dir / self.relative_paths[0]
            if not sample_p.exists():
                alt_dir = self._auto_locate_raw_images()
                if alt_dir is not None:
                    print(f"🎉 Kaggle detected! Đã tự động kết nối thư mục ảnh tại: {alt_dir}")
                    self.raw_images_dir = alt_dir

        # Khởi tạo Tensor Shared Memory trong RAM nếu kích hoạt preload_ram
        self.cached_images = None
        if self.preload_ram:
            self._preload_dataset()

        # Thiết lập transform mặc định nếu người dùng không truyền vào
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()

    def _auto_locate_raw_images(self) -> Optional[Path]:
        """Tự động tìm kiếm thư mục ảnh gốc trong /kaggle/input hoặc các thư mục lân cận."""
        if not self.relative_paths:
            return None
        test_rel = self.relative_paths[0]
        sub_folder = test_rel.split("/")[0] if "/" in test_rel else test_rel.split("\\")[0]

        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            # 1. Tìm bất kỳ thư mục nào chứa nhánh 'lower-gi-tract' hoặc 'upper-gi-tract'
            matches = list(kaggle_input.rglob(sub_folder))
            for m in matches:
                cand = m.parent
                if (cand / test_rel).exists():
                    return cand

            # 2. Tìm theo tên thư mục 'labeled-images'
            for cand_name in ["labeled-images", "labeled_images", "labeled images"]:
                for m in kaggle_input.rglob(cand_name):
                    if m.is_dir() and (m / test_rel).exists():
                        return m
        return None

    def _preload_dataset(self):
        """Nạp trước toàn bộ ảnh vào RAM dạng Tensor uint8 (256x256 RGB) bằng đa luồng C++.
        Tận dụng Shared Memory của PyTorch để tất cả DataLoader workers cùng chia sẻ mà không tốn thêm RAM.
        """
        from concurrent.futures import ThreadPoolExecutor
        import os
        from tqdm.auto import tqdm

        n_samples = len(self.relative_paths)
        if self.split == "train":
            # Nạp lớn hơn kích thước crop 10% để RandomCrop hoạt động tốt
            preload_h = max(256, int(self.img_size[1] * 1.1))
            preload_w = max(256, int(self.img_size[0] * 1.1))
        else:
            # Val/Test nạp trực tiếp đúng kích thước mục tiêu
            preload_h = self.img_size[1]
            preload_w = self.img_size[0]

        ram_gb = (n_samples * preload_h * preload_w * 3) / (1024**3)
        print(
            f"⚡ Đang nạp {n_samples} ảnh ({self.split}) vào RAM "
            f"({preload_w}x{preload_h} RGB ~ {ram_gb:.2f} GB) bằng đa luồng C++..."
        )
        self.cached_images = torch.empty((n_samples, preload_h, preload_w, 3), dtype=torch.uint8)

        def _load_single(idx: int):
            img_full_path = self.raw_images_dir / self.relative_paths[idx]
            img_bgr = cv2.imread(str(img_full_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(
                    img_rgb, (preload_w, preload_h), interpolation=cv2.INTER_LINEAR
                )
                self.cached_images[idx] = torch.from_numpy(img_resized)

        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                tqdm(
                    executor.map(_load_single, range(n_samples)),
                    total=n_samples,
                    desc=f"🚀 Preloading {self.split}",
                    leave=False,
                )
            )
        try:
            self.cached_images.share_memory_()
        except Exception:
            pass

    def _get_default_transform(self) -> A.Compose:
        """Tạo pipeline biến đổi theo từng chế độ split."""
        if self.split == "train":
            # Chế độ Huấn luyện: Ảnh đầu vào đã là 256x256, trực tiếp áp dụng Augmentation y tế
            return A.Compose(
                [
                    A.RandomCrop(
                        height=self.img_size[1],
                        width=self.img_size[0],
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.15,
                        hue=0.0,
                        p=0.5,
                    ),
                    A.Normalize(mean=self.mean, std=self.std),
                    ToTensorV2(),
                ]
            )
        else:
            # Chế độ Val/Test: Cố định, chỉ Resize 224x224 và Normalize
            return A.Compose(
                [
                    A.Resize(
                        height=self.img_size[1],
                        width=self.img_size[0],
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    A.Normalize(mean=self.mean, std=self.std),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Lấy một mẫu dữ liệu: Trả về (image_tensor, label_idx, filename)."""
        if self.cached_images is not None:
            # Truy xuất trực tiếp từ RAM (bảo đảm an toàn dữ liệu không bị albumentations sửa đổi in-place)
            img_rgb = self.cached_images[idx].numpy().copy()
        else:
            img_full_path = self.raw_images_dir / self.relative_paths[idx]
            img_bgr = cv2.imread(str(img_full_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Không thể đọc file ảnh tại: {img_full_path}")
            target_h = (
                max(256, int(self.img_size[1] * 1.1))
                if self.split == "train"
                else self.img_size[1]
            )
            target_w = (
                max(256, int(self.img_size[0] * 1.1))
                if self.split == "train"
                else self.img_size[0]
            )
            img_rgb = cv2.resize(
                img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        # Áp dụng Albumentations
        transformed = self.transform(image=img_rgb)
        image_tensor = transformed["image"]

        return image_tensor, self.labels[idx], self.filenames[idx]
