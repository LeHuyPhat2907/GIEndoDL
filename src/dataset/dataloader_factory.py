"""Module xây dựng DataLoader Factory chuẩn PyTorch với các tùy biến tối ưu hóa I/O."""

from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader

from src.dataset.hyperkvasir_dataset import HyperKvasirDataset


def get_dataloaders(
    processed_dir: str,
    raw_images_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (224, 224),
    pin_memory: bool = None,
) -> Dict[str, DataLoader]:
    """Khởi tạo bộ 3 DataLoaders (Train, Val, Test) với cấu hình tối ưu.

    Args:
        processed_dir: Thư mục chứa 3 file split CSV (data/processed).
        raw_images_dir: Thư mục chứa ảnh gốc (data/raw/labeled-images).
        batch_size: Kích thước batch cho mỗi bước huấn luyện.
        num_workers: Số luồng CPU đọc ảnh song song.
        img_size: Kích thước ảnh (W, H).
        pin_memory: Khóa bộ nhớ đệm (Tự động True nếu có GPU CUDA).
    """
    proc_path = Path(processed_dir)
    train_csv = proc_path / "train_split.csv"
    val_csv = proc_path / "val_split.csv"
    test_csv = proc_path / "test_split.csv"

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # 1. Khởi tạo 3 Datasets
    train_ds = HyperKvasirDataset(
        train_csv, raw_images_dir, split="train", img_size=img_size
    )
    val_ds = HyperKvasirDataset(val_csv, raw_images_dir, split="val", img_size=img_size)
    test_ds = HyperKvasirDataset(
        test_csv, raw_images_dir, split="test", img_size=img_size
    )

    # 2. Cấu hình DataLoader
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_to_idx": train_ds.class_to_idx,
        "idx_to_class": train_ds.idx_to_class,
    }
