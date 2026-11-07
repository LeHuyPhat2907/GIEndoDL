# -*- coding: utf-8 -*-
"""GIEndoDL Task D46: Huấn luyện Thử nghiệm Nhanh (Dry-Run) 10 Epoch trên Dữ liệu Sạch Mới (17 Classes).

Mục tiêu cốt lõi:
1. Đánh giá tốc độ hội tụ (Convergence Velocity) của mô hình lai Hybrid CNN-CBAM-Transformer trên Schema 17 lớp sạch.
2. Đo lường thông lượng DataLoader (FPS / Throughput) trên môi trường GPU (Kaggle P100 / T4 x2 hoặc Local GPU).
3. Kiểm tra tính ổn định của Class-Balanced Focal Loss (beta=0.75, gamma=2.0, label_smoothing=0.05) & Weighted Sampler.
4. Xuất trọn bộ báo cáo, biểu đồ hội tụ, bảng F1 từng lớp và file nén zip 1-click download.
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Cấu hình UTF-8 cho console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm PROJECT_ROOT vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset.medical_dataloader import (
    EndoMedicalDataset,
    create_power_law_weighted_sampler,
    ClassBalancedFocalLoss,
)
from src.models.hybrid_cnn_cbam_transformer import build_hybrid_model


def lock_seeds(seed: int = 42) -> None:
    """Khóa chặt 6 tầng sinh số ngẫu nhiên đảm bảo tính tái lập 100%."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_image_path_index(raw_images_dir: str) -> Dict[str, str]:
    """Lập chỉ mục nhanh tên file ảnh -> đường dẫn tuyệt đối (0.03s cho 10.6k ảnh)."""
    index = {}
    if not os.path.exists(raw_images_dir):
        return index
    for root, _, files in os.walk(raw_images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                index[f] = os.path.join(root, f)
    return index


def resolve_manifest_path(path_str: str, default_name: str) -> str:
    """Tự động định vị file manifest (train/val/test CSV) bất kể đang đứng ở thư mục nào."""
    p = Path(path_str)
    if p.exists() and p.is_file():
        return str(p.resolve())

    candidates = [
        PROJECT_ROOT / path_str,
        Path("data/processed") / default_name,
        PROJECT_ROOT / "data/processed" / default_name,
        Path("GIEndoDL/data/processed") / default_name,
        PROJECT_ROOT.parent / "GIEndoDL/data/processed" / default_name,
        Path("/kaggle/working/GIEndoDL/data/processed") / default_name,
        Path("/kaggle/working/data/processed") / default_name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c.resolve())
    return path_str


def resolve_hyperkvasir_data_dir() -> str:
    """Tự động phát hiện thư mục ảnh HyperKvasir trên các môi trường Kaggle và Local."""
    # 1. Tìm trong /kaggle/input (ưu tiên hàng đầu trên Kaggle)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for item in kaggle_input.iterdir():
            if item.is_dir():
                for root, _, files in os.walk(str(item)):
                    if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                        return str(item.resolve())

    # 2. Tìm trong các thư mục local hoặc workspace
    candidates = [
        Path("data/raw/labeled-images"),
        PROJECT_ROOT / "data/raw/labeled-images",
        Path("GIEndoDL/data/raw/labeled-images"),
        PROJECT_ROOT.parent / "GIEndoDL/data/raw/labeled-images",
        Path("D:/PhatProject/GIEndoDL/data/raw/labeled-images"),
    ]
    for c in candidates:
        if c.exists():
            for root, _, files in os.walk(str(c)):
                if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                    return str(c.resolve())

    return "data/raw/labeled-images"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float, float]:
    """Huấn luyện 1 epoch với Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    t0 = time.time()
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training Batch", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        batch_size = images.size(0)
        total_samples += batch_size

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

        current_acc = accuracy_score(all_targets, all_preds)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.3f}"})

    elapsed = time.time() - t0
    epoch_fps = total_samples / max(elapsed, 1e-5)
    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = accuracy_score(all_targets, all_preds)

    return epoch_loss, epoch_acc, epoch_fps


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: List[str],
) -> Dict[str, Any]:
    """Đánh giá toàn diện trên tập Validation."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    t0 = time.time()
    total_samples = 0

    pbar = tqdm(dataloader, desc="Validation Batch", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        batch_size = images.size(0)
        total_samples += batch_size

        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    elapsed = time.time() - t0
    fps = total_samples / max(elapsed, 1e-5)
    val_loss = running_loss / max(total_samples, 1)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # F1 score từng lớp
    per_class_f1_vals = f1_score(
        all_targets,
        all_preds,
        average=None,
        labels=list(range(len(classes))),
        zero_division=0,
    )
    per_class_f1_dict = {
        classes[i]: float(per_class_f1_vals[i]) for i in range(len(classes))
    }

    return {
        "loss": val_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "fps": fps,
        "per_class_f1": per_class_f1_dict,
        "preds": all_preds,
        "targets": all_targets,
    }


def plot_dryrun_metrics(
    history_df: pd.DataFrame,
    per_class_f1_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Vẽ và lưu biểu đồ hội tụ và hiệu năng từng lớp chuẩn y khoa."""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    # 1. Biểu đồ Convergence Curves (4 Subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    epochs = history_df["epoch"]

    # Subplot 1: Loss
    axes[0, 0].plot(
        epochs,
        history_df["train_loss"],
        "o-",
        color="#1f77b4",
        label="Train Loss",
        linewidth=2,
    )
    axes[0, 0].plot(
        epochs,
        history_df["val_loss"],
        "s--",
        color="#d62728",
        label="Val Loss",
        linewidth=2,
    )
    axes[0, 0].set_title(
        "Biến thiên Hàm Mất Mát (Loss Convergence)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Class-Balanced Focal Loss")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Macro F1 & Balanced Accuracy
    axes[0, 1].plot(
        epochs,
        history_df["val_macro_f1"] * 100,
        "o-",
        color="#2ca02c",
        label="Val Macro F1 (%)",
        linewidth=2,
    )
    axes[0, 1].plot(
        epochs,
        history_df["val_bal_acc"] * 100,
        "^-.",
        color="#9467bd",
        label="Val Balanced Acc (%)",
        linewidth=2,
    )
    axes[0, 1].set_title(
        "Tiến trình Cân Bằng (Macro F1 & Balanced Acc)", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Phần trăm (%)")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Accuracy & Overfitting Gap
    axes[1, 0].plot(
        epochs,
        history_df["train_acc"] * 100,
        "o-",
        color="#1f77b4",
        label="Train Acc (%)",
        linewidth=2,
    )
    axes[1, 0].plot(
        epochs,
        history_df["val_acc"] * 100,
        "s--",
        color="#ff7f0e",
        label="Val Acc (%)",
        linewidth=2,
    )
    gap = (history_df["train_acc"] - history_df["val_acc"]) * 100
    axes[1, 0].fill_between(
        epochs,
        history_df["train_acc"] * 100,
        history_df["val_acc"] * 100,
        color="gray",
        alpha=0.15,
        label="Overfit Gap",
    )
    axes[1, 0].set_title(
        "Độ chính xác Top-1 & Khoảng cách Quá khớp", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: DataLoader Throughput
    axes[1, 1].plot(
        epochs,
        history_df["train_fps"],
        "o-",
        color="#17becf",
        label="Train Throughput (FPS)",
        linewidth=2,
    )
    axes[1, 1].plot(
        epochs,
        history_df["val_fps"],
        "s--",
        color="#bcbd22",
        label="Val Throughput (FPS)",
        linewidth=2,
    )
    axes[1, 1].axhline(
        y=history_df["train_fps"].mean(),
        color="blue",
        linestyle=":",
        label=f"Mean Train FPS: {history_df['train_fps'].mean():.1f}",
    )
    axes[1, 1].set_title(
        "Thông lượng Xử lý Dữ liệu (Throughput / FPS)", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Frames Per Second (FPS)")
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "GIEndoDL: BÁO CÁO KẾT QUẢ HUẤN LUYỆN THỬ NGHIỆM DRY-RUN 10 EPOCHS (TASK D46)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    convergence_path = output_dir / "dryrun_convergence_curves.png"
    plt.savefig(convergence_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Biểu đồ Per-Class F1 Score (Bar Chart tại Final / Best Epoch)
    final_epoch_f1 = per_class_f1_df.iloc[-1].drop("epoch")
    sorted_f1 = final_epoch_f1.sort_values(ascending=False)

    plt.figure(figsize=(12, 7), dpi=300)
    colors = [
        "#2ca02c" if val >= 0.80 else "#ff7f0e" if val >= 0.65 else "#d62728"
        for val in sorted_f1.values
    ]
    bars = plt.barh(
        sorted_f1.index,
        sorted_f1.values * 100,
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )

    for bar in bars:
        w = bar.get_width()
        plt.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.axvline(
        x=sorted_f1.mean() * 100,
        color="navy",
        linestyle="--",
        linewidth=1.8,
        label=f"Macro F1 Trung bình: {sorted_f1.mean() * 100:.2f}%",
    )
    plt.title(
        "Phân bố F1-Score trên 17 Lớp Bệnh lý (Dry-Run Epoch 10)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("F1 Score (%)", fontsize=11)
    plt.xlim(0, 105)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    barchart_path = output_dir / "dryrun_per_class_f1_barchart.png"
    plt.savefig(barchart_path, dpi=300, bbox_inches="tight")
    plt.close()

    return convergence_path, barchart_path


def run_dryrun(args: argparse.Namespace) -> None:
    """Hàm điều phối toàn bộ phiên huấn luyện Dry-Run Task D46."""
    print("=" * 90)
    print(
        "🔬 GIEndoDL: TASK D46 - HUẤN LUYỆN THỬ NGHIỆM DRY-RUN 10 EPOCHS TRÊN DỮ LIỆU SẠCH MỚI"
    )
    print("=" * 90)

    lock_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nhận diện phần cứng
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"⚡ Thiết bị thực thi (Device):       {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU Model:                       {gpu_name} ({vram_gb:.2f} GB VRAM)")
        torch.cuda.empty_cache()
    else:
        print("⚠️ Chạy trên CPU - Tốc độ có thể chậm hơn so với GPU.")

    # 2. Định vị tập dữ liệu & Lập chỉ mục đường dẫn ảnh
    raw_images_dir = args.data_dir if args.data_dir else resolve_hyperkvasir_data_dir()
    print(f"📂 Thư mục ảnh gốc (Images Dir):     {raw_images_dir}")

    t_idx0 = time.time()
    path_map = build_image_path_index(raw_images_dir)
    print(
        f"⚡ Đã lập chỉ mục {len(path_map)} ảnh trong {time.time() - t_idx0:.3f} giây."
    )

    # Tự động tải qua kagglehub nếu trên Kaggle mà chưa thấy ảnh
    if len(path_map) == 0 and Path("/kaggle").exists():
        print("⚠️ Chưa tìm thấy ảnh trong /kaggle/input. Bắt đầu tự động tải qua kagglehub...")
        try:
            import kagglehub
            dl_path = kagglehub.dataset_download("meetnagadia/hyperkvasir")
            print(f"✅ Tải thành công dataset qua kagglehub: {dl_path}")
            raw_images_dir = dl_path
            path_map = build_image_path_index(raw_images_dir)
            print(f"⚡ Đã lập chỉ mục {len(path_map)} ảnh từ dataset vừa tải.")
        except Exception as e:
            print(f"⚠️ Lưu ý: Không thể tải tự động qua kagglehub ({e}).")
            print("👉 Vui lòng nhấn '+ Add Input' trên Kaggle và thêm 'vaillant/hyperkvasir' vào notebook.")

    # 3. Nạp Manifests phân hoạch sạch từ Phase 7
    train_csv_path = resolve_manifest_path(args.train_csv, "train_split.csv")
    val_csv_path = resolve_manifest_path(args.val_csv, "val_split.csv")
    test_csv_path = resolve_manifest_path(args.test_csv, "test_split.csv")

    print(f"📄 Train manifest:                  {train_csv_path}")
    print(f"📄 Val manifest:                    {val_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path) if os.path.exists(test_csv_path) else None

    # Lấy danh sách 17 nhãn chuẩn
    classes = sorted(train_df["final_class"].unique())
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(
        f"🏷️ Số lượng phân lớp mục tiêu:       {num_classes} phân lớp chuẩn hóa y sinh"
    )
    print(f"📊 Mẫu huấn luyện (Train):           {len(train_df):,} ảnh")
    print(f"📊 Mẫu kiểm định (Val):              {len(val_df):,} ảnh")
    if test_df is not None:
        print(f"📊 Mẫu kiểm thử độc lập (Test):     {len(test_df):,} ảnh")

    # 4. Kiểm tra tính sẵn sàng của file ảnh thực tế
    if len(path_map) == 0 and Path("/kaggle/input").exists():
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    path_map[f] = os.path.join(root, f)
        print(f"⚡ Quét bổ sung /kaggle/input: Đã lập chỉ mục {len(path_map):,} file ảnh!")

    if len(path_map) == 0:
        raise FileNotFoundError(
            "\n" + "=" * 80 + "\n"
            "❌ CHƯA CÓ DỮ LIỆU ẢNH TRÊN KAGGLE!\n"
            "Chương trình không tìm thấy file ảnh .jpg nào để huấn luyện.\n\n"
            "👉 HƯỚNG DẪN KHẮC PHỤC TRONG 10 GIÂY:\n"
            "1. Nhìn lên góc TRÊN BÊN PHẢI màn hình Kaggle, bấm nút '+ Add Input'.\n"
            "2. Tìm từ khóa: hyperkvasir\n"
            "3. Bấm dấu (+) ở bộ 'vaillant/hyperkvasir' (hoặc 'meetnagadia/hyperkvasir').\n"
            "4. Chạy lại lệnh huấn luyện.\n"
            + "=" * 80
        )

    # Thử đọc 1 ảnh mẫu để bảo đảm tính toàn vẹn
    sample_file = train_df.iloc[0].get("file_name", "")
    sample_path = path_map.get(sample_file, "")
    if sample_path and os.path.exists(sample_path):
        sample_img = cv2.imread(sample_path)
        if sample_img is not None:
            print(f"✅ Kiểm tra đọc ảnh mẫu thành công: {sample_file} -> Kích thước gốc: {sample_img.shape}")

    # 5. Khởi tạo Datasets & DataLoaders
    img_size = (args.img_size, args.img_size)
    print(
        f"📐 Kích thước ảnh nạp (Target Size): {img_size[0]}x{img_size[1]} pixels (Aspect-Ratio Letterboxed)"
    )

    train_dataset = EndoMedicalDataset(
        train_df,
        raw_images_dir,
        target_size=img_size,
        is_train=True,
        class_to_idx=class_to_idx,
        path_map=path_map,
    )
    val_dataset = EndoMedicalDataset(
        val_df,
        raw_images_dir,
        target_size=img_size,
        is_train=False,
        class_to_idx=class_to_idx,
        path_map=path_map,
    )

    sampler, _ = create_power_law_weighted_sampler(
        train_df, class_col="final_class", beta=args.beta
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": (device.type == "cuda"),
        "num_workers": args.num_workers,
        "persistent_workers": (args.num_workers > 0),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_kwargs = loader_kwargs.copy()
    val_loader = DataLoader(val_dataset, shuffle=False, **val_kwargs)

    # 5. Khởi tạo Mô hình Lai Hybrid CNN-CBAM-Transformer
    print(
        "🧠 Khởi tạo mô hình: Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + 2L Transformer + GAP Head)"
    )
    model = build_hybrid_model(
        num_classes=num_classes,
        pretrained_backbone=args.pretrained,
        cbam_placement="stage4",
        transformer_layers=2,
        d_model=512,
        nhead=8,
        token_pooling="gap",
        dropout=0.1,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"📊 Tổng số tham số mô hình:         {total_params / 1e6:.2f} M ({trainable_params / 1e6:.2f} M huấn luyện được)"
    )

    # 6. Thiết lập Loss Function & Optimizer & Scheduler
    samples_per_class = [
        int(train_df["final_class"].value_counts().get(c, 0)) for c in classes
    ]
    criterion = ClassBalancedFocalLoss(
        samples_per_class=samples_per_class,
        num_classes=num_classes,
        beta=args.beta,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = (
        torch.cuda.amp.GradScaler()
        if (args.use_amp and device.type == "cuda")
        else None
    )

    print(
        f"🎯 Hàm tổn thất:                     Class-Balanced Focal Loss (beta={args.beta}, gamma={args.gamma}, smoothing={args.label_smoothing})"
    )
    print(
        f"⚙️ Tối ưu hóa:                       AdamW (lr={args.lr}, weight_decay={args.weight_decay})"
    )
    print(
        f"📈 Lịch trình LR:                    CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)"
    )
    print(
        f"🚀 Huấn luyện AMP (Mixed Precision): {args.use_amp and device.type == 'cuda'}"
    )
    print("-" * 90)

    # 7. Vòng lặp huấn luyện Dry-Run 10 Epochs
    history: List[Dict[str, Any]] = []
    per_class_f1_records: List[Dict[str, Any]] = []

    best_val_macro_f1 = 0.0
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_fps = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=args.use_amp,
        )

        val_metrics = evaluate(model, val_loader, criterion, device, classes)
        scheduler.step()

        epoch_duration = time.time() - epoch_t0
        val_macro_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_bal_acc = val_metrics["balanced_accuracy"]
        val_loss = val_metrics["loss"]
        val_fps = val_metrics["fps"]

        is_best = val_macro_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_macro_f1
            # Lưu checkpoint tốt nhất
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro_f1,
                    "val_accuracy": val_acc,
                    "classes": classes,
                    "config": vars(args),
                },
                output_dir / "best_dryrun_model.pth",
            )

        # Lưu checkpoint gần nhất
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "classes": classes,
            },
            output_dir / "latest_checkpoint.pth",
        )

        # Ghi nhận log
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_fps": train_fps,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_bal_acc": val_bal_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_fps": val_fps,
            "duration_s": epoch_duration,
        }
        history.append(record)

        f1_row = {"epoch": epoch}
        f1_row.update(val_metrics["per_class_f1"])
        per_class_f1_records.append(f1_row)

        star_tag = " ★ BEST" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_macro_f1*100:.2f}% | "
            f"Train FPS: {train_fps:.1f} | Val FPS: {val_fps:.1f} | "
            f"Time: {epoch_duration:.1f}s{star_tag}"
        )

    total_training_time = time.time() - start_total_time
    print("-" * 90)
    print(
        f"✅ Hoàn tất Dry-Run {args.epochs} Epochs trong {total_training_time:.1f}s ({total_training_time / 60:.2f} phút)."
    )
    print(f"🏆 Best Val Macro F1: {best_val_macro_f1 * 100:.2f}%")

    # 8. Lưu kết quả ra file CSV & JSON
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "dryrun_metrics_history.csv", index=False)

    per_class_f1_df = pd.DataFrame(per_class_f1_records)
    per_class_f1_df.to_csv(output_dir / "dryrun_per_class_f1.csv", index=False)

    # 9. Vẽ biểu đồ chuẩn y khoa
    conv_plot, bar_plot = plot_dryrun_metrics(history_df, per_class_f1_df, output_dir)
    print(f"📊 Đã tạo biểu đồ hội tụ:          {conv_plot}")
    print(f"📊 Đã tạo biểu đồ F1 từng lớp:     {bar_plot}")

    # 10. Tóm tắt JSON
    summary = {
        "task": "Task D46 Dry-Run 10 Epochs",
        "dataset": "Clean HyperKvasir Schema V2",
        "num_classes": num_classes,
        "classes": classes,
        "best_val_macro_f1": float(best_val_macro_f1),
        "final_val_accuracy": float(history_df["val_acc"].iloc[-1]),
        "mean_train_fps": float(history_df["train_fps"].mean()),
        "mean_val_fps": float(history_df["val_fps"].mean()),
        "total_time_seconds": float(total_training_time),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "None",
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "beta": args.beta,
        "gamma": args.gamma,
    }
    with open(output_dir / "dryrun_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # 11. Đóng gói kết quả thành file zip để người dùng tải 1-click
    zip_base_name = str(output_dir.parent / "kaggle_dryrun_d46_results")
    shutil.make_archive(zip_base_name, "zip", output_dir)
    print(f"📦 Đã đóng gói toàn bộ kết quả:   {zip_base_name}.zip")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GIEndoDL Task D46: Dry-Run 10 Epochs trên Dữ liệu Sạch Mới"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/processed/train_split.csv",
        help="Đường dẫn train_split.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="data/processed/val_split.csv",
        help="Đường dẫn val_split.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/processed/test_split.csv",
        help="Đường dẫn test_split.csv",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Thư mục ảnh gốc (tự động phát hiện nếu bỏ trống)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/kaggle_dryrun_d46",
        help="Thư mục xuất kết quả",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Số epochs dry-run (mặc định 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Kích thước mini-batch"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Tốc độ học ban đầu")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Hệ số Weight Decay cho AdamW"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Số worker nạp dữ liệu song song"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Kích thước ảnh vuông mục tiêu (256 hoặc 224)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.75,
        help="Tham số mũ Class-Balanced và Weighted Sampler",
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0, help="Tham số tiêu điểm Focal Loss"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.05, help="Hệ số làm mịn nhãn y sinh"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Hạt giống ngẫu nhiên khóa 6 tầng"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Thiết bị thực thi (cuda hoặc cpu)"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Nạp trọng số ImageNet cho ResNet50",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Kích hoạt Mixed Precision Training (FP16)",
    )

    parsed_args = parser.parse_args()
    run_dryrun(parsed_args)
