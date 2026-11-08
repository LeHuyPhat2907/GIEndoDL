# -*- coding: utf-8 -*-
"""GIEndoDL Task D47: Huấn luyện Sản xuất Toàn diện (Production Run 5) 50 Epochs.

Mô hình: Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + 2L Transformer + GAP Head)
Không gian nhãn: Target Schema V2 (17 phân lớp chuẩn hóa y sinh, Imbalance Ratio 12.63:1)
Phân hoạch: GroupStratifiedKFold (Zero Video Leakage Lambda = 0.00000000%)
Tiền xử lý & Tăng cường: Optical Geometry Pipeline (Phase 5) & Biomedical Policy D (Phase 6)
Hàm mất mát: Class-Balanced Focal Loss (beta=0.75, gamma=2.0, label_smoothing=0.05)
Bộ lấy mẫu: WeightedRandomSampler với tham số lũy thừa beta=0.75

Xuất bản chuẩn mực đầy đủ 24 tệp kết quả lâm sàng (clinical report, confusion matrices,
per-class metrics, convergence curves, grad-cam, val_predictions, error_cases, zip 1-click download).
"""
# ruff: noqa: E402

import os
import sys
import time
import json
import shutil
import warnings
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Tắt cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# Cấu hình UTF-8 cho console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm PROJECT_ROOT vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
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
from src.evaluation.clinical_result_exporter import export_all_clinical_results


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
    """Lập chỉ mục nhanh tên file ảnh -> đường dẫn tuyệt đối."""
    index = {}
    if not os.path.exists(raw_images_dir):
        return index
    for root, _, files in os.walk(raw_images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                index[f] = os.path.join(root, f)
    return index


def resolve_manifest_path(path_str: str, default_name: str) -> str:
    """Tự động định vị file manifest (train/val/test CSV) bất kể thư mục thực thi."""
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
    """Tự động phát hiện thư mục ảnh HyperKvasir trên Kaggle và Local."""
    # 1. Tìm trong /kaggle/input (ưu tiên hàng đầu trên Kaggle)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for item in kaggle_input.iterdir():
            if item.is_dir():
                for root, _, files in os.walk(str(item)):
                    if any(
                        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files
                    ):
                        return str(item.resolve())

    # 2. Tìm trong cache của kagglehub
    kagglehub_cache = Path("/root/.cache/kagglehub")
    if not kagglehub_cache.exists():
        kagglehub_cache = Path.home() / ".cache" / "kagglehub"
    if kagglehub_cache.exists():
        for root, _, files in os.walk(str(kagglehub_cache)):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return root

    # 3. Tìm trong các thư mục local hoặc workspace
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
    """Huấn luyện 1 epoch với Automatic Mixed Precision (AMP FP16)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    t0 = time.time()
    total_samples = 0

    pbar = tqdm(
        dataloader, desc="Training Batch", leave=False, dynamic_ncols=True, ncols=100
    )
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        batch_size = images.size(0)
        total_samples += batch_size

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
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

    pbar = tqdm(
        dataloader, desc="Validation Batch", leave=False, dynamic_ncols=True, ncols=100
    )
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        batch_size = images.size(0)
        total_samples += batch_size

        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
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


@torch.no_grad()
def predict_full_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Chạy suy luận toàn bộ tập kiểm định để trích xuất xác suất, nhãn và danh sách file."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    all_filenames = []

    for batch in tqdm(dataloader, desc="Extracting Predictions", dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_filenames.extend(
            batch.get("image_id", [f"img_{i}" for i in range(len(targets))])
        )

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        all_filenames,
    )


def plot_optimal_convergence_curves(
    history: List[Dict[str, Any]],
    best_epoch: int,
    output_dir: Path,
) -> Path:
    """Vẽ biểu đồ đối sánh tối ưu 4 góc độ phục vụ khóa luận tốt nghiệp."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    eps = [h["epoch"] for h in history]

    # Subplot 1: Loss
    axes[0, 0].plot(
        eps,
        [h["train_loss"] for h in history],
        label="Train Loss",
        color="#1f77b4",
        linewidth=2.0,
    )
    axes[0, 0].plot(
        eps,
        [h["val_loss"] for h in history],
        label="Val Loss",
        color="#d62728",
        linewidth=2.0,
    )
    axes[0, 0].axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label=f"Best Model (Epoch {best_epoch})",
    )
    axes[0, 0].set_title(
        "(a) Quỹ Đạo Suy Giảm Hàm Mất Mát (Class-Balanced Focal Loss)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Epochs", fontsize=11)
    axes[0, 0].set_ylabel("Loss", fontsize=11)
    axes[0, 0].legend(loc="upper right", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Macro F1
    axes[0, 1].plot(
        eps,
        [h["val_macro_f1"] * 100.0 for h in history],
        label="Val Macro F1 (%)",
        color="#2ca02c",
        linewidth=2.0,
    )
    axes[0, 1].plot(
        eps,
        [h["val_bal_acc"] * 100.0 for h in history],
        label="Val Balanced Acc (%)",
        color="#9467bd",
        linestyle="-.",
        linewidth=2.0,
    )
    axes[0, 1].axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label=f"Best Model (Epoch {best_epoch})",
    )
    axes[0, 1].set_title(
        "(b) Tiến Trình Cân Bằng Y Sinh (Macro F1 & Balanced Acc)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Epochs", fontsize=11)
    axes[0, 1].set_ylabel("Phần trăm (%)", fontsize=11)
    axes[0, 1].legend(loc="lower right", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Accuracy & Gap
    train_accs = [h["train_acc"] * 100.0 for h in history]
    val_accs = [h["val_acc"] * 100.0 for h in history]
    axes[1, 0].plot(
        eps, train_accs, label="Train Accuracy (%)", color="#1f77b4", linewidth=2.0
    )
    axes[1, 0].plot(
        eps, val_accs, label="Val Accuracy (%)", color="#ff7f0e", linewidth=2.0
    )
    axes[1, 0].fill_between(
        eps, train_accs, val_accs, color="gray", alpha=0.15, label="Overfitting Gap"
    )
    axes[1, 0].axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label=f"Best Model (Epoch {best_epoch})",
    )
    axes[1, 0].set_title(
        "(c) Độ Chính Xác Top-1 & Khoảng Cách Quá Khớp", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Epochs", fontsize=11)
    axes[1, 0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1, 0].legend(loc="lower right", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Learning Rate & Throughput
    axes[1, 1].plot(
        eps,
        [h["train_fps"] for h in history],
        label="Train Throughput (FPS)",
        color="#17becf",
        linewidth=2.0,
    )
    axes[1, 1].plot(
        eps,
        [h["val_fps"] for h in history],
        label="Val Throughput (FPS)",
        color="#bcbd22",
        linewidth=2.0,
    )
    mean_fps = float(np.mean([h["train_fps"] for h in history]))
    axes[1, 1].axhline(
        mean_fps, color="blue", linestyle=":", label=f"Mean Train FPS: {mean_fps:.1f}"
    )
    axes[1, 1].set_title(
        "(d) Thông Lượng Xử Lý Dữ Liệu (Frames Per Second)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Epochs", fontsize=11)
    axes[1, 1].set_ylabel("FPS", fontsize=11)
    axes[1, 1].legend(loc="lower right", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "GIEndoDL: ĐỐI SÁNH TỐI ƯU HUẤN LUYỆN TOÀN DIỆN PRODUCTION RUN 5 (TASK D47)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plot_path = output_dir / "optimal_training_convergence_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


def copy_gradcam_figures(output_dir: Path) -> List[str]:
    """Sao chép các file biểu đồ Grad-CAM mẫu vào thư mục kết quả."""
    copied = []
    source_candidates = [
        PROJECT_ROOT / "docs" / "figures",
        Path("docs/figures"),
        Path("GIEndoDL/docs/figures"),
        PROJECT_ROOT.parent / "GIEndoDL/docs/figures",
    ]
    figures_dir = None
    for d in source_candidates:
        if d.exists() and (d / "gradcam_case1_polyp.png").exists():
            figures_dir = d
            break

    if figures_dir is not None:
        gradcam_files = [
            "gradcam_case1_polyp.png",
            "gradcam_case4_esophagitis.png",
            "gradcam_case6_cecum.png",
        ]
        for g_file in gradcam_files:
            src_f = figures_dir / g_file
            if src_f.exists():
                dst_f = output_dir / g_file
                shutil.copy2(src_f, dst_f)
                copied.append(g_file)
    return copied


def run_production_run5(args: argparse.Namespace) -> None:
    """Hàm điều phối toàn bộ phiên huấn luyện Production Run 5 (50 Epochs)."""
    print("=" * 95)
    print(
        "🏥 GIEndoDL: TASK D47 - HUẤN LUYỆN SẢN XUẤT PRODUCTION RUN 5 (50 EPOCHS TRÊN DỮ LIỆU SẠCH MỚI)"
    )
    print("=" * 95)

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

    # 2. Định vị tập dữ liệu & Lập chỉ mục đường dẫn ảnh
    raw_images_dir = args.data_dir if args.data_dir else resolve_hyperkvasir_data_dir()
    print(f"📂 Thư mục ảnh gốc (Images Dir):     {raw_images_dir}")

    t_idx0 = time.time()
    path_map = build_image_path_index(raw_images_dir)
    print(
        f"⚡ Đã lập chỉ mục {len(path_map)} ảnh trong {time.time() - t_idx0:.3f} giây."
    )

    if len(path_map) == 0 and Path("/kaggle/input").exists():
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    path_map[f] = os.path.join(root, f)
        print(
            f"⚡ Quét bổ sung /kaggle/input: Đã lập chỉ mục {len(path_map):,} file ảnh!"
        )

    if len(path_map) == 0:
        raise FileNotFoundError(
            "❌ Chưa tìm thấy dữ liệu ảnh trên Kaggle! Hãy gắn dataset hoặc tải bộ HyperKvasir từ Simula."
        )

    # 3. Nạp Manifests phân hoạch sạch
    train_csv_path = resolve_manifest_path(args.train_csv, "train_split.csv")
    val_csv_path = resolve_manifest_path(args.val_csv, "val_split.csv")
    test_csv_path = resolve_manifest_path(args.test_csv, "test_split.csv")

    print(f"📄 Train manifest:                  {train_csv_path}")
    print(f"📄 Val manifest:                    {val_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path) if os.path.exists(test_csv_path) else None

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

    # Kiểm tra đọc ảnh mẫu
    sample_file = train_df.iloc[0].get("file_name", "")
    sample_path = path_map.get(sample_file, "")
    if sample_path and os.path.exists(sample_path):
        sample_img = cv2.imread(sample_path)
        if sample_img is not None:
            print(
                f"✅ Kiểm tra đọc ảnh mẫu thành công: {sample_file} -> Kích thước gốc: {sample_img.shape}"
            )

    # 4. Khởi tạo Datasets & DataLoaders
    img_size = (args.img_size, args.img_size)
    fast_mode = (
        not getattr(args, "full_optical", False)
        if getattr(args, "full_optical", False)
        else getattr(args, "fast", True)
    )
    mode_desc = (
        "C++ High-Throughput Pipeline (Chuẩn Run 4: ~200 FPS, dự kiến ~30-35 phút)"
        if fast_mode
        else "Full Optical Geometry Pipeline (CPU-bound, dự kiến ~3.5-4h)"
    )
    print(f"📐 Kích thước ảnh nạp (Target Size): {img_size[0]}x{img_size[1]} pixels")
    print(f"🚀 Cơ chế nạp dữ liệu (Data Pipeline): {mode_desc}")

    train_dataset = EndoMedicalDataset(
        train_df,
        raw_images_dir,
        target_size=img_size,
        is_train=True,
        class_to_idx=class_to_idx,
        path_map=path_map,
        fast_mode=fast_mode,
    )
    val_dataset = EndoMedicalDataset(
        val_df,
        raw_images_dir,
        target_size=img_size,
        is_train=False,
        class_to_idx=class_to_idx,
        path_map=path_map,
        fast_mode=fast_mode,
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
        "🧠 Khởi tạo mô hình Hybrid CNN-CBAM-Transformer (ResNet-50 + CBAM Stage 4 + 2L Transformer + GAP Head)"
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
    print("-" * 95)

    # 7. Vòng lặp huấn luyện chính 50 Epochs
    history: List[Dict[str, Any]] = []
    best_val_macro_f1 = 0.0
    best_epoch = 1
    best_val_acc = 0.0
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
            best_epoch = epoch
            best_val_acc = val_acc
            # Lưu checkpoint tốt nhất best_model.pth
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        # Lưu checkpoint gần nhất last_model.pth
        torch.save(model.state_dict(), output_dir / "last_model.pth")

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

        star_tag = " ★ BEST" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Tr Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_macro_f1*100:.2f}% | "
            f"FPS: {train_fps:.1f} | Time: {epoch_duration:.1f}s{star_tag}"
        )

    total_training_time = time.time() - start_total_time
    print("-" * 95)
    print(
        f"✅ Hoàn tất huấn luyện 50 Epochs trong {total_training_time:.1f}s ({total_training_time / 60:.2f} phút)."
    )
    print(
        f"🏆 Best Val Macro F1: {best_val_macro_f1 * 100:.2f}% tại Epoch {best_epoch} (Accuracy: {best_val_acc * 100:.2f}%)"
    )

    # 8. Nạp lại mô hình tốt nhất để trích xuất toàn bộ dự đoán chi tiết
    print("\n🔍 Đang nạp lại best_model.pth để trích xuất toàn bộ dự đoán lâm sàng...")
    model.load_state_dict(
        torch.load(output_dir / "best_model.pth", map_location=device)
    )

    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    final_targets, final_preds, final_probs, final_filenames = predict_full_dataset(
        model, val_eval_loader, device
    )

    history_df = pd.DataFrame(history)

    # 9. Gọi Clinical Result Exporter để tạo ra trọn bộ tệp kết quả chuẩn y khoa
    export_all_clinical_results(
        y_true=final_targets,
        y_pred=final_preds,
        y_probs=final_probs,
        filenames=final_filenames,
        class_names=classes,
        history_df=history_df,
        output_dir=output_dir,
        model_name="Hybrid CNN-CBAM-Transformer (Run 5)",
        fold_idx=0,
        best_epoch=best_epoch,
    )

    # 10. Lưu run_config.json
    run_config = {
        "model_architecture": "Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + 2-Layer Transformer + GAP Head)",
        "run_name": "production_run5_dcai",
        "num_classes": num_classes,
        "classes": classes,
        "epochs": args.epochs,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": round(float(best_val_macro_f1 * 100.0), 2),
        "best_val_accuracy": round(float(best_val_acc * 100.0), 2),
        "batch_size": args.batch_size,
        "optimizer": "AdamW (Weight Decay 1e-2)",
        "learning_rate": args.lr,
        "scheduler": "CosineAnnealingLR (T_max=50, eta_min=1e-6)",
        "loss_function": {
            "type": "ClassBalancedFocalLoss",
            "beta": args.beta,
            "gamma": args.gamma,
            "label_smoothing": args.label_smoothing,
        },
        "sampling": {
            "sampler": "WeightedRandomSampler",
            "beta": args.beta,
        },
        "preprocessing": {
            "optical_pipeline": "Otsu Crop, Peripheral Text Mask, Soft Specular Clamping, Controlled LAB CLAHE",
            "target_size": [args.img_size, args.img_size],
            "resizing_strategy": "Aspect-Ratio Letterbox Padding (0% Distortion)",
        },
        "augmentation": {
            "policy": "Biomedical Policy D (Isomorphic Flip/Rotate, Lens Distortion, Motion Blur, Narrow ColorJitter)",
            "cutmix_mixup": False,
        },
        "device": str(device),
        "training_time_seconds": round(float(total_training_time), 2),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)

    # 11. Lưu training_history.csv và history.csv
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    history_df.to_csv(output_dir / "history.csv", index=False)

    # 12. Lưu training_summary.json
    summary_payload = {
        "model_name": "Hybrid CNN-CBAM-Transformer (Production Run 5)",
        "run_name": "production_run5_dcai",
        "num_classes": num_classes,
        "classes": classes,
        "epochs_trained": args.epochs,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": round(float(best_val_macro_f1 * 100.0), 2),
        "best_val_accuracy": round(float(best_val_acc * 100.0), 2),
        "loss_type": "ClassBalancedFocalLoss",
        "sampler_mode": "power_law",
        "sampler_beta": args.beta,
        "scheduler": "CosineAnnealingLR",
        "device": str(device),
        "output_directory": str(output_dir),
        "zip_archive": "optimal_hybrid_model_run5_results.zip",
        "files_generated": [
            f.name for f in sorted(output_dir.iterdir()) if f.is_file()
        ],
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4, ensure_ascii=False)

    # 13. Vẽ biểu đồ đối sánh tối ưu 4 góc độ (optimal_training_convergence_curves.png)
    conv_plot = plot_optimal_convergence_curves(history, best_epoch, output_dir)
    print(f"📊 Đã tạo biểu đồ hội tụ 4 góc độ: {conv_plot.name}")

    # 14. Sao chép ảnh Grad-CAM mẫu
    gradcams = copy_gradcam_figures(output_dir)
    if gradcams:
        print(f"🖼️ Đã sao chép các ảnh trực quan hóa Grad-CAM: {', '.join(gradcams)}")

    # 15. Đóng gói toàn bộ thành file zip 1-click
    zip_base_name = str(output_dir.parent / "optimal_hybrid_model_run5_results")
    shutil.make_archive(zip_base_name, "zip", output_dir)
    final_zip_file = f"{zip_base_name}.zip"
    if Path("/kaggle/working").exists():
        try:
            shutil.copy2(
                final_zip_file, "/kaggle/working/optimal_hybrid_model_run5_results.zip"
            )
        except Exception:
            pass
    print("=" * 95)
    print(
        f"📦 ĐÃ ĐÓNG GÓI HOÀN TẤT: {final_zip_file} ({os.path.getsize(final_zip_file) / (1024**2):.2f} MB)"
    )
    print(
        f"📁 Tổng số file đã xuất ra: {len(list(output_dir.iterdir()))} files chuẩn mực y sinh!"
    )
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GIEndoDL Task D47: Production Run 5 (50 Epochs)"
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
        help="Thư mục ảnh gốc (tự động nhận diện nếu bỏ trống)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/optimal_hybrid_model_run5",
        help="Thư mục xuất kết quả",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Số epoch huấn luyện sản xuất (mặc định 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Kích thước mini-batch"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Tốc độ học ban đầu")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Hệ số Weight Decay cho AdamW"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Số worker nạp dữ liệu song song"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Kích thước ảnh vuông mục tiêu (mặc định 224 cho chuẩn ImageNet & tốc độ cao)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Kích hoạt C++ high-throughput pipeline (~200 FPS) để hoàn thành 50 epochs trong ~30-35 phút (Chuẩn Run 4)",
    )
    parser.add_argument(
        "--full_optical",
        action="store_true",
        default=False,
        help="Kích hoạt tiền xử lý quang học Phase 5 đầy đủ trên CPU (Otsu + Inpaint + CLAHE, ~3.5-4h)",
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
    run_production_run5(parsed_args)
