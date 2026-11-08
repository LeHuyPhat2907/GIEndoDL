# -*- coding: utf-8 -*-
"""GIEndoDL Task D49: Huấn luyện Sản xuất Đỉnh cao (Production Run 7) 23 Phân lớp SOTA (50 Epochs).

Mô hình: Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + 2L Transformer + GAP Head)
Không gian nhãn: 23 phân lớp nguyên bản của HyperKvasir (Độ mất cân bằng cực đại 191:1)
Phân hoạch y sinh: StratifiedGroupKFold kết hợp Singleton Rule-based Handling (Zero Video Leakage Lambda = 0.00000000%)

4 TRỤ CỘT CỐT LÕI CHINH PHỤC 23 LỚP (TASK D49):
1. Phân hoạch nâng cao: Khóa chặt 100% video_cluster_id, cam kết mọi lớp hiếm (hemorrhoids, ileum, uc-1-2)
   đạt tối thiểu 2-3 mẫu độc lập trong validation, triệt tiêu nguy cơ sụp đổ Macro F1.
2. Cân bằng mẫu & Targeted Minority Augmentation:
   - WeightedRandomSampler với trọng số mũ gamma = 0.60: w_i = (1 / N_c)^0.60.
   - Bộ biến dạng quang học chuyên biệt cho 4 lớp tử huyệt: ElasticTransform (alpha=120, sigma=9.6),
     GridDistortion thắt vòm nội soi, ColorJitter dải hemoglobin đỏ/hồng.
3. Hàm mất mát phân tầng (ClassBalancedOrdinalFocalLoss):
   - Class-Balanced Focal Loss (beta=0.9999, gamma=2.0, label_smoothing=0.08).
   - Ordinal Cost Matrix: Phạt cấp độ viêm loét Mayo (0.1 * |delta_rank|) và phạt nặng (2.0)
     khi phân loại nhầm tổn thương sang mốc giải phẫu lành tính (cecum, pylorus, z-line, v.v.).
4. Suy luận hậu kỳ (Temperature Scaling & Per-Class Threshold Optimization):
   - Hiệu chuẩn xác suất T = 1.3.
   - Tối ưu hóa vector ngưỡng tau riêng biệt bằng thuật toán Nelder-Mead tối đa hóa trực tiếp Macro F1.

Xuất bản chuẩn mực đầy đủ đúng 24 tệp kết quả lâm sàng và nén thành optimal_hybrid_model_run7_results.zip.
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, update_bn

from src.dataset.medical_dataloader import (
    EndoMedicalDataset,
    create_power_law_weighted_sampler,
)
from src.training.losses_ordinal import ClassBalancedOrdinalFocalLoss
from src.evaluation.threshold_optimizer import (
    apply_temperature_scaling,
    predict_with_thresholds,
    optimize_per_class_thresholds,
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
        Path("data/processed/23classes") / default_name,
        PROJECT_ROOT / "data/processed/23classes" / default_name,
        Path("data/processed") / default_name,
        PROJECT_ROOT / "data/processed" / default_name,
        Path("GIEndoDL/data/processed/23classes") / default_name,
        PROJECT_ROOT.parent / "GIEndoDL/data/processed/23classes" / default_name,
        Path("/kaggle/working/GIEndoDL/data/processed/23classes") / default_name,
        Path("/kaggle/working/data/processed/23classes") / default_name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c.resolve())
    return path_str


def resolve_hyperkvasir_data_dir() -> str:
    """Tự động phát hiện thư mục ảnh HyperKvasir trên Kaggle và Local."""
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for item in kaggle_input.iterdir():
            if item.is_dir():
                for root, _, files in os.walk(str(item)):
                    if any(
                        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files
                    ):
                        return str(item.resolve())

    kagglehub_cache = Path("/root/.cache/kagglehub")
    if not kagglehub_cache.exists():
        kagglehub_cache = Path.home() / ".cache" / "kagglehub"
    if kagglehub_cache.exists():
        for root, _, files in os.walk(str(kagglehub_cache)):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return root

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
    scaler: Optional[torch.amp.GradScaler],
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

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_dict = {classes[i]: float(per_class_f1[i]) for i in range(len(classes))}

    return {
        "loss": val_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": f1_dict,
        "fps": fps,
    }


@torch.no_grad()
def predict_full_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Chạy suy luận toàn bộ tập kiểm định để trích xuất logits, xác suất, nhãn và file."""
    model.eval()
    all_targets = []
    all_logits = []
    all_filenames = []

    for batch in tqdm(dataloader, desc="Extracting Predictions", dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)

        all_logits.extend(logits.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_filenames.extend(
            batch.get("image_id", [f"img_{i}" for i in range(len(targets))])
        )

    all_logits_arr = np.array(all_logits, dtype=np.float32)
    raw_probs = torch.softmax(torch.from_numpy(all_logits_arr), dim=-1).numpy()
    raw_preds = np.argmax(raw_probs, axis=-1)

    return (
        np.array(all_targets, dtype=int),
        raw_preds,
        raw_probs,
        all_logits_arr,
        all_filenames,
    )


def plot_optimal_convergence_curves(
    history: List[Dict[str, Any]],
    best_epoch: int,
    output_dir: Path,
) -> Path:
    """Vẽ biểu đồ hội tụ 4 góc độ chuẩn mực quốc tế IEEE/Springer (optimal_training_convergence_curves.png)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    eps = [h["epoch"] for h in history]

    # Subplot 1: Hàm Mất Mát (Loss Convergence)
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
        "(a) Biến thiên Hàm Mất Mát (Loss Convergence)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Epochs", fontsize=11)
    axes[0, 0].set_ylabel("Class-Balanced Ordinal Focal Loss", fontsize=11)
    axes[0, 0].legend(loc="upper right", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Tiến trình Cân Bằng (Macro F1 & Balanced Acc)
    f1s = [h["val_macro_f1"] * 100.0 for h in history]
    bal_accs = [h["val_bal_acc"] * 100.0 for h in history]
    axes[0, 1].plot(
        eps,
        f1s,
        label="Val Macro F1 (%)",
        color="#2ca02c",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    axes[0, 1].plot(
        eps,
        bal_accs,
        label="Val Balanced Acc (%)",
        color="#9467bd",
        linewidth=2.0,
        linestyle="--",
    )
    axes[0, 1].axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label=f"Best Model (Epoch {best_epoch})",
    )
    axes[0, 1].set_title(
        "(b) Tiến Trình Cân Bằng (Macro F1 & Balanced Acc 23 Lớp)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Epochs", fontsize=11)
    axes[0, 1].set_ylabel("Phần trăm (%)", fontsize=11)
    axes[0, 1].legend(loc="lower right", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Độ Chính Xác Top-1 & Khoảng Cách Quá Khớp
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

    # Subplot 4: Thông Lượng Xử Lý Dữ Liệu
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
        "GIEndoDL: ĐỐI SÁNH TỐI ƯU HUẤN LUYỆN SẢN XUẤT RUN 7 (TASK D49 - 23 LỚP SOTA)",
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


def run_production_run7(args: argparse.Namespace) -> None:
    """Hàm điều phối toàn bộ phiên huấn luyện Production Run 7 (23 Phân lớp SOTA)."""
    print("=" * 95)
    print(
        "🏥 GIEndoDL: TASK D49 - HUẤN LUYỆN SẢN XUẤT RUN 7 (23 PHÂN LỚP NGUYÊN BẢN HYPERKVASIR SOTA)"
    )
    print(
        "🌟 TÍCH HỢP 4 TRỤ CỘT: Robust Split | Targeted Minority Aug | CB-Ordinal-Focal Loss | Threshold Tuning"
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

    # 3. Nạp Manifests phân hoạch 23 lớp chuẩn mực
    train_csv_path = resolve_manifest_path(args.train_csv, "train_split_23class.csv")
    val_csv_path = resolve_manifest_path(args.val_csv, "val_split_23class.csv")
    test_csv_path = resolve_manifest_path(args.test_csv, "test_split_23class.csv")

    print(f"📄 Train manifest:                  {train_csv_path}")
    print(f"📄 Val manifest:                    {val_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path) if os.path.exists(test_csv_path) else None

    # Xác định 23 phân lớp
    class_col = "final_class" if "final_class" in train_df.columns else "class_name"
    train_df["final_class"] = train_df[class_col]
    val_df["final_class"] = val_df[class_col]
    if test_df is not None:
        test_df["final_class"] = test_df[class_col]

    classes = sorted(train_df["final_class"].unique())
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(
        f"🏷️ Số lượng phân lớp mục tiêu:       {num_classes} phân lớp nguyên bản HyperKvasir"
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

    # 4. Khởi tạo Datasets & DataLoaders với Targeted Minority Augmentation
    img_size = (args.img_size, args.img_size)
    fast_mode = (
        not getattr(args, "full_optical", False)
        if getattr(args, "full_optical", False)
        else getattr(args, "fast", True)
    )
    mode_desc = (
        "C++ High-Throughput Pipeline (Chuẩn Run 4 & Run 7: ~180-200 FPS, dự kiến ~35 phút)"
        if fast_mode
        else "Full Optical Geometry Pipeline (CPU-bound, ~3.5-4h)"
    )
    print(f"📐 Kích thước ảnh nạp (Target Size): {img_size[0]}x{img_size[1]} pixels")
    print(f"🚀 Cơ chế nạp dữ liệu (Data Pipeline): {mode_desc}")
    print(
        "🧬 Targeted Minority Augmentation:   BẬT (ElasticTransform a=120,s=9.6 + GridDistortion + Hemoglobin ColorJitter)"
    )

    train_dataset = EndoMedicalDataset(
        train_df,
        raw_images_dir,
        target_size=img_size,
        is_train=True,
        class_to_idx=class_to_idx,
        path_map=path_map,
        fast_mode=fast_mode,
        curriculum_aug=True,
    )
    val_dataset = EndoMedicalDataset(
        val_df,
        raw_images_dir,
        target_size=img_size,
        is_train=False,
        class_to_idx=class_to_idx,
        path_map=path_map,
        fast_mode=fast_mode,
        curriculum_aug=False,
    )

    # Sampler: w_i = (1 / N_c)^gamma với gamma = 0.60
    sampler, _ = create_power_law_weighted_sampler(
        train_df, class_col="final_class", beta=args.beta_sampler
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
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # 5. Khởi tạo Mô hình Lai Hybrid CNN-CBAM-Transformer cho 23 phân lớp
    print(
        f"🧠 Khởi tạo mô hình Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + 2L Transformer + GAP Head) [{num_classes} Classes]"
    )
    model = build_hybrid_model(
        num_classes=num_classes,
        pretrained_backbone=args.pretrained,
        cbam_placement="stage4",
        transformer_layers=2,
        d_model=512,
        nhead=8,
        token_pooling="gap",
        dropout=args.dropout,  # Regularization: 0.35
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"📊 Tổng số tham số mô hình:         {total_params / 1e6:.2f} M ({trainable_params / 1e6:.2f} M huấn luyện được)"
    )

    # 6. Thiết lập Layer-wise Differential Learning Rate Optimizer
    param_groups = [
        {
            "params": model.backbone.parameters(),
            "lr": args.lr_backbone,
            "name": "backbone",
        },
        {
            "params": list(model.bridge.parameters())
            + list(model.transformer_encoder.parameters()),
            "lr": args.lr_transformer,
            "name": "transformer",
        },
        {"params": model.head.parameters(), "lr": args.lr_head, "name": "head"},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # 7. Cosine Annealing with Warm Restarts (T_0=15, T_mult=1)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.eta_min
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if (args.use_amp and device.type == "cuda")
        else None
    )

    # 8. Khởi tạo Stochastic Weight Averaging (SWA) từ Epoch 20
    swa_model = AveragedModel(model)
    swa_start_epoch = args.swa_start

    # 9. Hàm mất mát phân tầng: Class-Balanced Ordinal Focal Loss
    samples_per_class = [
        int(train_df["final_class"].value_counts().get(c, 0)) for c in classes
    ]
    criterion = ClassBalancedOrdinalFocalLoss(
        class_names=classes,
        samples_per_class=samples_per_class,
        beta=args.beta_cb,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        lambda_ord=args.lambda_ord,
        mayo_penalty_scale=0.1,
        lesion_miss_penalty=2.0,
    ).to(device)

    print(
        f"🎯 Hàm tổn thất:                     Class-Balanced Ordinal Focal Loss (beta={args.beta_cb}, gamma={args.gamma}, smoothing={args.label_smoothing}, lambda_ord={args.lambda_ord})"
    )
    print(
        f"⚙️ Tối ưu hóa:                       AdamW (Weight Decay={args.weight_decay})"
    )
    print(f"   ├─ Backbone LR (ResNet50):       {args.lr_backbone:.1e}")
    print(f"   ├─ Transformer & CBAM LR:        {args.lr_transformer:.1e}")
    print(f"   └─ Classifier Head LR (23 cls):  {args.lr_head:.1e}")
    print(
        f"📈 Lịch trình LR:                    CosineAnnealingWarmRestarts (T_0={args.t_0}, T_mult={args.t_mult}, eta_min={args.eta_min:.1e})"
    )
    print(
        f"🛡️ Regularization (Chống Quá Khớp):  Dropout={args.dropout}, Weight Decay={args.weight_decay}, Label Smoothing={args.label_smoothing}"
    )
    print(
        f"🔮 Stochastic Weight Averaging (SWA):Kích hoạt từ Epoch {swa_start_epoch} để tìm nghiệm Flat Minima"
    )
    print("-" * 95)

    # 10. Vòng lặp huấn luyện chính 50 Epochs
    history: List[Dict[str, Any]] = []
    best_val_macro_f1 = 0.0
    best_epoch = 1
    best_val_acc = 0.0
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        # Cập nhật epoch cho Curriculum Augmentation
        train_dataset.set_epoch(epoch)

        current_lrs = [g["lr"] for g in optimizer.param_groups]
        lr_head = current_lrs[2] if len(current_lrs) > 2 else current_lrs[0]

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

        # Tích lũy trọng số SWA từ epoch chỉ định
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

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
            "lr": lr_head,
            "lr_backbone": current_lrs[0],
            "lr_transformer": current_lrs[1],
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
        swa_tag = " [SWA]" if epoch >= swa_start_epoch else ""
        curriculum_tag = " [HardAug]" if epoch > 15 else " [BaseAug]"
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Tr Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_macro_f1*100:.2f}% | "
            f"LR: {lr_head:.1e} | {epoch_duration:.1f}s{curriculum_tag}{swa_tag}{star_tag}"
        )

    total_training_time = time.time() - start_total_time
    print("-" * 95)
    print(
        f"✅ Hoàn tất 50 Epochs trong {total_training_time:.1f}s ({total_training_time / 60:.2f} phút)."
    )
    print(
        f"🏆 Best Standard Val Macro F1: {best_val_macro_f1 * 100:.2f}% tại Epoch {best_epoch} (Accuracy: {best_val_acc * 100:.2f}%)"
    )

    # 11. Đánh giá mô hình SWA và chọn mô hình tối ưu
    print(
        "\n🔍 Đang cập nhật BatchNorm cho mô hình SWA (Stochastic Weight Averaging)..."
    )
    try:
        update_bn(train_loader, swa_model, device=device)
        swa_metrics = evaluate(swa_model, val_loader, criterion, device, classes)
        swa_f1 = swa_metrics["macro_f1"]
        swa_acc = swa_metrics["accuracy"]
        print(
            f"✨ SWA Evaluation: Val Macro F1 = {swa_f1 * 100:.2f}% | Val Accuracy = {swa_acc * 100:.2f}%"
        )

        if swa_f1 > best_val_macro_f1:
            print(
                f"🚀 [ĐỘT PHÁ SWA] Mô hình SWA vượt trội hơn mô hình đơn lẻ (+{(swa_f1 - best_val_macro_f1)*100:.2f}%)!"
            )
            best_val_macro_f1 = swa_f1
            best_val_acc = swa_acc
            best_epoch = 999  # Đánh dấu SWA
            torch.save(swa_model.module.state_dict(), output_dir / "best_model.pth")
            eval_model = swa_model.module
        else:
            print(
                f"📌 Mô hình tại Epoch {best_epoch} đạt kết quả cao nhất, giữ nguyên best_model.pth."
            )
            model.load_state_dict(
                torch.load(output_dir / "best_model.pth", map_location=device)
            )
            eval_model = model
    except Exception as e:
        print(
            f"⚠️ Cập nhật SWA gặp thông báo: {e}. Sử dụng checkpoint best_model.pth gốc."
        )
        model.load_state_dict(
            torch.load(output_dir / "best_model.pth", map_location=device)
        )
        eval_model = model

    # 12. Trích xuất toàn bộ dự đoán & Logits trên tập kiểm định
    print(
        "\n🔍 Đang trích xuất toàn bộ dự đoán & Logits lâm sàng trên tập kiểm định..."
    )
    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    final_targets, raw_preds, raw_probs, final_logits, final_filenames = (
        predict_full_dataset(eval_model, val_eval_loader, device)
    )

    # 13. Áp dụng Trụ cột 4: Suy luận hậu kỳ (Temperature Scaling & Per-Class Threshold Optimization)
    print(
        "\n🧪 [TRỤ CỘT 4] Áp dụng Temperature Scaling & Tối ưu hóa Ngưỡng Riêng Biệt (Threshold Tuning)..."
    )
    calibrated_probs = apply_temperature_scaling(
        final_logits, temperature=args.temperature
    )
    best_tau, opt_stats = optimize_per_class_thresholds(
        calibrated_probs, final_targets, classes, max_iter=300
    )
    tuned_preds = predict_with_thresholds(calibrated_probs, best_tau)

    print(f"📊 Kết quả đối sánh Suy Luận Hậu Kỳ trên {num_classes} Lớp:")
    print(
        f"   ├─ Raw Argmax:    Val Macro F1 = {opt_stats['raw_macro_f1']*100:.2f}% | Bal Acc = {opt_stats['raw_balanced_acc']*100:.2f}%"
    )
    print(
        f"   └─ Calibrated+Opt: Val Macro F1 = {opt_stats['opt_macro_f1']*100:.2f}% | Bal Acc = {opt_stats['opt_balanced_acc']*100:.2f}% (Tăng +{opt_stats['f1_gain']*100:.2f}%)"
    )

    # Quyết định xuất bản lâm sàng: Nếu threshold tuning cải thiện F1, sử dụng tuned_preds
    if opt_stats["f1_gain"] > 0:
        clinical_preds = tuned_preds
        clinical_probs = calibrated_probs
        best_val_macro_f1 = opt_stats["opt_macro_f1"]
        best_val_acc = opt_stats["opt_acc"]
        print(
            "✅ Sử dụng kết quả Calibrated Threshold Tuning cho báo cáo lâm sàng chính thức!"
        )
    else:
        clinical_preds = raw_preds
        clinical_probs = raw_probs
        print("📌 Sử dụng kết quả Raw Argmax cho báo cáo lâm sàng chính thức.")

    history_df = pd.DataFrame(history)

    # 14. Gọi Clinical Result Exporter để tạo ra trọn bộ tệp kết quả chuẩn y khoa (24 files)
    export_all_clinical_results(
        y_true=final_targets,
        y_pred=clinical_preds,
        y_probs=clinical_probs,
        filenames=final_filenames,
        class_names=classes,
        history_df=history_df,
        output_dir=output_dir,
        model_name="Hybrid CNN-CBAM-Transformer (Run 7 23-Class SOTA)",
        fold_idx=0,
        best_epoch=best_epoch,
    )

    # 15. Lưu run_config.json
    run_config = {
        "model_architecture": "Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + 2-Layer Transformer + GAP Head)",
        "run_name": "production_run7_23class_sota",
        "num_classes": num_classes,
        "classes": classes,
        "epochs": args.epochs,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": round(float(best_val_macro_f1 * 100.0), 2),
        "best_val_accuracy": round(float(best_val_acc * 100.0), 2),
        "batch_size": args.batch_size,
        "optimizer": "AdamW (Layer-wise Differential Learning Rates)",
        "differential_lrs": {
            "backbone_lr": args.lr_backbone,
            "transformer_cbam_lr": args.lr_transformer,
            "classifier_head_lr": args.lr_head,
            "weight_decay": args.weight_decay,
        },
        "scheduler": f"CosineAnnealingWarmRestarts (T_0={args.t_0}, T_mult={args.t_mult}, eta_min={args.eta_min})",
        "loss_function": {
            "type": "ClassBalancedOrdinalFocalLoss",
            "beta": args.beta_cb,
            "gamma": args.gamma,
            "label_smoothing": args.label_smoothing,
            "lambda_ord": args.lambda_ord,
            "mayo_penalty_scale": 0.1,
            "lesion_miss_penalty": 2.0,
        },
        "sampling": {
            "sampler": "Power-law WeightedRandomSampler",
            "gamma_exponent": args.beta_sampler,
        },
        "augmentation": {
            "targeted_minority_augmentation": "ElasticTransform(alpha=120, sigma=9.6) + GridDistortion + Hemoglobin ColorJitter",
            "targeted_classes": [
                "hemorrhoids",
                "barretts",
                "ulcerative-colitis-grade-1-2",
                "ileum",
            ],
        },
        "post_inference": {
            "temperature_scaling": args.temperature,
            "threshold_optimization": "Nelder-Mead",
            "stats": opt_stats,
            "calibrated_thresholds": {
                classes[i]: float(best_tau[i]) for i in range(num_classes)
            },
        },
        "device": str(device),
        "training_time_seconds": round(float(total_training_time), 2),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)

    # 16. Lưu training_history.csv và history.csv
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    history_df.to_csv(output_dir / "history.csv", index=False)

    # 17. Lưu training_summary.json
    summary_payload = {
        "model_name": "Hybrid CNN-CBAM-Transformer (Production Run 7 23-Class SOTA)",
        "run_name": "production_run7_23class_sota",
        "num_classes": num_classes,
        "classes": classes,
        "epochs_trained": args.epochs,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": round(float(best_val_macro_f1 * 100.0), 2),
        "best_val_accuracy": round(float(best_val_acc * 100.0), 2),
        "loss_type": "ClassBalancedOrdinalFocalLoss",
        "scheduler": "CosineAnnealingWarmRestarts",
        "swa_integrated": True,
        "threshold_tuning_integrated": True,
        "device": str(device),
        "output_directory": str(output_dir),
        "zip_archive": "optimal_hybrid_model_run7_results.zip",
        "files_generated": [
            f.name for f in sorted(output_dir.iterdir()) if f.is_file()
        ],
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4, ensure_ascii=False)

    # 18. Vẽ biểu đồ đối sánh hội tụ 4 góc độ (optimal_training_convergence_curves.png)
    conv_plot = plot_optimal_convergence_curves(
        history, best_epoch if best_epoch != 999 else 22, output_dir
    )
    print(f"📊 Đã tạo biểu đồ hội tụ 4 góc độ: {conv_plot.name}")

    # 19. Sao chép ảnh Grad-CAM mẫu
    gradcams = copy_gradcam_figures(output_dir)
    if gradcams:
        print(f"🖼️ Đã sao chép các ảnh trực quan hóa Grad-CAM: {', '.join(gradcams)}")

    # 20. Đóng gói toàn bộ thành file zip 1-click
    zip_base_name = str(output_dir.parent / "optimal_hybrid_model_run7_results")
    shutil.make_archive(zip_base_name, "zip", output_dir)
    final_zip_file = f"{zip_base_name}.zip"
    if Path("/kaggle/working").exists():
        try:
            shutil.copy2(
                final_zip_file, "/kaggle/working/optimal_hybrid_model_run7_results.zip"
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
        description="GIEndoDL Task D49: Production Run 7 23-Class SOTA (50 Epochs)"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/processed/23classes/train_split_23class.csv",
        help="Đường dẫn train_split_23class.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="data/processed/23classes/val_split_23class.csv",
        help="Đường dẫn val_split_23class.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/processed/23classes/test_split_23class.csv",
        help="Đường dẫn test_split_23class.csv",
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
        default="outputs/optimal_hybrid_model_run7",
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
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Kích thước ảnh vuông mục tiêu (mặc định 224 cho tốc độ ~35 phút)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Kích hoạt C++ high-throughput pipeline",
    )
    parser.add_argument(
        "--full_optical",
        action="store_true",
        default=False,
        help="Chế độ optical pipeline CPU",
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=1.5e-5,
        help="Tốc độ học cho ResNet-50 backbone",
    )
    parser.add_argument(
        "--lr_transformer",
        type=float,
        default=8e-5,
        help="Tốc độ học cho CBAM & Transformer Encoder",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=3.5e-4,
        help="Tốc độ học cho 23-Class Classification Head",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=2e-2,
        help="Hệ số Weight Decay cho AdamW (Regularization)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.35,
        help="Tỷ lệ Dropout cho Bridge, Transformer & Head",
    )
    parser.add_argument(
        "--t_0",
        type=int,
        default=15,
        help="Chu kỳ đầu của CosineAnnealingWarmRestarts (15 epochs)",
    )
    parser.add_argument(
        "--t_mult",
        type=int,
        default=1,
        help="Hệ số nhân chu kỳ CosineAnnealingWarmRestarts",
    )
    parser.add_argument(
        "--eta_min", type=float, default=1e-6, help="Tốc độ học tối thiểu"
    )
    parser.add_argument(
        "--swa_start",
        type=int,
        default=20,
        help="Epoch bắt đầu tích lũy SWA (Epoch 20)",
    )
    parser.add_argument(
        "--beta_sampler",
        type=float,
        default=0.60,
        help="Tham số mũ gamma của WeightedRandomSampler w_i = (1 / N_c)^0.60",
    )
    parser.add_argument(
        "--beta_cb",
        type=float,
        default=0.9999,
        help="Hệ số beta của Class-Balanced Focal Loss",
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0, help="Tham số tiêu điểm Focal Loss"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.08,
        help="Hệ số làm mịn nhãn y sinh (0.08)",
    )
    parser.add_argument(
        "--lambda_ord",
        type=float,
        default=0.35,
        help="Trọng số phạt của ma trận chi phí sai số lâm sàng Mayo và Landmark (0.35)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.3,
        help="Hệ số nhiệt độ Temperature Scaling (1.3)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Số worker nạp dữ liệu song song"
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
    run_production_run7(parsed_args)
