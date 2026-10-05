"""Script huấn luyện EfficientNetV2-S (Task #81 - 50 Epochs - Stratified 5-Fold Cross Validation chuẩn Y khoa).

Đặc điểm Task #81:
- Kiến trúc: EfficientNetV2-S (Fused-MBConv blocks) từ timm / torchvision.
- Chiến lược huấn luyện: Progressive Resizing Strategy (Tăng dần độ phân giải 256 -> 320 -> 384).
- Tối ưu Kaggle & Local: Tự động lưu trữ vào /kaggle/working và tự động ngắt GPU khi hoàn tất.
- Xuất đầy đủ 100% tất cả các dạng kết quả sau khi train xong.
- So sánh đối đầu trực tiếp: EfficientNetV2-S vs EfficientNet-B5 vs EfficientNet-B4 vs ResNet-50.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# Thiết lập đường dẫn root an toàn
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.dataset.dataloader_factory import get_dataloaders
    from src.evaluation.clinical_result_exporter import export_all_clinical_results
    from src.evaluation.cross_validation_reporter import CrossValidationReporter
    from src.models.efficientnet_v2_s import build_efficientnet_v2_s_baseline
    from src.training.checkpoint_manager import (
        ComprehensiveCheckpointManager,
        TrainingLogger,
    )
    from src.utils.reproducibility import set_seed
except ImportError:
    from checkpoint_manager import ComprehensiveCheckpointManager, TrainingLogger
    from clinical_result_exporter import export_all_clinical_results
    from cross_validation_reporter import CrossValidationReporter
    from dataloader_factory import get_dataloaders
    from efficientnet_v2_s import build_efficientnet_v2_s_baseline
    from reproducibility import set_seed


class MultiClassFocalLoss(nn.Module):
    """Hàm mất mát Focal Loss đa lớp tinh chỉnh gamma=1.5 tối ưu cho ảnh nội soi."""

    def __init__(
        self,
        weight: torch.Tensor = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing, reduction="none"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def calculate_smoothed_class_weights(
    train_csv: Path, idx_to_class: dict, power: float = 0.6
) -> torch.Tensor:
    """Tính trọng số nghịch đảo tần suất có làm mịn."""
    df = pd.read_csv(train_csv)
    counts = df["class_name"].value_counts().to_dict()
    num_classes = len(idx_to_class)
    weights = np.zeros(num_classes, dtype=np.float32)

    for idx, class_name in idx_to_class.items():
        count = counts.get(class_name, 1)
        weights[idx] = 1.0 / (count**power)

    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def get_progressive_resolution(epoch: int, total_epochs: int, max_size: int = 384, enabled: bool = True) -> int:
    """Tính toán kích thước ảnh train theo từng giai đoạn Progressive Resizing."""
    if not enabled:
        return max_size

    # Giai đoạn 1 (30% đầu): 256x256 (học nhanh cấu trúc đại thể)
    if epoch <= int(total_epochs * 0.3):
        return 256
    # Giai đoạn 2 (30%-65%): 320x320 (tinh chỉnh đặc trưng niêm mạc)
    elif epoch <= int(total_epochs * 0.65):
        return 320
    # Giai đoạn 3 (35% cuối): 384x384 (độ phân giải tối đa cho vi mạch, polyp)
    else:
        return max_size


def train_single_fold(
    fold_idx: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    raw_images_dir: Path,
    processed_dir: Path,
    checkpoints_base_dir: Path,
    eval_img_size: int = 384,
    use_progressive: bool = True,
    accum_steps: int = 1,
    num_workers: int = 4,
    preload_ram: bool = True,
):
    """Huấn luyện 1 Fold độc lập cho EfficientNetV2-S với Progressive Resizing."""
    print("=" * 80)
    print(
        f"🔥 BẮT ĐẦU HUẤN LUYỆN FOLD {fold_idx} (EFFICIENTNETV2-S - TASK #81 - {epochs} EPOCHS)..."
    )
    print(f"   Chiến lược: Progressive Resizing (256 -> 320 -> {eval_img_size}) | Eval Size: {eval_img_size}x{eval_img_size}")
    print(f"   Batch Size: {batch_size} | Gradient Accumulation: {accum_steps} | Luồng CPU: {num_workers}")
    print("=" * 80)

    set_seed(42 + fold_idx)
    fold_dir = processed_dir / f"fold_{fold_idx}"

    loaders = get_dataloaders(
        processed_dir=str(fold_dir),
        raw_images_dir=str(raw_images_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=(eval_img_size, eval_img_size),
        preload_ram=preload_ram,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    class_weights = calculate_smoothed_class_weights(
        fold_dir / "train.csv", train_loader.dataset.idx_to_class, power=0.6
    ).to(device)

    model = build_efficientnet_v2_s_baseline(
        num_classes=23, pretrained=True, drop_rate=0.3, freeze_backbone=False
    )
    model = model.to(device)

    criterion = MultiClassFocalLoss(
        weight=class_weights, gamma=1.5, label_smoothing=0.05
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=1, eta_min=5e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    chk_dir = checkpoints_base_dir / f"fold_{fold_idx}"
    chk_dir.mkdir(parents=True, exist_ok=True)

    chk_manager = ComprehensiveCheckpointManager(
        checkpoint_dir=str(chk_dir),
        metric_name="val_macro_f1",
        run_config={
            "fold": fold_idx,
            "model": f"EfficientNetV2-S Progressive ({eval_img_size}x{eval_img_size}) {epochs}ep",
            "batch_size": batch_size,
            "accum_steps": accum_steps,
            "progressive_resizing": use_progressive,
        },
    )
    logger = TrainingLogger(log_dir=str(chk_dir))

    history = []
    for ep in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        current_res = get_progressive_resolution(ep, epochs, eval_img_size, use_progressive)

        pbar = tqdm(
            train_loader,
            desc=f"Fold {fold_idx} [{ep:3d}/{epochs}] 🏋️ Train ({current_res}x{current_res})",
            leave=False,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:20}{r_bar}",
        )
        t_batch_start = time.time()
        for batch_i, batch in enumerate(pbar):
            t_data = time.time() - t_batch_start
            t_gpu_start = time.time()

            imgs, targets = (
                batch[0].to(device, non_blocking=True),
                batch[1].to(device, non_blocking=True),
            )

            # Áp dụng Progressive Resizing trên GPU
            if imgs.shape[-1] != current_res:
                imgs = F.interpolate(
                    imgs, size=(current_res, current_res), mode="bilinear", align_corners=False
                )

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outs = model(imgs)
                loss = criterion(outs, targets)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()

            if (batch_i + 1) % accum_steps == 0 or (batch_i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item()
            running_train_loss += loss_val
            t_gpu = time.time() - t_gpu_start

            pbar.set_postfix_str(
                f"Res:{current_res}|D:{t_data:.2f}s|G:{t_gpu:.2f}s|L:{loss_val:.3f}"
            )
            if pbar.n <= 3 or pbar.n % 20 == 0:
                pbar.write(
                    f"⏱️ Batch {pbar.n:3d}/{len(train_loader)} ➔ Res: {current_res}x{current_res} | Data: {t_data:.3f}s | GPU: {t_gpu:.3f}s"
                )
            t_batch_start = time.time()

        scheduler.step()
        train_loss = running_train_loss / len(train_loader)

        # Validation Phase (Luôn đánh giá cố định ở độ phân giải mục tiêu 384x384)
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                imgs, targets = (
                    batch[0].to(device, non_blocking=True),
                    batch[1].to(device, non_blocking=True),
                )
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outs = model(imgs)
                    v_loss = criterion(outs, targets)
                running_val_loss += v_loss.item()
                preds = outs.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds) * 100.0
        val_f1 = (
            f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
        )
        val_prec = (
            precision_score(all_targets, all_preds, average="macro", zero_division=0)
            * 100.0
        )
        val_rec = (
            recall_score(all_targets, all_preds, average="macro", zero_division=0)
            * 100.0
        )

        ep_duration = time.time() - start_t
        curr_lr = optimizer.param_groups[0]["lr"]

        chk_metrics = {
            "val_macro_f1": val_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
        }
        rec_log = {
            "epoch": ep,
            "resolution": current_res,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 2),
            "val_macro_f1": round(float(val_f1), 2),
            "learning_rate": curr_lr,
            "time_sec": round(float(ep_duration), 1),
        }
        logger.log_epoch(rec_log)
        is_best = chk_manager.step(ep, model, optimizer, chk_metrics, scaler=scaler)

        best_mark = "🌟 [BEST]" if is_best else "      "
        print(
            f"Epoch [{ep:3d}/{epochs}] {best_mark} (Res: {current_res}) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:6.2f}% | Val Macro F1: {val_f1:6.2f}% | "
            f"LR: {curr_lr:.2e} | Time: {ep_duration:4.1f}s"
        )

        history.append(
            {
                "epoch": ep,
                "train_resolution": current_res,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "lr": curr_lr,
                "duration": ep_duration,
            }
        )

    # ---------------------------------------------------------------------------------
    # ĐÁNH GIÁ LẠI CHECKPOINT TỐT NHẤT VÀ XUẤT TOÀN BỘ 100% CÁC DẠNG KẾT QUẢ
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"📥 Đang nạp Checkpoint tốt nhất của Fold {fold_idx} để thực hiện đánh giá toàn diện...")
    print("=" * 80)
    best_ckpt = chk_manager.load_best(model, device)
    model.eval()

    all_preds, all_targets, all_probs, all_filenames = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs, targets, filenames = (
                batch[0].to(device, non_blocking=True),
                batch[1].to(device, non_blocking=True),
                batch[2],
            )
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outs = model(imgs)
                probs = torch.softmax(outs, dim=-1)

            all_preds.extend(outs.argmax(dim=-1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_filenames.extend(filenames)

    class_names = [val_loader.dataset.idx_to_class[i] for i in range(len(val_loader.dataset.idx_to_class))]
    history_df = pd.DataFrame(history)
    history_df.to_csv(chk_dir / "history.csv", index=False)

    res50_base_dir = ROOT_DIR / "models" / "checkpoints" / "resnet50_5folds"

    # Gọi Clinical Result Exporter để xuất toàn bộ 100% kết quả
    fold_metrics = export_all_clinical_results(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        filenames=all_filenames,
        class_names=class_names,
        history_df=history_df,
        output_dir=chk_dir,
        model_name=f"EfficientNetV2-S (Progressive -> {eval_img_size}x{eval_img_size})",
        fold_idx=fold_idx,
        best_epoch=best_ckpt.get("epoch", -1),
        res50_comparison_dir=res50_base_dir if res50_base_dir.exists() else None,
    )

    return fold_metrics, history_df


def print_cross_model_benchmark(m: dict, f_idx: int, output_dir: Path):
    """Đối chiếu hiệu năng giữa các thế hệ: ResNet-50 vs EffNet-B4 vs EffNet-B5 vs EffNetV2-S."""
    models_to_check = {
        "ResNet-50 (224x224)": ROOT_DIR / "models" / "checkpoints" / "resnet50_5folds" / f"fold_{f_idx}" / "fold_metrics.json",
        "EfficientNet-B4 (380x380)": ROOT_DIR / "models" / "checkpoints" / "efficientnet_b4_5folds" / f"fold_{f_idx}" / "fold_metrics.json",
        "EfficientNet-B5 (456x456)": ROOT_DIR / "models" / "checkpoints" / "efficientnet_b5_5folds" / f"fold_{f_idx}" / "fold_metrics.json",
    }

    history_data = {}
    for name, path in models_to_check.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                history_data[name] = json.load(f)

    print("\n" + "=" * 95)
    print("🔬 BẢNG ĐỐI CHIẾU TIẾN HÓA KIẾN TRÚC: ResNet-50 -> EffNet V1 (B4, B5) -> EffNetV2-S (Task #81)")
    print("=" * 95)
    header = f"{'Chỉ số lâm sàng':<22} | {'ResNet-50':<12} | {'EffNet-B4':<12} | {'EffNet-B5':<12} | {'EffNetV2-S':<14} | {'Chênh vs B4':<12}"
    print(header)
    print("-" * 95)

    v2_acc = m.get("accuracy", 0.0)
    v2_f1 = m.get("macro_f1", 0.0)

    for k, label in [
        ("accuracy", "Accuracy (%)"),
        ("macro_f1", "Macro F1 (%)"),
        ("weighted_f1", "Weighted F1 (%)"),
        ("macro_precision", "Macro Precision (%)"),
        ("macro_recall", "Macro Recall (%)"),
    ]:
        r50_v = history_data.get("ResNet-50 (224x224)", {}).get(k, 0.0)
        b4_v = history_data.get("EfficientNet-B4 (380x380)", {}).get(k, 0.0)
        b5_v = history_data.get("EfficientNet-B5 (456x456)", {}).get(k, 0.0)
        v2_v = m.get(k, 0.0)

        diff = v2_v - b4_v if b4_v > 0 else 0.0
        diff_str = f"{diff:+.2f}%" if b4_v > 0 else "N/A"
        print(f"{label:<22} | {r50_v:>10.2f}% | {b4_v:>10.2f}% | {b5_v:>10.2f}% | {v2_v:>12.2f}% | {diff_str:>12}")

    print("-" * 95)
    print(f"{'Số tham số (Params)':<22} | {'23.5M':>11} | {'19.3M':>11} | {'30.4M':>11} | {'~21.5M':>13} | {'Fused-MBConv':>12}")
    print(f"{'Tốc độ huấn luyện':<22} | {'Chuẩn':>11} | {'Khá chậm':>11} | {'Rất nặng':>11} | {'Nhanh x2-x3':>13} | {'Progressive':>12}")
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện EfficientNetV2-S (Task #81)")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Số epochs huấn luyện (Mặc định: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Kích thước mini-batch (Mặc định: 24 tối ưu trên GPU T4/P100 16GB hoặc 8GB CMP 40HX)",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Số bước tích lũy gradient (Mặc định: 1)",
    )
    parser.add_argument(
        "--eval_img_size",
        type=int,
        default=384,
        help="Kích thước ảnh vuông kiểm định đích (Mặc định: 384 chuẩn EfficientNetV2-S)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Chọn Fold để chạy (Mặc định: 0 - chỉ chạy Fold 0, hoặc -1 chạy toàn bộ 5 Folds)",
    )
    parser.add_argument(
        "--no_progressive",
        dest="use_progressive",
        action="store_false",
        help="Tắt Progressive Resizing (huấn luyện cố định ở kích thước max)",
    )
    parser.set_defaults(use_progressive=True)

    default_workers = 4
    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_workers,
        help="Số luồng CPU nạp ảnh song song (Mặc định: 4)",
    )
    parser.add_argument(
        "--preload_ram",
        action="store_true",
        default=True,
        help="Nạp toàn bộ ảnh vào RAM để xóa sổ 100% độ trễ đọc đĩa (Mặc định: True)",
    )
    parser.add_argument(
        "--no_preload_ram",
        dest="preload_ram",
        action="store_false",
        help="Tắt nạp ảnh vào RAM nếu máy tính thiếu RAM",
    )
    parser.add_argument(
        "--raw_images_dir",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "labeled-images"),
        help="Đường dẫn tới thư mục labeled-images",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "5folds"),
        help="Đường dẫn tới thư mục 5folds",
    )

    # Tự động chọn output_dir phù hợp trên Kaggle vs Local
    default_out = (
        Path("/kaggle/working/models/checkpoints/efficientnet_v2_s_5folds")
        if Path("/kaggle/working").exists()
        else (ROOT_DIR / "models" / "checkpoints" / "efficientnet_v2_s_5folds")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_out),
        help="Đường dẫn lưu kết quả checkpoints",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(
        f"🖥️ Thiết bị: {device} ➔ {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    raw_images_dir = Path(args.raw_images_dir)
    # Tự động nhận diện thư mục dữ liệu trên Kaggle
    if Path("/kaggle/input").exists():
        lower_gi_dirs = list(Path("/kaggle/input").rglob("lower-gi-tract"))
        if lower_gi_dirs:
            raw_images_dir = lower_gi_dirs[0].parent
            print(f"🎉 Kaggle detected! Đã tự động kết nối ảnh tại: {raw_images_dir}")
        else:
            candidates = list(Path("/kaggle/input").rglob("*labeled*image*"))
            if candidates:
                raw_images_dir = candidates[0]
                print(f"🎉 Kaggle detected! Đã tự động kết nối ảnh tại: {raw_images_dir}")

    # Xác thực đường dẫn ảnh để tránh spam warning OpenCV
    if not (raw_images_dir / "lower-gi-tract").exists() and not (raw_images_dir / "upper-gi-tract").exists():
        print("\n" + "=" * 85)
        print("❌ LỖI: Không tìm thấy thư mục chứa ảnh HyperKvasir!")
        print(f"   Đường dẫn hiện tại: {raw_images_dir}")
        print("   👉 NGUYÊN NHÂN TRÊN KAGGLE: Bạn chưa gắn (Add Input) Dataset HyperKvasir vào Notebook!")
        print("   👉 CÁCH KHẮC PHỤC TRÊN KAGGLE:")
        print("      1. Ở cột bên phải màn hình Kaggle Notebook, bấm nút '+ Add Input' (hoặc 'Add Data').")
        print("      2. Tìm kiếm 'hyperkvasir' và bấm 'Add' để gắn dataset vào Notebook.")
        print("      3. Sau khi thấy dataset xuất hiện ở mục Input, chạy lại lệnh train.")
        print("=" * 85 + "\n")
        sys.exit(1)

    processed_dir = Path(args.processed_dir)
    # Nếu trên Kaggle chưa có thư mục 5folds, kiểm tra trong input hoặc tạo mới trong working
    if not (processed_dir / "fold_0" / "train.csv").exists() and Path("/kaggle/input").exists():
        cand_5f = list(Path("/kaggle/input").rglob("5folds"))
        if cand_5f:
            processed_dir = cand_5f[0]
            print(f"🎉 Kaggle detected! Đã tự động kết nối 5folds tại: {processed_dir}")

    output_dir = Path(args.output_dir)

    # Tự động tạo dữ liệu 5 Folds nếu chưa có
    if not (processed_dir / "fold_0" / "train.csv").exists():
        print(
            "⚠️ Chưa tìm thấy thư mục 5folds, đang tự động khởi tạo Stratified 5-Fold..."
        )
        try:
            from src.dataset.create_5fold_splits import generate_stratified_5folds
        except ImportError:
            from create_5fold_splits import generate_stratified_5folds
        generate_stratified_5folds(seed=42, n_splits=5)

    all_fold_metrics = []
    all_fold_histories = []

    folds_to_run = range(5) if args.fold == -1 else [args.fold]

    for f_idx in folds_to_run:
        f_metric, f_hist = train_single_fold(
            fold_idx=f_idx,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            raw_images_dir=raw_images_dir,
            processed_dir=processed_dir,
            checkpoints_base_dir=output_dir,
            eval_img_size=args.eval_img_size,
            use_progressive=args.use_progressive,
            accum_steps=args.accum_steps,
            num_workers=args.num_workers,
            preload_ram=args.preload_ram,
        )
        all_fold_metrics.append(f_metric)
        all_fold_histories.append(f_hist)

    # Báo cáo kết quả khi chỉ chạy 1 Fold duy nhất
    if len(all_fold_metrics) == 1:
        f_idx = folds_to_run[0]
        m = all_fold_metrics[0]
        print("\n" + "=" * 80)
        print(f"🏆 ĐÃ HOÀN THÀNH HUẤN LUYỆN FOLD {f_idx} (EFFICIENTNETV2-S - TASK #81)!")
        print(f"📊 Accuracy = {m.get('accuracy', 0):.2f}% | Macro F1 = {m.get('macro_f1', 0):.2f}% | Macro Recall = {m.get('macro_recall', 0):.2f}%")
        print(f"💾 Checkpoint và toàn bộ 10 bộ kết quả đã lưu tại: {output_dir / f'fold_{f_idx}'}")
        print("=" * 80)

        # In bảng so sánh đối đầu toàn diện các mô hình
        print_cross_model_benchmark(m, f_idx, output_dir)

    # Nếu chạy đủ 5 Folds, xuất báo cáo khoa học Mean ± Std và Box Plot
    if len(all_fold_metrics) == 5:
        print("\n" + "=" * 80)
        print("🏆 ĐÃ HOÀN THÀNH TOÀN BỘ 5-FOLD CROSS VALIDATION (EFFICIENTNETV2-S)!")
        print("📊 Đang tổng hợp thống kê Mean ± Std và xuất bản Box Plot...")
        print("=" * 80)

        reporter = CrossValidationReporter(
            model_name="EfficientNetV2-S (Progressive)",
            cv_result_dir=str(output_dir),
        )
        summary_df = reporter.aggregate_metrics(all_fold_metrics)
        try:
            print(
                summary_df[
                    ["Chỉ số lâm sàng", "Mean ± Std", "Khoảng dao động [Min, Max]"]
                ].to_markdown(index=False)
            )
        except Exception:
            print(
                summary_df[
                    ["Chỉ số lâm sàng", "Mean ± Std", "Khoảng dao động [Min, Max]"]
                ].to_string(index=False)
            )

        reporter.plot_boxplots(all_fold_metrics, "64_efficientnet_v2_s_5fold_boxplots.png")
        reporter.plot_5fold_learning_curves(
            all_fold_histories, "65_efficientnet_v2_s_5fold_learning_curves.png"
        )
        summary_df.to_csv(
            output_dir / "efficientnet_v2_s_5fold_summary_report.csv", index=False
        )
        print(f"\n💾 Đã lưu bảng báo cáo tổng kết 5-Folds tại: {output_dir}")

    # Thông báo hoàn tất và giải phóng bộ nhớ an toàn cho Kaggle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n🏁 Quá trình huấn luyện đã kết thúc trọn vẹn, GPU đã được giải phóng!")


if __name__ == "__main__":
    main()
