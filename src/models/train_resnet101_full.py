"""Script huấn luyện ResNet-101 (Task #77 - 50 Epochs - Stratified 5-Fold Cross Validation chuẩn Y khoa).

Áp dụng: 50 Epochs + Batch Size 64 + Focal Loss (gamma=1.5) + Dropout 0.45 + Weight Decay 5e-4.
Tương thích hoàn toàn cả trên PC lẫn Kaggle (Hỗ trợ GPU Tesla T4/P100).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
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
from tqdm.auto import tqdm

# Thiết lập đường dẫn root an toàn
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.dataset.dataloader_factory import get_dataloaders
    from src.evaluation.clinical_result_exporter import export_all_clinical_results
    from src.evaluation.cross_validation_reporter import CrossValidationReporter
    from src.models.resnet101 import build_resnet101_baseline
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
    from resnet101 import build_resnet101_baseline
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


def train_single_fold(
    fold_idx: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    raw_images_dir: Path,
    processed_dir: Path,
    checkpoints_base_dir: Path,
    num_workers: int = 4,
    preload_ram: bool = True,
):
    """Huấn luyện 1 Fold độc lập cho ResNet-101 trong hệ thống 5-Fold Cross Validation."""
    print("=" * 80)
    print(
        f"🔥 BẮT ĐẦU HUẤN LUYỆN FOLD {fold_idx} (RESNET-101 - TASK #77 - {epochs} EPOCHS - BATCH {batch_size})..."
    )
    print("=" * 80)

    set_seed(42 + fold_idx)
    fold_dir = processed_dir / f"fold_{fold_idx}"

    loaders = get_dataloaders(
        processed_dir=str(fold_dir),
        raw_images_dir=str(raw_images_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        preload_ram=preload_ram,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    class_weights = calculate_smoothed_class_weights(
        fold_dir / "train.csv", train_loader.dataset.idx_to_class, power=0.6
    ).to(device)

    model = build_resnet101_baseline(
        num_classes=23, pretrained=True, freeze_backbone=False
    )
    model.fc = nn.Sequential(nn.Dropout(p=0.45), nn.Linear(2048, 23))
    model = model.to(device)

    criterion = MultiClassFocalLoss(
        weight=class_weights, gamma=1.5, label_smoothing=0.05
    )
    # Tự động điều chỉnh LR theo Batch Size (Batch 64 -> LR 1.5e-4)
    base_lr = 2.5e-4 if batch_size >= 128 else (2.0e-4 if batch_size >= 96 else 1.5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=1, eta_min=5e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    chk_dir = checkpoints_base_dir / f"fold_{fold_idx}"
    chk_dir.mkdir(parents=True, exist_ok=True)

    chk_manager = ComprehensiveCheckpointManager(
        checkpoint_dir=str(chk_dir),
        metric_name="val_macro_f1",
        run_config={"fold": fold_idx, "model": f"ResNet-101 {epochs}ep"},
    )
    logger = TrainingLogger(log_dir=str(chk_dir))

    history = []
    for ep in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Fold {fold_idx} [{ep:3d}/{epochs}] 🏋️ Train",
            leave=False,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:20}{r_bar}",
        )
        t_batch_start = time.time()
        for batch in pbar:
            t_data = time.time() - t_batch_start
            t_gpu_start = time.time()

            imgs = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outs = model(imgs)
                loss = criterion(outs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            running_train_loss += loss_val
            t_gpu = time.time() - t_gpu_start
            pbar.set_postfix_str(
                f"D:{t_data:.2f}s|G:{t_gpu:.2f}s|L:{loss_val:.3f}"
            )
            if pbar.n <= 3 or pbar.n % 20 == 0:
                pbar.write(
                    f"⏱️ Batch {pbar.n:3d}/{len(train_loader)} ➔ Data Load: {t_data:.3f}s | GPU Compute: {t_gpu:.3f}s"
                )
            t_batch_start = time.time()

        scheduler.step()
        train_loss = running_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch[0].to(device, non_blocking=True)
                targets = batch[1].to(device, non_blocking=True)
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
            f"Epoch [{ep:3d}/{epochs}] {best_mark} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:6.2f}% | Val Macro F1: {val_f1:6.2f}% | "
            f"LR: {curr_lr:.2e} | Time: {ep_duration:4.1f}s"
        )

        history.append(
            {
                "epoch": ep,
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

    # Đánh giá lại checkpoint tốt nhất của Fold và xuất toàn bộ 100% kết quả
    best_ckpt = chk_manager.load_best(model, device)
    model.eval()
    all_preds, all_targets, all_probs, all_filenames = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = (
                batch[0].to(device, memory_format=torch.channels_last, non_blocking=True)
                if device.type == "cuda"
                else batch[0].to(device)
            )
            targets = batch[1].to(device, non_blocking=True)
            filenames = batch[2]
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

    fold_metrics = export_all_clinical_results(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        filenames=all_filenames,
        class_names=class_names,
        history_df=history_df,
        output_dir=chk_dir,
        model_name="ResNet-101",
        fold_idx=fold_idx,
        best_epoch=best_ckpt.get("epoch", -1),
        res50_comparison_dir=res50_base_dir if res50_base_dir.exists() else None,
    )

    return fold_metrics, history_df


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện ResNet-101 (Task #77)")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Số epochs huấn luyện (Mặc định: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Kích thước batch (Mặc định: 64 - tối ưu độ ổn định và gradient y tế)",
    )
    default_workers = 4
    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_workers,
        help="Số luồng CPU nạp ảnh song song (Mặc định: 4 để pre-fetch dữ liệu nhanh nhất từ RAM)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Chọn Fold để chạy (Mặc định: 0 - chỉ chạy Fold 0, hoặc -1 chạy toàn bộ 5 Folds)",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "models" / "checkpoints" / "resnet101_5folds"),
        help="Đường dẫn lưu kết quả checkpoints",
    )
    parser.add_argument(
        "--preload_ram",
        action="store_true",
        default=True,
        help="Nạp toàn bộ ảnh vào RAM (256x256 uint8) để xóa sổ 100% độ trễ đọc đĩa (Mặc định: True)",
    )
    parser.add_argument(
        "--no_preload_ram",
        dest="preload_ram",
        action="store_false",
        help="Tắt nạp ảnh vào RAM nếu máy tính thiếu RAM",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Kích hoạt cuDNN benchmark để Turing GPU tự chọn kernel convolution nhanh nhất
        torch.backends.cudnn.benchmark = True
    print(
        f"🖥️ Thiết bị: {device} ➔ {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    raw_images_dir = Path(args.raw_images_dir)
    # Tự động nhận diện thư mục dữ liệu trên Kaggle nếu không tìm thấy ở đường dẫn mặc định
    if not raw_images_dir.exists() and Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob("labeled-images"))
        if candidates:
            raw_images_dir = candidates[0]
            print(f"🎉 Kaggle detected! Đã tự động kết nối ảnh tại: {raw_images_dir}")
    processed_dir = Path(args.processed_dir)
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
        print(f"🏆 ĐÃ HOÀN THÀNH HUẤN LUYỆN FOLD {f_idx} (RESNET-101 - TASK #77)!")
        print(f"📊 Accuracy = {m.get('accuracy', 0):.2f}% | Macro F1 = {m.get('macro_f1', 0):.2f}% | Macro Recall = {m.get('macro_recall', 0):.2f}%")
        print(f"💾 Checkpoint tốt nhất và nhật ký đã lưu tại: {output_dir / f'fold_{f_idx}'}")
        print("=" * 80)

    # Nếu chạy đủ 5 Folds, tự động xuất báo cáo khoa học Mean ± Std và Box Plot
    if len(all_fold_metrics) == 5:
        print("\n" + "=" * 80)
        print("🏆 ĐÃ HOÀN THÀNH TOÀN BỘ 5-FOLD CROSS VALIDATION (RESNET-101)!")
        print("📊 Đang tổng hợp thống kê Mean ± Std và xuất bản Box Plot...")
        print("=" * 80)

        reporter = CrossValidationReporter(
            model_name="ResNet-101 Baseline",
            cv_result_dir=str(output_dir),
        )
        summary_df = reporter.aggregate_metrics(all_fold_metrics)

        try:
            print(
                summary_df[
                    ["Chỉ số lâm sàng", "Mean ± Std", "Khoảng dao động [Min, Max]"]
                ].to_markdown(index=False)
            )
        except (ImportError, ModuleNotFoundError):
            print(
                summary_df[
                    ["Chỉ số lâm sàng", "Mean ± Std", "Khoảng dao động [Min, Max]"]
                ].to_string(index=False)
            )

        # Xuất biểu đồ hộp Box Plot và biểu đồ đường 5 Folds
        reporter.plot_boxplots(all_fold_metrics, "56_resnet101_5fold_boxplots.png")
        reporter.plot_5fold_learning_curves(
            all_fold_histories, "57_resnet101_5fold_learning_curves.png"
        )
        summary_df.to_csv(
            output_dir / "resnet101_5fold_summary_report.csv", index=False
        )
        print(f"\n💾 Đã lưu bảng báo cáo tổng kết tại: {output_dir}")


if __name__ == "__main__":
    main()
