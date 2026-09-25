"""Script huấn luyện ResNet-50 RUN 4 (50 Epochs - Stratified 5-Fold Cross Validation chuẩn Y khoa).

Áp dụng: 50 Epochs + Batch Size 64 + Focal Loss (gamma=1.5) + Dropout 0.45 + Weight Decay 5e-4.
Tự động tính toán Mean ± Std và xuất biểu đồ Box Plot, 5-Fold Learning Curves.
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
    from src.evaluation.cross_validation_reporter import CrossValidationReporter
    from src.models.resnet50 import build_resnet50_baseline
    from src.training.checkpoint_manager import (
        ComprehensiveCheckpointManager,
        TrainingLogger,
    )
    from src.utils.reproducibility import set_seed
except ImportError:
    from checkpoint_manager import ComprehensiveCheckpointManager, TrainingLogger
    from cross_validation_reporter import CrossValidationReporter
    from dataloader_factory import get_dataloaders
    from resnet50 import build_resnet50_baseline
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

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def calculate_smoothed_class_weights(
    train_csv_path: Path, idx_to_class: dict, power: float = 0.6
) -> torch.Tensor:
    """Tính trọng số cân bằng lớp làm mịn lũy thừa 0.6 để tăng lực kéo cho lớp hiếm."""
    df = pd.read_csv(train_csv_path)
    counts = df["class_name"].value_counts().to_dict()
    num_classes = len(idx_to_class)

    class_counts = np.array(
        [counts.get(idx_to_class[i], 1) for i in range(num_classes)],
        dtype=np.float32,
    )
    max_count = np.max(class_counts)
    weights = (max_count / class_counts) ** power
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float)


def train_single_fold(
    fold_idx: int, epochs: int, batch_size: int, device: torch.device
):
    """Huấn luyện 1 Fold độc lập trong hệ thống 5-Fold Cross Validation."""
    print("=" * 80)
    print(
        f"🔥 BẮT ĐẦU HUẤN LUYỆN FOLD {fold_idx} (RESNET-50 RUN 4 - {epochs} EPOCHS)..."
    )
    print("=" * 80)

    set_seed(42 + fold_idx)
    fold_dir = ROOT_DIR / "data" / "processed" / "5folds" / f"fold_{fold_idx}"
    raw_images_dir = ROOT_DIR / "data" / "raw" / "labeled-images"

    loaders = get_dataloaders(
        processed_dir=str(fold_dir),
        raw_images_dir=str(raw_images_dir),
        batch_size=batch_size,
        num_workers=4 if os.name != "nt" else 2,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    class_weights = calculate_smoothed_class_weights(
        fold_dir / "train.csv", train_loader.dataset.idx_to_class, power=0.6
    ).to(device)

    model = build_resnet50_baseline(
        num_classes=23, pretrained=True, freeze_backbone=False
    )
    model.fc = nn.Sequential(nn.Dropout(p=0.45), nn.Linear(2048, 23))
    model = model.to(device)

    criterion = MultiClassFocalLoss(
        weight=class_weights, gamma=1.5, label_smoothing=0.05
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=1, eta_min=5e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    chk_dir = (
        ROOT_DIR / "models" / "checkpoints" / "resnet50_5folds" / f"fold_{fold_idx}"
    )
    chk_dir.mkdir(parents=True, exist_ok=True)

    chk_manager = ComprehensiveCheckpointManager(
        checkpoint_dir=str(chk_dir),
        metric_name="val_macro_f1",
        run_config={"fold": fold_idx, "model": "ResNet-50 Run 4 50ep"},
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
        for batch in pbar:
            imgs, targets = (
                batch[0].to(device, non_blocking=True),
                batch[1].to(device, non_blocking=True),
            )
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outs = model(imgs)
                loss = criterion(outs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item() * imgs.size(0)

            vram_mb = (
                torch.cuda.memory_reserved() / (1024**2)
                if torch.cuda.is_available()
                else 0
            )
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "vram": f"{vram_mb:.0f}MB"})

        train_loss = running_train_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        running_val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs, targets = (
                    batch[0].to(device, non_blocking=True),
                    batch[1].to(device, non_blocking=True),
                )
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outs = model(imgs)
                    v_loss = criterion(outs, targets)
                running_val_loss += v_loss.item() * imgs.size(0)
                all_preds.extend(torch.argmax(outs, dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_targets, all_preds) * 100.0
        val_f1 = (
            f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
        )
        elapsed = time.time() - start_t

        rec = {
            "epoch": ep,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 2),
            "val_macro_f1": round(float(val_f1), 2),
            "time_sec": round(float(elapsed), 1),
        }
        logger.log_epoch(rec)
        is_best = chk_manager.step(ep, model, optimizer, rec, scaler=scaler)
        history.append(rec)

        flag = "⭐ [BEST]" if is_best else ""
        print(
            f"Fold {fold_idx} Ep [{ep:3d}/{epochs}] ── TrLoss: {train_loss:.4f} ── ValLoss: {val_loss:.4f} ── Acc: {val_acc:.1f}% ── F1: {val_f1:.1f}% {flag}"
        )

    # Đánh giá kiểm thử Fold bằng Best Model
    chk_manager.load_best(model, device)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs, targets = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outs = model(imgs)
            test_preds.extend(torch.argmax(outs, dim=1).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    fold_metrics = {
        "fold": fold_idx,
        "accuracy": round(float(accuracy_score(test_targets, test_preds) * 100.0), 2),
        "macro_f1": round(
            float(
                f1_score(test_targets, test_preds, average="macro", zero_division=0)
                * 100.0
            ),
            2,
        ),
        "weighted_f1": round(
            float(
                f1_score(test_targets, test_preds, average="weighted", zero_division=0)
                * 100.0
            ),
            2,
        ),
        "macro_recall": round(
            float(
                recall_score(test_targets, test_preds, average="macro", zero_division=0)
                * 100.0
            ),
            2,
        ),
        "macro_precision": round(
            float(
                precision_score(
                    test_targets, test_preds, average="macro", zero_division=0
                )
                * 100.0
            ),
            2,
        ),
        "weighted_precision": round(
            float(
                precision_score(
                    test_targets, test_preds, average="weighted", zero_division=0
                )
                * 100.0
            ),
            2,
        ),
        "best_epoch": chk_manager.best_epoch,
    }

    # Lưu kết quả từng Fold
    with open(chk_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=4)
    pd.DataFrame(history).to_csv(chk_dir / "history.csv", index=False)

    return fold_metrics, pd.DataFrame(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=50, help="Số epochs huấn luyện (Mặc định: 50)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Kích thước batch (Mặc định: 64)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Chọn Fold để chạy (Mặc định: -1 chạy tự động toàn bộ 5 Folds)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(
        f"🖥️ Thiết bị: {device} ➔ {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # Tự động tạo dữ liệu 5 Folds nếu trên máy chưa có
    five_folds_dir = ROOT_DIR / "data" / "processed" / "5folds"
    if not (five_folds_dir / "fold_0" / "train.csv").exists():
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
            f_idx, epochs=args.epochs, batch_size=args.batch_size, device=device
        )
        all_fold_metrics.append(f_metric)
        all_fold_histories.append(f_hist)

    # Nếu chạy đủ 5 Folds, tự động xuất báo cáo khoa học Mean ± Std và Box Plot
    if len(all_fold_metrics) == 5:
        print("\n" + "=" * 80)
        print("🏆 ĐÃ HOÀN THÀNH TOÀN BỘ 5-FOLD CROSS VALIDATION!")
        print("📊 Đang tổng hợp thống kê Mean ± Std và xuất bản Box Plot...")
        print("=" * 80)

        reporter = CrossValidationReporter(
            model_name="ResNet-50 Run 4",
            cv_result_dir="models/checkpoints/resnet50_5folds",
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
        reporter.plot_boxplots(all_fold_metrics, "54_resnet50_5fold_boxplots.png")
        reporter.plot_5fold_learning_curves(
            all_fold_histories, "55_resnet50_5fold_learning_curves.png"
        )

        # Lưu file CSV bảng báo cáo
        summary_df.to_csv(
            ROOT_DIR / "data" / "processed" / "resnet50_5fold_summary_report.csv",
            index=False,
        )
        print(
            f"\n💾 Đã lưu bảng báo cáo tại: {ROOT_DIR / 'data' / 'processed' / 'resnet50_5fold_summary_report.csv'}"
        )


if __name__ == "__main__":
    main()
