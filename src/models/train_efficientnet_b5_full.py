"""Script huấn luyện EfficientNet-B5 (Task #80 - 50 Epochs - Stratified 5-Fold Cross Validation chuẩn Y khoa).

Đặc điểm Task #80:
- Kiến trúc: Pretrained EfficientNet-B5 từ thư viện timm (Compound Scaling).
- Độ phân giải ảnh đầu vào: 456x456 pixels (chuẩn nguyên bản của EfficientNet-B5).
- Cơ chế chống OOM: Batch Size 8 + Gradient Accumulation 4 -> Effective Batch Size 32 (an toàn tuyệt đối cho 8GB VRAM).
- Tích hợp kiểm tra tràn bộ nhớ (Check OOM Pre-flight dry-run).
- So sánh đối đầu trực tiếp 3 mô hình: ResNet-50 (224x224) vs EfficientNet-B4 (380x380) vs EfficientNet-B5 (456x456).
- Xuất đầy đủ 100% tất cả các dạng kết quả sau khi train xong.
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
from tqdm.auto import tqdm

# Thiết lập đường dẫn root an toàn
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.dataset.dataloader_factory import get_dataloaders
    from src.evaluation.clinical_result_exporter import export_all_clinical_results
    from src.evaluation.cross_validation_reporter import CrossValidationReporter
    from src.models.efficientnet_b5 import build_efficientnet_b5_baseline
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
    from efficientnet_b5 import build_efficientnet_b5_baseline
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


def preflight_oom_check(device: torch.device, batch_size: int, img_size: int = 456) -> bool:
    """Kiểm tra mô phỏng trước để đảm bảo VRAM GPU không bị tràn (OOM Protection)."""
    if not torch.cuda.is_available() or device.type != "cuda":
        print("ℹ️ Đang chạy trên CPU, bỏ qua kiểm tra VRAM GPU.")
        return True

    print("\n" + "=" * 80)
    print(f"🛡️ ĐANG KIỂM TRA CHỐNG TRÀN BỘ NHỚ VRAM (PRE-FLIGHT OOM CHECK)...")
    print(f"   Kích thước ảnh: {img_size}x{img_size} | Mini-Batch Size: {batch_size}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"   GPU: {torch.cuda.get_device_name(0)} (Tổng: {total_mem:.2f} GB | Đang trống: {free_mem:.2f} GB)")

    try:
        torch.cuda.empty_cache()
        model_test = build_efficientnet_b5_baseline(num_classes=23, pretrained=False).to(device)
        model_test.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model_test.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        # Giả lập 1 batch thử nghiệm forward + backward
        dummy_x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        dummy_y = torch.randint(0, 23, (batch_size,), device=device)

        with torch.amp.autocast("cuda"):
            out = model_test(dummy_x)
            loss = criterion(out, dummy_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"   ✅ Kiểm tra thành công! VRAM đỉnh ước tính: {peak_vram:.2f} GB / {total_mem:.2f} GB")
        print(f"   VRAM còn dư an toàn: {total_mem - peak_vram:.2f} GB. Sẵn sàng huấn luyện!")
        print("=" * 80 + "\n")

        del dummy_x, dummy_y, model_test, criterion, optimizer, scaler
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"   ❌ CẢNH BÁO: Batch Size {batch_size} ở độ phân giải {img_size}x{img_size} làm tràn VRAM (OOM)!")
            print(f"   💡 Hãy giảm Batch Size xuống 8 hoặc 4 và tăng accum_steps tương ứng.")
            torch.cuda.empty_cache()
            return False
        else:
            print(f"   ⚠️ Lỗi khác khi kiểm tra: {e}")
            return False


def train_single_fold(
    fold_idx: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    raw_images_dir: Path,
    processed_dir: Path,
    checkpoints_base_dir: Path,
    img_size: int = 456,
    accum_steps: int = 4,
    num_workers: int = 4,
    preload_ram: bool = True,
):
    """Huấn luyện 1 Fold độc lập cho EfficientNet-B5 với độ phân giải 456x456."""
    print("=" * 80)
    print(
        f"🔥 BẮT ĐẦU HUẤN LUYỆN FOLD {fold_idx} (EFFICIENTNET-B5 - TASK #80 - {img_size}x{img_size} - {epochs} EPOCHS)..."
    )
    print(f"   Batch Size: {batch_size} | Tích lũy Gradient: {accum_steps} bước -> Effective Batch Size = {batch_size * accum_steps}")
    print("=" * 80)

    set_seed(42 + fold_idx)
    fold_dir = processed_dir / f"fold_{fold_idx}"

    loaders = get_dataloaders(
        processed_dir=str(fold_dir),
        raw_images_dir=str(raw_images_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=(img_size, img_size),
        preload_ram=preload_ram,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    class_weights = calculate_smoothed_class_weights(
        fold_dir / "train.csv", train_loader.dataset.idx_to_class, power=0.6
    ).to(device)

    model = build_efficientnet_b5_baseline(
        num_classes=23, pretrained=True, drop_rate=0.4, freeze_backbone=False
    )
    model = model.to(device)

    criterion = MultiClassFocalLoss(
        weight=class_weights, gamma=1.5, label_smoothing=0.05
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=5e-4)
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
            "model": f"EfficientNet-B5 ({img_size}x{img_size}) {epochs}ep",
            "batch_size": batch_size,
            "accum_steps": accum_steps,
            "effective_batch_size": batch_size * accum_steps,
        },
    )
    logger = TrainingLogger(log_dir=str(chk_dir))

    history = []
    for ep in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"Fold {fold_idx} [{ep:3d}/{epochs}] 🏋️ Train",
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
        model_name=f"EfficientNet-B5 ({img_size}x{img_size})",
        fold_idx=fold_idx,
        best_epoch=best_ckpt.get("epoch", -1),
        res50_comparison_dir=res50_base_dir if res50_base_dir.exists() else None,
    )

    return fold_metrics, history_df


def print_comparative_table(m: dict, f_idx: int, output_dir: Path):
    """Đối chiếu hiệu năng trực tiếp 3 mô hình: ResNet-50 vs EfficientNet-B4 vs EfficientNet-B5."""
    res50_file = ROOT_DIR / "models" / "checkpoints" / "resnet50_5folds" / f"fold_{f_idx}" / "fold_metrics.json"
    b4_file = ROOT_DIR / "models" / "checkpoints" / "efficientnet_b4_5folds" / f"fold_{f_idx}" / "fold_metrics.json"

    r50_data = {}
    b4_data = {}
    if res50_file.exists():
        with open(res50_file, "r", encoding="utf-8") as f:
            r50_data = json.load(f)
    if b4_file.exists():
        with open(b4_file, "r", encoding="utf-8") as f:
            b4_data = json.load(f)

    print("\n" + "=" * 90)
    print("🔬 ĐỐI CHIẾU HIỆU NĂNG 3 KIẾN TRÚC: ResNet-50 vs EfficientNet-B4 vs EfficientNet-B5 (Task #80)")
    print("=" * 90)
    header = f"{'Chỉ số lâm sàng':<22} | {'ResNet-50 (224)':<16} | {'EffNet-B4 (380)':<16} | {'EffNet-B5 (456)':<16} | {'Chênh B5 - B4':<14}"
    print(header)
    print("-" * 90)

    records = []
    metrics_to_compare = [
        ("accuracy", "Accuracy (%)"),
        ("macro_f1", "Macro F1 (%)"),
        ("weighted_f1", "Weighted F1 (%)"),
        ("macro_precision", "Macro Precision (%)"),
        ("macro_recall", "Macro Recall (%)"),
    ]

    for k, label in metrics_to_compare:
        r_val = r50_data.get(k, 0.0)
        b4_val = b4_data.get(k, 0.0)
        b5_val = m.get(k, 0.0)
        diff_b4 = b5_val - b4_val if b4_val > 0 else 0.0
        diff_str = f"{diff_b4:+.2f}%" if b4_val > 0 else "N/A"

        row = f"{label:<22} | {r_val:>14.2f}% | {b4_val:>14.2f}% | {b5_val:>14.2f}% | {diff_str:>14}"
        print(row)
        records.append({
            "Chỉ số": label,
            "ResNet-50 (224x224)": f"{r_val:.2f}%",
            "EfficientNet-B4 (380x380)": f"{b4_val:.2f}%",
            "EfficientNet-B5 (456x456)": f"{b5_val:.2f}%",
            "Chênh lệch B5 vs B4": diff_str,
        })

    print("-" * 90)
    print(f"{'Số tham số (Params)':<22} | {'23.5 Triệu':>15} | {'19.3 Triệu':>15} | {'30.4 Triệu':>15} | {'+57.5%':>14}")
    print(f"{'Độ phân giải (Input)':<22} | {'224 x 224':>15} | {'380 x 380':>15} | {'456 x 456':>15} | {'+44% Pixels':>14}")
    print("=" * 90)

    # Lưu bảng đối chiếu ra file CSV
    df_comp = pd.DataFrame(records)
    comp_path = output_dir / f"fold_{f_idx}" / "b5_vs_b4_vs_resnet50_comparison.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"💾 Đã lưu bảng đối chiếu 3 mô hình tại: {comp_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện EfficientNet-B5 (Task #80)")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Số epochs huấn luyện (Mặc định: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Kích thước mini-batch (Mặc định: 8 an toàn chống OOM trên 8GB VRAM ở 456x456)",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Số bước tích lũy gradient (Mặc định: 4 -> Effective Batch Size = 32)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=456,
        help="Kích thước ảnh vuông đầu vào (Mặc định: 456 chuẩn EfficientNet-B5)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Chọn Fold để chạy (Mặc định: 0 - chỉ chạy Fold 0, hoặc -1 chạy toàn bộ 5 Folds)",
    )
    parser.add_argument(
        "--check_oom_only",
        action="store_true",
        help="Chỉ chạy thử nghiệm 1 batch kiểm tra tràn VRAM rồi thoát (Pre-flight check)",
    )
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "models" / "checkpoints" / "efficientnet_b5_5folds"),
        help="Đường dẫn lưu kết quả checkpoints",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(
        f"🖥️ Thiết bị: {device} ➔ {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # 1. Chạy Pre-flight OOM Check trước khi huấn luyện
    oom_ok = preflight_oom_check(device=device, batch_size=args.batch_size, img_size=args.img_size)
    if args.check_oom_only:
        print("🏁 Đã hoàn thành chế độ --check_oom_only. Kết thúc chương trình.")
        return

    if not oom_ok:
        print("❌ Dừng huấn luyện do không vượt qua kiểm tra bộ nhớ VRAM. Vui lòng giảm batch_size!")
        return

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
            img_size=args.img_size,
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
        print(f"🏆 ĐÃ HOÀN THÀNH HUẤN LUYỆN FOLD {f_idx} (EFFICIENTNET-B5 - TASK #80)!")
        print(f"📊 Accuracy = {m.get('accuracy', 0):.2f}% | Macro F1 = {m.get('macro_f1', 0):.2f}% | Macro Recall = {m.get('macro_recall', 0):.2f}%")
        print(f"💾 Checkpoint và toàn bộ 10 bộ kết quả đã lưu tại: {output_dir / f'fold_{f_idx}'}")
        print("=" * 80)

        # In bảng đối chiếu trực tiếp 3 mô hình (B5 vs B4 vs ResNet-50)
        print_comparative_table(m, f_idx, output_dir)

    # Nếu chạy đủ 5 Folds, xuất báo cáo khoa học Mean ± Std và Box Plot
    if len(all_fold_metrics) == 5:
        print("\n" + "=" * 80)
        print("🏆 ĐÃ HOÀN THÀNH TOÀN BỘ 5-FOLD CROSS VALIDATION (EFFICIENTNET-B5)!")
        print("📊 Đang tổng hợp thống kê Mean ± Std và xuất bản Box Plot...")
        print("=" * 80)

        reporter = CrossValidationReporter(
            model_name="EfficientNet-B5 (456x456)",
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

        reporter.plot_boxplots(all_fold_metrics, "62_efficientnet_b5_5fold_boxplots.png")
        reporter.plot_5fold_learning_curves(
            all_fold_histories, "63_efficientnet_b5_5fold_learning_curves.png"
        )
        summary_df.to_csv(
            output_dir / "efficientnet_b5_5fold_summary_report.csv", index=False
        )
        print(f"\n💾 Đã lưu bảng báo cáo tổng kết 5-Folds tại: {output_dir}")


if __name__ == "__main__":
    main()
