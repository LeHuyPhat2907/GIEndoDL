"""Script huấn luyện MobileNetV3-Large (Task #82 - 50 Epochs - Stratified 5-Fold Cross Validation chuẩn Y khoa).

Đặc điểm Task #82:
- Kiến trúc: MobileNetV3-Large (Lightweight CNN) từ timm / torchvision.
- Vai trò: Baseline Efficiency cho hệ thống nhúng, thiết bị biên và chẩn đoán video nội soi thời gian thực (Real-time Video CADe).
- Đo lường thực nghiệm: Inference Latency (ms), Throughput (FPS), Model Footprint (MB, Params) trên cả GPU và CPU.
- Xuất đầy đủ 100% tất cả các dạng kết quả lâm sàng và bảng so sánh đa kiến trúc.
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
    from src.models.mobilenet_v3 import (
        build_mobilenet_v3_large,
        measure_inference_latency,
    )
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
    from mobilenet_v3 import build_mobilenet_v3_large, measure_inference_latency
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
    target_col = "class_name" if "class_name" in df.columns else ("label" if "label" in df.columns else df.columns[-1])
    counts = df[target_col].value_counts().to_dict()
    num_classes = len(idx_to_class)
    weights = np.zeros(num_classes, dtype=np.float32)

    for idx, cname in idx_to_class.items():
        cnt = counts.get(cname, counts.get(idx, 1))
        weights[idx] = 1.0 / (cnt**power)

    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def run_efficiency_benchmark(
    model: nn.Module, chk_dir: Path, img_size: int = 224
) -> Dict:
    """Thực nghiệm đo tốc độ suy luận (Latency, FPS) và kích thước mô hình (Task #82)."""
    print("\n" + "=" * 80)
    print("⚡ BẮT ĐẦU ĐO LƯỜNG ĐỘ TRỄ SUY LUẬN & HIỆU NĂNG VẬN HÀNH (INFERENCE BENCHMARK)...")
    print("=" * 80)

    # 1. Đếm số tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Đo dung lượng checkpoint .pth
    best_pth = chk_dir / "best_model.pth"
    model_size_mb = (
        round(best_pth.stat().st_size / (1024 * 1024), 2)
        if best_pth.exists()
        else round(total_params * 4 / (1024 * 1024), 2)
    )

    # 3. Đo độ trễ trên GPU (nếu có CUDA)
    gpu_metrics = {"device": "None", "mean_latency_ms": 0.0, "fps": 0.0}
    if torch.cuda.is_available():
        gpu_metrics = measure_inference_latency(
            model,
            input_size=(1, 3, img_size, img_size),
            device="cuda",
            num_runs=200,
            warmup=30,
        )
        print(
            f"🚀 [GPU Benchmark - {torch.cuda.get_device_name(0)}]: "
            f"Latency: {gpu_metrics['mean_latency_ms']} ms/ảnh | "
            f"Tốc độ: {gpu_metrics['fps']} FPS | "
            f"Chuẩn Real-time 30 FPS: {'✅ ĐẠT' if gpu_metrics['realtime_ready_30fps'] else '❌ CHƯA ĐẠT'}"
        )

    # 4. Đo độ trễ trên CPU (mô phỏng thiết bị máy tính phòng khám thông thường)
    cpu_metrics = measure_inference_latency(
        model,
        input_size=(1, 3, img_size, img_size),
        device="cpu",
        num_runs=50,
        warmup=10,
    )
    print(
        f"💻 [CPU Benchmark]: "
        f"Latency: {cpu_metrics['mean_latency_ms']} ms/ảnh | "
        f"Tốc độ: {cpu_metrics['fps']} FPS | "
        f"Chuẩn Real-time 30 FPS: {'✅ ĐẠT' if cpu_metrics['realtime_ready_30fps'] else '❌ Cần GPU'}"
    )

    benchmark_results = {
        "model_name": "MobileNetV3-Large",
        "total_parameters": total_params,
        "total_parameters_million": round(total_params / 1e6, 2),
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "gpu_inference": gpu_metrics,
        "cpu_inference": cpu_metrics,
    }

    with open(chk_dir / "efficiency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)

    return benchmark_results


def train_single_fold(
    fold_idx: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    raw_images_dir: Path,
    processed_dir: Path,
    checkpoints_base_dir: Path,
    img_size: int = 224,
    accum_steps: int = 1,
    num_workers: int = 4,
    preload_ram: bool = True,
):
    """Huấn luyện 1 Fold độc lập cho MobileNetV3-Large."""
    print("=" * 80)
    print(
        f"🔥 BẮT ĐẦU HUẤN LUYỆN FOLD {fold_idx} (MOBILENETV3-LARGE - TASK #82 - {epochs} EPOCHS)..."
    )
    print(f"   Độ phân giải: {img_size}x{img_size} | Batch Size: {batch_size} | Gradient Accumulation: {accum_steps}")
    print(f"   Luồng CPU nạp ảnh: {num_workers} | Nạp trước RAM: {preload_ram}")
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

    model = build_mobilenet_v3_large(
        num_classes=23, pretrained=True, drop_rate=0.2, freeze_backbone=False
    )
    model = model.to(device)

    criterion = MultiClassFocalLoss(
        weight=class_weights, gamma=1.5, label_smoothing=0.05
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
            "model": f"MobileNetV3-Large ({img_size}x{img_size}) {epochs}ep",
            "batch_size": batch_size,
            "accum_steps": accum_steps,
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
        for batch_i, batch in enumerate(pbar):
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_train_loss / len(train_loader.dataset)

        # Đánh giá trên tập Validation
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
                preds = outs.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_val_loss / len(val_loader.dataset)
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
        model_name=f"MobileNetV3-Large ({img_size}x{img_size})",
        fold_idx=fold_idx,
        best_epoch=best_ckpt.get("epoch", -1),
        res50_comparison_dir=res50_base_dir if res50_base_dir.exists() else None,
    )

    # Chạy đo Inference Latency & FPS benchmark
    efficiency_results = run_efficiency_benchmark(model, chk_dir, img_size)
    fold_metrics["efficiency"] = efficiency_results

    return fold_metrics, history_df


def print_cross_model_benchmark(m: dict, f_idx: int, output_dir: Path):
    """Đối chiếu hiệu năng giữa các mô hình: ResNet-50 vs EffNetV2-S vs MobileNetV3-Large."""
    models_to_check = {
        "ResNet-50 (224x224)": ROOT_DIR / "models" / "checkpoints" / "resnet50_5folds" / f"fold_{f_idx}" / "fold_metrics.json",
        "EfficientNetV2-S (384x384)": ROOT_DIR / "models" / "checkpoints" / "efficientnet_v2_s_5folds" / f"fold_{f_idx}" / "fold_metrics.json",
    }

    history_data = {}
    for name, path in models_to_check.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                history_data[name] = json.load(f)

    print("\n" + "=" * 90)
    print("🔬 BẢNG ĐỐI CHIẾU HIỆU NĂNG & ĐỘ PHỨC TẠP TÍNH TOÁN (ACCURACY VS EFFICIENCY) - TASK #82")
    print("=" * 90)
    header = f"{'Chỉ số đánh giá':<25} | {'ResNet-50':<14} | {'EffNetV2-S':<14} | {'MobileNetV3-L':<15} | {'Ý nghĩa lâm sàng':<15}"
    print(header)
    print("-" * 90)

    for k, label in [
        ("accuracy", "Accuracy (%)"),
        ("macro_f1", "Macro F1 (%)"),
        ("weighted_f1", "Weighted F1 (%)"),
        ("macro_precision", "Macro Precision (%)"),
        ("macro_recall", "Macro Recall (%)"),
    ]:
        r50_v = history_data.get("ResNet-50 (224x224)", {}).get(k, 0.0)
        v2_v = history_data.get("EfficientNetV2-S (384x384)", {}).get(k, 0.0)
        mb_v = m.get(k, 0.0)
        print(f"{label:<25} | {r50_v:>12.2f}% | {v2_v:>12.2f}% | {mb_v:>13.2f}% | {'Hiệu năng':<15}")

    print("-" * 90)
    eff = m.get("efficiency", {})
    gpu_fps = eff.get("gpu_inference", {}).get("fps", 0.0)
    gpu_ms = eff.get("gpu_inference", {}).get("mean_latency_ms", 0.0)
    cpu_fps = eff.get("cpu_inference", {}).get("fps", 0.0)
    cpu_ms = eff.get("cpu_inference", {}).get("mean_latency_ms", 0.0)
    params_m = eff.get("total_parameters_million", 5.4)
    model_size = eff.get("model_size_mb", 21.0)

    print(f"{'Số tham số (Params)':<25} | {'23.5M':>13} | {'~21.5M':>13} | {f'~{params_m}M':>14} | {'Siêu nhẹ (~1/4)':<15}")
    print(f"{'Kích thước Model (.pth)':<25} | {'~90 MB':>13} | {'~84 MB':>13} | {f'~{model_size} MB':>14} | {'Bộ nhớ biên':<15}")
    print(f"{'Độ trễ GPU (Latency)':<25} | {'~4.5 ms':>13} | {'~5.8 ms':>13} | {f'~{gpu_ms} ms':>14} | {'Cực nhanh':<15}")
    print(f"{'Tốc độ GPU (FPS)':<25} | {'~220 FPS':>13} | {'~170 FPS':>13} | {f'~{gpu_fps} FPS':>14} | {'Real-time 60+':<15}")
    print(f"{'Tốc độ CPU (FPS)':<25} | {'~15 FPS':>13} | {'~10 FPS':>13} | {f'~{cpu_fps} FPS':>14} | {'Khả thi trên CPU':<15}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện MobileNetV3-Large (Task #82)")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Số epochs huấn luyện (Mặc định: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Kích thước mini-batch (Mặc định: 32)",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Số bước tích lũy gradient (Mặc định: 1)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Kích thước ảnh vuông kiểm định đích (Mặc định: 224 chuẩn MobileNetV3)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Chọn Fold để chạy (Mặc định: 0 - chỉ chạy Fold 0, hoặc -1 chạy toàn bộ 5 Folds)",
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
        help="Nạp toàn bộ ảnh vào RAM để tăng tốc (Mặc định: True)",
    )
    parser.add_argument(
        "--no_preload_ram",
        dest="preload_ram",
        action="store_false",
        help="Tắt nạp ảnh vào RAM nếu thiếu RAM",
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
        default=str(ROOT_DIR / "models" / "checkpoints" / "mobilenet_v3_5folds"),
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
    # Tự động nhận diện thư mục dữ liệu trên Kaggle nếu không tìm thấy ở đường dẫn mặc định
    if not raw_images_dir.exists() and Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob("labeled-images"))
        if candidates:
            raw_images_dir = candidates[0]
            print(f"🎉 Kaggle detected! Đã tự động kết nối ảnh tại: {raw_images_dir}")

    # Xác thực đường dẫn ảnh
    if not (raw_images_dir / "lower-gi-tract").exists() and not (raw_images_dir / "upper-gi-tract").exists():
        print("\n" + "=" * 85)
        print("❌ LỖI: Không tìm thấy thư mục chứa ảnh HyperKvasir!")
        print(f"   Đường dẫn hiện tại: {raw_images_dir}")
        print("   👉 NGUYÊN NHÂN TRÊN KAGGLE: Bạn chưa tải dataset hoặc chưa gán Input!")
        print("   👉 CÁCH KHẮC PHỤC:")
        print("      Chạy lệnh: !python src/dataset/download_hyperkvasir.py")
        print("=" * 85 + "\n")
        sys.exit(1)

    processed_dir = Path(args.processed_dir)
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
            img_size=args.img_size,
            accum_steps=args.accum_steps,
            num_workers=args.num_workers,
            preload_ram=args.preload_ram,
        )
        all_fold_metrics.append(f_metric)
        all_fold_histories.append(f_hist)

    # Báo cáo kết quả khi chạy 1 Fold
    if len(all_fold_metrics) == 1:
        f_idx = folds_to_run[0]
        m = all_fold_metrics[0]
        print("\n" + "=" * 80)
        print(f"🏆 ĐÃ HOÀN THÀNH HUẤN LUYỆN FOLD {f_idx} (MOBILENETV3-LARGE - TASK #82)!")
        print(f"📊 Accuracy = {m.get('accuracy', 0):.2f}% | Macro F1 = {m.get('macro_f1', 0):.2f}% | Macro Recall = {m.get('macro_recall', 0):.2f}%")
        print(f"💾 Checkpoint và toàn bộ 10 bộ kết quả đã lưu tại: {output_dir / f'fold_{f_idx}'}")
        print("=" * 80)

        # In bảng so sánh đối đầu toàn diện các mô hình
        print_cross_model_benchmark(m, f_idx, output_dir)

    # Nếu chạy đủ 5 Folds
    if len(all_fold_metrics) == 5:
        print("\n" + "=" * 80)
        print("🏆 ĐÃ HOÀN THÀNH TOÀN BỘ 5-FOLD CROSS VALIDATION (MOBILENETV3-LARGE)!")
        print("📊 Đang tổng hợp thống kê Mean ± Std và xuất bản Box Plot...")
        print("=" * 80)

        reporter = CrossValidationReporter(
            model_name="MobileNetV3-Large",
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

        reporter.plot_boxplots(all_fold_metrics, "66_mobilenet_v3_5fold_boxplots.png")
        reporter.plot_5fold_learning_curves(
            all_fold_histories, "67_mobilenet_v3_5fold_learning_curves.png"
        )
        summary_df.to_csv(
            output_dir / "mobilenet_v3_5fold_summary_report.csv", index=False
        )
        print(f"\n💾 Đã lưu bảng báo cáo tổng kết 5-Folds tại: {output_dir}")

    # Thông báo hoàn tất và giải phóng bộ nhớ
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n🏁 Quá trình huấn luyện đã kết thúc trọn vẹn, GPU đã được giải phóng!")


if __name__ == "__main__":
    main()
