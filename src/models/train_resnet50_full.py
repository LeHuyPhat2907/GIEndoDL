"""Script huấn luyện toàn diện ResNet-50 mở khóa toàn bộ các tầng trên GPU cá nhân (100 Epochs Real Training)."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
    from src.models.resnet50 import build_resnet50_baseline
    from src.training.checkpoint_manager import (
        ComprehensiveCheckpointManager,
        TrainingLogger,
    )
    from src.utils.reproducibility import set_seed
except ImportError:
    from checkpoint_manager import ComprehensiveCheckpointManager, TrainingLogger
    from dataloader_factory import get_dataloaders
    from resnet50 import build_resnet50_baseline
    from reproducibility import set_seed


def run_full_finetuning_resnet50(
    config_dir: str, fig_dir: str, doc_dir: str, epochs: int = 100
):
    cfg_path = Path(config_dir)
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    cfg_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(
        f"🔥 KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN RESNET-50 FULL FINE-TUNING ({epochs}"
        " EPOCHS)..."
    )
    print("=" * 80)

    # 1. Cố định tính tái lập khoa học
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU (Không tìm thấy GPU)"
    )
    print(f"🖥️ Phần cứng huấn luyện: {device} ➔ {gpu_name}")

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"⚡ Tổng bộ nhớ VRAM:     {vram_gb:.2f} GB GDDR6")

    # 2. Nạp dữ liệu ảnh thật
    proc_path = ROOT_DIR / "data" / "processed"
    raw_images_dir = ROOT_DIR / "data" / "raw" / "labeled-images"

    assert (
        raw_images_dir.exists()
    ), f"❌ Không tìm thấy thư mục ảnh tại: {raw_images_dir}"
    assert (
        proc_path / "train_split.csv"
    ).exists(), f"❌ Không tìm thấy train_split.csv tại: {proc_path}"

    print(f"📂 Nạp dữ liệu ảnh từ: {raw_images_dir}")
    loaders = get_dataloaders(
        processed_dir=str(proc_path),
        raw_images_dir=str(raw_images_dir),
        batch_size=32,
        num_workers=2 if os.name == "nt" else 4,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders.get("test")

    print(
        f"✅ Đã nạp thành công: {len(train_loader.dataset)} mẫu Train |"
        f" {len(val_loader.dataset)} mẫu Val (23 lớp)"
    )

    # 3. Khởi tạo mô hình ResNet-50 mở khóa 100% tầng
    print("📦 Khởi tạo ResNet-50 ImageNet và mở khóa 100% tham số...")
    model = build_resnet50_baseline(
        num_classes=23, pretrained=True, freeze_backbone=False
    )
    model = model.to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(
        f"✅ Tổng tham số: {total_params / 1e6:.2f} Triệu | Tham số huấn luyện:"
        f" {trainable_params / 1e6:.2f} Triệu (100% Unfrozen)"
    )

    # 4. Trình tối ưu hóa AdamW & Cosine Annealing Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    chk_dir = ROOT_DIR / "models" / "checkpoints" / "resnet50_full"
    chk_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        "model_name": "ResNet-50 (Full Fine-Tuned 100 Epochs)",
        "hardware": gpu_name,
        "epochs": epochs,
        "batch_size": 32,
        "optimizer": "AdamW (lr=1e-4, weight_decay=1e-4)",
        "scheduler": f"CosineAnnealingLR (T_max={epochs}, eta_min=1e-6)",
        "label_smoothing": 0.10,
    }

    logger = TrainingLogger(log_dir=str(chk_dir))
    chk_manager = ComprehensiveCheckpointManager(
        checkpoint_dir=str(chk_dir),
        metric_name="val_macro_f1",
        run_config=full_config,
    )

    history = []
    print("=" * 80)
    print(f"🚀 BẮT ĐẦU VÒNG LẶP {epochs} EPOCHS:")
    print("=" * 80)

    for ep in range(1, epochs + 1):
        start_t = time.time()

        # --- A. HUẤN LUYỆN (THANH TIẾN TRÌNH TRỰC QUAN ĐẸP MẮT) ---
        model.train()
        running_train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]

        pbar_train = tqdm(
            train_loader,
            desc=f"Epoch [{ep:3d}/{epochs}] 🏋️ Train",
            leave=False,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:25}{r_bar}",
        )

        for batch in pbar_train:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            running_train_loss += batch_loss * images.size(0)

            vram_use = (
                torch.cuda.memory_reserved() / (1024**2)
                if torch.cuda.is_available()
                else 0
            )
            pbar_train.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "vram": f"{vram_use:.0f}MB",
                    "lr": f"{current_lr:.1e}",
                }
            )

        train_loss = running_train_loss / len(train_loader.dataset)
        scheduler.step()

        # --- B. KIỂM ĐỊNH (VAL) ---
        model.eval()
        running_val_loss = 0.0
        all_preds, all_targets = [], []

        pbar_val = tqdm(
            val_loader,
            desc=f"Epoch [{ep:3d}/{epochs}] 🔍 Val  ",
            leave=False,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:25}{r_bar}",
        )

        with torch.no_grad():
            for batch in pbar_val:
                images = batch[0].to(device, non_blocking=True)
                targets = batch[1].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    v_loss = criterion(outputs, targets)

                running_val_loss += v_loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_targets, all_preds) * 100.0
        val_f1 = (
            f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
        )
        elapsed = time.time() - start_t

        metric_rec = {
            "epoch": ep,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 2),
            "val_macro_f1": round(float(val_f1), 2),
            "learning_rate": round(float(current_lr), 6),
            "time_sec": round(float(elapsed), 1),
        }

        logger.log_epoch(metric_rec)
        is_best = chk_manager.step(ep, model, optimizer, metric_rec, scaler=scaler)
        history.append(metric_rec)

        flag = "⭐ [KỶ LỤC MỚI ĐÃ LƯU]" if is_best else ""
        print(
            f"Epoch [{ep:3d}/{epochs}] ── Train Loss: {train_loss:.4f} ── Val"
            f" Loss: {val_loss:.4f} ── Val Acc: {val_acc:.1f}% ── Macro F1:"
            f" {val_f1:.1f}% ({elapsed:.0f}s) {flag}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 80)
    print("🏆 HOÀN THÀNH HUẤN LUYỆN XUẤT SẮC 100 EPOCHS!")
    print(
        f"🎯 Best Macro F1 đạt được: {chk_manager.best_metric_val:.2f}% (Tại"
        f" Epoch {chk_manager.best_epoch})"
    )
    print("=" * 80)

    # 5. Đánh giá kiểm thử chi tiết trên tập Test bằng mô hình tối ưu nhất
    print("🔬 Đang nạp trọng số tốt nhất để đánh giá trên tập Test độc lập...")
    chk_manager.load_best(model, device)
    eval_loader = test_loader if test_loader is not None else val_loader

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in eval_loader:
            imgs = batch[0].to(device)
            lbls = batch[1].to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outs = model(imgs)
            test_preds.extend(torch.argmax(outs, dim=1).cpu().numpy())
            test_targets.extend(lbls.cpu().numpy())

    class_names = [
        train_loader.dataset.idx_to_class[i] for i in range(len(set(test_targets)))
    ]
    report_dict = classification_report(
        test_targets,
        test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Xuất file CSV chi tiết 23 lớp bệnh
    per_class_data = []
    for cls_name in class_names:
        per_class_data.append(
            {
                "class_name": cls_name,
                "precision": round(report_dict[cls_name]["precision"] * 100, 2),
                "recall": round(report_dict[cls_name]["recall"] * 100, 2),
                "f1_score": round(report_dict[cls_name]["f1-score"] * 100, 2),
                "support": report_dict[cls_name]["support"],
            }
        )
    df_per_class = pd.DataFrame(per_class_data)
    per_class_csv_path = proc_path / "resnet50_100ep_per_class_metrics.csv"
    df_per_class.to_csv(per_class_csv_path, index=False)
    print(f"📊 Đã xuất báo cáo chi tiết 23 lớp bệnh tại: {per_class_csv_path}")

    # 6. Tự động vẽ Dashboard 4 Panel đối chuẩn (300 DPI)
    print("📈 Đang xuất bản Dashboard đồ thị 4 panel chuẩn mực...")
    df_h = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.set_theme(style="whitegrid")

    # Panel 1: Động lực Loss
    axes[0, 0].plot(
        df_h["epoch"],
        df_h["train_loss"],
        label="Train Loss",
        color="#3498db",
        lw=2.5,
    )
    axes[0, 0].plot(
        df_h["epoch"],
        df_h["val_loss"],
        label="Validation Loss",
        color="#e74c3c",
        lw=2.5,
    )
    axes[0, 0].set_title(
        "1. Động Lực Hội Tụ Loss (100 Epochs)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Panel 2: Hiệu năng lâm sàng Acc & F1
    axes[0, 1].plot(
        df_h["epoch"],
        df_h["val_acc"],
        label="Validation Accuracy (%)",
        color="#2ecc71",
        lw=2.5,
    )
    axes[0, 1].plot(
        df_h["epoch"],
        df_h["val_macro_f1"],
        label="Validation Macro F1 (%)",
        color="#f39c12",
        lw=2.5,
    )
    axes[0, 1].set_title(
        "2. Tăng Trưởng Hiệu Năng Lâm Sàng (100 Epochs)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Tỷ lệ (%)")
    axes[0, 1].legend()

    # Panel 3: Phân bố F1-Score 23 lớp bệnh học
    sorted_df = df_per_class.sort_values(by="f1_score", ascending=True)
    colors = ["#2ecc71" if val >= 85 else "#e74c3c" for val in sorted_df["f1_score"]]
    axes[1, 0].barh(sorted_df["class_name"], sorted_df["f1_score"], color=colors)
    axes[1, 0].axvline(
        85,
        color="red",
        linestyle="--",
        lw=1.5,
        label="Ngưỡng lâm sàng an toàn (85%)",
    )
    axes[1, 0].set_title(
        "3. Xếp Hạng F1-Score 23 Lớp Bệnh Học", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("F1-Score (%)")
    axes[1, 0].legend(loc="lower right")

    # Panel 4: Lịch trình suy giảm Learning Rate Cosine Annealing
    axes[1, 1].plot(
        df_h["epoch"],
        df_h["learning_rate"],
        color="#9b59b6",
        lw=2.5,
        label="Learning Rate",
    )
    axes[1, 1].set_title(
        "4. Lịch Trình Tốc Độ Học (Cosine Annealing LR)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].legend()

    plt.tight_layout()
    out_fig = fig_path / "49_resnet50_full_finetune_dynamics.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Đã lưu Dashboard 4 Panel tại: {out_fig}")

    # 7. Xuất cấu hình và báo cáo nghiên cứu
    full_config["final_test_accuracy"] = round(
        accuracy_score(test_targets, test_preds) * 100, 2
    )
    full_config["final_test_macro_f1"] = round(
        f1_score(test_targets, test_preds, average="macro") * 100, 2
    )
    full_config["final_test_macro_precision"] = round(
        precision_score(test_targets, test_preds, average="macro") * 100, 2
    )
    full_config["final_test_macro_recall"] = round(
        recall_score(test_targets, test_preds, average="macro") * 100, 2
    )
    full_config["best_val_macro_f1"] = chk_manager.best_metric_val
    full_config["best_epoch"] = chk_manager.best_epoch

    with open(
        cfg_path / "resnet50_full_finetune_config.json", "w", encoding="utf-8"
    ) as f:
        json.dump(full_config, f, indent=4)

    print("=" * 80)
    print("✅ TOÀN BỘ DỮ LIỆU ĐÃ ĐƯỢC XUẤT RA THÀNH CÔNG VÀ LƯU TRỮ AN TOÀN!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    config_dir_path = os.path.join(project_root, "configs")
    figures_dir_path = os.path.join(project_root, "docs", "figures")
    research_dir_path = os.path.join(project_root, "docs", "research")

    run_full_finetuning_resnet50(
        config_dir_path,
        figures_dir_path,
        research_dir_path,
        epochs=args.epochs,
    )
