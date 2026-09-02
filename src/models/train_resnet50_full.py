"""Script huấn luyện toàn diện ResNet-50 mở khóa toàn bộ các tầng (100 Epochs Real Training)."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

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
    from src.training.trainer import ModularTrainer
    from src.utils.reproducibility import set_seed
except ImportError:
    from checkpoint_manager import ComprehensiveCheckpointManager, TrainingLogger
    from dataloader_factory import get_dataloaders
    from resnet50 import build_resnet50_baseline
    from trainer import ModularTrainer
    from reproducibility import set_seed


def auto_extract_data():
    """Tự động tìm và giải nén dữ liệu trên Colab nếu chưa có."""
    target_data = Path("/content/data/labeled-images")
    if target_data.exists():
        return

    print("📦 Đang kiểm tra file nén dữ liệu trên Colab...")
    zip_candidates = [
        Path("/content/hyperkvasir_data.zip"),
        Path("/content/drive/MyDrive/hyperkvasir_data.zip"),
    ]

    for z in zip_candidates:
        if z.exists():
            print(
                f"⚡ Tìm thấy file zip tại: {z} ➔ Đang giải nén vào /content/data/..."
            )
            shutil.unpack_archive(str(z), "/content/data")
            print("✅ Giải nén dữ liệu thành công!")
            return


def run_full_finetuning_resnet50(
    config_dir: str, fig_dir: str, doc_dir: str, mode: str = "train", epochs: int = 100
):
    cfg_path = Path(config_dir)
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    cfg_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(
        f"🔥 BẮT ĐẦU TIẾN TRÌNH HUẤN LUYỆN RESNET-50 FULL FINE-TUNING ({epochs} EPOCHS)..."
    )
    print("=" * 75)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"🖥️ Thiết bị sử dụng: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Mode'})"
    )

    chk_dir = ROOT_DIR / "models" / "checkpoints" / "resnet50_full"
    chk_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tự động kiểm tra dữ liệu
    auto_extract_data()

    proc_path = ROOT_DIR / "data" / "processed"
    raw_images_dir = (
        Path("/content/data/labeled-images")
        if Path("/content/data/labeled-images").exists()
        else (ROOT_DIR / "data" / "raw" / "labeled-images")
    )

    has_real_data = raw_images_dir.exists() and (proc_path / "train_split.csv").exists()

    # 2. Khởi tạo DataLoaders
    if has_real_data and mode == "train":
        print(f"📂 Nạp tập dữ liệu thật từ: {raw_images_dir}")
        loaders = get_dataloaders(
            processed_dir=str(proc_path),
            raw_images_dir=str(raw_images_dir),
            batch_size=32 if torch.cuda.is_available() else 16,
            num_workers=2 if torch.cuda.is_available() else 0,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        print(
            f"✅ Đã nạp thành công: {len(train_loader.dataset)} mẫu Train, {len(val_loader.dataset)} mẫu Val"
        )
    else:
        print("⚡ Chế độ kiểm thử nhanh: Sử dụng Tensor giả lập...")
        from torch.utils.data import DataLoader, TensorDataset

        train_loader = DataLoader(
            TensorDataset(torch.randn(64, 3, 224, 224), torch.randint(0, 23, (64,))),
            batch_size=16,
        )
        val_loader = DataLoader(
            TensorDataset(torch.randn(32, 3, 224, 224), torch.randint(0, 23, (32,))),
            batch_size=16,
        )
        epochs = 5  # Rút ngắn nếu kiểm thử

    # 3. Khởi tạo mô hình
    print("📦 Khởi tạo ResNet-50 (Mở khóa 100% tất cả các tầng)...")
    model = build_resnet50_baseline(
        num_classes=23, pretrained=torch.cuda.is_available(), freeze_backbone=False
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    run_config = {
        "model": "ResNet-50 Full Fine-Tuning",
        "epochs": epochs,
        "device": str(device),
        "learning_rate": 1e-4,
        "label_smoothing": 0.10,
    }

    logger = TrainingLogger(log_dir=str(chk_dir))
    chk_manager = ComprehensiveCheckpointManager(
        checkpoint_dir=str(chk_dir),
        metric_name="val_macro_f1",
        run_config=run_config,
    )

    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config={"use_amp": torch.cuda.is_available()},
    )

    history = []
    print("=" * 75)
    print("🔄 BẮT ĐẦU CHẠY HUẤN LUYỆN TỪNG EPOCH THẬT SỰ TRÊN GPU:")

    for ep in range(1, epochs + 1):
        tr_loss = trainer.train_one_epoch()
        v_loss, v_acc, v_f1 = trainer.validate()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        metrics = {
            "epoch": ep,
            "train_loss": round(float(tr_loss), 4),
            "val_loss": round(float(v_loss), 4),
            "val_acc": round(float(v_acc), 2),
            "val_macro_f1": round(float(v_f1), 2),
            "learning_rate": round(float(current_lr), 6),
        }

        logger.log_epoch(metrics)
        is_best = chk_manager.step(ep, model, optimizer, metrics)
        history.append(metrics)

        flag = "⭐ [BEST MODEL SAVED]" if is_best else ""
        print(
            f"▶ Epoch [{ep:3d}/{epochs}] | Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.1f}% | Macro F1: {v_f1:.1f}% {flag}"
        )

    print("=" * 75)
    print("🏆 HOÀN THÀNH HUẤN LUYỆN XUẤT SẮC!")

    # 4. Xuất đồ thị Dashboard kết quả
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    df_h = pd.DataFrame(history)
    axes[0].plot(
        df_h["epoch"], df_h["train_loss"], label="Train Loss", color="#3498db", lw=2.2
    )
    axes[0].plot(
        df_h["epoch"], df_h["val_loss"], label="Val Loss", color="#e74c3c", lw=2.2
    )
    axes[0].set_title("Động Lực Hội Tụ Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        df_h["epoch"],
        df_h["val_acc"],
        label="Validation Accuracy (%)",
        color="#2ecc71",
        lw=2.2,
    )
    axes[1].plot(
        df_h["epoch"],
        df_h["val_macro_f1"],
        label="Validation Macro F1 (%)",
        color="#f39c12",
        lw=2.5,
    )
    axes[1].set_title("Tăng Trưởng Hiệu Năng Lâm Sàng", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Tỷ lệ (%)")
    axes[1].legend()

    plt.tight_layout()
    out_fig = fig_path / "49_resnet50_full_finetune_dynamics.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã lưu Dashboard kết quả tại: {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=100)
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
        mode=args.mode,
        epochs=args.epochs,
    )
