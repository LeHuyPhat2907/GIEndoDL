"""Script thực nghiệm kiểm chứng hệ thống Dual Checkpointing và vẽ Dashboard minh họa."""

import json
import os
from pathlib import Path
import sys
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# Thiết lập đường dẫn root an toàn
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.training.checkpoint_manager import (
        ComprehensiveCheckpointManager,
        TrainingLogger,
    )
except ImportError:
    from checkpoint_manager import ComprehensiveCheckpointManager, TrainingLogger


def run_checkpoint_demo(config_dir: str, fig_dir: str, doc_dir: str):
    cfg_path = Path(config_dir)
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    cfg_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("💾 ĐANG THỰC NGHIỆM HỆ THỐNG DUAL CHECKPOINTING & LOGGING CHUẨN Y KHOA...")
    print("=" * 75)

    # 1. Lưu file cấu hình chính sách checkpointing_policy.json
    policy_config = {
        "policy_name": "Macro_F1_Driven_Dual_Checkpoint_System",
        "primary_metric": "val_macro_f1",
        "rationale": "Sử dụng val_macro_f1 để cứu các lớp thiểu số, ngăn chặn mô hình học vẹt nhãn đa số để tối đa hóa Accuracy.",
        "artifacts_managed": [
            {
                "filename": "best_model.pth",
                "condition": "Kỷ lục val_macro_f1 mới",
                "purpose": "Đánh giá nghiệm thu tập Test",
            },
            {
                "filename": "last_model.pth",
                "condition": "Lưu sau mỗi Epoch",
                "purpose": "Khôi phục huấn luyện tức thì khi Colab ngắt phiên",
            },
            {
                "filename": "run_config.json",
                "condition": "Đóng gói đầu phiên",
                "purpose": "Lưu vết 100% siêu tham số khoa học",
            },
            {
                "filename": "training_history.csv",
                "condition": "Ghi nhận mỗi Epoch",
                "purpose": "Vẽ đường cong biểu đồ cho luận văn",
            },
        ],
    }

    opt_json_p = cfg_path / "checkpointing_policy.json"
    with open(opt_json_p, "w", encoding="utf-8") as f:
        json.dump(policy_config, f, indent=4)
    print(f"✅ Đã lưu chính sách Checkpointing tại: {opt_json_p}")

    # 2. Khởi tạo mô hình giả lập và kiểm thử chu trình lưu 6 Epochs
    dummy_model = nn.Linear(10, 23)
    dummy_optim = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)
    chk_dir = ROOT_DIR / "models" / "checkpoints" / "demo_run"

    logger = TrainingLogger(log_dir=str(chk_dir))
    chk_manager = ComprehensiveCheckpointManager(
        checkpoint_dir=str(chk_dir),
        metric_name="val_macro_f1",
        run_config=policy_config,
    )

    epochs_demo = [1, 2, 3, 4, 5, 6]
    f1_scores = [74.2, 83.1, 88.5, 91.2, 90.8, 92.8]
    acc_scores = [81.0, 86.5, 90.0, 92.4, 93.0, 94.1]

    print("🔄 Kiểm thử mô phỏng ghi nhận Checkpoints:")
    for i, ep in enumerate(epochs_demo):
        metrics = {
            "epoch": ep,
            "train_loss": round(float(1.5 / ep), 4),
            "val_loss": round(float(1.2 / ep), 4),
            "val_acc": acc_scores[i],
            "val_macro_f1": f1_scores[i],
            "learning_rate": round(float(0.0005 * (0.9**ep)), 6),
            "time_sec": 42.0,
        }
        logger.log_epoch(metrics)
        is_best = chk_manager.step(ep, dummy_model, dummy_optim, metrics)
        status_txt = (
            f"⭐ KỶ LỤC MỚI (Đã lưu best_model.pth với F1 = {f1_scores[i]:.1f}%)"
            if is_best
            else "Đã lưu last_model.pth"
        )
        print(
            f"   ▶ Epoch {ep} | Macro F1: {f1_scores[i]:.1f}% | Acc: {acc_scores[i]:.1f}% ➔ {status_txt}"
        )

    # Thử nghiệm Load lại best_model.pth
    chk_loaded = chk_manager.load_best(dummy_model, torch.device("cpu"))
    print(
        f"✅ Kiểm thử nạp lại Best Checkpoint thành công: Best Epoch = {chk_loaded['epoch']}, F1 = {chk_loaded['metrics']['val_macro_f1']}%"
    )
    print("=" * 75)

    # 3. Vẽ Dashboard 4 Panel
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.set_theme(style="whitegrid")

    # Panel 1: Cơ chế Kích hoạt Kỷ lục Macro F1 vs Accuracy
    axes[0, 0].plot(
        epochs_demo,
        acc_scores,
        marker="o",
        color="#3498db",
        label="Validation Accuracy (%)",
        lw=2,
    )
    axes[0, 0].plot(
        epochs_demo,
        f1_scores,
        marker="s",
        color="#2ecc71",
        label="Validation Macro F1 (%) - Metric Quyết Định",
        lw=2.5,
    )

    # Đánh dấu các điểm lưu best_model.pth
    for i in [0, 1, 2, 3, 5]:
        axes[0, 0].annotate(
            "Saved\nbest_model.pth",
            (epochs_demo[i], f1_scores[i]),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
        )

    axes[0, 0].set_title(
        "1. Cơ Chế Lưu Best Checkpoint Dựa Trên Macro F1 Kỷ Lục",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Tỷ lệ (%)")
    axes[0, 0].set_xticks(epochs_demo)
    axes[0, 0].set_ylim(70, 100)
    axes[0, 0].legend(loc="lower right")

    # Panel 2: Sơ đồ Cấu trúc Cặp File Dual Checkpoints
    axes[0, 1].set_xlim(0, 10)
    axes[0, 1].set_ylim(0, 10)
    axes[0, 1].axis("off")
    axes[0, 1].set_title(
        "2. Cấu Trúc Bộ Lưu Trữ Trọng Số & Nhật Ký (Artifacts Storage)",
        fontsize=11,
        fontweight="bold",
    )

    files_box = [
        (
            "best_model.pth",
            "Lưu khi val_macro_f1 đạt đỉnh (Epoch 6: 92.8%)\nDùng để đánh giá nghiệm thu cuối cùng",
            "#e8f8f5",
            "#27ae60",
            7.6,
        ),
        (
            "last_model.pth",
            "Lưu liên tục sau mỗi Epoch (Epoch 6)\nDùng để Resume khôi phục phiên Colab bị ngắt",
            "#ebf5fb",
            "#2980b9",
            5.2,
        ),
        (
            "run_config.json",
            "Đóng gói toàn bộ siêu tham số (lr, batch, loss)\nBảo đảm tính tái lập khoa học 100%",
            "#fef9e7",
            "#f39c12",
            2.8,
        ),
        (
            "training_history.csv",
            "Bảng dữ liệu Loss, Acc, F1 theo từng Epoch\nDùng để vẽ đồ thị xuất bản trong Luận văn",
            "#f5eef8",
            "#8e44ad",
            0.4,
        ),
    ]

    for fname, desc, bg_c, bdr_c, y_pos in files_box:
        rect = patches.FancyBboxPatch(
            (0.5, y_pos),
            9.0,
            1.9,
            boxstyle="round,pad=0.25",
            facecolor=bg_c,
            edgecolor=bdr_c,
            lw=2,
        )
        axes[0, 1].add_patch(rect)
        axes[0, 1].text(
            5.0,
            y_pos + 0.95,
            f"📄 {fname}\n{desc}",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=9.5,
        )

    # Panel 3: Động thái Loss từ file history.csv
    train_losses = [1.5, 0.75, 0.5, 0.375, 0.3, 0.25]
    val_losses = [1.2, 0.6, 0.4, 0.3, 0.24, 0.2]
    axes[1, 0].plot(
        epochs_demo, train_losses, marker="o", color="#e74c3c", label="Train Loss", lw=2
    )
    axes[1, 0].plot(
        epochs_demo,
        val_losses,
        marker="^",
        color="#3498db",
        label="Validation Loss",
        lw=2,
    )
    axes[1, 0].set_title(
        "3. Nhật Ký Động Lực Loss Được Ghi Tự Động Vào history.csv",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Giá trị Loss")
    axes[1, 0].set_xticks(epochs_demo)
    axes[1, 0].legend()

    # Panel 4: Khuyến nghị Kết luận
    axes[1, 1].text(
        0.5,
        0.5,
        "💾 NGUYÊN TẮC CHECKPOINTING CHUẨN Y KHOA\n\n"
        "✔ Tiêu chí vàng: val_macro_f1 (Tuyệt đối không dùng Accuracy)\n"
        "  ➔ Bảo đảm công bằng cho bệnh lý hiếm (Trĩ, Barretts).\n\n"
        "✔ Dual Checkpoints (best_model.pth & last_model.pth):\n"
        "  ➔ Luôn có bản tối ưu nhất VÀ bản mới nhất để Resume.\n\n"
        "✔ Tự động xuất CSV Logger (training_history.csv):\n"
        "  ➔ Sẵn sàng chèn vào Excel hoặc biểu đồ Luận án.\n\n"
        "✔ Khóa chặt run_config.json kèm trọng số:\n"
        "  ➔ Tránh tình trạng quên mất model này train bằng thông số nào.\n\n"
        "👉 HỆ THỐNG AN TOÀN, MINH BẠCH & ĐẠT CHUẨN QUỐC TẾ!",
        fontsize=11.5,
        va="center",
        ha="center",
        fontweight="bold",
        color="#145a32",
        bbox=dict(
            boxstyle="round,pad=0.8", facecolor="#e8f8f5", edgecolor="#27ae60", lw=2
        ),
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_fig = fig_path / "42_checkpointing_and_logging_system.png"
    plt.savefig(out_fig, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"✅ Đã lưu Dashboard Checkpointing tại: {out_fig}")

    # 4. Xuất tài liệu nghiên cứu kỹ thuật
    md_file = doc_path / "50_checkpointing_and_macro_f1_logging.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# 💾 Báo cáo Kỹ thuật: Thiết Lập Hệ Thống Checkpointing & Logging Theo Macro F1\n\n"
        )
        f.write(
            "> **File cấu hình:** `configs/checkpointing_policy.json` | **Hình minh họa:** `docs/figures/42_checkpointing_and_logging_system.png`\n\n---\n\n"
        )
        f.write(
            "## 1. Lý Do Chọn `val_macro_f1` Làm Tiêu Chí Quyết Định Lưu Trọng Số\n\n"
        )
        f.write(
            "Trong bài toán nội soi tiêu hóa đuôi dài (Long-tailed GI Endoscopy), Accuracy là một thước đo đánh lừa: mô hình có thể phớt lờ hoàn toàn tổn thương hiếm mà vẫn đạt độ chính xác 88.5%. "
            "Do đó, `ComprehensiveCheckpointManager` chỉ cập nhật file `best_model.pth` khi chỉ số Macro F1 (trung bình cộng F1 của 23 lớp) đạt giá trị cao nhất.\n\n---\n\n"
        )
        f.write("## 2. Cấu Trúc Bộ Quản Lý Trọng Số Kép (Dual Checkpoints)\n\n")
        f.write(
            "- **`best_model.pth`:** Chứa trọng số ở thời điểm mô hình đạt độ nhạy cân bằng tốt nhất giữa các tổn thương, dùng cho nghiệm thu lâm sàng.\n"
        )
        f.write(
            "- **`last_model.pth`:** Chứa trạng thái đầy đủ (Model + Optimizer + Scaler) của Epoch gần nhất, cho phép tiếp tục huấn luyện ngay lập tức nếu phiên đám mây bị ngắt quãng.\n"
        )
        f.write(
            "- **`training_history.csv`:** Lưu trữ minh bạch toàn bộ các đường cong mất mát và điểm số phục vụ vẽ đồ thị công bố.\n"
        )

    print(f"✅ Đã lưu tài liệu nghiên cứu tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    config_dir_path = os.path.join(project_root, "configs")
    figures_dir_path = os.path.join(project_root, "docs", "figures")
    research_dir_path = os.path.join(project_root, "docs", "research")

    run_checkpoint_demo(config_dir_path, figures_dir_path, research_dir_path)
