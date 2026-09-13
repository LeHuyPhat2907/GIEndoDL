"""Module tổng hợp, thống kê Mean ± Std và vẽ Box Plot, Learning Curves chuẩn khoa học cho 5-Fold Cross Validation."""

from pathlib import Path
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class CrossValidationReporter:
    """Bộ công cụ tổng hợp và trực quan hóa 5-Fold Cross Validation chuẩn bài báo khoa học."""

    def __init__(self, model_name: str, cv_result_dir: str):
        self.model_name = model_name
        self.cv_dir = Path(cv_result_dir)
        self.figures_dir = ROOT_DIR / "docs" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_metrics(self, fold_metrics: List[Dict[str, float]]) -> pd.DataFrame:
        """Tính toán Mean ± Std, Min, Max từ danh sách kết quả 5 Folds."""
        df = pd.DataFrame(fold_metrics)
        summary_rows = []

        metric_cols = [
            c for c in df.columns if c not in ["fold", "epoch", "model_name"]
        ]

        for col in metric_cols:
            vals = df[col].values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))

            summary_rows.append(
                {
                    "Chỉ số lâm sàng": col.upper().replace("_", " "),
                    "Mean ± Std": f"{mean_v:.2f}% ± {std_v:.2f}%",
                    "Mean": round(mean_v, 2),
                    "Std": round(std_v, 2),
                    "Min": round(min_v, 2),
                    "Max": round(max_v, 2),
                    "Khoảng dao động [Min, Max]": f"[{min_v:.2f}%, {max_v:.2f}%]",
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def plot_boxplots(
        self,
        fold_metrics: List[Dict[str, float]],
        out_filename: str = "cv_boxplots.png",
    ):
        """Vẽ biểu đồ hộp (Box Plot & Strip Plot) minh họa độ ổn định của 5 Folds."""
        df = pd.DataFrame(fold_metrics)
        # Lọc các chỉ số chính: accuracy, macro_f1, macro_recall, macro_precision
        key_metrics = [
            c
            for c in ["accuracy", "macro_f1", "macro_recall", "macro_precision"]
            if c in df.columns
        ]

        melted_df = df.melt(
            id_vars=["fold"],
            value_vars=key_metrics,
            var_name="Metric",
            value_name="Score (%)",
        )
        melted_df["Metric"] = melted_df["Metric"].str.upper().str.replace("_", " ")

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Vẽ Box Plot
        palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
        sns.boxplot(
            x="Metric",
            y="Score (%)",
            data=melted_df,
            palette=palette,
            width=0.4,
            boxprops=dict(alpha=0.7),
        )
        # Vẽ các điểm Fold cụ thể (Strip Plot)
        sns.stripplot(
            x="Metric",
            y="Score (%)",
            data=melted_df,
            color="black",
            size=8,
            jitter=0.1,
            alpha=0.9,
        )

        plt.title(
            f"Phân Bố & Độ Ổn Định 5-Fold Cross Validation ({self.model_name})",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("Chỉ Số Đánh Giá", fontsize=11)
        plt.ylabel("Điểm Số (%)", fontsize=11)
        plt.ylim(
            max(0, melted_df["Score (%)"].min() - 5),
            min(100, melted_df["Score (%)"].max() + 5),
        )

        out_path = self.figures_dir / out_filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Đã xuất biểu đồ Box Plot tại: {out_path}")
        return out_path

    def plot_5fold_learning_curves(
        self,
        fold_histories: List[pd.DataFrame],
        out_filename: str = "cv_learning_curves.png",
    ):
        """Vẽ biểu đồ đường trung bình 5 Folds với dải bóng mờ Mean ± Std qua từng Epoch."""
        # Ghép 5 history lại
        all_epochs = fold_histories[0]["epoch"].values

        train_losses = np.array([df["train_loss"].values for df in fold_histories])
        val_losses = np.array([df["val_loss"].values for df in fold_histories])
        val_accs = np.array([df["val_acc"].values for df in fold_histories])
        val_f1s = np.array([df["val_macro_f1"].values for df in fold_histories])

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        sns.set_theme(style="whitegrid")

        # 1. Đồ thị Loss (Train vs Val)
        mean_tr_l = np.mean(train_losses, axis=0)
        std_tr_l = np.std(train_losses, axis=0)
        mean_v_l = np.mean(val_losses, axis=0)
        std_v_l = np.std(val_losses, axis=0)

        axes[0].plot(
            all_epochs, mean_tr_l, label="Train Loss (Mean)", color="#3498db", lw=2.5
        )
        axes[0].fill_between(
            all_epochs,
            mean_tr_l - std_tr_l,
            mean_tr_l + std_tr_l,
            color="#3498db",
            alpha=0.2,
        )

        axes[0].plot(
            all_epochs, mean_v_l, label="Val Loss (Mean)", color="#e74c3c", lw=2.5
        )
        axes[0].fill_between(
            all_epochs,
            mean_v_l - std_v_l,
            mean_v_l + std_v_l,
            color="#e74c3c",
            alpha=0.2,
        )

        axes[0].set_title(
            f"1. Động Lực Loss 5 Folds Mean ± Std ({self.model_name})",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xlabel("Epochs", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].legend(fontsize=11)

        # 2. Đồ thị Accuracy & F1
        mean_v_acc = np.mean(val_accs, axis=0)
        std_v_acc = np.std(val_accs, axis=0)
        mean_v_f1 = np.mean(val_f1s, axis=0)
        std_v_f1 = np.std(val_f1s, axis=0)

        axes[1].plot(
            all_epochs, mean_v_acc, label="Val Accuracy (Mean)", color="#2ecc71", lw=2.5
        )
        axes[1].fill_between(
            all_epochs,
            mean_v_acc - std_v_acc,
            mean_v_acc + std_v_acc,
            color="#2ecc71",
            alpha=0.2,
        )

        axes[1].plot(
            all_epochs, mean_v_f1, label="Val Macro F1 (Mean)", color="#f39c12", lw=2.5
        )
        axes[1].fill_between(
            all_epochs,
            mean_v_f1 - std_v_f1,
            mean_v_f1 + std_v_f1,
            color="#f39c12",
            alpha=0.2,
        )

        axes[1].set_title(
            f"2. Tăng Trưởng Hiệu Năng 5 Folds Mean ± Std ({self.model_name})",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xlabel("Epochs", fontsize=11)
        axes[1].set_ylabel("Tỷ lệ (%)", fontsize=11)
        axes[1].legend(fontsize=11)

        plt.tight_layout()
        out_path = self.figures_dir / out_filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Đã xuất biểu đồ 5-Fold Learning Curves tại: {out_path}")
        return out_path


# --- VÍ DỤ CHẠY THỬ NGHIỆM ĐỂ XÁC NHẬN MODULE HOẠT ĐỘNG ---
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 KIỂM THỬ KHẢ NĂNG TÍNH TOÁN VÀ VẼ ĐỒ THỊ 5-FOLD CROSS VALIDATION...")
    print("=" * 80)

    # Dữ liệu mẫu giả định của 5 Folds
    mock_fold_metrics = [
        {
            "fold": 0,
            "accuracy": 91.5,
            "macro_f1": 68.2,
            "macro_recall": 68.5,
            "macro_precision": 68.1,
        },
        {
            "fold": 1,
            "accuracy": 91.8,
            "macro_f1": 68.6,
            "macro_recall": 68.9,
            "macro_precision": 68.7,
        },
        {
            "fold": 2,
            "accuracy": 91.2,
            "macro_f1": 67.9,
            "macro_recall": 68.1,
            "macro_precision": 67.8,
        },
        {
            "fold": 3,
            "accuracy": 92.1,
            "macro_f1": 69.1,
            "macro_recall": 69.4,
            "macro_precision": 69.0,
        },
        {
            "fold": 4,
            "accuracy": 91.6,
            "macro_f1": 68.4,
            "macro_recall": 68.7,
            "macro_precision": 68.3,
        },
    ]

    reporter = CrossValidationReporter(
        model_name="ResNet-50", cv_result_dir="models/checkpoints"
    )
    summary_table = reporter.aggregate_metrics(mock_fold_metrics)

    print("\n📋 BẢNG THỐNG KÊ KHOA HỌC XUẤT BẢN:")
    print(
        summary_table[
            ["Chỉ số lâm sàng", "Mean ± Std", "Khoảng dao động [Min, Max]"]
        ].to_markdown(index=False)
    )

    reporter.plot_boxplots(mock_fold_metrics, "demo_5fold_boxplots.png")
    print("\n✅ Module hoạt động hoàn hảo và sẵn sàng tích hợp!")
