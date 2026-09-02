"""Module trực quan hóa kết quả huấn luyện đạt chuẩn xuất bản khoa học (Publication-Quality Figures)."""

from pathlib import Path
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PublicationVisualizer:
    """Bộ công cụ vẽ biểu đồ khoa học chuẩn 300 DPI cho luận văn và bài báo quốc tế."""

    def __init__(self, style: str = "whitegrid", font_scale: float = 1.0):
        sns.set_theme(style=style, font_scale=font_scale)
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#2c3e50"
        plt.rcParams["axes.linewidth"] = 1.2

    @staticmethod
    def plot_training_dynamics(
        history_df: pd.DataFrame, output_path: str, title_prefix: str = "Model Training"
    ):
        """Vẽ biểu đồ đối sánh Loss, Accuracy, F1 và Learning Rate qua từng Epoch."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        epochs = history_df["epoch"]

        # Panel 1: Loss
        axes[0].plot(
            epochs,
            history_df["train_loss"],
            marker="o",
            color="#3498db",
            label="Train Loss",
            lw=2.2,
        )
        axes[0].plot(
            epochs,
            history_df["val_loss"],
            marker="s",
            color="#e74c3c",
            label="Val Loss",
            lw=2.2,
        )
        axes[0].set_title(
            f"{title_prefix}: Loss Convergence", fontsize=11.5, fontweight="bold"
        )
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss Value")
        axes[0].legend()

        # Panel 2: Accuracy & Macro F1
        axes[1].plot(
            epochs,
            history_df["val_acc"],
            marker="^",
            color="#2ecc71",
            label="Validation Accuracy (%)",
            lw=2.2,
        )
        axes[1].plot(
            epochs,
            history_df["val_macro_f1"],
            marker="d",
            color="#f39c12",
            label="Macro F1-Score (%)",
            lw=2.5,
        )
        axes[1].axhline(
            92.8, color="darkgreen", linestyle="--", label="Mục tiêu SOTA (92.8%)"
        )
        axes[1].set_title(
            f"{title_prefix}: Clinical Performance", fontsize=11.5, fontweight="bold"
        )
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Tỷ lệ (%)")
        axes[1].legend(loc="lower right")

        # Panel 3: Learning Rate
        axes[2].plot(
            epochs, history_df["learning_rate"], marker="o", color="#8e44ad", lw=2.2
        )
        axes[2].set_title(
            f"{title_prefix}: Learning Rate Schedule", fontsize=11.5, fontweight="bold"
        )
        axes[2].set_xlabel("Epochs")
        axes[2].set_ylabel("Learning Rate")

        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_confusion_matrix_heatmap(
        cm_normalized: np.ndarray,
        class_names: List[str],
        output_path: str,
        title: str = "23-Class Normalized Confusion Matrix",
    ):
        """Vẽ ma trận nhầm lẫn kích thước lớn với thang màu sắc nét chuẩn y khoa."""
        plt.figure(figsize=(15, 13))
        sns.heatmap(
            cm_normalized,
            xticklabels=class_names,
            yticklabels=class_names,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar=True,
            linewidths=0.5,
            linecolor="lightgray",
        )
        plt.title(title, fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Lớp dự đoán (Predicted Class)", fontsize=11, fontweight="bold")
        plt.ylabel("Lớp thực tế (Ground Truth)", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_per_class_f1_ranking(
        per_class_data: Union[pd.DataFrame, Dict[str, float]],
        output_path: str,
        title: str = "Per-Class F1-Score Ranking (23 Classes)",
    ):
        """Vẽ bảng xếp hạng điểm F1 của 23 lớp có tô màu phân bậc và vạch ngưỡng."""
        if isinstance(per_class_data, dict):
            df = pd.DataFrame(
                list(per_class_data.items()), columns=["Class_Name", "F1_Score"]
            )
        else:
            df = per_class_data.rename(
                columns={"Class_Name": "Class_Name", "F1-Score (%)": "F1_Score"}
            )

        df = df.sort_values(by="F1_Score", ascending=True)

        plt.figure(figsize=(12, 10))
        colors = [
            "#27ae60" if score >= 93.0 else "#f39c12" if score >= 90.0 else "#e74c3c"
            for score in df["F1_Score"]
        ]
        bars = plt.barh(
            df["Class_Name"], df["F1_Score"], color=colors, edgecolor="black", lw=0.8
        )
        plt.axvline(
            90.0,
            color="blue",
            linestyle="--",
            lw=1.5,
            label="Ngưỡng chuẩn y tế (90.0%)",
        )

        for bar in bars:
            w = bar.get_width()
            plt.annotate(
                f"{w:.1f}%",
                (w + 0.4, bar.get_y() + bar.get_height() / 2),
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.title(title, fontsize=12.5, fontweight="bold", pad=12)
        plt.xlabel("F1-Score (%)", fontsize=11, fontweight="bold")
        plt.xlim(80, 102)
        plt.legend(loc="lower right")

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close()
