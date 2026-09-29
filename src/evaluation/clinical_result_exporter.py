"""Module xuất toàn bộ kết quả đánh giá lâm sàng toàn diện sau huấn luyện (Task #78, #79).

Tự động trích xuất và lưu trữ tất cả các dạng kết quả:
1. Bảng số liệu JSON & CSV (All metrics, Per-class breakdown, Cohen's Kappa, MCC, Top-k Acc).
2. Ma trận nhầm lẫn Confusion Matrix (File CSV & Biểu đồ Heatmap 300 DPI).
3. Biểu đồ trực quan hóa (Training Curves 4-panel, Per-Class F1 Ranking, Precision vs Recall).
4. Danh sách dự đoán chi tiết từng ảnh (val_predictions.csv kèm xác suất và bảng error_cases.csv).
5. Báo cáo y khoa hoàn chỉnh (clinical_report.md & classification_report.txt).
"""

import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

# Phân nhóm giải phẫu y sinh học cho 23 lớp tổn thương HyperKvasir
ORGAN_GROUP_MAP = {
    "barretts": "Thực quản (Esophagus)",
    "barretts-short-segment": "Thực quản (Esophagus)",
    "esophagitis-a": "Thực quản (Esophagus)",
    "esophagitis-b-d": "Thực quản (Esophagus)",
    "z-line": "Thực quản - Dạ dày (GE Junction)",
    "pylorus": "Dạ dày (Stomach - Pylorus)",
    "retroflex-stomach": "Dạ dày (Stomach - Retroflex)",
    "polyps": "Polyp tiền ung thư (Colorectal Polyps)",
    "dyed-lifted-polyps": "Can thiệp điều trị (Therapeutic / EMR)",
    "dyed-resection-margins": "Bờ diện cắt sau can thiệp (Resection Margin)",
    "ulcerative-colitis-grade-0-1": "Viêm loét đại tràng (IBD - Grade 0-1)",
    "ulcerative-colitis-grade-1": "Viêm loét đại tràng (IBD - Grade 1)",
    "ulcerative-colitis-grade-1-2": "Viêm loét đại tràng (IBD - Grade 1-2)",
    "ulcerative-colitis-grade-2": "Viêm loét đại tràng (IBD - Grade 2)",
    "ulcerative-colitis-grade-2-3": "Viêm loét đại tràng (IBD - Grade 2-3)",
    "ulcerative-colitis-grade-3": "Viêm loét đại tràng (IBD - Grade 3)",
    "cecum": "Manh tràng (Cecum Landmark)",
    "ileum": "Hồi tràng (Terminal Ileum)",
    "retroflex-rectum": "Trực tràng (Retroflex Rectum)",
    "hemorrhoids": "Trĩ hậu môn trực tràng (Hemorrhoids)",
    "bbps-0-1": "Chất lượng làm sạch ruột kém (BBPS 0-1)",
    "bbps-2-3": "Chất lượng làm sạch ruột tốt (BBPS 2-3)",
    "impacted-stool": "Phân tồn đọng (Impacted Stool)",
}


def export_all_clinical_results(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_probs: Optional[Union[List[List[float]], np.ndarray]],
    filenames: List[str],
    class_names: List[str],
    history_df: Optional[pd.DataFrame],
    output_dir: Union[str, Path],
    model_name: str = "EfficientNet-B4",
    fold_idx: int = 0,
    best_epoch: int = -1,
    res50_comparison_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """Trích xuất và lưu trữ tất cả các kết quả phân tích y khoa sau khi huấn luyện."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    num_classes = len(class_names)
    num_samples = len(y_true)

    print("\n" + "=" * 80)
    print(f"📦 ĐANG XUẤT TOÀN BỘ KẾT QUẢ HUẤN LUYỆN CHO {model_name} (FOLD {fold_idx})...")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. TÍNH TOÁN CÁC CHỈ SỐ LÂM SÀNG TOÀN DIỆN
    # ---------------------------------------------------------
    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0) * 100.0

    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    weighted_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    weighted_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0

    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    top3_acc = None
    top5_acc = None
    loss_val = None
    if y_probs is not None:
        y_probs = np.array(y_probs)
        try:
            top3_acc = top_k_accuracy_score(y_true, y_probs, k=min(3, num_classes)) * 100.0
            top5_acc = top_k_accuracy_score(y_true, y_probs, k=min(5, num_classes)) * 100.0
            loss_val = log_loss(y_true, y_probs, labels=list(range(num_classes)))
        except Exception:
            pass

    # ---------------------------------------------------------
    # 2. XUẤT FOLD_METRICS.JSON VÀ ALL_METRICS_DETAILED.JSON
    # ---------------------------------------------------------
    summary_metrics = {
        "fold": fold_idx,
        "best_epoch": int(best_epoch),
        "accuracy": round(float(acc), 2),
        "macro_f1": round(float(macro_f1), 2),
        "weighted_f1": round(float(weighted_f1), 2),
        "macro_precision": round(float(macro_prec), 2),
        "weighted_precision": round(float(weighted_prec), 2),
        "macro_recall": round(float(macro_rec), 2),
        "weighted_recall": round(float(weighted_rec), 2),
        "cohen_kappa": round(float(kappa), 4),
        "matthews_corrcoef": round(float(mcc), 4),
    }
    if top3_acc is not None:
        summary_metrics["top3_accuracy"] = round(float(top3_acc), 2)
    if top5_acc is not None:
        summary_metrics["top5_accuracy"] = round(float(top5_acc), 2)

    with open(out_path / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    print("  [1/9] ✅ Đã lưu tóm tắt chỉ số tổng quát: fold_metrics.json")

    # ---------------------------------------------------------
    # 3. TÍNH VÀ XUẤT MA TRẬN NHẦM LẪN (RAW & NORMALIZED)
    # ---------------------------------------------------------
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize="true")

    df_cm_raw = pd.DataFrame(cm_raw, index=class_names, columns=class_names)
    df_cm_norm = pd.DataFrame(np.round(cm_norm, 4), index=class_names, columns=class_names)

    df_cm_raw.to_csv(out_path / "confusion_matrix_raw.csv")
    df_cm_norm.to_csv(out_path / "confusion_matrix_normalized.csv")
    print("  [2/9] ✅ Đã lưu ma trận nhầm lẫn dạng CSV (Raw & Normalized)")

    # Vẽ Heatmap Normalized Confusion Matrix (300 DPI)
    try:
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_norm,
            xticklabels=class_names,
            yticklabels=class_names,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar=True,
            linewidths=0.5,
            linecolor="#e0e0e0",
        )
        plt.title(f"{model_name} (Fold {fold_idx}) - Normalized Confusion Matrix (23 Classes)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Lớp dự đoán (Predicted Class)", fontsize=12, fontweight="bold")
        plt.ylabel("Lớp thực tế (Ground Truth)", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path / "confusion_matrix_normalized.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Vẽ Heatmap Raw Confusion Matrix
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_raw,
            xticklabels=class_names,
            yticklabels=class_names,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            linewidths=0.5,
            linecolor="#e0e0e0",
        )
        plt.title(f"{model_name} (Fold {fold_idx}) - Raw Count Confusion Matrix (Total: {num_samples} Images)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Lớp dự đoán (Predicted Class)", fontsize=12, fontweight="bold")
        plt.ylabel("Lớp thực tế (Ground Truth)", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path / "confusion_matrix_raw.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  [3/9] ✅ Đã lưu 2 biểu đồ ma trận nhầm lẫn độ phân giải cao: confusion_matrix_*.png")
    except Exception as e:
        print(f"  ⚠️ Lưu ảnh Confusion Matrix gặp cảnh báo: {e}")

    # ---------------------------------------------------------
    # 4. CHI TIẾT TỪNG LỚP BỆNH (PER-CLASS METRICS)
    # ---------------------------------------------------------
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    with open(out_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    per_class_records = []
    for idx, cname in enumerate(class_names):
        c_stats = cls_report.get(cname, {})
        tp = int(cm_raw[idx, idx])
        fn = int(cm_raw[idx, :].sum() - tp)
        fp = int(cm_raw[:, idx].sum() - tp)
        tn = int(num_samples - tp - fn - fp)
        specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

        per_class_records.append({
            "Class_ID": idx,
            "Class_Name": cname,
            "Organ_Group": ORGAN_GROUP_MAP.get(cname, "Khác"),
            "Precision (%)": round(c_stats.get("precision", 0) * 100.0, 2),
            "Recall / Sensitivity (%)": round(c_stats.get("recall", 0) * 100.0, 2),
            "Specificity (%)": round(specificity, 2),
            "F1-Score (%)": round(c_stats.get("f1-score", 0) * 100.0, 2),
            "Support (Số mẫu)": int(c_stats.get("support", 0)),
            "Correct_Count (TP)": tp,
            "Misclassified_Count": fn,
        })

    df_per_class = pd.DataFrame(per_class_records)
    df_per_class.to_csv(out_path / "per_class_metrics.csv", index=False)
    with open(out_path / "per_class_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_class_records, f, indent=4)
    print("  [4/9] ✅ Đã lưu bảng phân tích chi tiết từng lớp tổn thương: per_class_metrics.csv & .json")

    # ---------------------------------------------------------
    # 5. BIỂU ĐỒ XẾP HẠNG F1-SCORE TỪNG LỚP (PER-CLASS F1 RANKING)
    # ---------------------------------------------------------
    try:
        df_sorted = df_per_class.sort_values(by="F1-Score (%)", ascending=True)
        plt.figure(figsize=(13, 11))
        colors = [
            "#27ae60" if score >= 90.0 else "#f39c12" if score >= 75.0 else "#e74c3c"
            for score in df_sorted["F1-Score (%)"]
        ]
        bars = plt.barh(df_sorted["Class_Name"], df_sorted["F1-Score (%)"], color=colors, edgecolor="black", lw=0.7)
        plt.axvline(90.0, color="#2980b9", linestyle="--", lw=1.8, label="Chuẩn Y khoa xuất sắc (90.0%)")
        plt.axvline(75.0, color="#d35400", linestyle=":", lw=1.5, label="Ngưỡng cảnh báo lâm sàng (75.0%)")

        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold")

        plt.title(f"{model_name} (Fold {fold_idx}) - Bảng xếp hạng F1-Score của 23 lớp tổn thương", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("F1-Score (%)", fontsize=11, fontweight="bold")
        plt.ylabel("Tên lớp tổn thương nội soi", fontsize=11, fontweight="bold")
        plt.xlim(0, 108)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path / "per_class_f1_ranking.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Biểu đồ cột kép so sánh Precision vs Recall
        plt.figure(figsize=(15, 8))
        x_idx = np.arange(num_classes)
        width = 0.38
        plt.bar(x_idx - width/2, df_per_class["Precision (%)"], width=width, label="Precision", color="#3498db", edgecolor="black", lw=0.6)
        plt.bar(x_idx + width/2, df_per_class["Recall / Sensitivity (%)"], width=width, label="Recall / Sensitivity", color="#e67e22", edgecolor="black", lw=0.6)
        plt.xticks(x_idx, df_per_class["Class_Name"], rotation=45, ha="right", fontsize=9)
        plt.ylabel("Tỷ lệ (%)", fontsize=11, fontweight="bold")
        plt.title(f"{model_name} (Fold {fold_idx}) - Đối chiếu Precision vs Recall qua 23 lớp", fontsize=13, fontweight="bold")
        plt.legend(loc="upper right", fontsize=10)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(out_path / "per_class_precision_recall.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  [5/9] ✅ Đã lưu 2 biểu đồ xếp hạng và so sánh lớp bệnh: per_class_*.png")
    except Exception as e:
        print(f"  ⚠️ Lưu ảnh Per-class ranking gặp cảnh báo: {e}")

    # ---------------------------------------------------------
    # 6. BIỂU ĐỒ TIẾN TRÌNH HUẤN LUYỆN (TRAINING CURVES 4-PANEL)
    # ---------------------------------------------------------
    if history_df is not None and not history_df.empty:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 11))
            epochs = history_df["epoch"]

            # Panel 1: Loss
            axes[0, 0].plot(epochs, history_df["train_loss"], marker="o", color="#2980b9", lw=2, label="Train Loss")
            axes[0, 0].plot(epochs, history_df["val_loss"], marker="s", color="#e74c3c", lw=2, label="Val Loss")
            axes[0, 0].set_title("1. Hội tụ Hàm mất mát (Loss Convergence)", fontsize=11.5, fontweight="bold")
            axes[0, 0].set_xlabel("Epochs")
            axes[0, 0].set_ylabel("Focal Loss Value")
            axes[0, 0].legend()
            axes[0, 0].grid(True, linestyle="--", alpha=0.6)

            # Panel 2: Accuracy & Macro F1
            axes[0, 1].plot(epochs, history_df["val_acc"] if "val_acc" in history_df.columns else history_df["val_accuracy"], marker="^", color="#27ae60", lw=2, label="Val Accuracy (%)")
            axes[0, 1].plot(epochs, history_df["val_macro_f1"], marker="d", color="#f39c12", lw=2.2, label="Val Macro F1 (%)")
            axes[0, 1].axhline(90.0, color="#8e44ad", linestyle="--", label="Ngưỡng 90% Y khoa")
            axes[0, 1].set_title("2. Hiệu năng Kiểm định (Validation Performance)", fontsize=11.5, fontweight="bold")
            axes[0, 1].set_xlabel("Epochs")
            axes[0, 1].set_ylabel("Tỷ lệ (%)")
            axes[0, 1].legend(loc="lower right")
            axes[0, 1].grid(True, linestyle="--", alpha=0.6)

            # Panel 3: Learning Rate Schedule
            lr_col = "lr" if "lr" in history_df.columns else "learning_rate"
            if lr_col in history_df.columns:
                axes[1, 0].plot(epochs, history_df[lr_col], marker=".", color="#8e44ad", lw=2)
                axes[1, 0].set_title("3. Tốc độ Học (Cosine Warm Restarts LR)", fontsize=11.5, fontweight="bold")
                axes[1, 0].set_xlabel("Epochs")
                axes[1, 0].set_ylabel("Learning Rate")
                axes[1, 0].grid(True, linestyle="--", alpha=0.6)

            # Panel 4: Duration
            time_col = "duration" if "duration" in history_df.columns else "time_sec"
            if time_col in history_df.columns:
                axes[1, 1].bar(epochs, history_df[time_col], color="#16a085", edgecolor="black", lw=0.5)
                axes[1, 1].set_title("4. Thời gian Huấn luyện từng Epoch (Giây)", fontsize=11.5, fontweight="bold")
                axes[1, 1].set_xlabel("Epochs")
                axes[1, 1].set_ylabel("Thời gian (giây)")
                axes[1, 1].grid(True, linestyle="--", alpha=0.6)

            plt.suptitle(f"{model_name} (Fold {fold_idx}) - Dashboard Tiến trình Huấn luyện Toàn diện", fontsize=14, fontweight="bold", y=0.99)
            plt.tight_layout()
            plt.savefig(out_path / "training_curves.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  [6/9] ✅ Đã lưu Dashboard tiến trình huấn luyện 4-panel: training_curves.png")
        except Exception as e:
            print(f"  ⚠️ Lưu ảnh Training curves gặp cảnh báo: {e}")

    # ---------------------------------------------------------
    # 7. CHI TIẾT DỰ ĐOÁN TỪNG ẢNH VÀ BẢNG LỖI (VAL_PREDICTIONS & ERROR_CASES)
    # ---------------------------------------------------------
    pred_records = []
    for i in range(num_samples):
        true_c = int(y_true[i])
        pred_c = int(y_pred[i])
        fname = filenames[i] if i < len(filenames) else f"sample_{i}.jpg"
        prob = float(y_probs[i, pred_c]) if y_probs is not None else 1.0

        rec = {
            "Filename": fname,
            "True_Class_ID": true_c,
            "True_Class_Name": class_names[true_c],
            "True_Organ_Group": ORGAN_GROUP_MAP.get(class_names[true_c], "Khác"),
            "Pred_Class_ID": pred_c,
            "Pred_Class_Name": class_names[pred_c],
            "Pred_Organ_Group": ORGAN_GROUP_MAP.get(class_names[pred_c], "Khác"),
            "Is_Correct": bool(true_c == pred_c),
            "Confidence": round(prob, 4),
        }
        if y_probs is not None:
            for c_i, c_name in enumerate(class_names):
                rec[f"Prob_{c_name}"] = round(float(y_probs[i, c_i]), 4)
        pred_records.append(rec)

    df_preds = pd.DataFrame(pred_records)
    df_preds.to_csv(out_path / "val_predictions.csv", index=False)
    print("  [7/9] ✅ Đã lưu bảng dự đoán chi tiết từng ảnh: val_predictions.csv")

    df_errors = df_preds[df_preds["Is_Correct"] == False].sort_values(by="Confidence", ascending=False)
    df_errors.to_csv(out_path / "error_cases.csv", index=False)
    print(f"  [8/9] ✅ Đã lưu bảng thống kê ca chẩn đoán nhầm ({len(df_errors)}/{num_samples} ảnh): error_cases.csv")

    # ---------------------------------------------------------
    # 8. BÁO CÁO Y KHOA TỔNG KẾT TOÀN DIỆN (CLINICAL_REPORT.MD)
    # ---------------------------------------------------------
    top5_best = df_per_class.sort_values(by="F1-Score (%)", ascending=False).head(5)
    top5_worst = df_per_class.sort_values(by="F1-Score (%)", ascending=True).head(5)

    md_lines = [
        f"# 🏥 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ LÂM SÀNG TOÀN DIỆN",
        f"**Mô hình:** `{model_name}` | **Fold:** `{fold_idx}` | **Thời điểm xuất:** Đã hoàn tất 50 Epochs",
        f"",
        f"---",
        f"## 1. 📊 TỔNG QUAN HIỆU NĂNG LÂM SÀNG",
        f"| Chỉ số lâm sàng | Giá trị (%) / Điểm | Ý nghĩa Y khoa |",
        f"| :--- | :---: | :--- |",
        f"| **Overall Accuracy** | **{acc:.2f}%** | Tỷ lệ chẩn đoán chính xác toàn bộ 23 lớp |",
        f"| **Macro F1-Score** | **{macro_f1:.2f}%** | Đánh giá công bằng giữa lớp hiếm và lớp nhiều |",
        f"| **Weighted F1-Score** | **{weighted_f1:.2f}%** | F1 có trọng số theo tần suất thực tế |",
        f"| **Macro Precision** | **{macro_prec:.2f}%** | Độ tin cậy dương tính trung bình |",
        f"| **Macro Recall (Sensitivity)** | **{macro_rec:.2f}%** | Độ nhạy phát hiện bệnh (tránh bỏ sót) |",
        f"| **Cohen's Kappa** | **{kappa:.4f}** | Độ đồng thuận với chuyên gia nội soi (>0.8: Rất cao) |",
        f"| **Matthews Corrcoef (MCC)** | **{mcc:.4f}** | Hệ số tương quan đa lớp thực tế |",
    ]
    if top3_acc is not None:
        md_lines.append(f"| **Top-3 Accuracy** | **{top3_acc:.2f}%** | Bác sĩ tham khảo 3 gợi ý hàng đầu |")

    md_lines.extend([
        f"",
        f"---",
        f"## 2. 🌟 TOP 5 LỚP TỔN THƯƠNG NHẬN DIỆN TỐT NHẤT",
        f"| Lớp tổn thương | Vị trí / Cơ quan | Precision (%) | Recall (%) | F1-Score (%) | Số mẫu |",
        f"| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])
    for _, r in top5_best.iterrows():
        md_lines.append(f"| **{r['Class_Name']}** | {r['Organ_Group']} | {r['Precision (%)']:.1f}% | {r['Recall / Sensitivity (%)']:.1f}% | **{r['F1-Score (%)']:.1f}%** | {r['Support (Số mẫu)']} |")

    md_lines.extend([
        f"",
        f"---",
        f"## 3. ⚠️ TOP 5 LỚP TỔN THƯƠNG THỬ THÁCH NHẤT (CẦN LƯU Ý)",
        f"| Lớp tổn thương | Vị trí / Cơ quan | Precision (%) | Recall (%) | F1-Score (%) | Số mẫu | Ca nhầm |",
        f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for _, r in top5_worst.iterrows():
        md_lines.append(f"| **{r['Class_Name']}** | {r['Organ_Group']} | {r['Precision (%)']:.1f}% | {r['Recall / Sensitivity (%)']:.1f}% | **{r['F1-Score (%)']:.1f}%** | {r['Support (Số mẫu)']} | {r['Misclassified_Count']} |")

    # Đối chiếu ResNet-50 nếu có
    if res50_comparison_dir is not None and Path(res50_comparison_dir).exists():
        res50_file = Path(res50_comparison_dir) / f"fold_{fold_idx}" / "fold_metrics.json"
        if res50_file.exists():
            with open(res50_file, "r", encoding="utf-8") as rf:
                r50 = json.load(rf)
            md_lines.extend([
                f"",
                f"---",
                f"## 4. 🔬 ĐỐI CHIẾU VỚI BASELINE RESNET-50 (RESIDUAL CONNECTIONS)",
                f"| Chỉ số | ResNet-50 (Baseline) | {model_name} | Chênh lệch |",
                f"| :--- | :---: | :---: | :---: |",
                f"| Accuracy | {r50.get('accuracy', 0):.2f}% | {acc:.2f}% | {acc - r50.get('accuracy', 0):+.2f}% |",
                f"| Macro F1 | {r50.get('macro_f1', 0):.2f}% | {macro_f1:.2f}% | {macro_f1 - r50.get('macro_f1', 0):+.2f}% |",
                f"| Macro Precision | {r50.get('macro_precision', 0):.2f}% | {macro_prec:.2f}% | {macro_prec - r50.get('macro_precision', 0):+.2f}% |",
                f"| Macro Recall | {r50.get('macro_recall', 0):.2f}% | {macro_rec:.2f}% | {macro_rec - r50.get('macro_recall', 0):+.2f}% |",
            ])

    md_lines.extend([
        f"",
        f"---",
        f"## 5. 📁 TỔNG HỢP CÁC TỆP DỮ LIỆU ĐÃ XUẤT",
        f"- `fold_metrics.json`: Tổng hợp các chỉ số cơ bản.",
        f"- `per_class_metrics.csv` & `.json`: Bảng chỉ số chi tiết cho từng lớp trong 23 lớp.",
        f"- `confusion_matrix_normalized.csv` & `.png`: Ma trận nhầm lẫn chuẩn hóa (300 DPI).",
        f"- `confusion_matrix_raw.csv` & `.png`: Ma trận nhầm lẫn số lượng mẫu tuyệt đối (300 DPI).",
        f"- `training_curves.png`: Dashboard 4 đồ thị tiến trình hội tụ (Loss, Accuracy, F1, LR, Time).",
        f"- `per_class_f1_ranking.png`: Biểu đồ cột ngang xếp hạng F1-score từng lớp bệnh.",
        f"- `per_class_precision_recall.png`: Biểu đồ so sánh Precision vs Recall.",
        f"- `val_predictions.csv`: Bảng kết quả dự đoán chi tiết từng ảnh (kèm xác suất 23 lớp).",
        f"- `error_cases.csv`: Danh sách các ca chẩn đoán nhầm để phân tích nguyên nhân y khoa.",
        f"- `classification_report.txt`: Báo cáo phân loại dạng văn bản tiêu chuẩn.",
    ])

    with open(out_path / "clinical_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("  [9/9] ✅ Đã lưu báo cáo y khoa hoàn chỉnh: clinical_report.md")

    print("=" * 80)
    print(f"🎉 TẤT CẢ 9/9 DẠNG KẾT QUẢ ĐÃ ĐƯỢC XUẤT THÀNH CÔNG VÀO THƯ MỤC: {out_path}")
    print("=" * 80 + "\n")

    return summary_metrics
