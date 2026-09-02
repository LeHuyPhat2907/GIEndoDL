"""Module đánh giá hiệu năng mô hình học sâu nội soi tiêu hóa toàn diện (Clinical Evaluation Suite)."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClinicalEvaluator:
    """Bộ công cụ đánh giá đa chiều chuẩn y sinh học (Clinical AI Evaluation)."""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)

    def evaluate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Tính toán đồng thời toàn bộ các chỉ số Accuracy, Precision, Recall, F1, và AUC-ROC."""
        # 1. Các chỉ số tổng quan cơ sở
        acc = accuracy_score(y_true, y_pred) * 100.0

        # Precision (Macro, Micro, Weighted)
        p_macro = (
            precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        )
        p_micro = (
            precision_score(y_true, y_pred, average="micro", zero_division=0) * 100.0
        )
        p_weighted = (
            precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
        )

        # Recall / Sensitivity (Macro, Micro, Weighted)
        r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        r_micro = recall_score(y_true, y_pred, average="micro", zero_division=0) * 100.0
        r_weighted = (
            recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
        )

        # F1-Score (Macro, Micro, Weighted)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0) * 100.0
        f1_weighted = (
            f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
        )

        # 2. Multi-Class One-vs-Rest AUC-ROC
        auc_macro = None
        if y_prob is not None:
            try:
                auc_macro = (
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                    )
                    * 100.0
                )
            except ValueError:
                auc_macro = 0.0

        # 3. Ma trận nhầm lẫn Confusion Matrix
        cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        cm_norm = confusion_matrix(
            y_true, y_pred, labels=list(range(self.num_classes)), normalize="true"
        )

        # 4. Chi tiết từng lớp bệnh (Classification Report)
        cls_rep = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        results = {
            "overall_accuracy": round(float(acc), 2),
            "precision": {
                "macro": round(float(p_macro), 2),
                "micro": round(float(p_micro), 2),
                "weighted": round(float(p_weighted), 2),
            },
            "recall": {
                "macro": round(float(r_macro), 2),
                "micro": round(float(r_micro), 2),
                "weighted": round(float(r_weighted), 2),
            },
            "f1_score": {
                "macro": round(float(f1_macro), 2),
                "micro": round(float(f1_micro), 2),
                "weighted": round(float(f1_weighted), 2),
            },
            "auc_roc_ovr_macro": round(float(auc_macro), 2)
            if auc_macro is not None
            else None,
            "confusion_matrix_raw": cm_raw.tolist(),
            "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
            "classification_report_per_class": cls_rep,
        }

        return results

    def export_per_class_csv(self, results: Dict[str, Any], output_csv_path: str):
        """Xuất bảng chi tiết từng lớp bệnh ra file CSV phục vụ chèn vào Luận văn."""
        cls_rep = results["classification_report_per_class"]
        records = []

        for name in self.class_names:
            if name in cls_rep:
                d = cls_rep[name]
                records.append(
                    {
                        "Class_Name": name,
                        "Precision (%)": round(d["precision"] * 100.0, 1),
                        "Recall (%)": round(d["recall"] * 100.0, 1),
                        "F1-Score (%)": round(d["f1-score"] * 100.0, 1),
                        "Support (Số mẫu)": int(d["support"]),
                    }
                )

        df_out = pd.DataFrame(records)
        df_out.to_csv(output_csv_path, index=False)
        print(f"✅ Đã lưu bảng phân tích Per-Class CSV tại: {output_csv_path}")
