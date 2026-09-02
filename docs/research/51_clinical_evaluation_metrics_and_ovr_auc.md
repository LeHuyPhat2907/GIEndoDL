# 🩺 Báo cáo Kỹ thuật: Hệ Thống Đánh Giá Lâm Sàng Đa Chiều & OvR AUC-ROC

> **File cấu hình:** `configs/evaluation_metrics_report.json` | **Bảng dữ liệu:** `data/processed/per_class_evaluation_metrics.csv` | **Hình minh họa:** `docs/figures/43_clinical_evaluation_metrics_suite.png`

---

## 1. Ý Nghĩa Của Bộ Chỉ Số Đánh Giá Y Tế

Hệ thống đánh giá `ClinicalEvaluator` đo lường toàn diện các khía cạnh phân loại:
- **Sensitivity / Recall (Macro):** Đo lường tỷ lệ bệnh nhân có tổn thương thực tế được AI phát hiện kịp thời.
- **Precision (Macro):** Tránh hiện tượng báo động giả, giảm áp lực sinh thiết không cần thiết cho Bác sĩ.
- **One-vs-Rest AUC-ROC:** Đánh giá năng lực của vector Softmax ở mọi ngưỡng phân tách xác suất lâm sàng.

---

## 2. Kết Quả Đo Lường Tổng Hợp Trên Tập Kiểm Thử (Test Split: 1,600 ảnh)

- **Overall Accuracy:** `93.94%`
- **Macro Precision:** `93.82%`
- **Macro Recall (Độ nhạy):** `93.95%`
- **Macro F1-Score:** `93.84%`
- **Multi-Class OvR AUC-ROC:** `99.81%`
