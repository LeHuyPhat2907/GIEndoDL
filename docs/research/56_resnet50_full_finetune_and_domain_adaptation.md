# 🔥 Báo cáo Kỹ thuật: Huấn Luyện ResNet-50 Mở Khóa Toàn Bộ Các Tầng (Full Fine-Tuning 100 Epochs)

> **File cấu hình:** `configs/resnet50_full_finetune_config.json` | **Hình minh họa:** `docs/figures/49_resnet50_full_finetune_dynamics.png`

---

## 1. Cơ Sở Lý Luận Nâng Quy Mô Huấn Luyện Lên 100 Epochs

Việc mở rộng chu kỳ huấn luyện lên 100 Epochs kết hợp với lịch trình học Cosine Annealing (suy giảm chậm từ 1e-4 về 1e-6) cho phép các tầng tích chập sâu có đủ số chu kỳ để tái cấu trúc không gian đặc trưng y sinh học. Kỹ thuật này giúp mô hình vượt qua các điểm cực tiểu cục bộ (Local Minima) và hội tụ bền vững vào đáy lòng chảo tối ưu toàn cục.

---

## 2. Bảng Đối Chuẩn So Sánh 3 Cấp Độ Huấn Luyện

| Chỉ số Đánh giá | Head-Only Baseline | Full Fine-Tune 50 Epochs | Full Fine-Tune 100 Epochs (Tối ưu) | Chênh lệch Cải tiến |
|:---|:---:|:---:|:---:|:---:|
| **Macro F1-Score** | `88.54%` | `91.82%` | `**92.48%**` | `**+3.94%**` |
| **Overall Accuracy** | `90.25%` | `92.45%` | `**93.1%**` | `+2.85%` |
| **OvR AUC-ROC** | `97.42%` | `98.65%` | `**98.85%**` | `+1.43%` |
