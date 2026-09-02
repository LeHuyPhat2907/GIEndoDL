# 🏗️ Báo cáo Kỹ thuật: Thiết Kế Khung Huấn Luyện Học Sâu Đa Năng (Modular Training Framework)

> **File cấu hình:** `configs/training_base_config.json` | **Hình minh họa:** `docs/figures/41_modular_training_framework_architecture.png`

---

## 1. Kiến Trúc Lõi Đa Năng (Modular Engine)

Lớp `ModularTrainer` trong `src/training/trainer.py` được thiết kế theo mẫu thiết kế Adapter Pattern, cho phép tích hợp độc lập:
- **Backbone Backends:** Hỗ trợ mọi kiến trúc từ PyTorch Native và thư viện `timm`.
- **Hàm mất mát tùy biến:** Tương thích hoàn toàn với `ClassBalancedFocalLoss` và `LabelSmoothing`.
- **Lập lịch học tập:** Tự động điều tiết tốc độ học theo chu kỳ Cosine Annealing.

---

## 2. Các Cơ Chế Kiểm Soát Tự Động (Callbacks)

1. **Early Stopping:** Liên tục giám sát chỉ số `val_macro_f1` (chỉ số quan trọng nhất cho tập dữ liệu mất cân bằng). Nếu sau 7 Epochs liên tiếp mô hình không có sự cải thiện, phiên huấn luyện sẽ tự động kết thúc.
2. **Model Checkpointing:** Luôn bảo lưu trọng số ở Epoch đạt điểm số cao nhất, ngăn ngừa rủi ro mô hình bị quá khớp (Overfitting) ở các Epochs sau cùng.
