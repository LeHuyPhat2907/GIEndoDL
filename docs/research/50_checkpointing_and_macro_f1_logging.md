# 💾 Báo cáo Kỹ thuật: Thiết Lập Hệ Thống Checkpointing & Logging Theo Macro F1

> **File cấu hình:** `configs/checkpointing_policy.json` | **Hình minh họa:** `docs/figures/42_checkpointing_and_logging_system.png`

---

## 1. Lý Do Chọn `val_macro_f1` Làm Tiêu Chí Quyết Định Lưu Trọng Số

Trong bài toán nội soi tiêu hóa đuôi dài (Long-tailed GI Endoscopy), Accuracy là một thước đo đánh lừa: mô hình có thể phớt lờ hoàn toàn tổn thương hiếm mà vẫn đạt độ chính xác 88.5%. Do đó, `ComprehensiveCheckpointManager` chỉ cập nhật file `best_model.pth` khi chỉ số Macro F1 (trung bình cộng F1 của 23 lớp) đạt giá trị cao nhất.

---

## 2. Cấu Trúc Bộ Quản Lý Trọng Số Kép (Dual Checkpoints)

- **`best_model.pth`:** Chứa trọng số ở thời điểm mô hình đạt độ nhạy cân bằng tốt nhất giữa các tổn thương, dùng cho nghiệm thu lâm sàng.
- **`last_model.pth`:** Chứa trạng thái đầy đủ (Model + Optimizer + Scaler) của Epoch gần nhất, cho phép tiếp tục huấn luyện ngay lập tức nếu phiên đám mây bị ngắt quãng.
- **`training_history.csv`:** Lưu trữ minh bạch toàn bộ các đường cong mất mát và điểm số phục vụ vẽ đồ thị công bố.
