# 🎨 Báo cáo Kỹ thuật: Hệ Thống Trực Quan Hóa Kết Quả Huấn Luyện Đạt Chuẩn 300 DPI

> **File cấu hình:** `configs/visualization_config.json` | **Hình tổng hợp:** `docs/figures/44_training_visualization_dashboard.png` | **Heatmap ma trận:** `docs/figures/45_full_confusion_matrix_heatmap.png`

---

## 1. Tiêu Chuẩn Trực Quan Hóa Tạp Chí Khoa Học (IEEE / Springer / Elsevier)

- **Độ phân giải 300 DPI:** Bảo đảm hình ảnh không bị vỡ hạt khi in ấn hoặc phóng to trên tài liệu PDF.
- **Phối màu y sinh học:** Sử dụng bảng màu `YlGnBu` cho Ma trận nhầm lẫn giúp người đọc nhận biết ngay lập tức mật độ dự đoán đúng trên đường chéo chính.
- **Tính minh bạch:** Các đường cong Loss và F1-Score thể hiện chi tiết từng Epoch, chứng minh mô hình hội tụ thực chất và không bị hiện tượng quá khớp.
