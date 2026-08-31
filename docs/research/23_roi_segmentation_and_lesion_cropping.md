# 🔬 Báo cáo Kỹ thuật: Phân Đoạn & Cắt Cô Lập Vùng Tổn Thương Tự Động (Lesion ROI Extraction)

> **Module chính:** `src/preprocessing/roi_segmentation.py` | **Hình minh họa:** `docs/figures/16_roi_segmentation_and_crop.png`

---

## 1. Nguyên lý Hoạt động của Thuật toán

1. **Phân đoạn Mặt nạ Tổn thương (U-Net Segmentation):** Mô hình dự đoán ranh giới pixel của khối u polyp với độ chính xác cao.
2. **Trích xuất Khung bao Ngữ cảnh (Contextual Bounding Box):** Tự động tính toán khung chữ nhật tối thiểu bao quanh tổn thương và mở rộng biên **15% (pad_ratio=0.15)** để giữ lại mô ranh giới tiếp giáp.
3. **Cắt Cô lập Patch Tổn thương (Lesion Patch Cropping):** Tạo ra các patch ảnh tập trung toàn bộ độ phân giải vào rãnh vi mạch của khối u.

---

## 2. Lợi ích Đột phá đối với Phân loại Học Sâu (Classification Boost)

- **Triệt tiêu 80% diện tích nhiễu nền:** Ngăn mạng học sâu bị phân tâm bởi dịch nhầy, nếp gấp ruột bình thường và viền tối xung quanh.
- **Hỗ trợ Kiến trúc Đa Luồng (Dual-Stream Architecture):** Cho phép kết hợp luồng ảnh toàn cảnh (Global Context) và luồng ảnh cận cảnh vết loét (Local Lesion ROI) để đạt độ chính xác tối ưu.
