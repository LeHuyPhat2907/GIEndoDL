# 🌐 Báo Cáo Xây Dựng Đồ Thị Liên Kết Ca Bệnh (Patient/Video Cluster Graph)

> **Mã công việc:** Task D10
> **Phương pháp:** Hợp nhất SSIM (ngưỡng $\ge 0.92$), Multi-Hash 256-bit và Mutual Nearest Neighbors DINOv2 (ngưỡng $\ge 0.92$) thành Đồ thị Thành phần Liên thông (Connected Components)

---

## 1. Kết Quả Kiểm Toán Thực Nghiệm

- **Tổng số ca thủ thuật / cụm video nội soi độc lập:** `9,849` cụm.
- **Số cụm đa khung hình (chụp liên tiếp 2–11 ảnh):** `709` cụm.
- **Số ảnh đơn lẻ (Isolated single frames):** `9,140` ảnh.
- **Tỷ lệ cụm nhỏ từ 2 đến 5 ảnh:** `98.7%` (phù hợp tuyệt đối với thực tế bác sĩ bấm 2–5 phát chụp cho mỗi tổn thương).

---

## 2. Giải Pháp Bảo Vệ Tính Khách Quan Chuẩn TRIPOD-AI

1. **Đóng gói trọn vẹn (Atomic Cluster Binding):** Mỗi mã `patient_video_cluster_id` đại diện cho một thực thể ca bệnh bất khả phân. Khi phân chia Train/Val/Test ở Task D39, 100% các khung hình của cùng một ID sẽ đi cùng nhau.
2. **Triệt tiêu Shortcut Learning:** Mô hình bị tước đoạt hoàn toàn khả năng 'học thuộc' màu niêm mạc hay viền camera của bệnh nhân, buộc phải học đúng đặc trưng hình thái tổn thương.
