# Báo Cáo Kỹ Thuật: Tăng Cường Quang Học Sắc Thái Hẹp Bảo Toàn Hemoglobin (Narrow-Band Color Calibration)

> **Mã công việc:** Task D34
> **Tham số quy chuẩn:** `Brightness = ±0.10`, `Contrast = ±0.10`, `Saturation = ±0.10`, `Hue = ±0.015`.

---

## 1. Nguy Cơ Khi Tăng Cường Màu Quá Mức (Over-Color Augmentation)
- Việc điều chỉnh Hue vượt quá $\pm 0.05$ sẽ làm dịch chuyển phổ hấp thụ ánh sáng của Hemoglobin từ đỏ sang tím hoặc xanh lá cây. Điều này khiến mạng nơ-ron nhầm lẫn giữa ban đỏ xung huyết của viêm loét đại tràng (Mayo 2/3) với niêm mạc bình thường.
- Giới hạn biên độ trong khoảng hẹp $\pm 10\%$ giúp mô hình mô phỏng được sự biến thiên công suất chiếu sáng giữa các bóng đèn Xenon/LED của dây soi mà vẫn giữ nguyên màu sắc giải phẫu bệnh học.
