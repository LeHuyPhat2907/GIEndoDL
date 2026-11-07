# Báo Cáo Lý Thuyết: Điều Hòa Hiệu Chuẩn Bằng Label Smoothing (Task D44)

> **Tham số:** $\epsilon = 0.05$.

## 1. Giảm Thiểu Tự Tin Thái Quá (Overconfidence Calibration)
- Trong nội soi tiêu hóa, các tổn thương viêm nhẹ luôn tồn tại sự phân vân lâm sàng giữa các bác sĩ. Label Smoothing $\epsilon = 0.05$ phân bổ 5% xác suất đều cho các lớp khác, ngăn chặn trọng số lớp cuối cùng tăng vọt tới vô cực, giúp phân phối xác suất đầu ra có độ tin cậy lâm sàng cao.
