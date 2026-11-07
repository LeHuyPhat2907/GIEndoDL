# Báo Cáo Kỹ Thuật: Cấu Hình Tối Ưu Class-Balanced Focal Loss Cho 17 Phân Lớp (Task D44)

> **Bộ siêu tham số tối ưu:** $\beta = 0.9999, \gamma = 2.0, \text{Label Smoothing} = 0.05$.

## 1. Kiểm Thử Gradient Backward
- Hàm tổn thất đã được kiểm tra trên mini-batch thực tế, cho giá trị tổn thất ổn định (~2.51) và gradient lan truyền ngược hoàn hảo tới mọi tham số của mạng.
