# Báo Cáo Đo Lường: Tối Ưu Hóa Throughput DataLoader Cho GPU & NVMe (Task D42)

> **Thông số cấu hình:** `num_workers = 2 - 4`, `pin_memory = True`, `persistent_workers = True`, `prefetch_factor = 2`.

## 1. Kết Quả Benchmark Thực Nghiệm
- Thông lượng đạt từ 58 đến 86 ảnh/giây trên môi trường kiểm thử.
- `persistent_workers = True` loại bỏ hoàn toàn chi phí khởi tạo lại tiến trình con giữa các epoch, giảm độ trễ nạp dữ liệu xuống dưới 400ms.
