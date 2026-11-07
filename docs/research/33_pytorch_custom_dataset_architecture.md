# Báo Cáo Kỹ Thuật: Kiến Trúc Lớp PyTorch EndoMedicalDataset (Task D41)

> **Mục tiêu:** Tích hợp quy trình tiền xử lý quang học Phase 5 và tăng cường y sinh Phase 6 vào luồng nạp dữ liệu PyTorch song song.

## 1. Tính Năng Kỹ Thuật
- Nạp ảnh bằng OpenCV BGR -> RGB với cơ chế bắt lỗi an toàn (Zero-Tensor Fallback).
- Trả về Tensor chuẩn 3 kênh [3, 256, 256], nhãn số nguyên [0, 16], mã nhận dạng Image ID và cờ Bounding Box.
- Tương thích 100% với PyTorch multiprocessing spawn trên Windows và Linux/Kaggle.
