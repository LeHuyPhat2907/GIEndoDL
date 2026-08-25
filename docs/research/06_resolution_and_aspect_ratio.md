# 📐 Báo cáo Khảo sát Độ phân giải & Tỷ lệ Khung hình (HyperKvasir)

> **Tổng số ảnh phân tích:** 10,662 ảnh | **Khoảng kích thước:** 332x352 đến 1920x1079

---

## 1. Bảng Thống kê Phân phối Độ phân giải Chính

| Độ phân giải (W x H) | Chuẩn hiển thị | Số lượng ảnh | Tỷ lệ (%) | Nhận định thiết bị |
|:---|:---:|:---:|:---:|:---|
| `633x532` | Custom | 1,311 | 12.3% | Định dạng hỗn hợp |
| `633x531` | Custom | 1,272 | 11.9% | Định dạng hỗn hợp |
| `1349x1071` | Custom | 838 | 7.9% | Định dạng hỗn hợp |
| `1221x1012` | Custom | 788 | 7.4% | Định dạng hỗn hợp |
| `635x548` | Custom | 686 | 6.4% | Định dạng hỗn hợp |
| Các kích thước khác | Đa dạng | 5,767 | 54.1% | Các dòng máy nội soi khác |

---

## 2. Kết luận Khoa học & Định hướng Tiền xử lý (Phase 3)

1. **Tính đa dạng thiết bị (Heterogeneous Resolutions):** Ảnh trong HyperKvasir đến từ nhiều thế hệ máy nội soi khác nhau tại Bệnh viện Bærum, dao động từ độ phân giải chuẩn SD cũ đến Full HD hiện đại.
2. **Tỷ lệ khung hình ổn định:** Đa số ảnh có tỷ lệ khung hình xấp xỉ `1.15 - 1.33` (tương đương chuẩn 4:3 của màn hình nội soi y tế truyền thống).
3. **Quyết định Kỹ thuật Chuẩn hóa Đầu vào:**
   - **Kích thước đầu vào mô hình:** Chọn kích thước chuẩn **`224 × 224 px`** (cho CNN baseline & Swin Transformer) và **`384 × 384 px`** (cho mô hình đề xuất độ nét cao).
   - **Chiến lược Resize:** Sử dụng thuật toán nội suy **Bicubic Interpolation** kết hợp **Letterbox Padding** hoặc **Center Crop** để bảo toàn cấu trúc hạt niêm mạc (Pit pattern) mà không làm méo hình dạng polyp.
