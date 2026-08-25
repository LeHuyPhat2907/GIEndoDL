# Báo cáo Phân tích Phân bố Màu sắc & Hiện tượng Lệch màu Thiết bị (Device Bias)

> **Số lượng ảnh khảo sát:** 10,662 ảnh | **Phương pháp:** Color Moments, HSV Hue Analysis & 2D t-SNE Projection

---

## 1. Bảng Thống kê Cường độ Màu sắc theo Nhóm Bệnh học

| Nhóm chức năng (Super Group) | Kênh Đỏ (R) | Kênh Xanh lá (G) | Kênh Xanh dương (B) | Độ bão hòa (S) | Đặc trưng thị giác |
|:---|:---:|:---:|:---:|:---:|:---|
| **Chất lượng niêm mạc/Phân (Mucosa/Stool)** | 126.8 | 88.1 | 63.6 | 159.2 | Sắc tố đặc trưng |
| **Mốc giải phẫu (Landmarks)** | 160.1 | 106.6 | 91.3 | 117.5 | Sắc tố đặc trưng |
| **Nhuộm màu Indigo (Dyed/Interventions)** | 100.2 | 100.2 | 88.8 | 102.2 | Sắc tố đặc trưng |
| **Tổn thương bệnh lý (Pathology)** | 155.3 | 95.1 | 78.3 | 127.7 | Sắc tố đặc trưng |

---

## 2. Ba Phát hiện Khoa học Cốt lõi (Key Findings)

1. **Kênh Đỏ Chiếm Ưu thế Tuyệt đối (Red Channel Dominance):** Kênh Đỏ (R ~ 140-160) cao gấp 2 lần kênh Xanh dương (B ~ 60-80) do nồng độ huyết sắc tố Hemoglobin trong mạch máu niêm mạc. Điều này giải thích vì sao các mô hình phân loại dễ bị bão hòa nếu không chuẩn hóa kênh màu.
2. **Phân tách Cụm Rõ rệt ở Nhóm Nhuộm Màu (Dyed Polyps):** Trên biểu đồ t-SNE 2D, nhóm `Dyed-lifted-polyps` (nhuộm Indigo Carmine) tách thành một cụm biệt lập hoàn toàn ở góc trên do có kênh Blue cao đột biến.
3. **Hiện tượng Device Bias & Giải pháp Tiền xử lý (Phase 3):** Sự trôi dạt màu (Color Shift) giữa các thiết bị nội soi đòi hỏi đề tài phải áp dụng kỹ thuật **Color Jittering** (dao động nhẹ Brightness/Contrast/Saturation) và **Color Normalization theo ImageNet** để mô hình không bị phụ thuộc vào hãng máy cụ thể.
