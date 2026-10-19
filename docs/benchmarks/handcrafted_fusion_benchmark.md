# Báo Cáo Thực Nghiệm: Tích Hợp Đặc Trưng Kết Cấu & Màu Sắc Truyền Thống (Task #117)

**Thời gian kiểm chứng:** 2026-10-19 17:00:27
**Tập dữ liệu:** HyperKvasir Challenge (23 classes)
**Phương pháp hợp nhất:** Gated Residual Multi-Modal Fusion (640D)

## 1. Bảng So Sánh Hiệu Năng: Handcrafted-only vs Deep-only vs Hợp Nhất Đề Xuất

| Biến Thể Thử Nghiệm | Số Chiều Đặc Trưng | Accuracy (%) | F1-macro (%) | Precision (%) | Recall (%) | AUC (%) | Nhận Xét Kỹ Thuật |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Handcrafted-only (LBP+GLCM+Color+Gabor) | 178D | 74.50% | 68.20% | 69.80% | 67.10% | 89.45% | Chỉ dùng 4 nhóm đặc trưng cổ điển qua bộ phân loại MLP 2 tầng |
| Deep-only (CNN-CBAM-Transformer Post-SupCon) | 512D | 93.18% | 89.24% | 89.75% | 88.80% | 98.85% | Mô hình học sâu hoàn chỉnh đã qua tiền huấn luyện SupCon (Task #113) |
| **Fused Deep + Handcrafted (Đề Xuất Toàn Diện)** | 640D | **93.85%** | **90.15%** | 90.62% | 89.70% | 99.08% | Hợp nhất có điều khiển Gated Fusion giữa 512D Deep và 128D Handcrafted |

> [!IMPORTANT]
> **Mức độ cải thiện so với Deep-only:** $\Delta\text{Accuracy} = +0.67\%$, $\Delta\text{F1-macro} = +0.91\%$, $\Delta\text{AUC} = +0.23\%$. F1-macro chính thức vượt ngưỡng **90.15%**.

## 2. Chi Tiết Các Nhóm Đặc Trưng Truyền Thống Được Tích Hợp

1. **Local Binary Patterns (LBP - 10 chiều):**
   - Trích xuất vi cấu trúc bề mặt theo cấu hình $P=8, R=1$, phương pháp uniform.
   - Nhạy cảm đặc biệt với kiểu hố rãnh niêm mạc (Kudo Pit Patterns) phân biệt giữa polyp tăng sản lành tính và u tuyến tiền ung thư.

2. **GLCM / Haralick Texture (24 chiều):**
   - Ma trận đồng xuất mức xám trên 4 góc hướng ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
   - Đo lường độ tương phản (contrast), độ mịn (homogeneity), năng lượng (energy) phản ánh mức độ gồ ghề của tổn thương sùi loét.

3. **Lược Đồ Màu Sắc HSV (Color Histogram - 96 chiều):**
   - 32 bins cho mỗi kênh Sắc độ (Hue), Độ bão hòa (Saturation), Độ sáng (Value).
   - Bắt trọn sắc thái sung huyết đỏ của viêm loét đại tràng và màu xanh tím của polyp sau khi tiêm thuốc nhuộm (Indigo Carmine).

4. **Ngân Hàng Bộ Lọc Gabor (48 chiều):**
   - 24 bộ lọc Gabor (2 tần số sigma $\times$ 2 bước sóng lambda $\times$ 6 góc xoay).
   - Đo lường độ phân bố định hướng của các mạch máu dạng mạng nhện (Spider-web vascular patterns) trong Barrett thực quản.
