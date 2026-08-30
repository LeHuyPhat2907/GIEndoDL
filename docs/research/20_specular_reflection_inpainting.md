# ✨ Báo cáo Kỹ thuật: Phát Hiện & Tái Tạo Điểm Lóa Sáng Nội Soi (Specular Reflection Inpainting)

> **Module chính:** `src/preprocessing/specular_removal.py` | **Hình minh họa:** `docs/figures/13_specular_reflection_removal.png`

---

## 1. Nguyên lý Hoạt động của Thuật toán

1. **Phát hiện Vùng Lóa Bão Hòa (Dual-Space Masking):** Kết hợp đồng thời không gian màu HSV ($V \ge 240, S \le 45$) và kênh Grayscale ($I \ge 245$) để định vị các điểm ảnh bị bão hòa ánh sáng.
2. **Morphological Dilation:** Sử dụng kernel elip $3 \times 3$ nở biên mặt nạ để bao phủ toàn bộ gradient nhiễu xung quanh đốm lóa.
3. **Thuật toán Telea Fast Marching:** Nội suy lan truyền gradient từ các vùng niêm mạc lân cận lành lặn vào tâm đốm lóa, khôi phục kết cấu mô tự nhiên mà không làm biến dạng ranh giới polyp.

---

## 2. Giá trị Lâm sàng & Tối ưu cho Mạng Học Sâu

- **Ngăn chặn Gradient Giả mạo (Spurious Edge Artifacts):** Các đốm lóa trắng thường tạo ra các cạnh nhân tạo có gradient rất cao, dễ làm đánh lừa các bộ lọc tích chập (Convolutional Filters).
- **Tối ưu hóa Khối Attention CBAM:** Giúp bản đồ trọng số chú ý (Spatial Attention Map) không bị hút vào các đốm đèn nội soi mà tập trung 100% vào tổn thương bệnh lý.
