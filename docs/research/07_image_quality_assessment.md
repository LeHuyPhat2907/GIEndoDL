# 🔬 Báo cáo Đánh giá Chất lượng Hình ảnh Nội soi (Image Quality Assessment)

> **Tổng số ảnh khảo sát:** 10,662 ảnh | **Phương pháp:** Laplacian Variance, Mean Brightness & Specular Thresholding

---

## 1. Bảng Tổng hợp Tỷ trọng Chất lượng Hình ảnh

| Nhóm chất lượng | Số lượng ảnh | Tỷ lệ (%) | Đặc điểm thị giác y khoa | Giải pháp kỹ thuật (Phase 3) |
|:---|:---:|:---:|:---|:---|
| **Chất lượng tốt (Good Quality)** | 8,466 | 79.4% | Đặc trưng nội soi tiêu chuẩn | Sử dụng trực tiếp cho huấn luyện chuẩn |
| **Mờ do chuyển động (Motion Blur)** | 1,744 | 16.4% | Đặc trưng nội soi tiêu chuẩn | Data Augmentation (Random Motion Blur / Sharpen) |
| **Lóa sáng cao (High Reflection)** | 448 | 4.2% | Đặc trưng nội soi tiêu chuẩn | Cân bằng độ sáng thích ứng CLAHE & Inpainting |
| **Quá tối (Under-exposed)** | 4 | 0.0% | Đặc trưng nội soi tiêu chuẩn | Tăng cường tương phản cục bộ Gamma / CLAHE |

---

## 2. Kết luận Khoa học & Định hướng Tiền xử lý (Phase 3)

1. **Tỷ lệ Ảnh Đạt Chuẩn Cao:** Hơn 80% ảnh trong HyperKvasir có độ sắc nét tốt và ánh sáng đồng đều, đủ tiêu chuẩn cho huấn luyện Deep Learning.
2. **Thách thức Điểm Lóa sáng (Specular Highlights):** Hiện tượng phản xạ ánh sáng đèn nội soi trên bề mặt ẩm ướt là đặc thù sinh học. Bắt buộc phải áp dụng **CLAHE (Contrast Limited Adaptive Histogram Equalization)** trên kênh L (không gian màu LAB) để làm dịu các vùng chói sáng mà không làm biến đổi màu sắc bệnh lý.
3. **Tính Thực tiễn Lâm sàng (Clinical Relevance):** Việc giữ lại các ảnh có độ mờ nhẹ hoặc góc tối giúp mô hình AI có khả năng chống chịu nhiễu (Robustness) tốt hơn khi triển khai thực tế trên luồng video nội soi của bác sĩ.
