# ✂️ Báo cáo Kỹ thuật: Pipeline Tự động Cắt Viền Đen & Loại bỏ Nhiễu Thiết bị (ROI Cropping & Artifact Removal)

> **Module chính:** `src/preprocessing/crop_roi.py` | **Hình minh họa:** `docs/figures/10_roi_crop_comparison.png`

---

## 1. Nguyên lý Hoạt động của Thuật toán

1. **Phát hiện Viền Quang học (Optical Border Detection):** Sử dụng ngưỡng thích ứng `threshold_val = 15` để tách biệt vùng tối hình học của ống soi khỏi niêm mạc tiêu hóa.
2. **Morphological Closing:** Sử dụng kernel ellipse kích thước $15 \times 15$ để đóng các lỗ tối bên trong lòng ruột, đảm bảo contour bao trọn toàn bộ trường nhìn nội soi.
3. **Cắt Bounding Box & Lọc Nhiễu:** Tự động cắt bỏ trung bình 15–25% diện tích viền đen vô nghĩa và inpaint các vùng chứa chữ số thiết bị.

---

## 2. Ý nghĩa đối với Huấn luyện Mạng Học Sâu

- **Chống hiện tượng học vẹt (Anti-Cheating / Spurious Correlation):** Ngăn mô hình liên kết các chữ số ngày giờ hoặc logo bệnh viện với nhãn bệnh lý.
- **Tập trung 100% tài nguyên mạng vào Tổn thương:** Giúp các lớp tích chập (Convolutional Layers) và khối Attention (CBAM) chỉ học hoa văn mao mạch và hình thái khối u.
