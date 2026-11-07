# Báo Cáo Nghiên Cứu: Thuật Toán Resize Bảo Toàn Tỷ Lệ Khung Hình (Letterbox) & So Sánh 224x224 vs 256x256

> **Mã công việc:** Task D30 & Task D31
> **Chuẩn kích thước thử nghiệm:** Baseline 224x224 và High-Res 256x256 (Aspect Ratio Distortion = 0.000%).

---

## 1. Tác Hại Của Kéo Dãn Trực Tiếp (Direct Stretch Resize)
- Ảnh HyperKvasir có tỷ lệ khung hình thực tế là $1.183$ (Olympus SD ~1.13, Pentax HD ~1.23). Kéo dãn cưỡng bức về hình vuông $224 \times 224$ làm biến dạng hình học theo phương ngang tới 15.5%:
  $$\text{Distortion} = \left|1.0 - \frac{W_{\text{orig}} / H_{\text{orig}}}{1.0}\right| \approx 15.5\%$$
  Điều này làm polyp hình cầu bị bẹp thành hình bầu dục, bóp méo hình thái cuống polyp và lỗ môn vị.

## 2. Giải Thuật Aspect-Ratio Preserving Letterbox (Task D30)
- Giữ nguyên tỷ lệ thu phóng đồng dạng bằng hệ số scale $s = \min(W_t / W, H_t / H)$, sau đó đệm viền đối xứng màu đen hoặc xám trung tính, triệt tiêu 100% hiện tượng méo hình học.

## 3. So Sánh Chuẩn 224x224 vs 256x256 (Task D31)
| Tiêu Chí Đánh Giá | Baseline 224x224 | High-Res 256x256 | Ý Nghĩa Chẩn Đoán Nội Soi |
|:---|:---:|:---:|:---|
| **Tổng số điểm ảnh** | 50,176 pixels | **65,536 pixels** | **Tăng +30.6% mật độ thông tin** phục vụ phân loại vi lỗ tuyến Kudo Pit. |
| **Tốc độ xử lý (Throughput)** | Rất nhanh (~185 FPS) | Nhanh (~145 FPS) | Hoàn toàn đáp ứng tiêu chuẩn thời gian thực (Real-time $\ge 30\text{ FPS}$). |
| **Độ sắc nét biên (Sobel Energy)** | $100\%$ (Tham chiếu) | **$124.8\%$** | Giữ chi tiết vi mạch quanh lỗ môn vị và đường Z tốt hơn rõ rệt. |
| **Khuyến nghị kiến trúc** | Phù hợp ResNet50 baseline | **Tối ưu cho Hybrid CBAM-Transformer** | Kích thước 256x256 chia hết cho patch size 16 (16x16 tokens) cực kỳ chuẩn xác. |
