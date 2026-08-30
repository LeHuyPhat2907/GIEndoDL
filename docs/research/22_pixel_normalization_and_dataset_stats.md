# 📊 Báo cáo Kỹ thuật: Thống kê Điểm ảnh & Chiến lược Chuẩn hóa Tensor PyTorch

> **File cấu hình:** `configs/dataset_stats.json` | **Hình minh họa:** `docs/figures/15_pixel_normalization_comparison.png`

---

## 1. Bảng Thống kê So sánh Tham số Chuẩn hóa

| Bộ thông số (Normalization Stats) | Kênh Đỏ (R) Mean / Std | Kênh Xanh lá (G) Mean / Std | Kênh Xanh dương (B) Mean / Std | Đặc điểm phân phối |
|:---|:---:|:---:|:---:|:---|
| **ImageNet Defaults** | `0.4850` / `0.2290` | `0.4560` / `0.2240` | `0.4060` / `0.2250` | Chuẩn ảnh đời sống (Ánh sáng trắng) |
| **HyperKvasir Specific (Đo đạc)** | `0.5729` / `0.3105` | `0.3557` / `0.2116` | `0.2515` / `0.1834` | **Đặc thù Nội soi (Huyết sắc tố Đỏ)** |

---

## 2. Ý nghĩa Kỹ thuật đối với Huấn luyện Mạng Nơ-ron

1. **Triệt tiêu Độ lệch Kênh Đỏ (Zero-Centering):** Trong ảnh nội soi, kênh Đỏ chiếm tới >55% cường độ sáng. Khi dùng chuẩn hóa riêng của HyperKvasir, dữ liệu đầu vào của mạng nơ-ron thực sự đối xứng quanh tâm 0, giúp các hàm kích hoạt (ReLU/GELU) không bị lệch gradient.
2. **Quy trình Huấn luyện 2 Giai đoạn:**
   - *Giai đoạn Warm-up (GĐ 6):* Dùng ImageNet stats để tận dụng tối đa trọng số Pretrained ban đầu.
   - *Giai đoạn Fine-tune & Contrastive Learning (GĐ 8, 9):* Chuyển sang HyperKvasir stats để tối ưu hóa không gian nhúng biểu diễn vi mạch.
