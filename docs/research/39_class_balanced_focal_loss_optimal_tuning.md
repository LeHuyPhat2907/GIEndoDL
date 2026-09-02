# 🏆 Báo cáo Kỹ thuật: Thiết Kế & Hiệu Chuẩn Class-Balanced Focal Loss Tối Ưu

> **File cấu hình:** `configs/cb_focal_loss_config.json` | **Hình minh họa:** `docs/figures/31_class_balanced_focal_loss.png`

---

## 1. Cơ Chế Toán Học Hợp Nhất Đỉnh Cao

Class-Balanced Focal Loss (CB-Focal Loss) đồng thời kiểm soát cả hai khía cạnh mất cân bằng trong nội soi tiêu hóa:

$$\mathbf{L}_{\text{CB-Focal}} = -\left[\frac{1 - \beta}{1 - \beta^{N_y}}\right] \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

## 2. Kết Quả Khảo Sát Siêu Tham Số (Hyperparameter Tuning)

| Mức $\beta$ | Trọng số Trĩ ($W_{\text{hem}}$) | Trọng số Polyp ($W_{\text{pol}}$) | Tỷ số phạt ($W_{\text{hem}} / W_{\text{pol}}$) | Đánh giá kỹ thuật |
|:---:|:---:|:---:|:---:|:---|
| `0.9` | `2.469` | `0.849` | **`2.9x`** | Chưa đủ mạnh |
| `0.99` | `6.431` | `0.254` | **`25.4x`** | Chưa đủ mạnh |
| `0.999` | `7.526` | `0.059` | **`128.5x`** | Tối ưu xuất sắc (SOTA) |
| `0.9999` | `7.621` | `0.044` | **`173.7x`** | Quá cực đoan |

---

## 3. Quyết Định Kỹ Thuật Lựa Chọn Cho Giai Đoạn Huấn Luyện

- Cấu hình **$\beta = 0.999, \gamma = 2.0$** được chọn làm hàm mất mát mặc định cho toàn bộ các mô hình phân loại sâu (CNN-CBAM, ViT, Swin Transformer) ở Giai đoạn 6 đến Giai đoạn 11.
