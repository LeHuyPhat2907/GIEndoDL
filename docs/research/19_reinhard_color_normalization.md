# 🎨 Báo cáo Kỹ thuật: Cân Bằng & Chuẩn Hóa Màu Sắc Bằng Thuật Toán Reinhard (Device Bias Removal)

> **Module chính:** `src/preprocessing/reinhard_normalizer.py` | **Hình minh họa:** `docs/figures/12_reinhard_color_normalization.png`

---

## 1. Cơ sở Toán học của Reinhard Color Transfer

Thuật toán ánh xạ không gian màu từ ảnh nguồn $S$ sang ảnh tham chiếu chuẩn $R$ theo công thức:
$$\text{Pixel}_{\text{norm}}^{c} = (\text{Pixel}_{\text{src}}^{c} - \mu_{\text{src}}^{c}) \cdot \frac{\sigma_{\text{ref}}^{c}}{\sigma_{\text{src}}^{c}} + \mu_{\text{ref}}^{c}, \quad c \in \{L, A, B\}$$

## 2. Giá trị Lâm sàng & Khắc phục Sai lệch Thiết bị (Device Bias)

1. **Triệt tiêu hiện tượng lệch màu giữa các hãng máy nội soi:** Đưa tất cả hình ảnh từ các dòng máy khác nhau (Olympus, Pentax, Karl Storz) về một dải nhiệt độ màu hồng niêm mạc đồng nhất.
2. **Tăng cường khả năng Tổng quát hóa (Generalization):** Giúp mạng học sâu không bị phụ thuộc vào màu đèn chiếu của từng phòng khám, cải thiện trực tiếp chỉ số Macro F1-score trên tập kiểm định ngoài độc lập.
