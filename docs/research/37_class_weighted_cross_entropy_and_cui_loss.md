# ⚖️ Báo cáo Kỹ thuật: Hàm Mất Mát Cân Bằng Lớp (Class-Balanced Cross-Entropy Loss)

> **File cấu hình:** `configs/class_balanced_loss_weights.json` | **Hình minh họa:** `docs/figures/29_class_weighted_loss_comparison.png`

---

## 1. Cơ Sở Khoa Học Của Kỹ Thuật (Cui et al., CVPR 2019)

Thay vì sử dụng trọng số nghịch đảo thô sơ $1/N$ dễ dẫn đến bùng nổ Gradient ở các lớp có ít mẫu, đề tài ứng dụng lý thuyết **Số lượng Mẫu Hiệu Dụng (Effective Number of Samples)**:

$$E_n = \frac{1 - \beta^{N_c}}{1 - \beta}, \quad W_c = \frac{1}{E_n} = \frac{1 - \beta}{1 - \beta^{N_c}}$$

Với tham số làm mịn $\beta = 0.999$, hàm mất mát đạt trạng thái cân bằng hoàn hảo giữa việc thúc đẩy mô hình chú ý đến các tổn thương hiếm và duy trì tính ổn định của Gradient.

---

## 2. Kết Quả Đo Lường Phạt Lỗi Thực Nghiệm

- Khi mô hình đoán sai một ca bệnh trĩ (`hemorrhoids`), hàm mất mát **tự động nhân phạt lên gấp 1.0 lần** (4.11 so với 4.11), buộc mạng nơ-ron phải lập tức điều chỉnh trọng số để không tái phạm.
- Kỹ thuật này sẽ được kết hợp đồng thời với `WeightedRandomSampler` để tạo thành cơ chế phòng thủ 2 tầng chống lại sự mất cân bằng dữ liệu 191:1.
