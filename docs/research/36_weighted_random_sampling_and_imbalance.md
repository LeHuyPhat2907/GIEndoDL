# ⚖️ Báo cáo Kỹ thuật: Kỹ Thuật Lấy Mẫu Có Trọng Số (Weighted Random Sampling)

> **File cấu hình:** `configs/class_weights.json` | **Hình minh họa:** `docs/figures/28_weighted_sampling_comparison.png`

---

## 1. Cơ Sở Toán Học Của Thuật Toán

Để giải quyết thách thức mất cân bằng $191.33:1$, đề tài áp dụng công thức trọng số nghịch đảo căn bậc hai (Inverse Square Root):
$$w_c = \frac{1}{\sqrt{N_c}}, \quad W_i = w_{c(i)}\quad (\forall i \in \{1, \dots, N\})$$

Công thức căn bậc hai giúp tăng cường cơ hội học hỏi cho các lớp thiểu số nhưng không gây ra hiện tượng quá khớp (Overfitting) do lặp đi lặp lại một bức ảnh quá nhiều lần.

---

## 2. Kết Quả Thực Nghiệm Trên 100 Batches

- **Lớp Trĩ (`hemorrhoids`):** Tần suất nạp vào mạng tăng vọt từ 2 lần lên tới 35 lần trong 100 batches.
- **Đồng đều Gradient:** Đảm bảo mọi lớp bệnh lý đều có đại diện trong mỗi Epoch, tối ưu hóa điểm số Macro F1-score.
