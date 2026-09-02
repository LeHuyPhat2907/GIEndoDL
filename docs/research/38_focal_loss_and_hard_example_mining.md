# 🎯 Báo cáo Kỹ thuật: Nghiên Cứu & Cài Đặt Focal Loss Khai Phá Mẫu Khó (Hard Example Mining)

> **File cấu hình:** `configs/focal_loss_config.json` | **Hình minh họa:** `docs/figures/30_focal_loss_dynamics_and_tuning.png`

---

## 1. Cơ Sở Lý Luận Của Focal Loss (Lin et al., ICCV 2017)

Công thức toán học của Focal Loss:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Hệ số điều biến $(1 - p_t)^\gamma$ đóng vai trò như một bộ lọc động:
- **Khi mô hình tự tin và dự đoán đúng ($p_t \ge 0.95$):** $(1 - 0.95)^2 = 0.0025$, làm **suy giảm 400 lần** lượng mất mát, ngăn không cho các mẫu dễ làm chệch hướng Gradient.
- **Khi mô hình gặp ca bệnh khó ($p_t \le 0.15$):** $(1 - 0.15)^2 = 0.7225$, giữ lại phần lớn giá trị Loss để ép mạng nơ-ron phải học kỹ.

---

## 2. Kết Quả Tinh Chỉnh Tham Số (Hyperparameter Tuning)

- Khảo sát $\gamma \in \{0, 1, 2, 3, 5\}$ chứng minh **$\gamma = 2.0$** là trạng thái cân bằng lý tưởng nhất cho ảnh nội soi tiêu hóa HyperKvasir.
- Khi kết hợp với trọng số Class-Balanced $\alpha_t$ từ Task 57, hàm mất mát trở thành **Class-Balanced Focal Loss** – vũ khí tối thượng giải quyết đồng thời cả 2 vấn đề: Mất cân bằng số lượng lớp và Độ khó của tổn thương y tế.
