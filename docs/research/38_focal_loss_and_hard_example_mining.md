# Báo Cáo Lý Thuyết: Động Lực Học Triệt Tiêu Gradient Của Focal Loss (Task D44)

> **Công thức:** $\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$ với $\gamma = 2.0$.

## 1. Cơ Chế Khai Thác Mẫu Khó Tự Động (Hard Example Mining)
- Với các mẫu dễ nhận diện (niêm mạc bình thường, $p_t > 0.9$), hệ số $(1 - 0.9)^2 = 0.01$ triệt tiêu 99% gradient, ngăn các mẫu dễ chiếm ưu thế.
- Với các mẫu khó (ranh giới rách vi thể viêm thực quản, $p_t < 0.3$), hệ số $(1 - 0.3)^2 = 0.49$ giữ lại gradient mạnh mẽ để huấn luyện mô hình.
