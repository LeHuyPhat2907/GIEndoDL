# Báo Cáo Lý Thuyết: So Sánh Class-Weighted Cross-Entropy vs Cui Class-Balanced Loss (Task D44)

> **Lý thuyết thể tích hiệu dụng:** Cui et al. (CVPR 2019).

## 1. Thể Tích Hiệu Dụng Mẫu Dữ Liệu
$$E_{N} = \frac{1 - \beta^{N}}{1 - \beta}$$
- Khi số lượng mẫu $N$ tăng lên, thông tin cận biên mà mỗi mẫu mới mang lại giảm dần do sự chồng lấn không gian đặc trưng. Class-Balanced Loss đánh giá trọng số theo thể tích bao phủ thực tế thay vì đếm số lượng mẫu thuần túy.
