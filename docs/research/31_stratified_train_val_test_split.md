# Báo Cáo Kỹ Thuật: Phân Hoạch Dữ Liệu Train / Val / Test & 5-Fold Group-Stratified (Task D39 & D40)

> **Mã công việc:** Task D39 & Task D40
> **Tổng mẫu hợp lệ:** 10,644 mẫu trên 17 phân lớp mục tiêu (Loại bỏ 12 mẫu trùng tuyệt đối, 6 mẫu trĩ holdout).
> **Khóa hạt giống ngẫu nhiên:** `Seed = 42` (6 tầng: OS, Python, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN Deterministic).

---

## 1. Kết Quả Phân Chia Train / Val / Test (Tỷ lệ 70 / 15 / 15)
- **Tập Train (Huấn luyện):** `7,452` mẫu (70.01%).
- **Tập Val (Thẩm định):** `1,601` mẫu (15.04%).
- **Tập Test (Kiểm tra độc lập):** `1,591` mẫu (14.95%).

## 2. Kiểm Toán Rò Rỉ Ca Bệnh (Zero Data Leakage)
- **Chỉ số Rò rỉ Ranh giới:** $\Lambda = 0.00000000\%$ (Trùng lặp 0 cụm ca bệnh giữa các tập).
- Mọi phân lớp đều có tối thiểu 13 mẫu thẩm định trong tập Val và 14 mẫu trong tập Test (ngay cả lớp nhỏ nhất `barretts-esophagus`).
