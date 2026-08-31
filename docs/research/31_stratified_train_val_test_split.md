# ✂️ Báo cáo Kỹ thuật: Phân Chia Tập Dữ Liệu Phân Tầng Chuẩn Y Khoa (Stratified Split)

> **Tập dữ liệu:** HyperKvasir (10,662 ảnh) | **Hình minh họa:** `docs/figures/24_stratified_split_distribution.png`

---

## 1. Bảng Thống Kê Chi Tiết Phân Chia Tập Dữ Liệu

| Phân vùng (Split) | Tỷ lệ (%) | Số lượng ảnh | Vai trò trong Pipeline Huấn Luyện |
|:---|:---:|:---:|:---|
| **Train Split** | `70.0%` | **7,463** | Huấn luyện mạng nơ-ron kết hợp Data Augmentation đa dạng |
| **Validation Split** | `15.0%` | **1,599** | Tinh chỉnh Hyperparameters, kiểm tra Early Stopping, chống Overfitting |
| **Test Split** | `15.0%` | **1,600** | **Khóa độc lập**, chỉ đánh giá hiệu năng cuối cùng (Final Benchmark) |
| **Tổng cộng** | `100.0%` | **10,662** | Phân tầng chính xác 100% trên cả 23 lớp bệnh học |

---

## 2. Cam Kết Chuẩn Mực Nghiên Cứu Y Sinh (Reproducibility & Zero-Leakage)

1. **Fixed Seed (`seed=42`):** Đảm bảo bất kỳ nhà nghiên cứu nào trên thế giới cũng có thể tái lập chính xác 100% cùng một tập Train/Val/Test.
2. **Bảo Toàn Lớp Thiểu Số:** Các lớp khó như trĩ (`hemorrhoids`) và Barretts được phân bổ chặt chẽ theo tỷ lệ, đảm bảo không có lớp nào bị 'bỏ rơi' khỏi tập Test.
