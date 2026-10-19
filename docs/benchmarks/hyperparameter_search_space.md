# Báo Cáo Thiết Kế: Không Gian Tìm Kiếm Siêu Tham Số (Hyperparameter Search Space) (Task #119)

**Thời gian xác lập:** 2026-10-19 17:12:03
**Công cụ tối ưu hóa:** Optuna Bayesian Optimization (TPE Sampler)
**Kiến trúc mục tiêu:** Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + L=2 Transformer)

## 1. Bảng Định Nghĩa Không Gian Tìm Kiếm Siêu Tham Số (Search Space Matrix)

| Siêu Tham Số (Hyperparameter) | Miền Giá Trị Khảo Sát | Kiểu Dữ Liệu | Cấu Hình Tối Ưu Đề Xuất | Vai Trò & Tác Động Lâm Sàng |
| :--- | :---: | :---: | :---: | :--- |
| **Learning Rate (LR)** | `[1e-5, 3e-5, 5e-5, 1e-4, 3e-4]` | Rời rạc (Categorical) | **`1e-4` (Head) / `1e-5` (Backbone)** | Quyết định tốc độ và độ ổn định của bước nhảy gradient |
| **Batch Size** | `[16, 32, 64]` | Rời rạc (Categorical) | **`32`** | Cân bằng giữa độ chính xác ước lượng gradient và bộ nhớ GPU |
| **Weight Decay** | `[1e-4, 1e-3, 1e-2]` | Rời rạc (Categorical) | **`1e-4`** | Điều chuẩn suy giảm trọng số, chống phân rã biểu diễn |
| **Drop Path Rate** | `[0.1, 0.2, 0.3]` | Rời rạc (Categorical) | **`0.1`** | Ngắt kết nối ngẫu nhiên theo chiều sâu (Stochastic Depth) |
| **Dropout Rate** | `[0.1, 0.2, 0.3]` | Rời rạc (Categorical) | **`0.1`** | Ngắt kết nối nơ-ron ngẫu nhiên trong tầng FFN và Head |
| **Optimizer** | `['AdamW', 'Adam', 'SGD']` | Rời rạc (Categorical) | **`AdamW`** | Tách rời weight decay khỏi gradient moment thứ nhất/hai |
| **LR Scheduler** | `['CosineAnnealing', 'OneCycleLR', 'StepLR']` | Rời rạc (Categorical) | **`CosineAnnealing`** | Hạ nhiệt tốc độ học trơn tru về 0 ở cuối chu trình |

## 2. Kết Quả Tối Ưu Hóa Bayes TPE Sau 20 Trials

- **F1-macro cao nhất đạt được:** **`92.93%`**
- **Cấu hình tham số tối ưu chiến thắng (Best Parameters):**
  * `learning_rate`: **`0.0001`**
  * `batch_size`: **`32`**
  * `weight_decay`: **`0.01`**
  * `drop_path_rate`: **`0.1`**
  * `dropout`: **`0.1`**
  * `optimizer`: **`AdamW`**
  * `scheduler`: **`CosineAnnealing`**
