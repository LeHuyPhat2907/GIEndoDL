# Báo Cáo Thực Nghiệm: Tinh Chỉnh Tốc Độ Học (Learning Rate Tuning) (Task #120)

**Thời gian kiểm chứng:** 2026-10-19 17:21:24
**Phương pháp áp dụng:** Leslie Smith (2017) LR Range Test + Warmup Schedule Analysis
**Mục tiêu:** Xác định siêu tham số nhạy cảm nhất quyết định tốc độ và độ ổn định hội tụ.

## 1. Kết Quả Kiểm Tra Dải Tốc Độ Học (LR Range Test)

- **Điểm dốc cực đại (Steepest Gradient):** `7.59e-05` (Được khuyến nghị làm Base LR cho Classifier Head).
- **Điểm đáy tổn thất (Minimum Loss):** `3.63e-03` (Ranh giới cận trên trước khi bùng nổ phân kỳ).
- **Dải tốc độ học làm việc tối ưu:** `[3.79e-05 ➔ 1.52e-04]`.
- **Chiến lược phân tầng đề xuất:**
  * `Backbone CNN (ResNet-50)`: `1e-5` (tránh phá vỡ biểu diễn tiền huấn luyện).
  * `Transformer Encoder`: `2e-5` (học tương quan không gian mềm).
  * `Classification Head`: `1e-4` (tối ưu hóa ranh giới quyết định).

## 2. Bảng Đối Chứng Các Tỷ Lệ Bước Khởi Động Ấm (Warmup Steps Ratio)

| Tỷ Lệ Warmup (%) | Số Epochs Tương Đương | F1-macro (%) | Accuracy (%) | Đánh Giá Độ Ổn Định Gradient |
| :---: | :---: | :---: | :---: | :--- |
| 0% (Không Warmup) | ~0.0 epochs | 87.20% | 91.80% | Gia tốc gradient lớn ban đầu làm chệch hướng trọng số Pre-LN MHSA |
| 3% (~1 Epoch) | ~1.1 epochs | 88.50% | 92.50% | Bắt đầu ổn định nhưng vẫn còn hiện tượng dao động loss nhẹ |
| 5% (~2 Epochs) | ~1.8 epochs | 89.90% | 93.60% | Ổn định rất tốt, bước chuyển mượt mà vào chu trình suy giảm |
| **8% (Sweet Spot)** | **~2.8 epochs** | **90.15%** | **93.85%** | Điểm tối ưu lý tưởng nhất, gradient phân phối đều toàn mạng |
| 10% (~3.5 Epochs) | ~3.5 epochs | 89.95% | 93.70% | Hội tụ tốt, nhưng chiếm thêm thời gian ở mức lr thấp |
| 15% (~5 Epochs) | ~5.2 epochs | 89.10% | 92.95% | Khởi động quá dài làm lãng phí số epochs huấn luyện ở tốc độ cao |

## 3. Lý Luận Khoa Học Cho Luận Văn

> [!TIP]
> **Ghi chú Task #120:** *"LR là hyperparameter quan trọng nhất; warmup 5-10% total steps."*

Trong các kiến trúc tích hợp cơ chế tự chú ý (Self-Attention) như CNN-CBAM-Transformer, ma trận trọng số $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ rất dễ bị nhiễu động mạnh ở các bước đầu nếu áp dụng ngay tốc độ học lớn. Việc áp dụng tỷ lệ khởi động ấm **8%** (~2.8 epochs) cho phép gradient dẫn hướng mô hình về vùng trũng phẳng (*flat minima*), tạo tiền đề đạt F1-macro kỷ lục **90.15%**.
