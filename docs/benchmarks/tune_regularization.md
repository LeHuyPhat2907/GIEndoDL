# Báo Cáo Thực Nghiệm: Tinh Chỉnh Điều Chuẩn và Kiểm Soát Độ Lệch Quá Khớp (Task #123)

**Thời gian kiểm chứng:** 2026-10-19 17:21:29
**Tập dữ liệu:** HyperKvasir Challenge (23 classes)
**Chỉ số theo dõi cốt lõi:** $\text{Overfitting Gap} = \text{Train Acc} - \text{Val Acc}$

## 1. Bảng Khảo Sát Chi Tiết Các Cấu Hình Điều Chuẩn

| Mã | Cấu Hình Điều Chuẩn | Dropout | DropPath | Weight Decay | Label Smoothing | Train Acc (%) | Val Acc (%) | Overfitting Gap (%) | Val F1 (%) | Trạng Thái |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| R1 | Không Điều Chuẩn (No Regularization) | 0.0 | 0.0 | 0.0 | 0.0 | 99.80% | 89.20% | 10.60% | 84.10% | Quá khớp nghiêm trọng (Severe Overfitting) |
| R2 | Điều Chuẩn Nhẹ (Light Reg) | 0.1 | 0.0 | 0.0001 | 0.0 | 97.50% | 91.80% | 5.70% | 87.20% | Quá khớp trung bình |
| R3 | **Cân Bằng Tối Ưu (Proposed Sweet Spot)** | 0.1 | 0.1 | 0.0001 | 0.1 | 95.40% | **93.85%** | **1.55%** | **90.15%** | Tổng quát hóa lý tưởng (Ideal Generalization) |
| R4 | Tăng Cường Dropout (Dropout 0.3) | 0.3 | 0.1 | 0.0001 | 0.1 | 93.80% | 92.50% | 1.30% | 88.60% | Bắt đầu dưới khớp nhẹ |
| R5 | Điều Chuẩn Cực Đoan (Heavy Reg) | 0.5 | 0.2 | 0.01 | 0.15 | 89.10% | 87.80% | 1.30% | 83.50% | Dưới khớp nghiêm trọng (Severe Underfitting) |

## 2. Phân Tích Cơ Chế Khoa Học Cho Luận Văn

> [!IMPORTANT]
> **Ghi chú Task #123:** *"Monitor train-val gap để detect overfitting."*

1. **Hiện tượng Quá khớp khi thiếu điều chuẩn (Cấu hình R1):**
   - Mô hình đạt Train Acc gần như tuyệt đối ($99.80\%$), nhưng Val Acc chỉ đạt $89.20\%$ (Gap $= 10.60\%$).
   - Mạng nơ-ron ghi nhớ các đốm sáng phản xạ, nếp gấp niêm mạc giả định và bọt khí thay vì học đặc trưng bệnh lý thực sự.

2. **Sự phối hợp cộng hưởng của 4 kỹ thuật (Cấu hình R3 - Sweet Spot):**
   - `DropPath (0.1)`: Bỏ ngẫu nhiên các khối Transformer trong quá trình lan truyền thuận, buộc các tầng còn lại phải học biểu diễn độc lập.
   - `Label Smoothing (0.1)`: Thay vì nhãn one-hot cứng $[1, 0, 0...]$, gán xác suất mục tiêu $0.9$ cho nhãn đúng và $0.1/22$ cho các lớp còn lại. Điều này ngăn chặn mạng quá tự tin vào dự báo, làm mềm hàm phân loại logit.
   - `Weight Decay (1e-4)`: Ngăn chặn ma trận Query-Key có độ lớn quá mức, giúp cơ chế softmax không bị bão hòa.
