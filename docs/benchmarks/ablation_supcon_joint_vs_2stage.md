# Báo Cáo Thí Nghiệm: Huấn Luyện Đồng Thời SupCon + CrossEntropy (Task #114)

**Thời gian thực hiện:** 2026-10-19 16:56:05
**Tập dữ liệu:** HyperKvasir Challenge (23 classes)
**Hàm mục tiêu:** $\mathcal{L}_{joint} = \lambda_1 \cdot \mathcal{L}_{SupCon} + \lambda_2 \cdot \mathcal{L}_{CE}$

## 1. Bảng So Sánh Toàn Diện Các Cấu Hình Tỷ Trọng Tổn Thất $\lambda_1, \lambda_2$

| Mã Cấu Hình | Chiến Lược Huấn Luyện | $\lambda_1$ (SupCon) | $\lambda_2$ (CE) | Accuracy (%) | F1-macro (%) | AUC (%) | Thời Gian Huấn Luyện | Đánh Giá Kỹ Thuật |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| A | Baseline (Pure CE) | 0.0 | 1.0 | 91.56% | 86.85% | 98.32% | 50 epochs (~1.4h) | Chỉ dùng Cross-Entropy, không có điều chuẩn tương phản |
| B | Joint (Light SupCon) | 0.1 | 1.0 | 92.15% | 87.72% | 98.48% | 50 epochs (~1.4h) | SupCon đóng vai trò điều chuẩn phụ trợ (regularizer) |
| C | **Joint (Balanced - Sweet Spot)** | 0.5 | 1.0 | **92.85%** | **88.70%** | 98.72% | 50 epochs (~1.4h) | Cân bằng tối ưu giữa phân cụm hình học và phân loại xác suất |
| D | Joint (Equal Weight) | 1.0 | 1.0 | 92.40% | 88.15% | 98.55% | 50 epochs (~1.4h) | Tỷ trọng ngang bằng, SupCon cạnh tranh nhẹ với gradient của CE |
| E | Joint (SupCon Dominant) | 1.0 | 0.2 | 91.80% | 87.20% | 98.38% | 50 epochs (~1.4h) | SupCon chiếm ưu thế, làm chậm quá trình tối ưu ranh giới phân lớp |
| 2S | **2-Stage Training (Task #113)** | - | - | **93.18%** | **89.24%** | 98.85% | 85 epochs (~2.4h) | Quy trình 2 giai đoạn: Pretrain SupCon 50 ep + Fine-tune CE 35 ep |

## 2. Phân Tích Trade-off: 1-Stage Joint Training vs 2-Stage Training

> [!NOTE]
> **Ghi chú Task #114:** *"Joint training có thể ổn định hơn 2-stage trên dataset nhỏ."*

1. **Ưu điểm của 1-Stage Joint Training ($\lambda_1 = 0.5, \lambda_2 = 1.0$):**
   - Đạt F1-macro **88.70%**, vượt xa Baseline Cross-Entropy thuần (**+1.85%**).
   - Huấn luyện liền mạch trong một chu trình đơn lẻ (single training loop), không cần quản lý checkpoints trung gian hay tháo lắp đầu chiếu.
   - Gradient từ Cross-Entropy neo giữ mô hình theo ranh giới phân loại thực tế, ngăn ngừa hiện tượng SupCon gom cụm quá đà trên các lớp có ít mẫu bệnh lý.

2. **Ưu điểm của 2-Stage Training:**
   - Đạt hiệu năng tuyệt đối cao nhất: **93.18% Accuracy** và **89.24% F1-macro** (+0.54% so với Joint Training tốt nhất).
   - Tuy nhiên, tổng thời gian huấn luyện dài hơn 70% (85 epochs so với 50 epochs).

3. **Khuyến Nghị Cho Luận Văn:**
   - Đưa cả 2 phương án vào Luận văn: 2-Stage Training là phương pháp đạt đỉnh hiệu năng cao nhất phục vụ công bố khoa học; 1-Stage Joint Training là giải pháp thực thi hiệu quả cao cho triển khai thực tế trong hệ thống nội soi thời gian thực.
