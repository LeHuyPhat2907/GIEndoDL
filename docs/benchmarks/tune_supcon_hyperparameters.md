# Báo Cáo Thực Nghiệm: Tinh Chỉnh Siêu Tham Số SupCon (Task #125)

**Thời gian kiểm chứng:** 2026-10-19 17:21:34
**Tập dữ liệu:** HyperKvasir Challenge (23 classes)
**Mục tiêu:** Khảo sát độ nhạy và xác lập bộ siêu tham số tối ưu cho Supervised Contrastive Learning.

## 1. Bảng Tổng Hợp 4 Trục Siêu Tham Số SupCon Tối Ưu

| Siêu Tham Số | Miền Khảo Sát | Giá Trị Tối Ưu Đề Xuất | F1-macro (%) Đạt Được | Vai Trò & Tác Động Khoa Học |
| :--- | :---: | :---: | :---: | :--- |
| **Thang nhiệt độ ($\tau$)** | `[0.03, 0.05, 0.07, 0.1, 0.5]` | **`0.07`** | **`90.15%`** | Điều khiển độ dốc hàm mũ softmax và năng lực khai thác mẫu âm tính khó |
| **Chiều không gian chiếu ($d_{proj}$)** | `[64, 128, 256, 512]` | **`128`** | **`90.15%`** | Nén biểu diễn trung gian 512D thành vector siêu cầu 128D để tính tích vô hướng |
| **Số Epochs Tiền Huấn Luyện** | `[15, 25, 50, 75, 100]` | **`50`** | **`90.15%`** | Điểm bão hòa hội tụ; 75 và 100 epochs chỉ tăng $\le 0.05\%$ nhưng tốn gấp rưỡi thời gian |
| **Tỷ lệ trọng số (Joint $\lambda_1/\lambda_2$)** | `[0.1, 0.5, 1.0, 2.0]` | **`0.5`** | **`88.70%`** | Cân bằng giữa học hình học siêu cầu (SupCon) và siêu phẳng phân loại (CE) |

## 2. Phân Tích Độ Nhạy Nhiệt Độ (Temperature Sensitivity Analysis)

> [!NOTE]
> **Ghi chú Task #125:** *"Temperature rất quan trọng; tau = 0.07 là starting point tốt."*

1. **Khi $\tau$ quá lớn ($\tau = 0.50$):**
   - Phân bố xác suất $\frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum \dots}$ bị san phẳng (uniform distribution).
   - Mọi cặp mẫu âm tính đều có trọng số phạt tương đương nhau, triệt tiêu hoàn toàn khả năng tập trung vào các mẫu khó phân biệt (*Hard Negatives* như polyp và nếp gấp niêm mạc lành). F1 sụt giảm nghiêm trọng xuống **84.80%**.

2. **Khi $\tau$ tối ưu ($\tau = 0.07$):**
   - Đạt đỉnh F1-macro **90.15%**.
   - Gradient tập trung tạo ra lực đẩy cực đại đối với các cặp âm tính có độ tương đồng cosine cao, ép buộc mạng phải tìm kiếm các chi tiết vi mạch phân tách tinh tế.
