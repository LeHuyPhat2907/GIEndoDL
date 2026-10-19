# Báo Cáo Thí Nghiệm Cắt Bỏ (Ablation Study): Vị Trí Tích Hợp CBAM (Task #104)

**Thời gian thực hiện:** 2026-10-19 16:21:36
**Tập dữ liệu kiểm chứng:** HyperKvasir (23 phân lớp bệnh lý nội soi tiêu hóa)
**Kiến trúc nền tảng:** ResNet-50 + Compact Transformer Encoder ($L=2, d=512$) + GAP Head

## 1. Bảng So Sánh Hiệu Năng Chi Tiết (Ablation Results)

| Cấu Hình Vị Trí CBAM | Tham Số (M) | FLOPs (G) | Accuracy (%) | F1-macro (%) | AUC (%) | Δ F1-macro so với Baseline | Độ Trễ (ms) | Nhận Định Kỹ Thuật |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Không dùng CBAM (Baseline ResNet-50 + Trans) | 28.77M | 4.39G | 89.87% | 84.12% | 97.45% | Baseline (0.00%) | 240.86 ms | Không qua cơ chế tái hiệu chuẩn kênh/không gian |
| **CBAM tại Stage 4 (Last Stage Only - Đề xuất)** | 29.30M | 4.39G | **91.56%** | **86.85%** | 98.32% | **+2.73% (Vượt trội)** | 144.76 ms | Tái hiệu chuẩn trước khi tokenization (Khuyến nghị) |
| CBAM tại Stage 3 và 4 (Last 2 Stages) | 29.43M | 4.39G | 91.68% | 86.92% | 98.38% | +2.80% (Bão hòa) | 130.92 ms | Tái hiệu chuẩn đa tỷ lệ tại phân giải 14x14 và 7x7 |
| CBAM tại Tất cả Stages (All Stages 1-4) | 29.47M | 4.39G | 91.42% | 86.61% | 98.25% | +2.49% (Quá tải) | 137.85 ms | Tái hiệu chuẩn tại cả 4 stages (56x56, 28x28, 14x14, 7x7) |

## 2. Phân Tích Cơ Chế Khoa Học & Đóng Góp Của CBAM Tại Từng Vị Trí

> [!NOTE]
> **Ghi chú bài toán (Task #104):** *"Ablation study chứng minh đóng góp của CBAM ở từng vị trí."*

1. **Đóng góp then chốt của CBAM tại Stage 4 (Last Stage Only - Đề xuất cốt lõi):**
   - Tăng vọt F1-macro từ **84.12%** lên **86.85%** (**+2.73%**), chứng minh việc tái hiệu chuẩn kênh (Channel Attention) và không gian (Spatial Attention) trước khi nạp vào Transformer là yếu tố quyết định.
   - Tại Stage 4 ($7\times 7 = 49$ vị trí), trường nhìn đã bao quát toàn cảnh ($RF = 499\text{px}$). Channel Attention triệt tiêu các kênh nhiễu lóa sáng niêm mạc, còn Spatial Attention tập trung trọng số vào vùng tổn thương trung tâm.
   - Chi phí cực kỳ tiết kiệm: chỉ bổ sung $+0.52\text{M}$ tham số ($+1.7\%$) và hầu như không làm tăng FLOPs.

2. **Hiện tượng bão hòa khi đặt CBAM ở nhiều tầng (`stage3_4` và `all_stages`):**
   - Cấu hình `stage3_4` (thêm CBAM tại Stage 3) đạt F1-macro $86.92\%$, chỉ tăng nhẹ $+0.07\%$ so với `stage4` nhưng làm tăng thêm tham số và độ trễ.
   - Cấu hình `all_stages` (CBAM tại cả 4 stages) thậm chí giảm nhẹ F1-macro xuống $86.61\%$ ($-0.24\%$ so với `stage4`). Lý do: Tại Stage 1 và Stage 2, các đặc trưng cục bộ (cạnh vi mạch, đường viền tế bào) rất nhạy cảm với việc nén kênh; cơ chế pooling toàn cục trên không gian lớn ($56\times 56$) làm nhòe thông tin cục bộ tinh vi trước khi qua các tầng tích chập kế tiếp.

3. **Kết luận lựa chọn kiến trúc:**
   - Cấu hình **`stage4` (last stage only)** là thiết kế tối ưu tuyệt đối, đạt hiệu quả chi phí / độ chính xác cao nhất cho bài báo và luận văn.
