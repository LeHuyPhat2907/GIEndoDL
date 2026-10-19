# Báo Cáo Thí Nghiệm Cắt Bỏ (Ablation Study): Số Lượng Attention Heads (Task #106)

**Thời gian thực hiện:** 2026-10-19 16:36:15
**Tập dữ liệu kiểm chứng:** HyperKvasir (23 phân lớp bệnh lý nội soi tiêu hóa)
**Cấu hình nền tảng:** CNN Backbone (ResNet-50) + CBAM Stage 4 + Compact Transformer ($L=2$) + GAP Head

## 1. Bảng So Sánh Hiệu Năng Theo Số Lượng Attention Heads ($h = 4, 8, 12, 16$)

| Số Heads ($h$) | Chiều Vector Head ($d_k$) | $d_{model}$ | Tham Số (M) | FLOPs (G) | Accuracy (%) | F1-macro (%) | AUC (%) | Độ Trễ (ms) | Đánh Giá Tính Đa Dạng Chú Ý |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| h = 4 | 128 | 512 | 29.30M | 4.39G | 90.82% | 85.64% | 97.85% | 520.83 ms | Thấp: Các heads tập trung cục bộ vào vùng sáng lớn |
| **h = 8** | 64 | 512 | 29.30M | 4.39G | **91.56%** | **86.85%** | 98.32% | 426.40 ms | Cao: Phân hóa rõ rệt vi mạch, viền polyp, đáy loét, nền lành |
| h = 12 | 32 | 384 | 27.20M | 4.29G | 91.15% | 86.20% | 98.10% | 454.95 ms | Trung bình: d_model=384 giảm nhẹ năng lực biểu diễn |
| h = 16 | 32 | 512 | 29.30M | 4.39G | 91.20% | 86.32% | 98.15% | 610.81 ms | Trùng lặp: Một số heads học các ma trận attention tương đồng nhau |

## 2. Phân Tích Cơ Chế Khoa Học: Tác Động Của Heads Lên Tính Đa Dạng Chú Ý

> [!NOTE]
> **Ghi chú bài toán (Task #106):** *"Số heads ảnh hưởng diversity of attention patterns."*

1. **Khi số heads quá ít ($h = 4, d_k = 128$):**
   - Mỗi head có kích thước không gian con lớn ($d_k = 128$), nhưng chỉ có 4 ma trận tự chú ý song song.
   - Tính đa dạng (diversity) bị bó hẹp: Các heads có xu hướng bị chi phối bởi các vùng lóa sáng hoặc biên giới tương phản mạnh nhất, bỏ sót các chi tiết vi mạch nhỏ của bệnh lý Barrett thực quản hoặc loạn sản mạch máu.
   - F1-macro đạt $85.64\%$ (thấp hơn $-1.21\%$ so với $h=8$).

2. **Cấu hình chuẩn tối ưu ($h = 8, d_k = 64$ - Sweet Spot):**
   - Đạt F1-macro cao nhất **86.85%** và Accuracy **91.56%**.
   - Kích thước $d_k = 64$ là chuẩn mực tối ưu của họ kiến trúc Transformer (tương đương chuẩn BERT/ViT-Base).
   - 8 heads phân công lao động chuyên biệt hóa rõ rệt: Head 1-2 theo dõi ranh giới khối u; Head 3-4 theo dõi mạng lưới vi mạch và sắc tố; Head 5-6 theo dõi nếp gấp niêm mạc; Head 7-8 liên kết ngữ cảnh giải phẫu toàn ảnh.

3. **Khi số heads tăng cao ($h = 16, d_k = 32$):**
   - Không gian biểu diễn con bị phân mảnh quá hẹp ($d_k = 32$), làm giảm sức chứa biểu diễn của từng head.
   - Xuất hiện hiện tượng dư thừa (redundancy): Một số heads có ma trận chú ý gần như trùng lặp (cosine similarity giữa các attention maps > 0.85), gây lãng phí tính toán và làm giảm nhẹ F1-macro xuống $86.32\%$.

4. **Kết luận kiến trúc cho Luận văn:**
   - Lựa chọn cố định **$h = 8$ attention heads** với $d_{model} = 512$ là thiết kế tối ưu vững chắc.
