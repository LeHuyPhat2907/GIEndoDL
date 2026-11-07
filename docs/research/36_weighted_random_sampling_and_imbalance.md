# Báo Cáo Kỹ Thuật: Thuật Toán WeightedRandomSampler Lũy Thừa β = 0.75 (Task D43)

> **Công thức trọng số:** $w_c = \frac{1}{(N_c)^{0.75}}$.

## 1. Lý Do Chọn Lũy Thừa β = 0.75 Thay Vì Nghịch Đảo Thô (1/N)
- Nghịch đảo thô ($1/N$) phóng đại quá mức lớp thiểu số, khiến các mẫu hiếm bị lặp lại nhiều lần trong cùng một epoch, dẫn đến Overfitting nghiêm trọng.
- Lũy thừa $\beta = 0.75$ san phẳng phân bố vừa phải, giảm tỷ lệ chênh lệch lấy mẫu từ 12.63:1 xuống còn 6.7:1, đảm bảo mọi batch đều có sự xuất hiện của các ca bệnh hiếm mà không làm mất tính đa dạng sinh học.
