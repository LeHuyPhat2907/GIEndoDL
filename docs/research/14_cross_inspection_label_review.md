# 🔬 Báo Cáo Đối Chiếu Thủ Công & Phân Tích Chéo Các Ca Nhãn Nghi Vấn (Cross-Inspection Audit)

> **Mã công việc:** Task D14
> **Tệp phân tầng xử lý:** [`data/processed/cross_inspection_label_review.csv`](file:///d:/PhatProject/GIEndoDL/data/processed/cross_inspection_label_review.csv)
> **Số ca kiểm toán:** `915` ca nghi vấn

---

## 1. Kết Quả Phân Tầng Xử Lý (Triage Distribution)

| Mức độ sai lệch | Số lượng ca | Tỷ lệ (%) | Tiêu chuẩn toán học | Hành động đề xuất |
|:---|:---:|:---:|:---|:---|
| **`CLINICAL_BORDERLINE`** | `431` | 47.1% | Gap từ $30\%$ đến $60\%$ | **Gom nhóm phân cấp hoặc làm mềm nhãn (Label Smoothing)** |
| **`LOW_CONFIDENCE_AMBIGUITY`** | `428` | 46.8% | Gap $< 30\%$ | **Giữ nguyên nhãn gốc bảo toàn độ đa dạng** |
| **`HIGH_CONFIDENCE_ERROR`** | `56` | 6.1% | Xác suất dự đoán $\ge 80\%$, Gap $\ge 60\%$ | **Sửa nhãn sang phân lớp dự đoán (Re-label)** |

---

## 2. Phân Tích Sâu 2 Nhóm Bệnh Học Xung Đột Điển Hình

### A. Xung đột giữa Viêm Thực Quản Độ A (`esophagitis-a`) và Đường Z-line (`z-line`)
- **Thực tế giải phẫu:** Đường Z-line là ranh giới tự nhiên giữa niêm mạc thực quản và dạ dày. Theo phân loại Los Angeles, Viêm thực quản độ A yêu cầu phải có một hoặc nhiều vết trợt rách niêm mạc $\le 5$mm.
- **Phát hiện:** Có tới **`99 ca`** được gán nhãn `esophagitis-a` nhưng qua kiểm tra hình ảnh hoàn toàn không có vết rách niêm mạc nào, bề mặt đường Z trơn láng bình thường. Người gán nhãn đã nhầm lẫn vùng sung huyết sinh lý thành tổn thương trợt!
- **Hậu quả Run 4:** Khi mô hình thấy ảnh Z-line bình thường nhưng bị phạt vì không đoán ra `esophagitis-a`, mô hình bị 'loạn thần' ở vùng thực quản, kéo tụt F1 của cả 2 lớp xuống dưới 50%!

### B. Xung đột giữa các phân độ Viêm Loét Đại Tràng Mayo (Grade 1 vs Grade 2)
- **Thực tế lâm sàng:** Thang điểm Mayo là đánh giá bán định lượng (*Semi-quantitative*). Grade 1 (viêm nhẹ) và Grade 2 (viêm vừa) có sự bất đồng ý kiến lên tới 35-40% giữa các bác sĩ chuyên khoa tiêu hóa.
- **Phát hiện:** Có **`72 ca`** gán Grade 1 nhưng mạng DINOv2 phát hiện mạng vi mạch máu đã biến mất hoàn toàn và có chấm xuất huyết li ti (đặc trưng rõ của Grade 2).
- **Giải pháp kiến trúc:** Đây là cơ sở khoa học hoàn hảo để ở Task D20, chúng ta xem xét hợp nhất các phân độ ranh giới này hoặc áp dụng cơ chế phân loại đa tầng có trọng số.
