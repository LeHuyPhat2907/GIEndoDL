# 🛠️ Báo Cáo Chuẩn Hóa Nhãn Sạch (Label Noise Remediation)

> **Mã công việc:** Task D15
> **Tệp kết xuất cốt lõi:** [`data/processed/cleaned_hyperkvasir_labels.csv`](file:///d:/PhatProject/GIEndoDL/data/processed/cleaned_hyperkvasir_labels.csv)
> **Tệp nhật ký sửa đổi:** [`data/processed/remediation_summary.csv`](file:///d:/PhatProject/GIEndoDL/data/processed/remediation_summary.csv)

---

## 1. Kết Quả Chuẩn Hóa Nhãn

- **Tổng số mẫu ảnh:** `10,662` ảnh.
- **Số ca gán sai được sửa sang nhãn đúng:** `58` ảnh.
- **Cặp ảnh xung đột SSIM = 1.0000 đã đồng nhất:** 2 cặp (4 ảnh) giữa Mayo Grade 1 và 2 đã được chuẩn hóa về Mayo Grade 2.
- **Số ca giữ nguyên có gắn cờ ranh giới lâm sàng:** 431 ca (sẽ hỗ trợ cơ chế Label Smoothing ở Phase 7).

---

## 2. Các Hướng Sửa Đổi Trọng Yếu Phục Vụ Nâng Cao F1-Score

1. **Giải cứu phân lớp Thực quản (`esophagitis-a` ➔ `z-line`):** 27 ca Z-line bình thường bị gán nhầm sang viêm thực quản đã được trả về đúng nhãn `z-line`. Điều này giúp mô hình không còn bị phạt oan khi nhìn thấy đường Z bình thường.
2. **Chuẩn hóa thủ thuật tiêm nâng & diện cắt (`dyed-lifted-polyps` ⟷ `dyed-resection-margins`):** 15 ca bị dán nhầm chéo giữa bờ diện cắt và polyp đã tiêm nâng được trả về đúng nhãn hình thái học.
3. **Sẵn sàng cho Phase 4 (Schema Quyết Định Cuối Cùng):** Bộ nhãn sạch mới này bảo đảm độ chính xác ground-truth ở mức cao nhất từ trước đến nay, sẵn sàng cho việc tái cấu trúc phân cấp ở Task D19-D24.
