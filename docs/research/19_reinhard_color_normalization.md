# Báo Cáo Khoa Học: Quyết Định Loại Bỏ Reinhard Color Normalization Để Bảo Tồn Huyết Sắc Tố Đỏ

> **Mã công việc:** Task D27
> **Quyết định chiến lược:** Formally Eliminate Reinhard Color Normalization → Giữ nguyên không gian màu sRGB tự nhiên.

---

## 1. Nghịch Lý Sinh Hóa Của Reinhard Normalization Trong Nội Soi Tiêu Hóa
- Thuật toán Reinhard ánh xạ trung bình và độ lệch chuẩn của các kênh màu về một bức ảnh tham chiếu (Reference Image).
- Trong nội soi đường tiêu hóa, **Độ đỏ xung huyết (Erythema) và Mạng mạch máu (Vascularity)** là tiêu chuẩn chẩn đoán cốt lõi của Viêm loét đại tràng (Mayo 1, 2, 3) và Viêm thực quản trào ngược (Los Angeles A/B/C/D). Khi áp dụng Reinhard, thuật toán hiểu lầm sự sung huyết viêm loét nghiêm trọng là 'sự sai lệch màu sắc' và cưỡng bức tẩy trắng nó về mức trung bình của niêm mạc lành!
- Đối với các ca thủ thuật nhuộm màu Indigo Carmine (`dyed-lifted-polyps`, `dyed-resection-margins`), Reinhard biến màu xanh lam nhân tạo thành màu xám đục dị thường.

## 2. Giải Pháp Thay Thế: Data Augmentation Nhẹ + CBAM Attention
- Giữ nguyên màu sắc sRGB thực tế, hỗ trợ mô hình thích nghi thông qua tăng cường ColorJitter nhẹ nhàng và khối chú ý CBAM tự động thích ứng với các dải màu sinh học.
