# Báo Cáo Kỹ Thuật: Cân Bằng Ánh Sáng Thích Ứng Cục Bộ Có Kiểm Soát (Controlled LAB CLAHE)

> **Mã công việc:** Task D28
> **Tham số chuẩn hóa:** `clipLimit = 1.35`, `tileGridSize = (8, 8)`, thực thi độc quyền trên kênh L của không gian CIE LAB.

---

## 1. Lý Do Giảm clipLimit Từ 3.0 Xuống 1.35
- Trong các nghiên cứu thị giác máy tính thông thường, CLAHE thường đặt `clipLimit` từ 2.5 đến 4.0. Tuy nhiên, trong nội soi tiêu hóa, giá trị này phóng đại nhiễu cảm biến trong các vùng tối của lòng ruột và tạo ra các quầng sáng nhân tạo (Halo Artifacts) quanh các polyp.
- Bằng cách đặt `clipLimit = 1.35`, thuật toán chỉ tăng cường nhẹ độ tương phản vi mô của các rãnh niêm mạc (crypt architecture) mà không làm biến dạng biên độ sáng tự nhiên.

## 2. Bảo Tồn Tuyệt Đối Màu Sắc Trên Kênh a* và b*
- Việc chuyển đổi sang không gian CIE $L^*a^*b^*$ và chỉ xử lý kênh $L^*$ (Luminance) đảm bảo kênh $a^*$ (xanh lá - đỏ) và kênh $b^*$ (xanh dương - vàng) giữ nguyên vẹn 100%. Toàn bộ sắc đỏ xung huyết của viêm loét đại tràng và sắc xanh của phẩm Indigo Carmine được bảo toàn nguyên bản.
