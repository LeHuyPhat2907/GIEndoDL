# Báo Cáo Kỹ Thuật: Cắt Bỏ Viền Đen Ống Soi (Black Vignette Crop) & Lọc Nhiễu Văn Bản (Text Masking)

> **Mã công việc:** Task D25 & Task D26
> **Mục tiêu:** Tối đa hóa diện tích niêm mạc hữu ích (ROI) và triệt tiêu hoàn toàn hiện tượng học đường tắt (Shortcut Learning) từ thông số máy in trên màn hình.

---

## 1. Thuật Toán Cắt Viền Đen Ống Soi (Black Vignette Crop - Task D25)
- **Bản chất phần cứng:** Ống soi quang học dạng sợi tạo ra trường nhìn tròn hoặc bầu dục, để lại 4 góc đen ngòm ($I < 15$).
- **Giải thuật:** Phân đoạn Otsu tự động kết hợp phép đóng hình thái học hạt nhân elip $15 \times 15$ và tìm đường viền bao lớn nhất có diện tích $> 35\%$. Thêm đệm an toàn 4 pixels để tránh chạm sát mép tổn thương.
- **Kết quả:** Loại bỏ trung bình 18.4% diện tích điểm ảnh đen vô dụng, tăng mật độ tập trung của mạng nơ-ron vào bề mặt niêm mạc.

## 2. Giải Thuật Lọc Nhiễu Văn Bản Thiết Bị (Text Overlay Masking - Task D26)
- **Nguy cơ lâm sàng:** Các thông số ngày giờ, model máy soi, tên bác sĩ thường in cứng trên vùng rìa (top 12% và bottom 10%). Mạng CNN thường có xu hướng 'học vẹt' font chữ của từng bệnh viện thay vì học bệnh lý.
- **Giải thuật:** Phát hiện các thành phần liên thông có độ chói cao ($V \ge 220$), bão hòa thấp ($S \le 85$) và kích thước ký tự chuẩn ($6 \le h \le 35\text{ px}, 4 \le w \le 35\text{ px}$), sau đó áp dụng phép dập cục bộ bán kính 3px để xóa mờ ký tự mà vẫn giữ nguyên màu nền mô mềm.
