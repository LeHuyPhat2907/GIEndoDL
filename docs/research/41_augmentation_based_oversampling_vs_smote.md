# 🧬 Báo cáo Kỹ thuật: Nghiên Cứu Over-Sampling Cho Dữ Liệu Ảnh Y Tế (Thay Thế SMOTE)

> **File cấu hình:** `configs/oversampling_config.json` | **Hình minh họa:** `docs/figures/33_augmentation_oversampling_distribution.png`

---

## 1. Lý Do Loại Bỏ SMOTE Truyền Thống Trong Thị Giác Máy Tính Y Khoa

- **Nội suy Pixel gây hủy hoại đặc trưng:** Thuật toán SMOTE (Chawla et al., 2002) được thiết kế cho dữ liệu bảng (Tabular Data). Khi áp dụng lên không gian pixel, việc nội suy tuyến tính tạo ra hiện tượng chồng ảnh (Ghosting), phá hủy hoa văn vi mạch và vi cấu trúc bề mặt mô.
- **Giải pháp Augmentation-Based Oversampling:** Áp dụng chuỗi biến đổi hình học, quang học và biến dạng cơ sinh học (Elastic Deformation) để mô phỏng các góc nhìn thực tế của ống nội soi, bảo toàn 100% tính chất y sinh học của tổn thương.

---

## 2. Kết Quả Cân Bằng Tập Dữ Liệu Huấn Luyện

- Thiết lập ngưỡng tối thiểu **120 mẫu** cho mọi lớp thiểu số.
- Lớp Trĩ (`hemorrhoids`) từ **4 ảnh** ban đầu được bổ sung thêm **116 biến thể chất lượng cao**, đạt tròn **120 mẫu**.
- Tổng số mẫu tập huấn luyện nâng từ **7,463 ảnh** lên **8,230 ảnh**, triệt tiêu triệt để tình trạng 'đói dữ liệu' của các lớp đuôi dài.
