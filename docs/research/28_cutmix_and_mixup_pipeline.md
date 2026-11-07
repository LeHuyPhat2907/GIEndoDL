# Báo Cáo Khoa Học: Quyết Định Loại Bỏ Hoàn Toàn CutMix & MixUp Trên Tập Dữ Liệu Đa Lớp

> **Mã công việc:** Task D36
> **Quyết định:** Formally Eliminate CutMix & MixUp from Multi-class Endoscopic Pipeline.

---

## 1. Cắt Ghép Lệch Lạc Ranh Giới Giải Phẫu (Anatomical Boundary Violation)
- CutMix cắt một mảng hình chữ nhật của ảnh này dán đè lên ảnh khác. Khi áp dụng trên dữ liệu nội soi, thuật toán vô tình dán niêm mạc chuyển sản Barrett của thực quản lên lòng manh tràng, hoặc dán phân lên polyp.
- Về mặt y khoa, đây là những **hình ảnh tổn thương dị dạng phi thực tế (Medically Absurd Artifacts)** làm phá vỡ phân phối dữ liệu tự nhiên, khiến mạng nơ-ron học các đặc trưng ranh giới cắt ghép giả tạo thay vì học vi mạch bệnh lý.
## 2. Phá Hỏng Biên Độ Tách Biệt Trong Không Gian Đa Tạp
- Nội suy tuyến tính MixUp ($y = \lambda y_1 + (1-\lambda) y_2$) tạo ra các mẫu nằm lơ lửng chính giữa ranh giới hai lớp. Trong khi ở Task D16 ta đã chứng minh các lớp Mayo Colitis vốn đã có Margin âm ($-0.22$), MixUp làm trầm trọng thêm sự chồng lấn, khiến hàm mất mát Cross-Entropy bị mất phương hướng hoàn toàn.
