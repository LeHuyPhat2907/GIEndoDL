# Báo Cáo Kỹ Thuật: Chuẩn Hóa Phản Quang Mềm (Soft Reflection Suppression) Thay Thế Inpainting

> **Mã công việc:** Task D29
> **Phương pháp:** Soft Saturation Clamping với bộ lọc Bilateral và Gaussian Mask Smoothing.

---

## 1. Khuyết Điểm Nghiêm Trọng Của Specular Inpainting Cổ Điển
- Các thuật toán Inpainting truyền thống (Telea / Navier-Stokes) cố gắng vẽ lại vùng lóe sáng bằng cách nội suy từ điểm ảnh lân cận. Trong lòng ruột có dịch nhầy, điều này tạo ra các 'mảng nhựa mờ nhân tạo' (blurry synthetic patches), xóa mất các lỗ tuyến pit-pattern bên dưới và đánh lừa mạng CNN.

## 2. Cơ Chế Chuẩn Hóa Phản Quang Mềm
- Thay vì xóa bỏ điểm ảnh, thuật toán nén nhẹ 45% độ chói đỉnh của điểm lóe sáng về gần mức trung bình của niêm mạc xung quanh bằng mặt nạ Gaussian mềm ($\sigma=2.0$).
- Nhờ đó, mắt người và mạng nơ-ron không còn bị lóa mắt bởi các điểm phản xạ ánh sáng trắng bão hòa, trong khi cấu trúc vi mô niêm mạc không hề bị bóp méo hay mờ nhòe.
