# Báo Cáo Kỹ Thuật: Pipeline Biến Đổi Hình Học Đẳng Cấu (Isomorphic Geometric Augmentation)

> **Mã công việc:** Task D32
> **Phương pháp:** Random Horizontal / Vertical Flip, Random Rotate 90°, ShiftScaleRotate.

---

## 1. Cơ Sở Lý Thuyết Y Sinh: Tính Đẳng Cấu Trong Lòng Ống Tiêu Hóa
- Trong thị giác máy tính tự nhiên (ImageNet/COCO), hình ảnh chịu lực hấp dẫn vật lý (trời ở trên, đất ở dưới). Lật dọc (Vertical Flip) thường tạo ra hình ảnh phi thực tế.
- Ngược lại, trong lòng ruột hoặc thực quản, camera ống soi quay tự do $360^\circ$ quanh trục sợi quang. Khái niệm 'trên' hay 'dưới' hoàn toàn phụ thuộc vào tư thế xoay cổ tay của bác sĩ nội soi. Do đó, các phép biến đổi lật ngang, lật dọc và xoay $90^\circ, 180^\circ, 270^\circ$ là các phép biến đổi **đẳng cấu sinh học 100% (Isomorphic Geometric Invariance)**, tăng gấp 8 lần không gian mẫu tự nhiên mà không làm biến dạng tổn thương.
