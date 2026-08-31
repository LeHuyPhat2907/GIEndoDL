# 🧬 Báo cáo Kỹ thuật: Tăng Cường Dữ Liệu Biến Dạng Cơ Sinh Học Mô Mềm (Medical Biomechanical Augmentation)

> **Module chính:** `src/preprocessing/augmentations.py` | **Hình minh họa:** `docs/figures/20_medical_specific_augmentations.png`

---

## 1. Cơ sở Vật lý & Y học của các Phép biến đổi

| Phép biến đổi (Transform) | Hiện tượng vật lý / Y khoa tương ứng | Ý nghĩa đối với Mạng Học Sâu |
|:---|:---|:---|
| **Elastic Deformation** | Sóng nhu động ruột (Peristalsis) làm cơ trơn co bóp phi tuyến tính | Giúp mô hình nhận diện được polyp ở các trạng thái co bóp khác nhau |
| **Grid Distortion** | Thao tác bơm khí $\text{CO}_2$ làm căng giãn thành niêm mạc cục bộ | Rèn luyện tính bất biến kích thước rãnh niêm mạc |
| **Optical Distortion** | Độ cong hình học của thấu kính góc rộng Fisheye ($140^\circ - 170^\circ$) | Giúp nhận diện chính xác tổn thương nằm ở rìa viền ống kính |

---

## 2. Kết luận Kỹ thuật cho Khóa luận

Khác với các đối tượng cứng (ô tô, nhà cửa), cơ quan nội tạng người là mô mềm đàn hồi. Việc tích hợp các phép biến dạng Elastic và Optical Distortion đã tạo ra những mẫu huấn luyện có độ chân thực sinh học tuyệt đối, giúp mạng CNN và Vision Transformer đạt độ bền vững (Robustness) cao khi triển khai trên video nội soi động.
