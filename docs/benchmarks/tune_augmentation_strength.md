# Báo Cáo Thực Nghiệm: Tinh Chỉnh Cường Độ Tăng Cường Dữ Liệu (Task #124)

**Thời gian kiểm chứng:** 2026-10-19 17:21:30
**Tập dữ liệu:** HyperKvasir Challenge (23 classes)
**Chiến lược:** Khảo sát đồng thời Tăng cường dữ liệu (Data Augmentation) và Điều chuẩn (Regularization)

## 1. Bảng So Sánh Chi Tiết Các Mức Độ Tăng Cường Dữ Liệu

| Mức Độ (Level) | Danh Mục Phép Biến Đổi | Train Acc (%) | Val Acc (%) | Val F1 (%) | Overfitting Gap (%) | Đánh Giá Ý Nghĩa Y Sinh |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Light (Cơ Bản) | `RandomHorizontalFlip, RandomVerticalFlip, Resize(224)` | 96.00% | 91.20% | 86.40% | 4.80% | Chưa đủ độ đa dạng thị giác; mô hình vẫn nhạy cảm với góc quay camera ống soi |
| Moderate (Task #50) | `RandomFlip + RandomRotation(15°) + ColorJitter(0.2) + Affine(0.9-1.1)` | 95.40% | 92.50% | 88.20% | 2.90% | Tốt cho các tổn thương hình thái lớn; chưa bao quát được biến dạng thấu kính nội soi |
| **Strong Medical (Sweet Spot)** | `RandomResizedCrop(0.75-1.0) + Flips + ColorJitter(0.3) + GaussianBlur(0.2) + OpticalDistortion(0.2)` | 95.40% | **93.85%** | **90.15%** | 1.55% | Cân bằng hoàn hảo: Mô phỏng dịch chuyển góc soi và ánh sáng quang học thực tế trong phòng mổ |
| Extreme (Cực Đoan) | `Heavy Crop(0.4) + Heavy Jitter(0.6) + CoarseDropout(Cutout) + ElasticTransform` | 91.20% | 90.40% | 85.30% | 0.80% | Phá hủy đặc trưng bệnh lý: Che khuất khối polyp và làm méo các mao mạch dạng mạng nhện |

## 2. Phân Tích Cơ Chế Khoa Học Cho Luận Văn

> [!NOTE]
> **Ghi chú Task #124:** *"Augmentation và regularization cùng chống overfitting; tune jointly."*

1. **Tại sao không nên áp dụng Extreme Augmentation trong bài toán nội soi?**
   - Khác với tập ảnh tự nhiên ImageNet (nơi chú chó hay ô tô bị che khuất 50% vẫn nhận diện được), ảnh nội soi tiêu hóa dựa vào các dấu hiệu vi thể cực kỳ nhỏ (ví dụ polyp phẳng Paris IIb chỉ dày vài milimet, hoặc loạn sản mạch máu chỉ chiếm một vài pixel).
   - Phép biến đổi Cutout vô tình che lấp trọn vẹn tổn thương, gán nhãn sai lệch cho mô bình thường, khiến F1 sụt giảm nghiêm trọng (-4.85%).

2. **Đóng góp của Optical Distortion:**
   - Ống nội soi góc rộng luôn gây ra hiện tượng méo hình mắt cá (*barrel distortion*). Phép biến đổi `OpticalDistortion` giúp mô hình bất biến với vị trí tổn thương (nằm ở trung tâm hay rìa thấu kính).
