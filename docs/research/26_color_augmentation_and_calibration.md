# 🎨 Báo cáo Kỹ thuật: Hiệu Chuẩn Tăng Cường Màu Sắc Chuẩn Y Khoa (Calibrated Color Augmentation)

> **Module chính:** `src/preprocessing/augmentations.py` | **Hình minh họa:** `docs/figures/19_color_data_augmentations.png`

---

## 1. Bảng Hiệu Chuẩn Biên Độ Tham Số (Hyperparameter Calibration)

| Kỹ thuật (Transform) | Biên độ thiết lập | Cơ sở lý luận Y học | Đánh giá an toàn |
|:---|:---:|:---|:---:|
| **Brightness Jitter** | `[-0.15, +0.15]` | Mô phỏng sự dao động công suất nguồn sáng đèn Xenon/LED | 🟢 Hoàn toàn an toàn |
| **Contrast Jitter** | `[-0.15, +0.15]` | Mô phỏng độ nhạy dải động khác nhau của chip CCD cảm biến | 🟢 Hoàn toàn an toàn |
| **Hue Shift** | `[-0.04, +0.04]` ($\pm 8^\circ$) | **Khống chế nghiêm ngặt** để không biến niêm mạc hồng thành xanh tím | 🟢 **Bắt buộc hiệu chuẩn** |
| **Random Gamma** | `[0.85, 1.15]` | Mô phỏng tính chất phi tuyến của thấu kính quang học | 🟢 Rất tốt |
| **Channel Shuffle** | *Không sử dụng* | Phá hủy tỷ lệ quang phổ hấp thụ Hemoglobin sinh học | 🔴 **Cấm tuyệt đối** |

---

## 2. Kết luận Thực nghiệm cho Khóa luận

Thực nghiệm đối chứng đã chứng minh rằng: Việc áp dụng bừa bãi Channel Shuffle hoặc dịch chuyển Hue quá đà (>30 độ) sẽ biến mô lành thành mô hoại tử giả tạo, làm hỏng quá trình học đặc trưng của mạng nơ-ron. Đề tài đã hiệu chuẩn chính xác phạm vi Hue trong giới hạn ±8 độ, vừa tạo ra độ phong phú dữ liệu vừa bảo toàn 100% tính chân thực của bệnh học.
