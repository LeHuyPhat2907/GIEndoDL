# 🔄 Báo cáo Kỹ thuật: Pipeline Tăng Cường Dữ Liệu Hình Học Y Khoa (Albumentations)

> **Module chính:** `src/preprocessing/augmentations.py` | **Hình minh họa:** `docs/figures/18_basic_data_augmentations.png`

---

## 1. Cơ sở Lý luận Y học (Domain Knowledge Rationale)

1. **Tính Bất biến Không gian 3D (Spatial Invariance):** Ruột người có dạng hình trụ uốn lượn tự do. Khối polyp ở vị trí 12 giờ hay vị trí 6 giờ trong lòng ruột đều có chung bản chất bệnh lý. Do đó, các phép lật (Flip) và xoay (Rotate) hoàn toàn an toàn và phản ánh đúng thao tác xoay ống soi thực tế của bác sĩ.
2. **Tăng cường Khả năng Hội tụ cho Lớp Hiếm:** Giúp các lớp thiểu số (như Barretts chỉ có 41 ảnh, Trĩ chỉ có 6 ảnh) được nhân bản góc nhìn, chống lại hiện tượng Overfitting khi train 100 epochs.
