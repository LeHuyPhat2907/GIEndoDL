# 🔁 Báo cáo Kỹ thuật: Pipeline Tăng Cường Hai Góc Nhìn Cho Học Tương Phản (SupCon Two-View Augmentation)

> **Module chính:** `src/preprocessing/contrastive_augmenter.py` | **Hình minh họa:** `docs/figures/22_contrastive_learning_augmentations.png`

---

## 1. Cơ sở Phương pháp Luận của Contrastive Learning

Trong kiến trúc **Supervised Contrastive Learning (SupCon)** ở Giai đoạn 9, mục tiêu của hàm mất mát là kéo gần biểu diễn vector không gian (Embedding Vectors) của 2 góc nhìn $(v_1, v_2)$ cùng nhãn bệnh lý và đẩy xa các góc nhìn khác nhãn.

## 2. Vì sao Cần Kỹ thuật Random Grayscale & Gaussian Blur?

1. **Random Grayscale ($p=0.2$):** Vì ảnh nội soi có kênh Đỏ chiếm tới 57%, nếu không có Grayscale, bộ mã hóa (Encoder) sẽ có xu hướng học mẹo bằng cách so khớp màu đỏ thay vì học hoa văn vi mạch. Grayscale ép mạng phải trích xuất các đặc trưng hình thái học sâu sắc.
2. **Gaussian Blur ($p=0.5$):** Loại bỏ bẫy so khớp tần số cao của các đốm lóa sáng, giúp vector nhúng tập trung vào bản chất rãnh u.
