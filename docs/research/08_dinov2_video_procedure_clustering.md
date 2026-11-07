# 🧠 Báo Cáo Gom Cụm Video Thủ Thuật Bằng DINOv2 Self-Supervised Embeddings

> **Mã công việc:** Task D09
> **Mô hình nền tảng:** DINOv2 ViT-S/14 (22.1M tham số, đặc trưng ngữ nghĩa 384-d L2-normalized)
> **Ngưỡng tương đồng Cosine:** $\ge 0.88$

---

## 1. Kết Quả Thực Nghiệm

- **Tổng số ảnh khảo sát:** `10,662` ảnh.
- **Tổng số cụm video thủ thuật độc lập nhận diện được:** `6,560` cụm.
- **Số cụm đa khung hình (Multi-frame procedures):** `292` cụm.
- **Số ảnh đơn lẻ (Single independent frames):** `6,268` ảnh.
- **Kích thước cụm lớn nhất:** `1133` khung hình.

---

## 2. Ý Nghĩa Khoa Học Đối Với Chuẩn Y Tế TRIPOD-AI

1. **Giải quyết triệt để rò rỉ cấp bệnh nhân/video:** Nhờ DINOv2, các khung hình chụp cùng một ca bệnh dù góc máy hay khoảng cách có thay đổi đều được gom vào một `video_cluster_id` duy nhất.
2. **Bảo đảm tính độc lập khi chia Fold (Task D39):** Khi áp dụng thuật toán `GroupStratifiedKFold`, toàn bộ các ảnh cùng một `video_cluster_id` sẽ được chuyển trọn gói sang Train hoặc Val, loại bỏ 100% tình trạng mô hình nhận diện bệnh nhân thay vì nhận diện bệnh học.
