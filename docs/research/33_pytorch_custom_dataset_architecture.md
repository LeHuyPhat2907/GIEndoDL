# 🧪 Báo cáo Kỹ thuật: Kiến Trúc Lớp PyTorch HyperKvasirDataset & Kiểm Thử Đơn Vị

> **Module chính:** `src/dataset/hyperkvasir_dataset.py` | **Hình minh họa:** `docs/figures/26_pytorch_dataset_batch_sample.png`

---

## 1. Thiết Kế Hỗ Trợ 3 Chế Độ Dữ Liệu

| Chế độ (Split Mode) | Số lượng mẫu | Chính sách Transform áp dụng | Mục đích sử dụng |
|:---|:---:|:---|:---|
| **`train`** | `7,463` | Augmentation toàn diện (Flip, Rotate, Crop, Color, Deform) | Rèn luyện mạng nơ-ron chống Overfitting |
| **`val`** | `1,599` | Cố định (Resize Bicubic + Normalize) | Giám sát độ mất mát và Early Stopping |
| **`test`** | `1,600` | Cố định (Resize Bicubic + Normalize) | Đánh giá khách quan năng lực tổng quát hóa |

---

## 2. Kết Quả Kiểm Thử Đơn Vị (Unit Test Summary)

- **Tensor Dimensions:** `torch.Size([B, 3, 224, 224])` chuẩn kênh `[Channel, Height, Width]`.
- **Label Format:** `torch.int64` chuẩn hóa trong không gian nhãn rời rạc $[0, 22]$.
- **Bộ nhớ & Multi-processing:** Đọc ảnh bằng OpenCV kết hợp ToTensorV2, cho phép nạp dữ liệu siêu tốc trên `num_workers=4` mà không gây tràn RAM.
