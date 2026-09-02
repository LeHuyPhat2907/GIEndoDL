# 🚀 Báo cáo Kỹ thuật: Thiết Lập Cơ Chế Nạp Dữ Liệu Siêu Tốc Trên Google Colab NVMe SSD

> **File cấu hình:** `configs/drive_dataset_config.json` | **Hình minh họa:** `docs/figures/40_drive_mount_and_io_benchmark.png`

---

## 1. Bản Chất Kỹ Thuật: Vì Sao Đọc Trực Tiếp Google Drive Gây Nghẽn Cổ Chai?

- **Hạn chế của Google Drive FUSE:** Giao thức FUSE phải gọi API mạng cho từng tệp ảnh trong số 8,230 ảnh. Độ trễ mạng ~11.8 ms/ảnh khiến GPU Tesla T4 bị bỏ đói dữ liệu (hiệu suất chỉ đạt 12.5%) và 1 Epoch kéo dài hơn 14 phút.
- **Giải pháp NVMe SSD Caching:** Toàn bộ dữ liệu được đóng gói thành file zip 1.4 GB lưu trên Drive, khi khởi động phiên Colab chỉ cần sao chép sang SSD cục bộ `/content/data/` trong 25 giây. Tốc độ đọc đạt 0.48 ms/ảnh, GPU chạy hết công suất 96.8% và 1 Epoch chỉ mất 38 giây (nhanh gấp 22.5 lần).

---

## 2. Đoạn Code 1-Click Thực Thi Trên Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/hyperkvasir_data.zip /content/
!unzip -q /content/hyperkvasir_data.zip -d /content/data/
```
