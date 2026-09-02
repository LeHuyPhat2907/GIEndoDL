# ⚡ Báo cáo Kỹ thuật: Cấu Hình Tối Ưu Hóa DataLoader & Đo Lường Băng Thông Nạp (Throughput)

> **File cấu hình:** `configs/dataloader_config.json` | **Hình minh họa:** `docs/figures/27_dataloader_throughput_benchmark.png`

---

## 1. Bảng Kết Quả Đo Lường Hiệu Năng Thực Tế

| Cấu hình Luồng (Workers) | Pin Memory | Tốc độ nạp (Ảnh/giây) | Độ trễ mỗi Batch (ms) | Hệ số Tăng tốc |
|:---|:---:|:---:|:---:|:---:|
| **1. Single Worker (Baseline)** | `True` | **`49.5 samples/s`** | `646.3 ms` | **`1.00x`** |
| **2. Dual Workers (Balanced)** | `True` | **`106.2 samples/s`** | `301.2 ms` | **`2.15x`** |
| **3. Quad Workers (High Speed)** | `True` | **`181.0 samples/s`** | `176.8 ms` | **`3.66x`** |

---

## 2. Giải Pháp Triệt Tiêu Hiện Tượng Nghẽn Cổ Chai (Anti-Bottleneck Strategy)

1. **DMA Memory Pinning:** Khóa cứng bộ nhớ đệm giúp GPU sao chép Tensor trực tiếp từ RAM mà không qua trung gian CPU, tăng tốc truyền dữ liệu lên Card đồ họa.
2. **Prefetch Queue:** Cơ chế nạp đón đầu giúp triệt tiêu hoàn toàn thời gian chết (GPU Idle Time) giữa các bước tính toán Gradient.
