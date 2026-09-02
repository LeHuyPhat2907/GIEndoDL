# 🚀 Báo cáo Kỹ thuật: Cấu Hình Môi Trường Google Colab Free & Tối Ưu Hóa GPU Tesla T4 (0 Đồng)

> **File cấu hình:** `configs/colab_env_config.json` | **Hình minh họa:** `docs/figures/36_colab_free_gpu_t4_audit.png`

---

## 1. Phân Bổ Ngân Sách Phần Cứng Colab Free (NVIDIA Tesla T4)

| Tài nguyên (Resource) | Thông số kỹ thuật | Cơ chế tối ưu hóa cho Đề tài |
|:---|:---|:---|
| **GPU Model** | **NVIDIA Tesla T4 (Turing)** | Nhân Tensor Cores hỗ trợ tính toán số thực nửa chính xác FP16 |
| **VRAM Dung lượng** | **15.3 GB GDDR6** | Dư dả cho cả CNN (3.1GB) lẫn Swin Transformer (5.2GB) khi bật AMP |
| **Tốc độ Tính toán** | **65.0 TFLOPs (FP16)** | Nhanh gấp 8 lần so với FP32 truyền thống |
| **Thời gian Phiên** | 4 - 12 tiếng liên tục | Tự động đồng bộ Checkpoint về Google Drive sau mỗi Epoch |

---

## 2. Các Quy Tắc Vàng Vận Hành Huấn Luyện

1. **Bật Automatic Mixed Precision (AMP):** Giảm tải bộ nhớ xuống 50% và tăng tốc độ hội tụ.
2. **Lưu trữ Checkpoint Đám mây:** Mọi trọng số `best_model.pth` được lưu thẳng vào `/content/drive/MyDrive/GIEndoDL_Models/` đảm bảo an toàn tuyệt đối.
