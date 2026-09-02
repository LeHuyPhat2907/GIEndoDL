# 💻 Báo cáo Kỹ thuật: Thiết Lập Môi Trường Kép Đồng Bộ Local & Google Colab

> **File cấu hình:** `environment.yml` | **Hình minh họa:** `docs/figures/38_local_vs_colab_dual_environment.png`

---

## 1. Phân Định Rạch Ròi Nhiệm Vụ Giữa Hai Môi Trường

| Môi trường | Nền tảng phần cứng | Nhiệm vụ chính trong Đề tài | Thiết lập tối ưu |
|:---|:---|:---|:---|
| **Local Machine** | Windows 11 / WSL2 (CPU) | Viết code, tái cấu trúc, debug, kiểm thử Pytest, quản trị Git | `device='cpu'`, `num_workers=0` |
| **Google Colab** | Ubuntu Linux (GPU Tesla T4 16GB) | Huấn luyện quy mô lớn 50-100 Epochs, đồng bộ trọng số Google Drive | `device='cuda'`, `num_workers=2`, `AMP FP16` |

---

## 2. Tính Năng Tự Thích Ứng Runtime (Automatic Environment Adaptation)

Hàm `detect_runtime_environment()` trong module `src/utils/environment_mirror.py` tự động phát hiện môi trường thực thi để chuyển đổi cấu hình phù hợp, giúp cùng một mã nguồn có thể chạy liền mạch cả trên máy cá nhân lẫn trên máy chủ đám mây mà không cần chỉnh sửa thủ công.
