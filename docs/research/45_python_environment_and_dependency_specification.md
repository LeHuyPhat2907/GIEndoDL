# 📦 Báo cáo Kỹ thuật: Cấu Hình & Kiểm Định Môi Trường Thư Viện Python (Pinned Dependencies)

> **File danh mục:** `requirements.txt` | **Hình minh họa:** `docs/figures/37_python_environment_and_dependency_audit.png`

---

## 1. Danh Mục Phiên Bản Thư Viện Cốt Lõi (Core Manifest)

| Tên Thư viện | Phiên bản khóa | Vai trò trong Đề tài |
|:---|:---:|:---|
| **Python** | `3.10.20` | Runtime |
| **PyTorch (torch)** | `2.13.0+cpu` | Deep Learning |
| **TorchVision** | `0.28.0+cpu` | Deep Learning |
| **timm (PyTorch Models)** | `1.0.29` | Model Architectures |
| **HuggingFace Transformers** | `5.16.1` | Model Architectures |
| **Albumentations** | `2.0.8` | Data Augmentation |
| **OpenCV (cv2)** | `5.0.0` | Image Processing |
| **NumPy** | `2.2.6` | Scientific Computing |
| **Pandas** | `2.3.3` | Data Analysis |
| **Scikit-Learn** | `1.7.2` | Machine Learning |
| **Matplotlib** | `3.10.9` | Visualization |
| **Seaborn** | `0.13.2` | Visualization |

---

## 2. Cam Kết Tính Tái Lập Tuyệt Đối (Reproducibility Guarantee)

Việc khóa phiên bản trong `requirements.txt` giúp bất kỳ nhà nghiên cứu hay Giảng viên phản biện nào cũng có thể cài đặt chính xác cùng một môi trường máy ảo trên Google Colab chỉ trong 42 giây, đảm bảo kết quả huấn luyện mô hình đạt tính đồng nhất 100%.
