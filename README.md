# 🔬 GIEndoDL: Deep Learning for Gastrointestinal Lesion Detection & Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TRIPOD-AI](https://img.shields.io/badge/Compliance-TRIPOD--AI-orange.svg)](docs/)
[![Dataset](https://img.shields.io/badge/Dataset-HyperKvasir-brightgreen.svg)](https://endoskop.kvasir.no/)

> **TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM (HUIT)**  
> **KHOA CÔNG NGHỆ THÔNG TIN**  
> **Khóa luận Cử nhân Công nghệ Thông tin (Hướng Nghiên cứu Khoa học)**  
> **Đề tài:** Ứng dụng học sâu trong nhận diện và phân loại tổn thương đường tiêu hóa từ ảnh nội soi.

---

## 📌 1. Giới thiệu & Mục tiêu Nghiên cứu

Nội soi tiêu hóa là phương pháp tiêu chuẩn vàng trong chẩn đoán các bệnh lý thực quản, dạ dày và đại tràng. Tuy nhiên, việc bỏ sót tổn thương (miss-rate) do mệt mỏi thị giác hoặc biến thiên độ sáng/nhiễu thiết bị vẫn là một thách thức lớn.

Dự án **GIEndoDL** được xây dựng nhằm phát triển một hệ thống hỗ trợ chẩn đoán AI tiên tiến, kết hợp giữa khả năng trích xuất đặc trưng cục bộ mịn của CNN, cơ chế chú ý kênh-không gian của CBAM, mối liên kết ngữ cảnh toàn cục của Vision Transformer, cùng kỹ thuật Học tương phản (Contrastive Learning).

### 🎯 4 Mục tiêu chính:
1. **Chuẩn hóa, Phân đoạn & Xử lý mất cân bằng dữ liệu:** Tự động tiền xử lý ảnh nội soi, phân đoạn vùng tổn thương (ROI), giải quyết hiện tượng mất cân bằng lớp nghiêm trọng bằng *Weighted Sampling* và *Class-Balanced Focal Loss*.
2. **Cải tiến Thuật toán & Đóng góp Kỹ thuật:** Đề xuất kiến trúc lai **CNN-CBAM-Transformer** kết hợp Supervised Contrastive Learning (SupCon) và đặc trưng kết cấu/màu sắc truyền thống.
3. **Đánh giá Toàn diện & Thẩm định Thống kê:** Đánh giá đa chiều qua các chỉ số y tế (*Accuracy, Precision, Recall, F1-score, AUC-ROC*), thẩm định độc lập (*External Validation*) trên tập dữ liệu ngoài và kiểm thử ý nghĩa thống kê ($p < 0.05$). Tuân thủ hướng dẫn báo cáo y tế **TRIPOD-AI**.
4. **Tích hợp Web App, Interactive XAI & CBMIR:** Xây dựng ứng dụng Web real-time tích hợp **Interactive Grad-CAM++** (cho phép bác sĩ tương tác phản hồi Active Learning) và module **CBMIR** (truy vấn tìm kiếm hình ảnh y học tương đồng làm chứng cứ đối chiếu lâm sàng).

---

## 📊 2. Bộ Dữ liệu (Dataset Attribution)

Nghiên cứu sử dụng các bộ dữ liệu ảnh nội soi tiêu hóa tiêu chuẩn quốc tế do **Phòng thí nghiệm Nghiên cứu Simula (Simula Research Laboratory)** và **Bệnh viện Bærum (Na Uy)** công bố:

- **HyperKvasir:** 10,662 ảnh nội soi được gắn nhãn thuộc 23 lớp bệnh lý và mốc giải phẫu (Polyps, Ulcerative Colitis, Esophagitis, Barrett's, Pylorus, Cecum...).
- **Kvasir-v2:** 8,000 ảnh (8 lớp) phục vụ mở rộng tập huấn luyện và thẩm định chéo.
- **Kvasir-SEG:** 1,000 ảnh polyp đi kèm mặt nạ phân đoạn chuẩn (Ground-truth segmentation masks) phục vụ huấn luyện module tự động phân đoạn ROI.

---

## 🏗️ 3. Cấu trúc Thư mục Dự án

```text
GIEndoDL/
├── checkpoints/          # Trọng số mô hình (.pth, .onnx) & Checkpoints huấn luyện
├── configs/              # Các file cấu hình siêu tham số (YAML/JSON)
├── data/                 # Dữ liệu ảnh nội soi (đã phân chia Train/Val/Test)
│   ├── raw/              # Ảnh gốc tải từ Simula Lab
│   └── processed/        # Ảnh đã qua tiền xử lý, CLAHE & Crop ROI
├── docs/                 # Quyển báo cáo khóa luận, Slide, TRIPOD-AI Checklist
├── notebooks/            # Jupyter Notebooks cho EDA, thử nghiệm pilot
├── src/                  # Mã nguồn chính phát triển mô hình PyTorch
│   ├── dataset/          # Dataset Loader, Data Augmentation (Albumentations)
│   ├── losses/           # Focal Loss, Class-Balanced Loss, SupCon Loss
│   ├── models/           # CNN Backbones, CBAM Module, Vision Transformer, Hybrid
│   ├── utils/            # Thống kê (t-test, McNemar), Metrics, Seed configuration
│   └── xai/              # Module trực quan hóa Grad-CAM++, Attention Maps
├── web_app/              # Ứng dụng Web hỗ trợ bác sĩ lâm sàng
│   ├── backend/          # RESTful API bằng FastAPI, ONNX Runtime Inference Engine
│   └── frontend/         # Giao diện tương tác người dùng (HTML/CSS/JS)
├── .gitattributes        # Cấu hình Git LFS (Large File Storage)
├── .gitignore            # Cấu hình chặn push file nặng/tạm lên GitHub
├── LICENSE               # Giấy phép mã nguồn mở MIT
├── requirements.txt      # Thư viện Python phụ thuộc
└── README.md             # Tài liệu hướng dẫn dự án
```

## 🚀 4. Hướng dẫn Cài đặt & Khởi chạy
### Yêu cầu Hệ thống
- Python >= 3.9
- CUDA >= 11.8 (Khuyên dùng GPU NVIDIA T4/V100/A100 hoặc RTX series)
- PyTorch >= 2.0

### Clone repository
- git clone https://github.com/username/GIEndoDL.git
- cd GIEndoDL

### Tạo môi trường ảo (tùy chọn)
python -m venv venv
#### Trên Windows:
.\venv\Scripts\activate
#### Trên Linux/macOS:
source venv/bin/activate

### Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

## 📜 Giấy phép & Trích dẫn (License & Citation)
- GIEndoDL Project - Bachelor Thesis (2026)
- HCMC University of Industry and Trade (HUIT)
- Phát hành theo giấy phép MIT License.



