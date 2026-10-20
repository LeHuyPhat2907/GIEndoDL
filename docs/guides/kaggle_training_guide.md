# Hướng Dẫn Huấn Luyện Sản Xuất Mô Hình Đề Xuất Trên Kaggle GPU (Task #127)

**Dự án:** Luận văn Thạc sĩ / Cử nhân: *"Ứng dụng Học Sâu trong Nhận diện và Phân loại Tổn thương Bệnh lý Đường Tiêu hóa qua Ảnh Nội soi"*
**Repository:** `GIEndoDL` (HUIT, Sinh viên: Lê Huy Phát)
**Kiến trúc:** `Hybrid CNN-CBAM-Transformer` (ResNet50 + CBAM + Transformer 4 layers, 8 heads, 512D)
**Mục tiêu:** Huấn luyện mô hình đề xuất trên Kaggle GPU (Tesla P100 / T4 x2) với tùy chọn 50 epochs thử nghiệm trước hoặc 150 epochs toàn diện, xuất đầy đủ 100% artifacts, số liệu và biểu đồ khoa học.

---

## 1. Thiết Lập Môi Trường Kaggle (Chỉ Mất 30 Giây)

1. Truy cập [Kaggle](https://www.kaggle.com/) ➔ Bấm **"+ Create"** ➔ Chọn **"New Notebook"**.
2. Ở cột **Notebook Settings** bên phải:
   - **Accelerator:** Chọn **GPU P100** (khuyến nghị ưu tiên 1) hoặc **GPU T4 x2**.
   - **Language:** Python.
   - **Persistence:** Variables and Files.
   - **Internet:** **Bật (ON)** *(Bắt buộc để tải code từ GitHub và dữ liệu từ Simula)*.

---

## 2. Quy Trình 3 Ô Code Huấn Luyện Chuẩn (Giống Các Lần Trước)

Bạn chỉ cần tạo lần lượt 3 ô code trong Notebook Kaggle và bấm chạy:

### Ô Cell 1: Kéo Code từ GitHub & Cài Thư Viện
```bash
# 1. Kéo mã nguồn dự án về máy ảo Kaggle
!git clone https://github.com/LeHuyPhat2907/GIEndoDL.git /kaggle/working/GIEndoDL
%cd /kaggle/working/GIEndoDL

# 2. Cài đặt các thư viện phụ thuộc
!pip install -q timm albumentations scikit-learn seaborn matplotlib tabulate wandb
```

### Ô Cell 2: Tải Dữ Liệu HyperKvasir Tự Động Bằng Code (~1.5 đến 2 phút)
```bash
# Tự động tải 3.9 GB ảnh có nhãn từ máy chủ Simula, giải nén và dọn dẹp file zip
!python src/dataset/download_hyperkvasir.py
```
> [!NOTE]
> Script `download_hyperkvasir.py` tải trực tiếp từ máy chủ chính thức của **Simula Research Laboratory**:
> - Hiển thị thanh tiến trình `tqdm` trực quan.
> - Giải nén trực tiếp vào `data/raw/labeled-images`.
> - Tự động xóa file zip sau khi giải nén để tiết kiệm tối đa dung lượng đĩa của Kaggle.
> - **Không cần Kaggle API Key**, **không bị lỗi 403 Forbidden**!

### Ô Cell 3: Bắt Đầu Huấn Luyện 50 Epochs Trước
```bash
# Huấn luyện 50 epochs (Warmup 4 epochs, Early Stopping patience = 15)
!python src/training/train_optimal_model.py --epochs 50
```

---

## 3. Danh Mục Toàn Bộ Kết Quả Xuất Ra (100% Giống Các Lần Trước)

Sau khi chạy xong, toàn bộ kết quả phục vụ viết luận văn tốt nghiệp sẽ được lưu đầy đủ tại thư mục `/kaggle/working/` (hoặc `docs/benchmarks/` & `docs/figures/`):

### 1. Trọng Số Mô Hình (Checkpoints)
- `checkpoints/best_model.pth`: Trọng số tại epoch có Macro F1 cao nhất.
- `checkpoints/last_model.pth`: Trọng số tại epoch dừng cuối cùng.

### 2. Số Liệu Báo Cáo Định Lượng (CSV & JSON & Markdown)
- `docs/benchmarks/training_history.csv`: Bảng lịch sử từng epoch (`train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1`, `learning_rate`, `is_best`).
- `docs/benchmarks/per_class_metrics.csv`: Bảng đánh giá chi tiết toàn bộ 23 lớp bệnh lý (`support`, `precision`, `recall`, `f1_score`, `auc_roc`).
- `docs/benchmarks/train_optimal_model.json`: JSON tổng kết toàn bộ siêu tham số, phần cứng, peak metrics.
- `docs/benchmarks/train_optimal_model.md`: Báo cáo học thuật định dạng chuẩn APA/LaTeX.

### 3. Biểu Đồ Khoa Học Phục Vụ Luận Văn (Figures - 300 DPI)
- `docs/figures/optimal_training_convergence_curves.png`: Đồ thị 4 ô trực quan hóa suy giảm hàm mất mát, tăng trưởng độ chính xác/F1, lịch trình hạ nhiệt Cosine LR có Warmup, và F1 nhóm bệnh trọng tâm.
- `docs/figures/confusion_matrix_optimal.png`: Ma trận nhầm lẫn 23x23 chuẩn hóa tỷ lệ nhận diện.
- `docs/figures/per_class_f1_bar_chart.png`: Biểu đồ cột ngang sắp xếp toàn bộ 23 phân lớp bệnh lý kèm đường chuẩn trung bình.

### Tải Kết Quả Về Máy:
Chỉ cần mở bảng **Output** ở cột bên phải giao diện Kaggle ➔ Bấm nút **Download** hoặc tải file zip về máy tính cá nhân.
