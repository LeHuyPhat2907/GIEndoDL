# Hướng Dẫn Huấn Luyện Sản Xuất Mô Hình Đề Xuất Trên Kaggle GPU (Task #127)

**Dự án:** Luận văn Thạc sĩ / Cử nhân: *"Ứng dụng Học Sâu trong Nhận diện và Phân loại Tổn thương Bệnh lý Đường Tiêu hóa qua Ảnh Nội soi"*
**Repository:** `GIEndoDL` (HUIT, Sinh viên: Lê Huy Phát)
**Kiến trúc:** `Hybrid CNN-CBAM-Transformer` (ResNet50 + CBAM + Transformer 4 layers, 8 heads, 512D)
**Mục tiêu:** Huấn luyện mô hình đề xuất trên Kaggle GPU (Tesla P100 / T4 x2) với tùy chọn 50 epochs thử nghiệm trước hoặc 150 epochs toàn diện, xuất đầy đủ 100% artifacts, số liệu và biểu đồ khoa học.

---

## 1. Hai Cách Khởi Tạo Code Trên Kaggle

Bạn có thể chọn **1 trong 2 cách** dưới đây tùy theo thói quen:

### Cách A: Kéo code từ GitHub về (Khuyến nghị - Nhanh gọn & Đồng bộ nhất)
Mở một Notebook trống mới trên Kaggle, sau đó tạo 1 ô code và chạy lệnh:
```bash
!git clone https://github.com/LeHuyPhat2907/GIEndoDL.git
%cd GIEndoDL
!git pull origin main
```
> [!TIP]
> Với cách này, toàn bộ code chuẩn, các file configs, script huấn luyện và notebook đều tự động tải về đầy đủ trong `/kaggle/working/GIEndoDL`. Khi mã nguồn trên GitHub có cập nhật, bạn chỉ cần gõ `!git pull` là có ngay phiên bản mới nhất!

### Cách B: Upload trực tiếp file Notebook lên Kaggle
1. Nhấn **"+ Create"** ➔ **"New Notebook"**.
2. Chọn menu **File** ➔ **Upload Notebook**.
3. Chọn file [`notebooks/kaggle_train_optimal_model.ipynb`](file:///d:/PhatProject/GIEndoDL/notebooks/kaggle_train_optimal_model.ipynb) từ máy tính của bạn.

---

## 2. Thiết Lập Phần Cứng Tăng Tốc (Accelerator)

Ở bảng điều khiển bên phải (Notebook Settings):
1. **Accelerator:** Chọn **GPU P100** (ưu tiên hàng đầu) hoặc **GPU T4 x2**.
2. **Language:** Python.
3. **Persistence:** Variables and Files.
4. **Environment:** Always use latest environment.
5. **Internet:** **Bật (ON)** (bắt buộc để tải weights và dữ liệu).

---

## 3. Cách Tải Bộ Dữ Liệu HyperKvasir

Bạn có thể chuẩn bị dữ liệu bằng **Lệnh tự động** hoặc **Giao diện người dùng**:

### Cách 1: Tải bằng lệnh Python (Hoàn toàn tự động, không cần bấm chuột)
Chạy đoạn code sau trong một ô Notebook:
```python
import kagglehub

# Tải tự động dataset HyperKvasir qua thư viện chính thức của Kaggle
path = kagglehub.dataset_download("meetnagadia/hyperkvasir")
print("✅ Đường dẫn dataset đã sẵn sàng:", path)
```
Hoặc dùng lệnh Kaggle CLI trong terminal/cell:
```bash
!kaggle datasets download -d meetnagadia/hyperkvasir -p /kaggle/working/data --unzip
```

### Cách 2: Gắn qua giao diện "+ Add Input" (Tiết kiệm thời gian & bộ nhớ nhất)
1. Bấm nút **"+ Add Input"** ở góc trên bên phải màn hình Notebook.
2. Tại thanh tìm kiếm, gõ: `hyperkvasir` hoặc `hyperkvasir-dataset-segmented`.
3. Bấm dấu **"+"** màu xanh bên cạnh dataset.
*Ưu điểm: Kaggle gắn trực tiếp dữ liệu ảo qua mạng nội bộ chỉ trong 1 giây mà không phải chờ tải hàng chục GB về ổ đĩa làm đầy bộ nhớ.*

> [!NOTE]
> Hệ thống pipeline và notebook đã được lập trình cơ chế **đa đường dẫn (Multi-path Resolver)**: Tự động nhận diện dữ liệu dù bạn dùng nút "+ Add Input" hay tải bằng lệnh `kagglehub`!

---

## 4. Huấn Luyện 50 Epochs Trước Theo Yêu Cầu

### Nếu chạy qua Script Dòng Lệnh (Dành cho Cách Git Clone):
```bash
# Cài đặt các thư viện cần thiết
!pip install -q albumentations timm wandb scikit-learn matplotlib seaborn

# Chạy huấn luyện đúng 50 epochs (Warmup tự động thích ứng 4 epochs, Early stopping = 15)
!python src/training/train_optimal_model.py --epochs 50
```

### Nếu chạy qua Giao Diện Notebook:
Trong ô cấu hình của notebook [`notebooks/kaggle_train_optimal_model.ipynb`](file:///d:/PhatProject/GIEndoDL/notebooks/kaggle_train_optimal_model.ipynb):
```python
EPOCHS = 50                              # Huấn luyện 50 epochs thử nghiệm trước
WARMUP_EPOCHS = max(3, int(EPOCHS * 0.08)) # Tự động điều chỉnh Warmup = 4 epochs
PATIENCE = 15                            # Dừng sớm nếu không cải thiện sau 15 epochs
```
Nhấn **Run All** hoặc chạy tuần tự từng ô từ trên xuống dưới.

---

## 5. Cấu Hình Giám Sát Weights & Biases (Tùy Chọn)

Để theo dõi đồ thị trực tiếp trên điện thoại / máy tính thời gian thực:
1. Đăng ký tài khoản miễn phí tại [wandb.ai](https://wandb.ai/).
2. Lấy API Key tại [wandb.ai/authorize](https://wandb.ai/authorize).
3. Trong Kaggle Notebook: **Add-ons** ➔ **Secrets** ➔ Thêm Label `WANDB_API_KEY`, dán Key vào.
*(Nếu không điền, hệ thống sẽ tự động chuyển sang chế độ W&B Offline an toàn, không báo lỗi).*

---

## 6. Danh Mục Toàn Bộ Kết Quả Xuất Ra (100% Giống Các Lần Trước)

Sau khi chạy xong, toàn bộ kết quả học thuật sẽ nằm trong thư mục `/kaggle/working/`:

### 1. Trọng Số Mô Hình (Checkpoints)
- `checkpoints/best_model.pth`: Trọng số tại epoch có Macro F1 cao nhất.
- `checkpoints/last_model.pth`: Trọng số tại epoch dừng cuối cùng.

### 2. Số Liệu Báo Cáo Định Lượng (CSV & JSON & Markdown)
- `metrics/training_history.csv`: Bảng lịch sử từng epoch gồm `[epoch, train_loss, val_loss, val_accuracy, val_macro_f1, learning_rate, is_best]`.
- `metrics/per_class_metrics.csv`: Bảng đánh giá chi tiết toàn bộ 23 lớp bệnh lý HyperKvasir (`support_samples`, `precision_pct`, `recall_pct`, `f1_score_pct`, `auc_roc_pct`).
- `metrics/training_summary.json`: File tổng hợp siêu tham số, cấu hình phần cứng, số lượng epoch tốt nhất, Macro F1, Macro Precision, Recall, AUC-ROC, ECE.
- `metrics/train_optimal_model.md`: Báo cáo học thuật định dạng chuẩn luận văn.

### 3. Biểu Đồ Khoa Học Phục Vụ Luận Văn (Figures - 300 DPI)
- `figures/optimal_training_convergence_curves.png`: Đồ thị 4 ô trực quan hóa suy giảm hàm mất mát, tăng trưởng độ chính xác/F1, lịch trình hạ nhiệt Cosine LR có Warmup, và F1 nhóm bệnh trọng tâm.
- `figures/confusion_matrix_optimal.png`: Ma trận nhầm lẫn 23x23 chuẩn hóa tỷ lệ nhận diện.
- `figures/per_class_f1_bar_chart.png`: Biểu đồ cột ngang sắp xếp toàn bộ 23 phân lớp bệnh lý kèm đường chuẩn trung bình.

### Tải Kết Quả Về Máy:
Chỉ cần mở bảng **Output** ở cột bên phải giao diện Kaggle ➔ Bấm nút **Download** hoặc tải file zip về máy tính cá nhân.
