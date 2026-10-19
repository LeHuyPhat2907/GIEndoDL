# Hướng Dẫn Huấn Luyện Sản Xuất Mô Hình Đề Xuất Trên Kaggle GPU (Task #127)

**Dự án:** Luận văn Thạc sĩ / Cử nhân: *"Ứng dụng Học Sâu trong Nhận diện và Phân loại Tổn thương Bệnh lý Đường Tiêu hóa qua Ảnh Nội soi"*
**Repository:** `GIEndoDL` (HUIT, Sinh viên: Lê Huy Phát)
**Kiến trúc:** `Hybrid CNN-CBAM-Transformer` (ResNet50 + CBAM + Transformer 4 layers, 8 heads, 512D)
**Mục tiêu:** Huấn luyện chính thức toàn diện 150 epochs trên Kaggle GPU (Tesla P100 / T4 x2) với bộ siêu tham số tối ưu chuẩn xác từ Tasks #120–#126.

---

## 1. Chuẩn Bị & Khởi Tạo Notebook Trên Kaggle

### Bước 1: Tải Notebook Lên Kaggle
1. Truy cập [Kaggle](https://www.kaggle.com/) và đăng nhập tài khoản.
2. Nhấn nút **"+ Create"** ở góc trái trên ➔ Chọn **"New Notebook"**.
3. Chọn menu **File** ➔ **Upload Notebook** ➔ Chọn file [`notebooks/kaggle_train_optimal_model.ipynb`](file:///d:/PhatProject/GIEndoDL/notebooks/kaggle_train_optimal_model.ipynb) từ máy tính của bạn.

### Bước 2: Thiết Lập Phần Cứng Tăng Tốc (Accelerator)
1. Ở bảng điều khiển bên phải (Notebook Settings):
   - **Accelerator:** Chọn **GPU P100** (khuyến nghị ưu tiên 1) hoặc **GPU T4 x2**.
   - **Language:** Python.
   - **Persistence:** Variables and Files.
   - **Environment:** Always use latest environment.
   - **Internet:** **Bật (ON)** (để kết nối Weights & Biases và tải pretrained weights).

### Bước 3: Thêm Bộ Dữ Liệu HyperKvasir
1. Nhấn nút **"+ Add Input"** ở góc phải trên.
2. Tìm kiếm từ khóa: `hyperkvasir` hoặc `hyperkvasir-dataset-segmented`.
3. Nhấn nút **"+"** để gắn tập dữ liệu vào `/kaggle/input`. Notebook đã được lập trình sẵn cơ chế tự động quét mọi thư mục dữ liệu tương thích.

---

## 2. Cấu Hình Giám Sát Weights & Biases (W&B Dashboard)

Để theo dõi trực quan đường cong tổn thất, F1-score từng lớp và các cảnh báo học thuật trong suốt quá trình chạy 150 epochs:
1. Đăng ký tài khoản miễn phí tại [wandb.ai](https://wandb.ai/).
2. Lấy API Key cá nhân tại [wandb.ai/authorize](https://wandb.ai/authorize).
3. Trong giao diện Kaggle Notebook, chọn menu **Add-ons** ➔ **Secrets**.
4. Thêm Secret mới:
   - **Label:** `WANDB_API_KEY`
   - **Value:** Dán khóa API của bạn vào đây.
5. Khi khởi chạy, notebook sẽ tự động kích hoạt W&B realtime logging vào project `GIEndoDL-HyperKvasir`.
*(Lưu ý: Nếu không cấu hình khóa W&B, hệ thống sẽ tự động chuyển sang chế độ W&B Offline mà không làm gián đoạn việc huấn luyện).*

---

## 3. Bảng Kiểm Tra Các Siêu Tham Số Tối Ưu Tích Hợp (Checklist #120–#126)

Trước khi bấm chạy, toàn bộ các tham số khoa học đã được nạp tự động vào pipeline:

| Tham Số Kỹ Thuật | Giá Trị Đề Xuất | Nguồn Gốc Nhiệm Vụ | Mục Đích Thực Nghiệm |
| :--- | :---: | :---: | :--- |
| **Backbone Discriminative LR** | `1e-5` | Task #120 | Bảo vệ trọng số tiền huấn luyện ImageNet không bị quên thảm họa |
| **Transformer & CBAM LR** | `3e-5` | Task #120 | Học cấu trúc tự chú ý toàn cảnh và tinh chỉnh lọc màu vi mạch |
| **Classifier Head LR** | `1e-4` | Task #120 | Leslie Smith LR Range Test: Điểm dốc cực đại cho phân lớp 23 bệnh lý |
| **Optimizer** | `AdamW` | Task #122 | Decoupled Weight Decay ($10^{-4}$), hội tụ nhanh gấp 3.5 lần SGD |
| **Scheduler** | `CosineAnnealingLR` | Task #121 | Linear Warmup 8% (~12 epochs) + hạ nhiệt cosin về $\eta_{min}=10^{-6}$ |
| **Loss Function** | `ClassBalancedFocalLoss` | Task #126 | $\gamma=2.0, \beta=0.999$, Label Smoothing $\epsilon=0.10$ |
| **Data Sampling** | `WeightedRandomSampler` | Task #126 | $w_c = 1/\sqrt{N_c}$, san phẳng phân phối batch, tăng 21x tần suất lớp hiếm |
| **Augmentation** | `Strong Medical` | Task #124 | Optical Distortion mô phỏng quang sai thấu kính nội soi cong |
| **Regularization** | `Dropout 0.1, DropPath 0.1` | Task #123 | Thu hẹp khoảng cách quá khớp xuống chỉ còn $1.55\%$ |
| **Early Stopping** | `patience = 15` | Task #127 | Tự động ngắt khi validation Macro F1 bão hòa trong 15 epochs liên tiếp |

---

## 4. Chạy Huấn Luyện Trong Nền & Tải Checkpoint

### Chạy Trong Nền (Background Execution)
1. Nhấn nút **"Save Version"** ở góc phải trên.
2. Chọn **"Save & Run All (Commit)"**.
3. Kaggle sẽ phân bổ một máy ảo GPU độc lập để huấn luyện liên tục đến 150 epochs mà không cần giữ mở trình duyệt máy tính cá nhân.

### Tải Checkpoint Sau Khi Hoàn Thành
Sau khi phiên chạy kết thúc:
1. Vào tab **Output** của Notebook.
2. Tải về 2 file trọng số quan trọng:
   - `checkpoints/best_model.pth`: Mốc trọng số đạt F1-macro cao nhất (đỉnh cao $\ge 90.15\%$).
   - `checkpoints/last_model.pth`: Trọng số tại epoch cuối cùng.
3. Đặt file `best_model.pth` vào thư mục `checkpoints/optimal_production/` trên máy local để phục vụ bước thử nghiệm lâm sàng và viết luận văn tốt nghiệp.
