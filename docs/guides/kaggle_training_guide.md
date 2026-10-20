# Hướng Dẫn Huấn Luyện & Xuất Báo Cáo Lâm Sàng Mô Hình Lai Đề Xuất (Task #127)

**Dự án:** Luận văn Thạc sĩ / Cử nhân: *"Ứng dụng Học Sâu trong Nhận diện và Phân loại Tổn thương Bệnh lý Đường Tiêu hóa qua Ảnh Nội soi"*
**Repository:** `GIEndoDL` (HUIT, Sinh viên: Lê Huy Phát)
**Mô hình:** `Hybrid CNN-CBAM-Transformer` (ResNet50 + CBAM + Transformer 4 layers, 8 heads, 512D)
**Mục tiêu:** Huấn luyện thực tế trên GPU và xuất trọn bộ báo cáo y khoa chuyên sâu (21 files độc lập, không trộn lẫn bất kỳ mô hình hay task nào khác).

---

## 1. Thiết Lập Môi Trường Kaggle

1. Trên Kaggle, mở Notebook của bạn và kiểm tra:
   - **Accelerator:** GPU P100 (hoặc GPU T4 x2).
   - **Internet:** Bật (ON).

---

## 2. Quy Trình Chạy Lệnh

### Ô Cell 1: Kéo Code Mới Nhất từ GitHub & Cài Thư Viện
```bash
!git clone https://github.com/LeHuyPhat2907/GIEndoDL.git /kaggle/working/GIEndoDL
%cd /kaggle/working/GIEndoDL
!git pull origin main
!pip install -q timm albumentations scikit-learn seaborn matplotlib tabulate wandb
```

### Ô Cell 2: Tải Dữ Liệu HyperKvasir Tự Động (Nếu chưa tải)
```bash
!python src/dataset/download_hyperkvasir.py
```

### Ô Cell 3: Huấn Luyện Thật & Xuất Trọn Bộ Kết Quả Riêng Biệt

#### 👉 Lựa chọn 1: Huấn luyện lại từ đầu 50 Epochs THẬT trên GPU
```bash
!python src/training/train_optimal_model.py --epochs 50
```

#### 👉 Lựa chọn 2: Trích xuất NGAY LẬP TỨC từ 50 Epochs bạn vừa chạy xong (Chỉ mất ~30 giây)
```bash
!python src/training/train_optimal_model.py --export-only
```

---

## 3. Danh Sách 21 File Kết Quả Độc Lập Cho Mô Hình Đề Xuất

Sau khi chạy xong, toàn bộ 21 files sẽ nằm gọn trong thư mục riêng biệt `/kaggle/working/optimal_hybrid_model/`:

| Tên File | Định Dạng | Mô Tả Ý Nghĩa Khoa Học / Lâm Sàng |
| :--- | :---: | :--- |
| `best_model.pth` | PTH (~146 MB) | Trọng số mô hình tối ưu nhất tại epoch đạt kỷ lục |
| `last_model.pth` | PTH (~146 MB) | Trọng số mô hình tại epoch kết thúc |
| `classification_report.txt` | TXT | Báo cáo phân loại tổng hợp chuẩn Scikit-Learn |
| `clinical_report.md` | Markdown | Báo cáo đánh giá y khoa toàn diện (Top 5 lớp tốt/kém, Cohen's Kappa, MCC) |
| `confusion_matrix_normalized.csv` | CSV | Ma trận nhầm lẫn chuẩn hóa tỷ lệ phần trăm nhận diện |
| `confusion_matrix_normalized.png` | PNG (300 DPI) | Heatmap ma trận nhầm lẫn chuẩn hóa độ phân giải cao |
| `confusion_matrix_raw.csv` | CSV | Ma trận nhầm lẫn đếm số lượng ảnh thực tế |
| `confusion_matrix_raw.png` | PNG (300 DPI) | Heatmap ma trận đếm số lượng ca nhận diện |
| `error_cases.csv` | CSV | Danh sách từng ca chẩn đoán nhầm xếp theo độ tự tin để bác sĩ phân tích |
| `val_predictions.csv` | CSV | Bảng chi tiết toàn bộ ảnh test kèm xác suất của 23 lớp |
| `per_class_metrics.csv` | CSV | Bảng chi tiết 23 lớp (Precision, Sensitivity, Specificity, F1, Support) |
| `per_class_metrics.json` | JSON | Dữ liệu per-class định dạng JSON phục vụ vẽ biểu đồ |
| `per_class_f1_ranking.png` | PNG (300 DPI) | Biểu đồ xếp hạng F1-Score 23 lớp có vạch chuẩn y khoa 90% |
| `per_class_precision_recall.png` | PNG (300 DPI) | Biểu đồ cột kép đối chiếu Precision vs Recall từng lớp |
| `training_curves.png` | PNG (300 DPI) | Dashboard 4 khung (Hội tụ Loss, Acc & F1, Tốc độ học, Thời gian epoch) |
| `optimal_training_convergence_curves.png` | PNG (300 DPI) | Biểu đồ hội tụ mở rộng có nhóm bệnh lý trọng tâm |
| `training_history.csv` | CSV | Lịch sử chi tiết 50 epochs (Loss, Acc, F1, LR, Time) |
| `history.csv` | CSV | File lịch sử đồng bộ tương thích |
| `training_summary.json` | JSON | Tóm lược cấu hình phần cứng, tham số tối ưu và các mốc kỷ lục |
| `fold_metrics.json` | JSON | Bảng chỉ số tổng thể (Accuracy, Macro F1, Kappa, MCC) |
| `run_config.json` | JSON | File cấu hình tham số đảm bảo tính tái lập 100% |

---

## 4. Tải File Nén Về Máy Tính

Tất cả 21 files trên được nén độc lập duy nhất vào file **`optimal_hybrid_model_results.zip`**.

### Cách A: Tải trực tiếp siêu tốc bằng 1 dòng lệnh (Khuyên dùng)
```bash
!curl -T /kaggle/working/optimal_hybrid_model_results.zip https://transfer.sh/optimal_hybrid_model_results.zip
```
*(Bấm vào link rút gọn in ra trên màn hình để tải ngay về máy tính).*

### Cách B: Tải qua bảng Output của Kaggle
- Ở cột phải Notebook ➔ Kéo xuống mục **Output** ➔ Bấm nút **Refresh 🔄** ➔ Bấm dấu **3 chấm `...`** cạnh file `optimal_hybrid_model_results.zip` ➔ Chọn **Download**.
