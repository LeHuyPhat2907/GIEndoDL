# Báo Cáo Huấn Luyện Sản Xuất Mô Hình Đề Xuất (Task #127)

**Thời gian kiểm chứng:** 2026-10-19 18:03:47
**Môi trường thực thi:** Kaggle GPU Accelerator (Tesla P100 / T4 x2)
**Tập dữ liệu:** HyperKvasir Challenge (23 classes, 10,662 ảnh)

## 1. Bảng Thông Số Huấn Luyện Tối Ưu Tích Hợp (Tasks #120–#126)

| Thành Phần Hệ Thống | Giá Trị Cấu Hình Tối Ưu | Nguồn Gốc Thực Nghiệm | Vai Trò Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| **Kiến Trúc Mô Hình** | `CNN-CBAM-Transformer` | Tasks #94–#101 | ResNet50 + CBAM + Transformer 4 layers 8 heads |
| **Số Epochs Cấu Hình** | `150 epochs` | Task #127 | Huấn luyện sâu toàn diện với cơ chế Early Stopping |
| **Early Stopping** | `patience = 15` | Task #127 | Dừng sớm dựa trên metric `val_macro_f1` |
| **Bộ Tối Ưu Hóa** | `AdamW (WD = 1e-4)` | Task #122 | Decoupled Weight Decay chuẩn mực cho Vision Transformers |
| **Tốc Độ Học Phân Biệt** | `Head: 1e-4, Backbone: 1e-5` | Task #120 | Xác lập chính xác qua Leslie Smith LR Range Test |
| **Bộ Điều Phối LR** | `CosineAnnealingLR (Warmup 8%)` | Task #121 | Hạ nhiệt cosin trơn tru về $10^{-6}$ |
| **Hàm Mất Mát** | `CB-Focal Loss (γ=2.0, β=0.999)` | Task #126 | Hóa giải tỷ lệ mất cân bằng cực đoan 191:1 |
| **Cơ Chế Lấy Mẫu** | `WeightedRandomSampler` | Task #126 | $w_c = 1 / \sqrt{N_c}$, san phẳng phân bố mini-batch |
| **Tăng Cường Dữ Liệu** | `Strong Medical Augmentation` | Task #124 | Bổ sung Optical Distortion & Blur mô phỏng ống soi |
| **Điều Chuẩn** | `Dropout 0.1, DropPath 0.1` | Task #123 | Kiểm soát Overfitting Gap ở mức tối ưu $1.55\%$ |

## 2. Kết Quả Đạt Được Tại Mốc Kỷ Lục

- **Epoch Kỷ Lục (Best Epoch):** `Epoch 146`
- **Validation F1-macro:** **`90.22%`** (Vượt chuẩn 90%)
- **Validation Accuracy:** **`93.84%`**
- **ROC-AUC Toàn Cục:** **`99.08%`**
- **Dual Checkpoints:** Lưu trữ thành công tại `checkpoints/optimal_production/best_model.pth`
