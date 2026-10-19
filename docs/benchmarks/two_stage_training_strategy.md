# Báo Cáo Thiết Kế: Chiến Lược Huấn Luyện 2 Giai Đoạn (Two-Stage Training Strategy) (Task #111)

**Thời gian xác lập:** 2026-10-19 16:42:46
**Kiến trúc áp dụng:** Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + L=2 Transformer Encoder)
**Tập dữ liệu mục tiêu:** HyperKvasir Challenge (23 classes)

## 1. Tổng Quan Quy Trình Huấn Luyện 2 Giai Đoạn

```
========================================================================================================
  [STAGE 1: CONTRASTIVE REPRESENTATION PRETRAINING]
  Dữ liệu: 2-View Augmentations (Crop, Flip, ColorJitter, Blur)
  Mô hình: CNN-CBAM + Bridge + Transformer Encoder ➔ MLP Projection Head (512 -> 512 -> 128)
  Mục tiêu: Tối ưu hóa Supervised Contrastive Loss (SupCon, tau=0.07, Khosla et al., 2020)
  Lịch trình: 50 epochs, AdamW (lr=5e-4), Cosine Annealing
          │
          ▼  THÁO BỎ (DISCARD) PROJECTION HEAD ➔ GIỮ LẠI TRỌNG SỐ BIỂU DIỄN TỐI ƯU
          │
  [STAGE 2: SUPERVISED CLASSIFICATION FINE-TUNING]
  Dữ liệu: Đơn ảnh (Single View, Standard Endo Augmentation)
  Mô hình: CNN-CBAM + Bridge + Transformer Encoder ➔ Classification Head (Linear 512 -> 23)
  Tốc độ học phân biệt (Discriminative LRs):
    - Backbone: 1e-5 (bảo toàn không gian đặc trưng tương phản)
    - Transformer: 2e-5 (điều chỉnh nhẹ ngữ cảnh toàn cục)
    - Classifier Head: 1e-3 (hội tụ nhanh năng lực phân lớp)
  Mục tiêu: Focal Loss (gamma=2.0) + Label Smoothing Cross-Entropy
  Lịch trình: 35 epochs, Linear Warmup (5 ep) + Cosine Decay (30 ep)
========================================================================================================
```

## 2. Bảng Đối Chiếu Thông Số Kỹ Thuật Giữa 2 Giai Đoạn

| Tham Số Huấn Luyện | Giai Đoạn 1 (Stage 1: SupCon Pretraining) | Giai Đoạn 2 (Stage 2: Fine-Tuning) |
| :--- | :--- | :--- |
| **Mục đích** | Học không gian đặc trưng phân tách (Separable Feature Space) | Tối ưu hóa xác suất phân lớp bệnh lý 23 lớp |
| **Số Epochs** | 50 epochs | 35 epochs |
| **Batch Size** | 32 (Hiệu dụng: 64 views) | 32 (Đơn góc nhìn) |
| **Hàm Tổn Thất** | SupCon Loss ($\tau = 0.07$) | Focal Loss (gamma=2.0, alpha=balanced) / Label Smoothing CE |
| **Đầu Ra (Head)** | Non-linear MLP Projection Head ($512 \to 512 \to 128$) | Linear Classification Head ($512 \to 23$) |
| **Tốc Độ Học (LR)** | Đơn nhất: $\eta = 0.0005$ | Phân tầng: Backbone $10^{-5}$, Transformer $2 \times 10^{-5}$, Head $10^{-3}$ |
| **Tối Ưu Hóa** | AdamW ($wd = 0.0001$) | AdamW with Discriminative Learning Rates |
| **Lịch Trình LR** | CosineAnnealingLR (T_max=50, eta_min=1e-6) | Linear Warmup (5 epochs) + Cosine Decay (30 epochs) |
| **Tăng Cường Dữ Liệu** | RandomResizedCrop(224) + ColorJitter(0.4, 0.4, 0.4, 0.1) + RandomHorizontalFlip + GaussianBlur | Standard Medical Endo Augmentations (Resize 224, RandomFlip, Mild ColorJitter) |

## 3. Lý Luận Khoa Học Cho Luận Văn

1. **Giải quyết vấn đề mất cân bằng dữ liệu y tế (Long-tailed Imbalance):**
   - Tập dữ liệu HyperKvasir có các lớp phổ biến (cecum, retroflex) lên tới hàng ngàn ảnh, trong khi các lớp bệnh lý hiểm nghèo (Barrett, dysplasia) chỉ có vài chục ảnh.
   - Huấn luyện trực tiếp bằng Cross-Entropy thường khiến mô hình thiên lệch về các lớp đông. SupCon giải quyết triệt để điều này bằng cách chuẩn hóa $\ell_2$ và chia đều sức ảnh hưởng của từng mẫu dương tính qua mẫu số $|P(i)|$.

2. **Triệt tiêu hiện tượng Quên Lãng Thảm Họa (Catastrophic Forgetting):**
   - Việc áp dụng Discriminative Learning Rates với tỷ lệ $1:100$ giữa Backbone và Classifier Head đảm bảo cấu trúc hình học của không gian biểu diễn học được từ Stage 1 không bị phá vỡ trong quá trình thích ứng với phân loại nhãn ở Stage 2.
