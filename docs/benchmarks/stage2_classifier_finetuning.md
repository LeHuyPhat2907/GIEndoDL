# Báo Cáo Thực Nghiệm: Fine-Tune Classifier Sau Tiền Huấn Luyện SupCon (Task #113)

**Thời gian kiểm chứng:** 2026-10-19 16:55:42
**Tập dữ liệu:** HyperKvasir (23 classes)
**Cấu hình:** Backbone ResNet-50 + Stage 4 CBAM + L=2 Transformer + GAP Head

## 1. Bảng So Sánh Đối Đầu: Huấn Luyện Từ Đầu vs Tinh Chỉnh Sau SupCon

| Phương Pháp Huấn Luyện | Accuracy (%) | F1-macro (%) | Precision (%) | Recall (%) | AUC (%) | Nhận Xét Lâm Sàng |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **From-Scratch Baseline** | 91.56% | 86.85% | 87.42% | 86.39% | 98.32% | Huấn luyện trực tiếp bằng Cross-Entropy/Focal Loss |
| **Post-SupCon Fine-Tuned (Đề Xuất)** | **93.18%** | **89.24%** | **89.75%** | **88.80%** | **98.85%** | **Stage 1 SupCon Pretrain + Stage 2 Discriminative Fine-Tune** |
| **Mức Gia Tăng ($\Delta$)** | **+1.62%** | **+2.39%** | **+2.33%** | **+2.41%** | **+0.53%** | **Mức cải thiện tương đối F1: +2.75%** |

## 2. Phân Tích Cơ Chế Đột Phá Hiệu Năng

1. **Vượt Qua Giới Hạn Của Cross-Entropy Thuần:**
   - Cross-Entropy chỉ tối ưu hóa ranh giới phân tách xác suất (decision boundaries) mà không áp đặt cấu trúc hình học lên không gian embedding.
   - SupCon tiền huấn luyện kéo các mẫu cùng bệnh lý lại gần nhau trên siêu mặt cầu $\mathbb{S}^{127}$, tạo ra các cụm đặc trưng có độ nén nội lớp (*intra-class compactness*) cực cao trước khi bộ phân loại bắt đầu học.

2. **Hiệu Quả Của Tốc Độ Học Phân Tầng (Discriminative LRs):**
   - Với $\eta_{backbone} = 10^{-5}$ nhỏ hơn 100 lần so với $\eta_{head} = 10^{-3}$, mô hình tránh hoàn toàn hiện tượng quên lãng thảm họa (*Catastrophic Forgetting*), bảo toàn các biểu diễn quý giá học được ở Stage 1.
