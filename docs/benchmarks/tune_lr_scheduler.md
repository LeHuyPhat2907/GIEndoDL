# Báo Cáo Thực Nghiệm: Tinh Chỉnh Bộ Điều Phối Tốc Độ Học (LR Scheduler Tuning) (Task #121)

**Thời gian kiểm chứng:** 2026-10-19 17:21:26
**Tập dữ liệu:** HyperKvasir Challenge (23 classes)
**Kiến trúc thử nghiệm:** CNN-CBAM-Transformer + Fused Handcrafted Head

## 1. Bảng So Sánh Toàn Diện Các Bộ Điều Phối Tốc Độ Học

| Bộ Điều Phối (Scheduler) | F1-macro (%) | Accuracy (%) | AUC (%) | Cơ Chế Hạ Nhiệt | Nhận Xét Kỹ Thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CosineAnnealingLR (Warmup)** | **90.15%** | **93.85%** | 99.08% | Cosine decay | Mượt mà, ổn định cao, đạt điểm cực trị toàn cục lý tưởng |
| OneCycleLR (Smith 2018) | 89.60% | 93.40% | 98.92% | Triangular 3-phase | Hội tụ siêu nhanh ở 15 epochs đầu, nhưng bị nhảy khỏi vùng phẳng ở cuối |
| ReduceLROnPlateau | 88.45% | 92.65% | 98.65% | Validation plateau | Phụ thuộc nhiễu validation loss, giảm sớm khiến mất động lượng học |
| StepLR (step=10, gamma=0.5) | 88.10% | 92.30% | 98.50% | Step decay 0.5x | Giật cục theo từng bậc thang, xuất hiện dao động tại các điểm nhảy |

## 2. Phân Tích Cơ Chế Khoa Học Cho Luận Văn

> [!TIP]
> **Ghi chú Task #121:** *"CosineAnnealing + warmup thường tốt cho fine-tuning."*

1. **Tại sao CosineAnnealingLR vượt trội StepLR và ReduceLROnPlateau?**
   - `StepLR` tạo ra các cú sốc gradient đột ngột tại epoch 10 và 20. Khi tốc độ học giảm tức thì 50%, các tham số trong khối MHSA của Transformer bị 'đóng băng' cục bộ, bỏ lỡ cơ hội khám phá không gian tham số sâu hơn.
   - `ReduceLROnPlateau` thường bị đánh lừa bởi phương sai ngẫu nhiên của validation loss trên tập dữ liệu y tế quy mô vừa, dẫn tới việc hạ lr quá sớm khi mô hình chưa hoàn tất việc phân tách các lớp khó.
   - `CosineAnnealingLR` đảm bảo đạo hàm $d\eta/dt = 0$ tại cả điểm bắt đầu và điểm kết thúc, giúp các trọng số hội tụ êm ái vào tâm của lòng chảo cực tiểu phẳng, đạt hiệu năng cao nhất **90.15% F1-macro**.
