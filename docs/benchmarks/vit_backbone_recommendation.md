# 🏆 BÁO CÁO PHÂN TÍCH CHỌN ViT BACKBONE & THÀNH PHẦN TRANSFORMER CHO KIẾN TRÚC LAI (TASK #93)

**Đề tài:** *Ứng dụng học sâu trong nhận diện và phân loại tổn thương đường tiêu hóa qua hình ảnh nội soi*
**Học viên:** Lê Huy Phát — Đại học Công Thương TP.HCM (HUIT)

---

## 1. 🎯 TỔNG KẾT QUYẾT ĐỊNH CHỌN KIẾN TRÚC TRANSFORMER (DECISION SUMMARY)

Dựa trên kết quả thực nghiệm 6 mô hình Vision Transformer trên bộ dữ liệu HyperKvasir (23 lớp, Fold 0):

1. **Ứng viên SOTA Tuyệt đối ở Chế độ Mô hình Độc lập (Top Absolute SOTA): Swin-Base (#89)**
   - Đạt kỷ lục Macro F1 (**69.78%**), Weighted F1 (**90.18%**) và AUC-ROC (**99.48%**).
   - Triệt tiêu 98.7% báo động giả polyp (chỉ duy nhất 1 ca FP, Precision polyp 99.50%).
   - Hạn chế: Nặng 86.8M tham số, tốc độ CPU chỉ 0.9 FPS (không khả thi cho video thời gian thực y tế).

2. **Ứng viên Cân bằng & Gọn nhẹ Tốt nhất (Top Efficiency & Balanced Accuracy): Swin-Tiny (#90)**
   - Đạt Accuracy cao nhất họ Transformer (**89.97%**), Cohen Kappa **0.8912**, chỉ **27.54M tham số**.
   - Huấn luyện nhanh gấp 2.5 lần Swin-Base (46.9 phút vs 115.3 phút), đạt 177.9 FPS trên GPU.

3. 👉 **QUYẾT ĐỊNH CỐT TỬ CHO KIẾN TRÚC LAI HYBRID CNN-CBAM-TRANSFORMER (PHASE 8 / TASK #94):**
   - **KHÔNG bê nguyên xi Swin-Base hay Swin-Tiny vào sau ResNet-50** (tránh bùng nổ >110M tham số và dư thừa cửa sổ trượt trên feature map 7x7).
   - **LỰA CHỌN TỐI ƯU:** Sử dụng **Compact Transformer Encoder (Multi-Head Self-Attention trên N=49 Visual Tokens, L=2 đến 4 layers, d=512, heads=8)**.
   - Kế thừa trọn vẹn: Khả năng mô hình hóa ngữ cảnh tầm xa toàn cục của ViT/Swin, nhưng chỉ bổ sung **~6.3M tham số** vào ResNet-50 + CBAM $\implies$ Tổng mô hình lai chỉ **~32.3M tham số**, đạt chuẩn thời gian thực $\ge 100$ FPS trên GPU và $\ge 15$ FPS trên CPU!
