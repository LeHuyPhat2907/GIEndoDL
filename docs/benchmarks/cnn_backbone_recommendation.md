# 🏆 BÁO CÁO PHÂN TÍCH CHỌN CNN BACKBONE TỐI ƯU CHO KIẾN TRÚC LAI (TASK #85)
**Đề tài:** *Ứng dụng học sâu trong nhận diện và phân loại tổn thương đường tiêu hóa qua hình ảnh nội soi*
**Học viên:** Lê Huy Phát — Đại học Công Thương TP.HCM (HUIT)
**Tài liệu luận giải chi tiết:** [`docs/research/58_optimal_cnn_backbone_selection_for_hybrid_model.md`](file:///d:/PhatProject/GIEndoDL/docs/research/58_optimal_cnn_backbone_selection_for_hybrid_model.md)

---

## 1. 🎯 TỔNG KẾT QUYẾT ĐỊNH CHỌN BACKBONE (DECISION SUMMARY)

Dựa trên phân tích đánh đổi đa tiêu chí giữa **Độ chính xác lâm sàng (Accuracy/Macro F1)**, **Hiệu quả tính toán (FLOPs/FPS)**, **Tính tương thích không dư thừa với khối CBAM** và **Độ dài chuỗi Visual Tokens cho Transformer**:

> 🏆 **BACKBONE TỐI ƯU NHẤT ĐƯỢC CHỌN:** **ResNet-50 (Pretrained ImageNet-1k)**
> * Được lựa chọn làm bộ trích xuất đặc trưng nền tảng (CNN Feature Extractor) cho **Kiến trúc Lai Đề xuất CNN-CBAM-Transformer** ở Giai đoạn 8 (Task #94).

---

## 2. 🔬 VÌ SAO RESNET-50 VƯỢT TRỘI EFFICIENTNET-B4?

Theo gợi ý ban đầu của lộ trình nghiên cứu (*Roadmap Note: "Backbone tốt nhất có thể là EfficientNet-B4 hoặc ResNet-50 tùy kết quả"*), thực nghiệm đối đầu trực tiếp đã chứng minh **ResNet-50 hoàn toàn áp đảo EfficientNet-B4** trên 4 phương diện then chốt:

1. **Hiệu năng chẩn đoán trên lớp thiểu số:**
   * ResNet-50 đạt **Macro F1 = 65.51%** (vượt trội $+2.81\%$ so với mức $62.70\%$ của B4).
   * ResNet-50 đạt **Macro Precision = 68.20%** (vượt trội $+6.48\%$ so với mức $61.72\%$ của B4), giảm thiểu tối đa các ca dương tính giả.
2. **Triệt tiêu hiện tượng Dư thừa Attention (Non-redundancy):**
   * EfficientNet-B4 đã tích hợp sẵn cơ chế Squeeze-and-Excitation (SE) bên trong các khối MBConv. Việc đặt thêm CBAM sẽ gây ra hiện tượng **trùng lặp cơ chế Channel Attention**.
   * ResNet-50 là mạng tích chập thuần túy, giúp khối CBAM (Channel + Spatial Attention) phát huy **100% giá trị gia tăng** mà không có bất kỳ xung đột nào.
3. **Tương thích kích thước Visual Tokens cho Transformer ($\mathcal{O}(N^2)$):**
   * Output Stage 4 của ResNet-50 có kích thước không gian chuẩn **$7 \times 7 = 49$ tokens**, tương đương chỉ **2,401 phép tính ma trận Attention**.
   * EfficientNet-B4 (ảnh $380 \times 380$) sinh ra $12 \times 12 = 144$ tokens, tương đương **20,736 phép tính Attention (gấp gần 9 lần!)**, dẫn đến nguy cơ tràn bộ nhớ VRAM khi huấn luyện mô hình lai.
4. **Tốc độ huấn luyện và suy luận:**
   * ResNet-50 huấn luyện nhanh gấp **$1.84\times$** (4.3 giờ vs 7.9 giờ).
   * ResNet-50 có thông lượng suy luận nhanh gấp **$3.52\times$ trên GPU** (148.2 FPS vs 42.1 FPS) và nhanh gấp **$4.02\times$ trên CPU** (18.5 FPS vs 4.6 FPS).

---

## 3. 📊 MA TRẬN CHẤM ĐIỂM ĐA TIÊU CHÍ (AHP MULTI-CRITERIA DECISION MATRIX)

| Ứng viên Backbone | Lâm sàng (F1/AUC) [25%] | Tốc độ / FLOPs [20%] | Tương thích CBAM [20%] | Tokenizer Transformer [20%] | Ngân sách Params [15%] | **ĐIỂM TỔNG KẾT (Max 10)** | XẾP HẠNG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (Baseline)** | **8.5** | **9.0** | **10.0** | **10.0** | **9.5** | **9.35** 🏆 | **HẠNG NHẤT (CHỌN)** |
| **ResNet-101** | 8.8 | 6.5 | 10.0 | 10.0 | 5.5 | **8.33** | Hạng 2 |
| **DenseNet-121** | 8.8 | 8.5 | 6.5 | 8.5 | 9.0 | **8.25** | Hạng 3 |
| **EfficientNetV2-S** | 9.5 | 8.0 | 6.5 | 7.0 | 8.0 | **7.88** | Hạng 4 |
| **EfficientNet-B4** | 7.8 | 6.0 | 5.5 | 6.0 | 8.5 | **6.73** | Hạng 5 |

---

## 4. 📐 THIẾT KẾ PHỐI HỢP CHO MÔ HÌNH LAI (PHASE 8 - TASK #94)

* **Backbone:** ResNet-50 (4 Stages: 256, 512, 1024, 2048 kênh).
* **Attention:** Tích hợp khối **CBAM** ở cuối Stage 3 và Stage 4 để định vị đồng thời kênh màu và không gian tổn thương.
* **Bridge Projection:** $1 \times 1$ Conv chiếu tensor $7 \times 7 \times 2048 \to 7 \times 7 \times 512$, trải phẳng thành **$N=49$ Visual Tokens**.
* **Transformer Encoder:** 4 tầng Transformer Encoder với Multi-Head Self-Attention (MHSA, $h=8$) và 1 `[CLS]` token học được.
* **Tổng tham số:** **~35 - 38 M tham số** (Tối ưu tuyệt đối cho tính toán và hội tụ).
