# 🏥 BÁO CÁO NGHIÊN CỨU & ĐỐI SOÁT TOÀN DIỆN 8 KIẾN TRÚC MẠNG TÍCH CHẬP (CNN) TRONG PHÂN LOẠI NỘI SOI TIÊU HÓA
**Mã Task:** `#84` (Tổng hợp & Phân tích toàn bộ CNN) | **Phụ thuộc:** `#76` $\rightarrow$ `#83`
**Đề tài:** *Ứng dụng học sâu trong nhận diện và phân loại tổn thương đường tiêu hóa qua hình ảnh nội soi*
**Học viên:** Lê Huy Phát — Đại học Công Thương TP.HCM (HUIT)
**Bộ dữ liệu thực nghiệm:** HyperKvasir (10,662 ảnh nội soi, 23 lớp giải phẫu & bệnh lý, Fold 0, $N_{val} = 2,133$ ảnh)

---

## 1. GIỚI THIỆU & MỤC TIÊU HỌC THUẬT CỦA TASK #84

Giai đoạn 6 của đề tài khóa luận tập trung triển khai và đánh giá có hệ thống 8 kiến trúc mạng tích chập đại diện cho 4 trường phái thị giác máy tính tiêu biểu:
1. **Trường phái Kết nối phần dư (Residual Networks):** ResNet-50 (#76), ResNet-101 (#77), SE-ResNet-50 (#83).
2. **Trường phái Kết nối dày đặc & Tái sử dụng đặc trưng (Dense Connectivity):** DenseNet-121 (#78).
3. **Trường phái Mở rộng đồng tỷ lệ & Học tăng tiến (Compound Scaling & Progressive Learning):** EfficientNet-B4 (#79), EfficientNet-B5 (#80), EfficientNetV2-S (#81).
4. **Trường phái Điện toán biên & Nhẹ hóa (Edge & Mobile Computing):** MobileNetV3-Large (#82).

Mục tiêu cốt lõi của Task #84 là:
* Xây dựng **Master Benchmark Table (Bảng số liệu trung tâm của bài báo khoa học)** đối sánh toàn bộ 8 mô hình trên cùng hệ quy chiếu thực nghiệm chuẩn hóa.
* Khảo sát đa chiều: Độ chính xác lâm sàng (**Accuracy**, **F1-macro**, **AUC-ROC**), quy mô tính toán (**Parameters**, **FLOPs**) và chi phí vận hành phần cứng (**Training Duration**, **Inference Latency GPU/CPU**, **Throughput FPS**).
* Đúc rút các quy luật thị giác máy tính y sinh học và làm tiền đề vững chắc cho việc lựa chọn Backbone trong mô hình đề xuất lai (Hybrid Model - Phase 8).

---

## 2. BẢNG MASTER BENCHMARK CHÍNH (MAIN PAPER TABLE)

| Task ID | Mô hình (Model) | Trường phái kiến trúc | Cơ chế Chú ý | Kích thước ảnh | Tham số (Params) | FLOPs | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | AUC-ROC (OvR %) | Thời gian Train (50ep) | Latency GPU (FPS) | Latency CPU (FPS) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#76** | **ResNet-50 (Baseline)** | Standard Residual | None | $224 \times 224$ | 23.51 M | 4.12 G | 88.70% | 65.51% | 88.88% | 97.42% | ~4.3 h (311s/ep) | 6.75 ms (148.2) | 54.1 ms (18.5) |
| **#77** | **ResNet-101** | Deep Residual | None | $224 \times 224$ | 44.55 M | 7.85 G | 89.45% | **68.04%** | 89.24% | 97.85% | ~7.6 h (549s/ep) | 11.30 ms (88.5) | 89.3 ms (11.2) |
| **#78** | **DenseNet-121** | Dense Connectivity | None | $224 \times 224$ | 7.98 M | 2.88 G | 89.83% | **68.04%** | 89.54% | 98.12% | ~3.5 h (254s/ep) | 8.90 ms (112.4) | 67.6 ms (14.8) |
| **#79** | **EfficientNet-B4** | Compound Scaling | SE (Internal) | $380 \times 380$ | 19.34 M | 4.40 G | 88.61% | 62.70% | 88.91% | 97.21% | ~7.9 h (570s/ep) | 23.75 ms (42.1) | 217.4 ms (4.6) |
| **#80** | **EfficientNet-B5** | Compound Scaling | SE (Internal) | $456 \times 456$ | 30.40 M | 9.90 G | 88.61% | 66.11% | 89.01% | 97.68% | ~16.8 h (1207s/ep)| 41.15 ms (24.3) | 476.2 ms (2.1) |
| **#81** | **EfficientNetV2-S** | Fused-MBConv / Progressive | SE (Internal) | $384 \times 384$ | 21.46 M | 8.40 G | **90.44%** 🏆| 67.97% ⭐ | **90.16%** 🏆| **98.45%** 🏆| **~1.49 h** (107s/ep)⚡| 13.07 ms (76.5) | 112.4 ms (8.9) |
| **#82** | **MobileNetV3-Large** | Lightweight Edge | SE (Internal) | $224 \times 224$ | **4.23 M** 🏆| **0.22 G** 🏆| 88.37% | 62.55% | 88.10% | 97.10% | **~0.25 h** (18s/ep) ⚡| **6.39 ms (156.6)** 🏆| **21.4 ms (46.7)** 🏆|
| **#83** | **SE-ResNet-50** | Channel Attention Residual | Squeeze & Excite | $224 \times 224$ | 26.14 M | 4.14 G | 88.42% | 64.65% | 88.32% | 97.35% | ~0.65 h (46s/ep) | 7.12 ms (140.4) | 58.8 ms (17.0) |

---

## 3. PHÂN TÍCH CHUYÊN SÂU 4 TRƯỜNG PHÁI KIẾN TRÚC

### 3.1. Trường phái Residual Networks (ResNet-50 vs ResNet-101 vs SE-ResNet-50)
* **Quy luật độ sâu (Depth Scaling):** Khi tăng độ sâu từ 50 lên 101 tầng (#77), số tham số tăng gần gấp đôi ($23.51\text{M} \to 44.55\text{M}$), giúp Macro F1 bứt phá mạnh từ $65.51\% \to 68.04\%$ (+2.53%). ResNet-101 có năng lực trừu tượng hóa tốt hơn ở các tổn thương niêm mạc mạn tính (Viêm đại tràng). Tuy nhiên, thời gian huấn luyện tăng gần gấp đôi (7.6 giờ vs 4.3 giờ), thể hiện rõ quy luật hiệu suất biên giảm dần.
* **Tác động của Channel Attention (SE-ResNet-50 #83):** Việc bổ sung khối SE tái hiệu chuẩn trọng số kênh màu giúp cải thiện độ nhạy ở một số lớp viêm thực quản và giảm báo động giả polyp. Tuy nhiên, do phép gom trung bình toàn cục (Global Average Pooling) làm mất tọa độ không gian vi thể, SE-ResNet-50 ($88.42\%$) không vượt qua được baseline ($88.70\%$) trên độ chính xác toàn cục. Điều này khẳng định cơ chế chú ý kênh đơn lẻ chưa đủ giải quyết triệt để bài toán nội soi.

### 3.2. Trường phái Dense Connectivity (DenseNet-121 #78) - Hiện tượng "Nhỏ mà có võ"
* DenseNet-121 đạt kết quả kinh ngạc: **Chỉ với 7.98M tham số** (chưa bằng $1/5$ ResNet-101 và $1/3$ ResNet-50), mô hình đạt **Accuracy 89.83%** và **Macro F1 68.04%** (ngang ngửa ResNet-101 và tiệm cận EfficientNetV2-S).
* **Cơ chế lý giải:** Cơ chế kết nối dày đặc (Dense Blocks) ghép nối đặc trưng $\mathbf{x}_{\ell} = H_{\ell}([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}])$ cho phép các tầng sâu tái sử dụng trực tiếp các đặc trưng biên bờ, màu sắc thô từ tầng nông mà không bị suy giảm thông tin qua các phép cộng dồn (như ResNet). Nhờ đó, DenseNet-121 cực kỳ nhạy bén với cấu trúc mạng mạch máu niêm mạc đại tràng.

### 3.3. Trường phái Compound Scaling & Progressive Learning (EfficientNet-B4/B5 vs EfficientNetV2-S)
* **EfficientNet-B4 (#79) & B5 (#80):**
  * Tăng độ phân giải lên $380 \times 380$ và $456 \times 456$ đem lại trường nhìn cực kỳ sắc nét, giúp B5 đạt **Macro Recall cao nhất toàn nhánh CNN (68.64%)** và kích hoạt thành công các lớp siêu hiếm như `hemorrhoids` (F1 100%).
  * Tuy nhiên, chi phí tính toán bùng nổ: B5 ngốn tới **9.9 GFLOPs**, thời gian huấn luyện lên đến **16.8 giờ** và tốc độ suy luận tụt xuống **24.3 FPS (GPU)** và **2.1 FPS (CPU)**, hoàn toàn không khả thi cho triển khai video thời gian thực thông thường.
* **Đột phá vượt bậc của EfficientNetV2-S (#81):**
  * Thiết lập **Kỷ lục Toàn diện của Nhánh CNN**: **Accuracy 90.44%**, **Weighted F1 90.16%**, **Macro Precision 70.33%**, **AUC-ROC 98.45%**.
  * Thay vì dùng MBConv ở tất cả các tầng như bản V1, EfficientNetV2-S thay thế các tầng nông bằng **Fused-MBConv** (gộp $1 \times 1$ conv và depthwise conv thành $3 \times 3$ standard conv) giúp tận dụng tối đa năng lực tăng tốc phần cứng. Kết hợp kỹ thuật huấn luyện lũy tiến **Progressive Learning** (tăng dần kích thước ảnh và độ mạnh Data Augmentation), mô hình hội tụ chỉ sau **1.49 giờ huấn luyện** (nhanh hơn B5 gấp 11 lần!).

### 3.4. Trường phái Lightweight Edge Computing (MobileNetV3-Large #82)
* **Kỷ lục về Tốc độ và Dung lượng:** Với chỉ **4.23M tham số** và **0.22 GFLOPs**, MobileNetV3-Large đạt tốc độ **156.6 FPS trên GPU** và **46.7 FPS trên CPU**.
* **Ý nghĩa thực tiễn lâm sàng:** Đây là mô hình duy nhất trong nghiên cứu đạt chuẩn xử lý video thời gian thực ($\ge 30$ FPS) trên **CPU máy tính văn phòng thông thường**, mở ra tiềm năng triển khai hệ thống CADe/CADx giá rẻ tại các trạm y tế vùng sâu, vùng xa.

---

## 4. MA TRẬN PHÂN TÍCH ĐÁNH ĐỔI (PARETO FRONTIER TRADE-OFFS)

```
   Accuracy (%)
       ^
 91.0 -|                                          [EfficientNetV2-S] 🏆 (90.44%, 13.1ms)
       |
 90.0 -|                    [DenseNet-121] ⭐ (89.83%, 8.9ms)
       |                                          [ResNet-101] (89.45%, 11.3ms)
 89.0 -|        [ResNet-50] (88.70%, 6.8ms)
       |        [SE-ResNet-50] (88.42%, 7.1ms)    [EffNet-B4] (88.61%, 23.8ms)   [EffNet-B5] (88.61%, 41.2ms)
 88.0 -|   [MobileNetV3] ⚡ (88.37%, 6.4ms)
       |
       +----------------------------------------------------------------------------------------->
       0       5       10      15      20      25      30      35      40      45   Inference Latency (ms)
```

1. **Biên hiệu quả Pareto (Pareto-Optimal Models):**
   * **MobileNetV3-Large:** Tối ưu tuyệt đối ở góc phần tư tốc độ / nhẹ nhất (Edge Deployable).
   * **DenseNet-121:** Tối ưu cân bằng giữa kích thước nhỏ (8M params) và độ chính xác cao (89.83% Acc).
   * **EfficientNetV2-S:** Tối ưu tuyệt đối về độ chính xác lâm sàng và năng lực tổng quát hóa (SOTA Accuracy 90.44%).
2. **Các mô hình nằm dưới biên Pareto (Sub-optimal):**
   * **EfficientNet-B5:** Chi phí tính toán quá đắt đỏ so với mức cải thiện hiệu năng thu được.
   * **ResNet-101:** Quá nặng và train lâu so với hiệu năng mà DenseNet-121 và EfficientNetV2-S mang lại.

---

## 5. KẾT LUẬN & ĐỀ XUẤT CHO TASK #85 VÀ GIAI ĐOẠN TIẾP THEO

Bảng Master Table của Task #84 đã hoàn thành xuất sắc sứ mệnh làm bằng chứng thực nghiệm trung tâm cho bài báo khoa học. Dựa trên các phát hiện trên:
1. **Lựa chọn Backbone cho Kiến trúc Lai Hybrid (Phase 8):**
   * **EfficientNetV2-S** là ứng viên số 1 nếu bài toán ưu tiên tối đa độ chính xác lâm sàng (Top-1 Accuracy).
   * **ResNet-50 / SE-ResNet-50** là ứng viên số 1 nếu bài toán ưu tiên tính mô-đun hóa phân cấp (Hierarchical 4 Stages) để kết nối trực tiếp với khối không gian **CBAM** và chuyển đổi thành **Visual Tokens** cho Transformer.
2. **Khởi đầu Giai đoạn 7 (Vision Transformers):**
   * Toàn bộ nhánh CNN đã thiết lập trần độ chính xác tại **90.44% (EfficientNetV2-S)** và trần F1-macro tại **68.04% (DenseNet-121 & ResNet-101)**.
   * Bước tiếp theo là đưa các kiến trúc **Vision Transformers (ViT, Swin Transformer, CaiT)** vào thực nghiệm đối đầu để đánh giá xem cơ chế Self-Attention toàn cục có thể phá vỡ trần hiệu năng này hay không!
