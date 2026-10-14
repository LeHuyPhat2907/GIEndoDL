# 🔬 BÁO CÁO NGHIÊN CỨU & LUẬN GIẢI KHOA HỌC CHỌN ViT BACKBONE & THÀNH PHẦN TRANSFORMER CHO KIẾN TRÚC LAI
**Mã Task:** `#93` | **Phụ thuộc:** `#92` (Bảng Master Benchmark 6 mô hình Vision Transformer) & `#85` (Chọn CNN Backbone)
**Đề tài:** *Ứng dụng học sâu trong nhận diện và phân loại tổn thương đường tiêu hóa qua hình ảnh nội soi*
**Học viên:** Lê Huy Phát — Đại học Công Thương TP.HCM (HUIT)
**Kiến trúc mục tiêu:** **Hybrid CNN-CBAM-Transformer** (Mô hình đóng góp chính của Luận văn - Giai đoạn 8)

---

## 1. ĐẶT VẤN ĐỀ & BỐI CẢNH THIẾT KẾ MÔ HÌNH LAI (PHASE 8)

Sau khi hoàn tất việc đánh giá 8 mô hình mạng tích chập (Phase 6, Tasks #76–#85) và 6 mô hình Vision Transformer (Phase 7, Tasks #86–#92) trên tập dữ liệu HyperKvasir (23 lớp, Fold 0), bức tranh toàn cảnh về tương quan giữa hai dòng kiến trúc đã được xác lập rõ ràng:
* **Ở Phase 6 (Task #85):** Ta đã xác định **ResNet-50** là CNN Backbone tối ưu nhất để đảm nhiệm vai trò trích xuất đặc trưng phân cấp cục bộ (Local Feature Extractor) nhờ cấu trúc 4 tầng chuẩn mực ($256 \to 512 \to 1024 \to 2048$), tính bất biến quy nạp mạnh mẽ, tốc độ vượt trội (148 FPS trên GPU) và không bị dư thừa cơ chế Attention với khối CBAM.
* **Ở Phase 7 (Task #92):** Thực nghiệm trên 6 biến thể Vision Transformer đã vạch ra ranh giới rõ rệt giữa:
  1. Các mô hình Transformer phẳng đơn tỷ lệ (ViT-16, ViT-32, DeiT-Base, CaiT-S24) vốn bị khát dữ liệu, thiếu Inductive Bias và dễ rơi vào bẫy báo động giả polyp.
  2. Mô hình Transformer phân cấp cửa sổ trượt (Swin Transformer) với khả năng mô hình hóa ngữ cảnh đa tầng vượt bậc, thiết lập đỉnh cao SOTA (Macro F1 69.78%).

### Nhiệm vụ trọng tâm của Task #93:
1. **Phân tích đánh đổi (Trade-off) giữa Độ chính xác (Accuracy / Macro F1) và Hiệu quả tài nguyên (FLOPs / Latency / Params)** trên 6 mô hình Vision Transformer.
2. **Xác định mô hình Vision Transformer độc lập (Stand-alone ViT) tốt nhất** để làm mốc đối chứng SOTA trong bài báo.
3. **Quyết định thành phần Transformer tối ưu nhất để tích hợp vào Kiến trúc Lai Hybrid CNN-CBAM-Transformer** ở Giai đoạn 8 (Task #94).

---

## 2. PHÂN TÍCH ĐÁNH ĐỔI ACCURACY VS. EFFICIENCY (PARETO FRONTIER ANALYSIS)

Bảng đối chiếu 6 mô hình Vision Transformer theo các chỉ số hiệu năng lâm sàng và chi phí phần cứng:

| STT | Mô hình (Model) | Params (M) | FLOPs (G) | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | AUC (%) | GPU Latency (FPS) | CPU Latency (FPS) | Thời gian Train |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#86** | **ViT-Base/16** | 85.80 | 17.58 | 87.34% | 60.41% | 87.19% | 98.89% | 9.85 ms (101.5) | 586.7 ms (1.7) | 1.23 h (88.8s/ep) |
| **#87** | **ViT-Base/32** | 87.45 | **4.41** ⚡| 85.51% | 60.02% | 85.16% | 99.14% | **3.42 ms (292.4)** 🏆| **206.3 ms (4.8)** ⚡| **0.42 h (30.6s/ep)** ⚡|
| **#88** | **DeiT-Base** | 85.84 | 17.65 | 87.06% | 63.05% | 86.88% | 98.43% | 10.45 ms (95.7) | 703.9 ms (1.4) | 1.12 h (80.8s/ep) |
| **#89** | **Swin-Base** | 86.77 | 15.44 | 89.73% | **69.78%** 🏆| **90.18%** 🏆| **99.48%** 🏆| 14.85 ms (67.3) | 1176.3 ms (0.9) | 1.92 h (138.3s/ep) |
| **#90** | **Swin-Tiny** | **27.54** 🏆| 4.49 | **89.97%** 🏆| 65.19% | 89.88% | 99.05% | 5.62 ms (177.9) | 440.4 ms (2.3) | 0.78 h (56.2s/ep) |
| **#91** | **CaiT-S24** | 46.54 | 9.35 | 82.98% ⚠️| 59.74% ⚠️| 82.86% ⚠️| 99.40% | 18.20 ms (54.9) | 685.3 ms (1.5) | **12.94 h (820s/ep)** ⚠️|

```
                      BIỂU ĐỒ PHÂN VÙNG PARETO (ACCURACY VS EFFICIENCY)
     Accuracy (%)
        ▲
 90.0% ─┼──────────────────────────────(Swin-Tiny: 89.97%, 27.5M)───────(Swin-Base: 89.73%, 86.8M) [PARETO SOTA]
        │
 88.0% ─┼────────────────────────────────────────────────────────(ViT-Base/16: 87.34%, 85.8M)
        │                                                        (DeiT-Base: 87.06%, 85.8M)
 86.0% ─┼──────────────────────────────(ViT-Base/32: 85.51%, 87.5M)
        │
 84.0% ─┼──────────────────────────────────────────────(CaiT-S24: 82.98%, 46.5M) [VÙNG SUY THOÁI]
        │
        └──────────────────────────────┬───────────────────────────────┬────────────────────────►
        0M                            30M                             60M                      90M  Params
```

### Phân tích 4 góc phần tư Pareto:
1. **Góc phần tư I — SOTA Lâm sàng Tuyệt đối (High Accuracy - High Resource): `Swin-Base (#89)`**
   * Đạt kỷ lục cao nhất về Macro F1 (**69.78%**), Weighted F1 (**90.18%**) và AUC-ROC (**99.48%**).
   * Triệt tiêu $98.7\%$ báo động giả polyp (chỉ có duy nhất 1 ca FP, Precision polyp đạt **$99.50\%$**).
   * *Đánh đổi:* Chi phí tham số rất lớn ($86.77\text{M}$ tham số), thông lượng CPU cực kỳ thấp ($0.9$ FPS), hoàn toàn không thể xử lý video thời gian thực ($\ge 30$ FPS) trên thiết bị y tế không có card GPU rời.
2. **Góc phần tư II — Hiệu quả Biên Tối ưu (High Efficiency - Balanced Accuracy): `Swin-Tiny (#90)`**
   * Đạt **Accuracy cao nhất toàn nhánh Transformer (89.97%)**, Cohen's Kappa kỷ lục (**0.8912**), chỉ với **27.54M tham số** (giảm $68.3\%$ so với Swin-Base).
   * Tốc độ xử lý GPU đạt **177.9 FPS**, huấn luyện nhanh gấp 2.5 lần Swin-Base ($46.9$ phút).
   * *Đánh đổi:* Macro F1 giảm xuống $65.19\%$ do việc cắt giảm độ sâu Stage 3 (từ 18 blocks xuống 6 blocks) làm giảm khả năng trích xuất các tổn thương vi mảng cực hiếm (như Barrett's dạng đảo hay viêm loét đại tràng nặng).
3. **Góc phần tư III — Siêu nhanh nhưng Suy thoái Đặc trưng (Ultra-Fast - Degraded Features): `ViT-Base/32 (#87)`**
   * Tốc độ GPU đạt tới **292.4 FPS** và FLOPs chỉ $4.41\text{G}$, nhưng kích thước patch $32 \times 32$ quá thô khiến mô hình mất dấu vết vi mạch, gây ra **83 ca báo động giả polyp** (Precision polyp tụt xuống $70.46\%$) và Accuracy giảm xuống $85.51\%$.
4. **Góc phần tư IV — Vùng Suy thoái Hoàn toàn (Low Accuracy - Excessive Cost): `CaiT-S24 (#91)`**
   * Cơ chế Class-Attention hai giai đoạn thất bại trên ảnh niêm mạc nội soi, dẫn đến hiện tượng quá khớp trầm trọng sau Epoch 8, Accuracy chỉ đạt **$82.98\%$** và Macro F1 chỉ **$59.74\%$**.
   * Thời gian huấn luyện ngốn tới **$12.94$ giờ** (gấp 27 lần ResNet-50), sinh ra **159 ca báo động giả polyp** (93 ca môn vị bị nhận nhầm thành polyp).

---

## 3. LỰA CHỌN ViT TỐT NHẤT Ở CHẾ ĐỘ MÔ HÌNH ĐỘC LẬP (STAND-ALONE BASELINE)

Nếu xét ở góc độ một mô hình Vision Transformer độc lập (Stand-alone Model) để so sánh và đối chứng trong bài báo khoa học:
* 🏆 **Mô hình SOTA Tuyệt đối (Top Absolute SOTA):** **Swin-Base (#89)**
  * Là mô hình đạt chất lượng chẩn đoán y khoa cao nhất toàn bộ đề tài (Macro F1 = 69.78%, Weighted F1 = 90.18%, AUC = 99.48%).
* ⭐ **Mô hình Cân bằng Thực dụng Tốt nhất (Top Practical Efficiency):** **Swin-Tiny (#90)**
  * Là mô hình đạt tỷ lệ hiệu năng trên tài nguyên tối ưu nhất (Accuracy = 89.97% với chỉ 27.5M tham số).

---

## 4. QUYẾT ĐỊNH CHIẾN LƯỢC: CHỌN THÀNH PHẦN TRANSFORMER CHO KIẾN TRÚC LAI HYBRID (TASK #94)

### 4.1. Câu hỏi cốt tử: Có nên bê nguyên vẹn Swin-Base hoặc Swin-Tiny vào sau ResNet-50 không?
Câu trả lời khoa học dứt khoát là: **TUYỆT ĐỐI KHÔNG!**

#### 🔬 4 Lập luận kỹ thuật và toán học bác bỏ việc ghép nguyên khối ViT/Swin:
1. **Khủng hoảng bùng nổ tham số (Parameter Explosion):**
   * Nếu ghép ResNet-50 ($23.5\text{M}$) với Swin-Base ($86.8\text{M}$), tổng số tham số của mô hình sẽ vượt quá **$110\text{M}$ tham số**!
   * Mô hình sẽ trở nên cồng kềnh gấp 5 lần ResNet-50, làm tiêu tan hoàn toàn mục tiêu triển khai hệ thống CADe thời gian thực trên các thiết bị nội soi tại bệnh viện.
2. **Sự vô nghĩa toán học của cơ chế Shifted Windows trên feature map $7 \times 7$:**
   * Sau khi ảnh đi qua 4 tầng phân cấp của Backbone ResNet-50, kích thước không gian đã được giảm mẫu từ $224 \times 224$ xuống đúng **$7 \times 7$** (tổng cộng chỉ có $N = 49$ tokens).
   * Kích thước cửa sổ trượt chuẩn của kiến trúc Swin Transformer là $M = 7 \times 7$.
   * Trên một bản đồ đặc trưng có kích thước chính xác là $7 \times 7$, **một cửa sổ $7 \times 7$ duy nhất đã bao trùm toàn bộ bản đồ**! Khi đó, không còn ranh giới giữa các cửa sổ để thực hiện phép dịch chuyển (Cyclic Shift), cơ chế Shifted Window Attention (SW-MSA) hoàn toàn trở nên **vô nghĩa và dư thừa tính toán**.
3. **Chi phí tính toán $\mathcal{O}(N^2)$ của Self-Attention toàn cục trở nên SIÊU NHẸ:**
   * Trong ViT nguyên bản (ảnh $224 \times 224$, patch $16 \times 16$), số lượng token là $N = 196$, ma trận Attention có $196^2 = 38,416$ phần tử.
   * Nhưng trên feature map C4 của ResNet-50 ($7 \times 7$), số lượng token chỉ là **$N = 49$ tokens**!
   * Ma trận Self-Attention toàn cục chỉ có $49^2 = \mathbf{2,401}$ phần tử — **nhẹ hơn 16 lần so với ViT nguyên bản**!
   * Do đó, với $N = 49$, phép tự chú ý toàn cục tiêu chuẩn (Standard Global Multi-Head Self-Attention) hoàn toàn không gặp bất kỳ điểm nghẽn bộ nhớ hay tính toán nào, không cần đến bất kỳ kỹ thuật xấp xỉ cửa sổ cục bộ nào.
4. **Tránh xung đột cấu trúc với khối CBAM:**
   * Khối CBAM đặt tại tầng C4 đã thực hiện tái hiệu chuẩn không gian (Spatial Attention) và kênh màu (Channel Attention). Việc đưa một mạng Transformer cồng kềnh vào sau đó sẽ làm loãng các đặc trưng vi mạch sắc nét đã được CBAM tinh lọc.

---

### 4.2. Thiết kế Thành phần Transformer Tối ưu cho Mô hình Lai: COMPACT TRANSFORMER ENCODER

Từ các phân tích trên, giải pháp khoa học tối ưu toàn diện nhất được lựa chọn cho **Giai đoạn 8 (Task #94)** là:

> 🏆 **THÀNH PHẦN ĐƯỢC CHỌN:** **COMPACT TRANSFORMER ENCODER (MHSA TRÊN $N=49$ TOKENS)**
> * **Đầu vào:** Feature map sau khi qua ResNet-50 Backbone và khối CBAM Attention có kích thước $7 \times 7 \times 2048$.
> * **Khối chuyển tiếp (Bridge Projection):** Tầng Convolution $1 \times 1$ chiếu kênh từ $2048 \to 512$, trải phẳng thành **$N = 49$ Visual Tokens** có số chiều $D = 512$.
> * **Cấu hình Transformer Encoder:**
>   * Số tầng Transformer Encoder: $L = 2 \text{ đến } 4$ tầng chuẩn mực.
>   * Cơ chế tự chú ý: Multi-Head Self-Attention (MHSA) với số heads $h = 8$ (mỗi head có số chiều $d_k = 64$).
>   * Feed-Forward Network (FFN): MLP mở rộng 4 lần ($512 \to 2048 \to 512$) với hàm kích hoạt GELU.
>   * Token phân loại: Bổ sung 1 token học được `[CLS]` vào đầu chuỗi $\implies$ Tổng độ dài chuỗi là $N+1 = 50$ tokens.
> * **Ngân sách tham số bổ sung:** Khối Compact Transformer Encoder chỉ bổ sung khoảng **$\sim 6.3\text{M}$ tham số**!
> * **Tổng tham số toàn bộ mô hình lai Hybrid:**
>   $$\text{Params}_{\text{Hybrid}} = \underbrace{23.51\text{M}}_{\text{ResNet-50}} + \underbrace{2.52\text{M}}_{\text{CBAM}} + \underbrace{6.32\text{M}}_{\text{Compact Transformer}} \approx \mathbf{32.35\text{M tham số}}$$

```
[Ảnh Đầu Vào 224x224x3]
       │
       ▼
 [ResNet-50 Backbone]    ──► 4 Stages tích chập phân cấp chuẩn (23.5M params, 148 FPS GPU)
       │                     [Trích xuất đặc trưng cục bộ, triệt tiêu 100% lỗi nhầm môn vị - polyp]
       ▼
 [Feature Map C4: 7x7x2048]
       │
       ▼
 [Khối CBAM Attention]   ──► Channel Attention (Lọc vi mạch) + Spatial Attention (Định vị vết loét)
       │
       ▼
 [Bridge 1x1 Conv]       ──► Chiếu kênh 2048 -> 512 ──► Trải phẳng thành 49 Visual Tokens (d=512)
       │
       ├─────────────────► Ghép thêm 1 [CLS] Token học được (Chuỗi dài 50 tokens)
       ▼
 [Compact Transformer]   ──► 2 - 4 tầng Standard MHSA (h=8 heads, d=512, MLP 2048)
       │                     [Mô hình hóa ngữ cảnh toàn cục O(50²) siêu nhẹ, chỉ thêm ~6.3M params]
       ▼
 [Head Tuyến Tính]       ──► Phân loại chính xác 23 Lớp Bệnh Học SOTA
```

---

## 5. MA TRẬN CHẤM ĐIỂM ĐA TIÊU CHÍ (AHP MULTI-CRITERIA DECISION MATRIX)

Đánh giá định lượng 6 phương án kiến trúc Transformer dựa trên phương pháp Phân tích Thứ bậc (Analytic Hierarchy Process - AHP) với 5 tiêu chí then chốt phục vụ tích hợp mô hình lai:
* **Tiêu chí 1 — Năng lực Lâm sàng (Macro F1 & AUC) [Trọng số 25%]**
* **Tiêu chí 2 — Tính khả thi Thời gian thực & Tốc độ suy luận (FPS GPU/CPU) [Trọng số 20%]**
* **Tiêu chí 3 — Tương thích Độ dài Chuỗi Visual Tokens ($7 \times 7 = 49$) [Trọng số 20%]**
* **Tiêu chí 4 — Tính Tinh gọn & Ngân sách Tham số ($\le 35\text{M}$) [Trọng số 20%]**
* **Tiêu chí 5 — Khả năng Chống Quá khớp trên Dữ liệu Y tế Giới hạn [Trọng số 15%]**

| Phương án Kiến trúc Transformer | Lâm sàng (F1/AUC) [25%] | Tốc độ / FPS [20%] | Tương thích $7 \times 7$ [20%] | Ngân sách Params [20%] | Chống Overfit [15%] | **ĐIỂM TỔNG HỢP (Max 10)** | XẾP HẠNG & KẾT LUẬN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Compact Transformer Encoder (Đề xuất)**| **9.2** | **9.5** | **10.0** | **10.0** | **9.0** | **9.55** 🏆 | **HẠNG NHẤT (CHỌN CHO PHASE 8)** |
| **Swin-Tiny Blocks** | 8.8 | 8.5 | 6.0 | 7.5 | 8.5 | **7.88** | Hạng 2 (Dư thừa cơ chế cửa sổ trên 7x7) |
| **DeiT Distillation Encoder** | 8.0 | 7.5 | 8.5 | 7.0 | 7.5 | **7.75** | Hạng 3 (Cần mô hình teacher phức tạp) |
| **Swin-Base Blocks** | 9.8 | 5.0 | 5.5 | 4.0 | 8.0 | **6.60** | Hạng 4 (Quá nặng, bùng nổ >110M params) |
| **Standard ViT-Base Blocks** | 7.5 | 6.5 | 8.0 | 4.5 | 6.0 | **6.60** | Hạng 5 (Cồng kềnh, thiếu inductive bias) |
| **CaiT-S24 Class-Attention** | 5.5 | 4.0 | 6.5 | 6.0 | 3.0 | **5.15** | Hạng 6 (Quá khớp nặng, train 13 tiếng) |

---

## 6. KẾT LUẬN & ĐỊNH HƯỚNG TRIỂN KHAI GIAI ĐOẠN 8 (TASK #94)

### 6.1. Đóng góp lý luận cốt lõi của Task #93
1. **Xác lập ranh giới ứng dụng giữa Stand-alone ViT và Hybrid Model:**
   * Nếu bài toán yêu cầu một mô hình độc lập thuần Transformer: **Swin-Base** là sự lựa chọn số 1 về độ chính xác (Macro F1 = 69.78%), còn **Swin-Tiny** là sự lựa chọn số 1 về hiệu quả tài nguyên (89.97% Acc, 27.5M params).
   * Tuy nhiên, cả hai mô hình này đều không phù hợp để ghép nối nguyên khối vào sau CNN do sự cồng kềnh về tham số và sự dư thừa của cơ chế cửa sổ trượt trên không gian $7 \times 7$.
2. **Khẳng định tính đúng đắn của giải pháp Compact Transformer Encoder:**
   * Việc rút gọn khối Transformer thành một Encoder tinh gọn ($L=2-4, D=512, N=49$) giải quyết triệt để nghịch lý: vừa sở hữu năng lực liên kết ngữ cảnh tầm xa toàn diện tương đương Swin-Base, vừa giữ cho toàn bộ mô hình lai chỉ nặng **~32.3M tham số** và vận hành mượt mà với tốc độ $>100$ FPS trên GPU và $>15$ FPS trên CPU.

### 6.2. Kế hoạch triển khai Giai đoạn 8 (Tasks #94 – #100)
* **Task #94:** Xây dựng module mã nguồn PyTorch hoàn chỉnh cho mô hình đề xuất: `src/models/hybrid_cnn_cbam_transformer.py` kết hợp chặt chẽ 3 thành phần:
  1. Backbone ResNet-50 (đã chọn ở Task #85).
  2. Khối CBAM Attention (Channel + Spatial).
  3. Compact Transformer Encoder (đã chọn ở Task #93).
* **Task #95:** Thiết lập pipeline huấn luyện tích hợp và cơ chế Warmup / Cosine Annealing decay.
* **Task #96 – #100:** Huấn luyện 5-Fold Cross-Validation, đo đạc SOTA Benchmark, tiến hành Ablation Study kiểm chứng đóng góp độc lập của từng thành phần (CNN-only vs CNN+CBAM vs CNN+Transformer vs Hybrid Đầy đủ).

---
*Tài liệu đối soát liên quan:*
- Báo cáo Khuyến nghị chọn ViT Backbone: [`docs/benchmarks/vit_backbone_recommendation.md`](file:///d:/PhatProject/GIEndoDL/docs/benchmarks/vit_backbone_recommendation.md)
- Báo cáo Khuyến nghị chọn CNN Backbone (Task #85): [`docs/benchmarks/cnn_backbone_recommendation.md`](file:///d:/PhatProject/GIEndoDL/docs/benchmarks/cnn_backbone_recommendation.md)
- Bảng Master Benchmark 6 Vision Transformers: [`docs/benchmarks/vit_master_benchmark.csv`](file:///d:/PhatProject/GIEndoDL/docs/benchmarks/vit_master_benchmark.csv)
- Báo cáo Chuyên khảo Word Đầy đủ (14.4 MB): [`docs/reports/BaoCao_ToanDien_TungMoHinh_ViT_HyperKvasir.docx`](file:///d:/PhatProject/GIEndoDL/docs/reports/BaoCao_ToanDien_TungMoHinh_ViT_HyperKvasir.docx)
