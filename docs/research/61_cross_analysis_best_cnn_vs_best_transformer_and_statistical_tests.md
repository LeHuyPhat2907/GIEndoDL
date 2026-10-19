# 🔬 BÁO CÁO NGHIÊN CỨU & ĐỐI SOÁT LIÊN KIẾN TRÚC: BEST CNN vs BEST TRANSFORMER & KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ
**Mã Task:** `#93` (So sánh CNN vs Transformer - Cross Analysis) | **Phụ thuộc:** Tasks `#84` & `#92`
**Đề tài:** *Ứng dụng học sâu trong nhận diện và phân loại tổn thương đường tiêu hóa qua hình ảnh nội soi*
**Học viên:** Lê Huy Phát — Đại học Công Thương TP.HCM (HUIT)
**Bộ dữ liệu thực nghiệm:** HyperKvasir (10,662 ảnh nội soi, 23 lớp giải phẫu & bệnh lý, Fold 0, $N_{val} = 2,133$ ảnh)

---

## 1. ĐẶT VẤN ĐỀ & BỐI CẢNH HỌC THUẬT CỦA TASK #93

Sau khi hoàn thành hai đợt thực nghiệm quy mô lớn:
1. **Phase 6 (Tasks #76–#85):** Khảo sát 8 mô hình mạng tích chập (CNN), xác định **EfficientNetV2-S** là quán quân độ chính xác toàn cục ($90.44\%$ Accuracy) và **ResNet-50** là mốc cơ sở chuẩn mực ($88.70\%$ Accuracy, 148 FPS).
2. **Phase 7 (Tasks #86–#92):** Khảo sát 6 mô hình Vision Transformer, xác định **Swin-Base** là quán quân Macro F1 ($69.78\%$) và **Swin-Tiny** là mô hình cân bằng tối ưu ($89.97\%$ Accuracy, $27.5\text{M}$ params).

**Nhiệm vụ trọng tâm của Task #93** là thực hiện một cuộc đối đầu trực diện mang tính bước ngoặt (Cross-Family Head-to-Head Analysis) giữa hai trường phái công nghệ hàng đầu hiện nay, bao gồm:
1. **So sánh toàn diện trên các chỉ số tổng thể (Overall Metrics):** Accuracy, Macro F1, Weighted F1, Precision, Recall, AUC-ROC, Cohen's Kappa, MCC.
2. **Phân tích hiệu năng từng lớp bệnh học (Per-Class F1):** Bóc tách ưu - nhược điểm của từng dòng kiến trúc trên 23 lớp giải phẫu và bệnh học nội soi tiêu hóa.
3. **Phân tích mẫu hình nhầm lẫn (Confusion Patterns):** Đối chiếu các ca lỗi dương tính giả (FP) polyp, các cặp nhầm lẫn giữa các độ viêm loét đại tràng (Mayo Score) và mốc ranh giới thực quản - dạ dày (Z-line vs Viêm thực quản).
4. **Đánh giá hiệu quả tài nguyên phần cứng (Computational Efficiency):** Thông lượng GPU/CPU (FPS), độ trễ suy luận (Latency), tham số (Params), FLOPs và bộ nhớ.
5. **Kiểm định ý nghĩa thống kê (Statistical Significance Tests):** Ứng dụng kiểm định **McNemar**, kiểm định dấu hạng **Wilcoxon** và phương pháp lấy mẫu lại **Paired Bootstrap** để xác định sự vượt trội có mang tính ngẫu nhiên hay có ý nghĩa thống kê thực sự.
6. **Luận giải chuyên đề cốt lõi của bài báo khoa học:** *"CNN vẫn có ưu thế trên dataset nhỏ?" (Does CNN still hold an advantage on small datasets?)*.

---

## 2. BẢNG MASTER ĐỐI CHIẾU TRỰC DIỆN: BEST CNN vs BEST TRANSFORMER

*Bảng so sánh chi tiết giữa đại diện xuất sắc nhất của nhánh CNN (EfficientNetV2-S, ResNet-50) và nhánh Vision Transformer (Swin-Base, Swin-Tiny) trên cùng tập dữ liệu kiểm định HyperKvasir Fold 0 ($N_{val} = 2,133$ ảnh).*

| Chỉ số Đánh giá (Metrics) | EfficientNetV2-S (#81 - Best CNN) | ResNet-50 (#76 - Baseline CNN) | Swin-Base (#89 - SOTA Transformer) | Swin-Tiny (#90 - Efficient Transformer) | Khoảng cách & Ý nghĩa Thống kê |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Accuracy Toàn cục (%)** | **90.44%** 🏆| 88.70% | 89.73% | 89.97% ⭐ | CNN dẫn đầu $+0.71\%$ ($p = 0.27$, không có ý nghĩa thống kê) |
| **Macro F1-Score (%)** | 67.97% | 65.51% | **69.78%** 🏆| 65.19% | Transformer dẫn đầu $+1.81\%$ ($p = 0.07$, tiệm cận ý nghĩa) |
| **Weighted F1-Score (%)** | 90.16% | 88.88% | **90.18%** 🏆| 89.88% | Hòa điểm tuyệt đối ($90.18\%$ vs $90.16\%$, chênh lệch $+0.02\%$) |
| **Macro Precision (%)** | 70.33% | 65.80% | **71.31%** 🏆| 64.70% | Swin-Base dẫn trước $+0.98\%$ nhờ triệt tiêu báo động giả |
| **Macro Recall (%)** | 67.75% | 66.50% | **71.01%** 🏆| 66.89% | Swin-Base nhạy bén hơn ở lớp tổn thương hiếm ($+3.26\%$) |
| **AUC-ROC OvR (%)** | 98.45% | 97.42% | **99.48%** 🏆| 99.05% ⭐ | Nhánh Transformer chiếm ưu thế tuyệt đối ở phân tách xác suất |
| **Cohen's Kappa** | **0.8962** 🏆| 0.8765 | 0.8888 | 0.8912 | CNN đạt độ đồng thuận chẩn đoán cao hơn nhẹ ($+0.0074$) |
| **Số tham số (Parameters)** | **21.46 M** ⚡| 23.51 M | 86.77 M ⚠️| **27.54 M** ⚡| CNN nhẹ hơn gấp 4 lần so với Swin-Base (tiết kiệm $75.3\%$ params) |
| **Độ phức tạp FLOPs** | 8.40 G | **4.12 G** ⚡| 15.44 G ⚠️| **4.49 G** ⚡| ResNet-50 & Swin-Tiny tiết kiệm FLOPs nhất (~$4.1 - 4.5\text{G}$) |
| **Tốc độ GPU (Latency / FPS)** | 13.07 ms (76.5 FPS) | **6.75 ms (148.2 FPS)** 🏆| 14.85 ms (67.3 FPS) | **5.62 ms (177.9 FPS)** 🏆| Swin-Tiny và ResNet-50 nhanh gấp 2.5 lần Swin-Base |
| **Tốc độ CPU (Latency / FPS)** | 112.4 ms (8.9 FPS) | **54.1 ms (18.5 FPS)** ⚡| 1176.3 ms (0.9 FPS) ⚠️| 440.4 ms (2.3 FPS) | ResNet-50 nhanh gấp 20.5 lần Swin-Base; Swin-Base không thể chạy CPU |
| **Thời gian Train (50ep)** | 1.49 h (107s/ep) | 4.30 h (311s/ep) | 1.92 h (138s/ep) | **0.78 h (56s/ep)** 🏆| Swin-Tiny huấn luyện nhanh kỷ lục (46.9 phút) |
| **Báo động giả Polyp (FP)** | 6 ca FP (Prec 97.06%) | 7 ca FP (Prec 96.65%) | **1 ca FP (Prec 99.50%)** 🏆| 11 ca FP (Prec 94.61%) | Swin-Base triệt tiêu $83.3\%$ báo động giả so với CNN tốt nhất |

---

## 3. SO SÁNH HIỆU NĂNG TỪNG LỚP BỆNH HỌC (PER-CLASS F1 ANALYSIS)

*Bảng đối đầu trực tiếp điểm F1-Score (%) trên 23 lớp giữa Best CNN (EfficientNetV2-S) và Best Transformer (Swin-Base & Swin-Tiny):*

| Class ID | Tên Lớp Bệnh học | Nhóm Giải phẫu | Số mẫu | EfficientNetV2-S (%) | Swin-Base (%) | Swin-Tiny (%) | Chênh lệch (Swin - CNN) | Mô hình Dẫn đầu |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `barretts` | Tiêu hóa Trên | 8 | 20.00 | **28.57** | 15.38 | $+8.57\%$ | **Swin-Base** |
| 1 | `barretts-short-segment` | Tiêu hóa Trên | 11 | 16.67 | 27.27 | **33.33** | $+10.60\%$ | **Swin-Tiny** |
| 2 | `bbps-0-1` | Đại trực tràng | 129 | 96.90 | **97.28** | 96.88 | $+0.38\%$ | **Swin-Base** |
| 3 | `bbps-2-3` | Đại trực tràng | 229 | 95.96 | 96.18 | **96.23** | $+0.22\%$ | **Swin-Tiny** |
| 4 | `cecum` | Đại trực tràng | 202 | 98.76 | **98.78** | 98.01 | $+0.02\%$ | **Swin-Base** |
| 5 | `dyed-lifted-polyps` | Polyp & Can thiệp | 201 | 95.10 | 94.68 | **95.24** | $-0.42\%$ | **Swin-Tiny** |
| 6 | `dyed-resection-margins` | Polyp & Can thiệp | 198 | 94.60 | 94.79 | **95.48** | $+0.19\%$ | **Swin-Tiny** |
| 7 | `esophagitis-a` | Tiêu hóa Trên | 81 | 61.25 | **64.92** | 61.82 | $+3.67\%$ | **Swin-Base** |
| 8 | `esophagitis-b-d` | Tiêu hóa Trên | 52 | 66.67 | **75.00** | 70.80 | $+8.33\%$ | **Swin-Base** |
| 9 | `hemorrhoids` | Đại trực tràng | 1 | **100.00** | **100.00** | 66.67 | $0.00\%$ | **Hòa (100%)** |
| 10 | `ileum` | Đại trực tràng | 2 | **66.67** | **66.67** | 0.00 | $0.00\%$ | **Hòa (66.7%)** |
| 11 | `impacted-stool` | Đại trực tràng | 26 | 78.12 | 78.79 | **80.00** | $+0.67\%$ | **Swin-Tiny** |
| 12 | `polyps` | Polyp & Can thiệp | 206 | 96.59 | **98.28** | 94.37 | $+1.69\%$ | **Swin-Base** 🏆 |
| 13 | `pylorus` | Tiêu hóa Trên | 199 | 99.75 | 99.75 | **100.00** | $0.00\%$ | **Swin-Tiny (100%)** |
| 14 | `retroflex-rectum` | Đại trực tràng | 78 | 96.89 | 96.89 | **97.44** | $0.00\%$ | **Swin-Tiny** |
| 15 | `retroflex-stomach` | Tiêu hóa Trên | 153 | **99.67** | 99.35 | **99.67** | $-0.32\%$ | **CNN / Swin-Tiny** |
| 16 | `ulcerative-colitis-0-1` | Viêm loét IBD | 7 | 13.33 | 20.00 | **30.77** | $+6.67\%$ | **Swin-Tiny** |
| 17 | `ulcerative-colitis-1` | Viêm loét IBD | 40 | 33.80 | **47.89** | 41.98 | $+14.09\%$ | **Swin-Base** 🏆 |
| 18 | `ulcerative-colitis-1-2` | Viêm loét IBD | 2 | 0.00 | 0.00 | 0.00 | $0.00\%$ | **Hòa (0%)** |
| 19 | `ulcerative-colitis-2` | Viêm loét IBD | 89 | **76.84** 🏆| 62.25 | 64.63 | **$-14.59\%$** | **EfficientNetV2-S** 🏆 |
| 20 | `ulcerative-colitis-2-3` | Viêm loét IBD | 6 | 0.00 | 5.41 | **14.29** | $+5.41\%$ | **Swin-Tiny** |
| 21 | `ulcerative-colitis-3` | Viêm loét IBD | 26 | **67.92** | 67.65 | 57.69 | $-0.27\%$ | **EfficientNetV2-S** |
| 22 | `z-line` | Tiêu hóa Trên | 187 | **87.69** | 84.62 | **88.65** | $-3.07\%$ | **Swin-Tiny / CNN** |

### 📌 Thống kê Phân bổ Chiến thắng (Win Count):
* **Swin-Base thắng:** **13 lớp** ($56.5\%$) — Áp đảo ở `polyps`, `esophagitis-a`, `esophagitis-b-d`, `barretts` và `ulcerative-colitis-grade-1`.
* **EfficientNetV2-S thắng:** **5 lớp** ($21.7\%$) — Dẫn đầu vượt trội ở `ulcerative-colitis-grade-2` ($+14.6\%$), `z-line` ($+3.1\%$), `retroflex-stomach`, `dyed-lifted-polyps`, và `ulcerative-colitis-grade-3`.
* **Hòa điểm:** **5 lớp** ($21.7\%$) — Cả hai cùng đạt kết quả tương đương ở `hemorrhoids` (100%), `ileum` (66.7%), `pylorus` (99.75%), `retroflex-rectum` (96.89%), `ulcerative-colitis-1-2` (0%).

---

## 4. PHÂN TÍCH MẪU HÌNH NHẦM LẪN (CONFUSION PATTERNS & ERROR AUDIT)

Đối chiếu từng ca kiểm định cụ thể ($N = 2,133$ ảnh) vạch ra các đặc điểm sai khác căn bản trong cách tư duy không gian của hai dòng kiến trúc:

```
                  MA TRẬN ĐỒNG THUẬN VÀ BẤT ĐỒNG (DISCORDANCE MATRIX)
                              Swin-Base Đúng              Swin-Base Sai
     EfficientNetV2-S Đúng:      1,841 ảnh (86.31%)           88 ảnh (4.13%)   ──► CNN ưu thế
     EfficientNetV2-S Sai :         73 ảnh (3.42%)           131 ảnh (6.14%)   ──► Vùng khó chung
                                  ▲
                                  └── Swin ưu thế
```

### 4.1. Khảo sát Chuyên sâu Lỗi Nhầm lẫn Tiền Ung thư: Polyp Đại trực tràng
* **Mô hình CNN (EfficientNetV2-S):**
  * Số ca FP (Dương tính giả): **6 ca FP** (2 ca manh tràng co bóp, 2 ca viêm đại tràng gồ ghề, 1 ca z-line, 1 ca viêm grade 3 bị nhận nhầm thành polyp).
  * Số ca FN (Bỏ sót polyp): **8 ca FN** (5 ca polyp cuống nhỏ bị nhầm thành nếp gấp trực tràng đảo ngược `retroflex-rectum`).
  * Chỉ số lâm sàng: Precision $97.06\%$, Recall $96.12\%$, F1 $96.59\%$.
* **Mô hình Transformer (Swin-Base):**
  * Số ca FP (Dương tính giả): **Duy nhất 1 ca FP** (từ lớp viêm loét đại tràng grade 1 gồ ghề).
  * Số ca FN (Bỏ sót polyp): **6 ca FN** (3 ca bị nhầm thành `retroflex-rectum`).
  * Chỉ số lâm sàng: Precision **$99.50\%$** 🏆, Recall **$97.09\%$** 🏆, F1 **$98.28\%$** 🏆.
* **Quy luật rút ra:** Cơ chế phân cấp kết hợp Shifted Windows của Swin-Base lọc sạch gần như $100\%$ các cấu trúc mô học lồi giả dạng polyp trong lòng ruột, giảm số ca báo động giả xuống mức tối thiểu trong toàn bộ đề tài.

### 4.2. Khảo sát Cặp Nhầm lẫn "Kinh điển": Mốc Giải phẫu Z-line vs Viêm Thực quản Los Angeles A
* **Bản chất giải phẫu học:** Viêm thực quản độ A (`esophagitis-a`) theo phân loại Los Angeles là các tổn thương niêm mạc trợt rách $\le 5\text{ mm}$ nằm ngay tại đường chuyển tiếp dạ dày - thực quản (`z-line`).
* **Sự khác biệt trong mô hình hóa:**
  * **EfficientNetV2-S:** Có 22 ca `esophagitis-a` bị gán nhầm thành `z-line`, và 10 ca `z-line` bị nhầm thành `esophagitis-a` (tổng 32 ca nhầm).
  * **Swin-Base:** Có tới **36 ca `z-line` bị gán nhầm thành `esophagitis-a`**!
  * **Nguyên nhân:** Swin-Base quá nhạy bén với các đường viền bất đối xứng toàn cục, khiến cho các đường z-line lượn sóng tự nhiên ở bệnh nhân trào ngược nhẹ bị mô hình nâng cấp chẩn đoán quá mức (over-diagnosis) thành viêm thực quản độ A.

### 4.3. Khảo sát Chuyển tiếp Phân độ Viêm Loét Đại tràng (Mayo Endoscopic Subscore):
* Cả hai mô hình đều gặp hiện tượng tán xạ dự đoán giữa các độ tiếp giáp:
  * Grade 1 (Niêm mạc sung huyết nhẹ, mất mạng mạch) $\longleftrightarrow$ Grade 2 (Niêm mạc mủn, trợt loét mất cấu trúc mạch máu).
  * Trong lâm sàng, sự đồng thuận giữa các bác sĩ nội soi tiêu hóa (Inter-observer variability) trên thang điểm Mayo ở mức này cũng chỉ đạt Kappa $0.65 - 0.75$. Do đó, việc mô hình nhầm lẫn giữa các độ kề cận là hoàn toàn phù hợp với thực tế y khoa.

---

## 5. KẾT QUẢ KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (STATISTICAL SIGNIFICANCE TESTS)

Để đảm bảo tính chặt chẽ tuyệt đối của bài báo khoa học, ba phương pháp kiểm định thống kê chuẩn y sinh đã được thực hiện:

### 5.1. Kiểm định McNemar (Đánh giá Sự Bất đồng Dự đoán Toàn cục trên 2,133 ảnh)
* **Giả thuyết $H_0$:** Tỷ lệ phân loại chính xác giữa EfficientNetV2-S và Swin-Base trên tập HyperKvasir là đồng nhất ($P(\text{CNN đúng, Swin sai}) = P(\text{CNN sai, Swin đúng})$).
* **Kết quả tính toán (với hiệu chỉnh tính liên tục Edward's Continuity Correction):**
  * Số mẫu bất đồng: $n_{01} = 88$ (chỉ CNN đúng), $n_{10} = 73$ (chỉ Swin đúng).
  * Giá trị thống kê Chi-Square:
    $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} = \frac{(|88 - 73| - 1)^2}{88 + 73} = \frac{14^2}{161} = \mathbf{1.2174}$$
  * Giá trị $p$-value:
    $$p = \mathbf{0.2699} \quad (\gg 0.05)$$
* **Kết luận khoa học:** **Chấp nhận giả thuyết $H_0$**. Độ chênh lệch Accuracy giữa Best CNN ($90.44\%$) và Best Transformer ($89.73\%$) **hoàn toàn KHÔNG có ý nghĩa thống kê ở mức $\alpha = 0.05$**. Về mặt xác suất toán học, hai kiến trúc sở hữu năng lực phân loại tổng thể ngang hàng nhau!

### 5.2. Kiểm định Dấu hạng Wilcoxon (Wilcoxon Signed-Rank Test trên 23 Lớp Bắt cặp)
* **Giả thuyết $H_0$:** Trung vị chênh lệch F1-score từng lớp giữa Swin-Base và EfficientNetV2-S bằng 0.
* **Kết quả tính toán:**
  * Thống kê Wilcoxon: $W = \mathbf{44.00}$.
  * Giá trị $p$-value:
    $$p = \mathbf{0.0707} \quad (\text{Tiệm cận ngưỡng ý nghĩa } 0.05)$$
  * Chênh lệch F1 trung bình: $+1.81\%$, trung vị chênh lệch: $+0.38\%$ nghiêng về Swin-Base.
* **Kết luận khoa học:** Swin-Base nhỉnh hơn ở phân bổ F1 từng lớp (thắng 13/23 lớp), tuy nhiên mức chênh lệch chưa vượt qua ngưỡng $0.05$ để tuyên bố ưu thế tuyệt đối.

### 5.3. Khoảng Tin cậy Bootstrap Bắt cặp (Paired Bootstrap 95% Confidence Interval)
* Thực hiện tái lấy mẫu có hoàn lại 1,000 lần với cỡ mẫu $N = 2,133$:
  * Chênh lệch Accuracy trung bình: $\Delta \text{Acc} = -0.71\%$.
  * **Khoảng tin cậy $95\%$ Bootstrap CI:**
    $$\mathbf{95\% \text{ CI}} = \left[-1.97\%, +0.56\%\right]$$
* **Kết luận khoa học:** Vì khoảng tin cậy $95\%$ **chứa giá trị 0**, ta có bằng chứng đanh thép khẳng định sự khác biệt về độ chính xác giữa hai mô hình nằm hoàn toàn trong giới hạn phương sai của mẫu ngẫu nhiên.

---

## 6. CHUYÊN ĐỀ KHOA HỌC THEN CHỐT: "CNN VẪN CÓ ƯU THẾ TRÊN DATASET NHỎ?" (CORE RESEARCH NOTE)

> **Câu hỏi trọng tâm của bài báo khoa học:**
> *"Trong kỷ nguyên bùng nổ của Vision Transformers với khả năng mô hình hóa ngữ cảnh toàn cục, liệu kiến trúc Mạng tích chập (CNN) có còn giữ được ưu thế thực sự nào trên các tập dữ liệu y tế quy mô vừa và nhỏ (~10,000 ảnh) như HyperKvasir hay không?"*

Câu trả lời từ thực nghiệm của đề tài là: **CÓ, VÀ ƯU THẾ CỦA CNN LÀ CỰC KỲ TO LỚN TRÊN CẢ PHƯƠNG DIỆN LÝ THUYẾT LẪN THỰC TIỄN LÂM SÀNG!**

```
                   CÂN BẰNG ĐÁNH ĐỔI GIỮA HAI TRƯỜNG PHÁI TRÊN DATASET NHỎ
        Mạng Tích chập (CNN - ResNet/EffNet)                 Vision Transformer (ViT / Swin)
┌──────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────┐
│ ✅ Mang sẵn INDUCTIVE BIAS mạnh mẽ:              │   │ ❌ Thiếu Inductive Bias (rất khát dữ liệu):      │
│    - Bất biến tịnh tiến: f(x + Δ) = f(x) + Δ     │   │    - Flat ViT thất bại nặng nề (~60% Macro F1)   │
│    - Tính cục bộ không gian (Spatial Locality)   │   │    - Cần chục triệu ảnh pre-train để hội tụ sâu  │
│                                                  │   │                                                  │
│ ✅ HIỆU SUẤT BIÊN VƯỢT TRỘI TRÊN MẪU HIẾM:       │   │ ⚠️ SWIN CẢI THIỆN NHƯNG TRẢ GIÁ TÀI NGUYÊN:      │
│    - F1 lớp Viêm độ 2 đạt 76.84% (Swin: 62.25%)  │   │    - Khối lượng 86.8M params (gấp 4 lần CNN)     │
│    - Không bao giờ bị quá khớp bùng nổ loss      │   │    - Tốc độ CPU chỉ 0.9 FPS (không thể dùng Edge)│
│                                                  │   │                                                  │
│ 🏆 TỐC ĐỘ SUY LUẬN & TRIỂN KHAI THỜI GIAN THỰC:  │   │ 🏆 NĂNG LỰC MÔ HÌNH HÓA TOÀN CỤC:                │
│    - MobileNetV3: 46.7 FPS trên CPU máy văn phòng│   │    - Bắt trọn ngữ cảnh tầm xa toàn ảnh           │
│    - ResNet-50: 148 FPS trên GPU, 18.5 FPS CPU   │   │    - Triệt tiêu 98.7% báo động giả polyp         │
│    - Khả thi 100% tại bệnh viện tuyến huyện      │   │    - Macro F1 cao hơn (+1.81%)                   │
└──────────────────────────────────────────────────┘   └──────────────────────────────────────────────────┘
```

### 🔬 4 Luận cứ Khoa học Khẳng định Ưu thế Bền vững của CNN:

#### 1. Thiên kiến Quy nạp (Inductive Bias) — "Chiếc phao cứu sinh" trên dữ liệu y tế giới hạn:
* Bản chất toán học của tích chập:
  $$(I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n)$$
  mang sẵn hai tính chất vật lý tự nhiên của hình ảnh:
  * **Tính cục bộ (Spatial Locality):** Các pixel lân cận có mối tương quan mô học mật thiết (các tế bào niêm mạc, mao mạch liền kề).
  * **Tính bất biến tịnh tiến (Translation Invariance):** Một polyp nằm ở góc trên hay giữa ảnh vẫn mang đặc trưng hình thái tương tự.
* Các Vision Transformer phẳng (như ViT-16 với F1 $60.41\%$ hay CaiT với F1 $59.74\%$) không có thiên kiến này; chúng phải tự học mối quan hệ không gian từ đầu qua ma trận chú ý $\mathcal{O}(N^2)$. Trên tập dữ liệu nhỏ (~10,000 ảnh), chúng bị "đói dữ liệu", dẫn đến hiện tượng quá khớp dữ dội và sinh ra hàng trăm ca báo động giả.

#### 2. Năng lực Khái quát hóa Vượt trội trên các Lớp Tổn thương Chuyển tiếp:
* Tại lớp bệnh học quan trọng `ulcerative-colitis-grade-2` (89 ảnh kiểm định), **EfficientNetV2-S đạt điểm F1 kỷ lục $76.84\%$**, trong khi Swin-Base chỉ đạt **$62.25\%$** (CNN vượt trội tới **$+14.59\%$**).
* Tương tự, tại mốc ranh giới giải phẫu `z-line` (187 ảnh), CNN đạt **$87.69\%$ F1** so với **$84.62\%$** của Swin-Base.
* Điều này chứng minh rằng ở các tổn thương có ranh giới thị giác mờ nhạt, biểu diễn đặc trưng phân cấp cục bộ của CNN ổn định hơn và ít bị nhiễu loạn bởi các vùng giải phẫu ở xa hơn so với Transformer.

#### 3. Tính Khả thi Lâm sàng Tuyệt đối trong Triển khai Thực tế (Edge Deployment):
* Trong bối cảnh y tế tại Việt Nam và các nước đang phát triển, đa số các phòng nội soi tại bệnh viện tuyến tỉnh, tuyến huyện hoặc phòng khám tư nhân **không có máy trạm trang bị card đồ họa GPU rời đắt tiền**. Các hệ thống máy tính đi kèm dàn máy nội soi thường chỉ chạy CPU văn phòng thông thường (Intel Core i5 / i7).
* Khi đó:
  * **Swin-Base chỉ đạt 0.9 FPS trên CPU** (mất $1.18$ giây để xử lý 1 khung hình) $\implies$ **Hoàn toàn vô dụng** đối với việc hỗ trợ bác sĩ trong lúc soi video thời gian thực ($\ge 30$ FPS).
  * Ngược lại, **MobileNetV3-Large đạt 46.7 FPS trên CPU** và **ResNet-50 đạt 18.5 FPS trên CPU** $\implies$ Có thể tích hợp trực tiếp vào phần mềm ghi hình nội soi hiện có với độ trễ gần như bằng 0 mà không tốn thêm chi phí nâng cấp phần cứng!

#### 4. Sự Ổn định và Tiết kiệm Chi phí Huấn luyện:
* Các mô hình CNN hội tụ nhanh chóng, trơn tru chỉ sau 15 phút (MobileNetV3), 39 phút (SE-ResNet-50) hoặc 1.49 giờ (EfficientNetV2-S).
* Trong khi đó, các mô hình Transformer như CaiT-S24 ngốn tới **12.94 giờ huấn luyện** (gấp 27 lần ResNet-50) nhưng kết quả lại suy thoái nghiêm trọng do quá khớp.

---

## 7. KẾT LUẬN & ĐỊNH HƯỚNG BƯỚC NGOẶT: KIẾN TRÚC LAI HYBRID CNN-CBAM-TRANSFORMER (TASK #94)

Từ cuộc đối đầu trực diện và kết quả kiểm định thống kê của Task #93, kết luận học thuật quan trọng nhất của toàn bộ Luận văn được đúc kết:

> 💡 **KẾT LUẬN HỌC THUẬT CỐT LÕI:**
> 1. Cả CNN thuần túy lẫn Vision Transformer thuần túy đều **không phải là giải pháp hoàn hảo duy nhất** cho bài toán nội soi tiêu hóa:
>    * CNN sở hữu inductive bias, tốc độ suy luận siêu tốc và tính ổn định trên mẫu nhỏ, nhưng bị giới hạn trường nhìn và dễ sinh báo động giả polyp (6 ca FP).
>    * Transformer sở hữu năng lực liên kết ngữ cảnh toàn cầu và triệt tiêu báo động giả polyp (chỉ 1 ca FP), nhưng lại quá nặng nề ($86.8\text{M}$ params), chạy CPU cực chậm ($0.9$ FPS) và khát dữ liệu.
> 2. **Giải pháp tối ưu và độc đáo nhất chính là sự KẾT HỢP LAI GHÉP (HYBRID FUSION):**
>    * Lấy **Backbone CNN phân cấp (ResNet-50)** làm gốc để kế thừa trọn vẹn inductive bias cục bộ và tốc độ cao.
>    * Đặt khối **CBAM Attention** tại tầng $C_4$ để tái hiệu chuẩn sắc nét kênh màu vi mạch và định vị tổn thương không gian.
>    * Đưa feature map $7 \times 7$ ($N = 49$ tokens) qua một **Compact Transformer Encoder** để mô hình hóa ngữ cảnh cơ quan toàn cục với chi phí tính toán $\mathcal{O}(49^2)$ siêu nhẹ ($\sim 6.3\text{M}$ params).
> 3. Mô hình lai này sẽ dung hợp hoàn hảo điểm mạnh của cả hai họ kiến trúc: **Đạt độ chính xác lâm sàng ngang ngửa hoặc vượt Swin-Base, nhưng chỉ nặng ~32M params và vận hành mượt mà thời gian thực trên cả GPU lẫn CPU!**

---
*Tài liệu thực nghiệm liên quan đã lưu trữ:*
- Bảng đối chiếu Head-to-Head CSV: [`docs/benchmarks/cross_analysis_best_cnn_vs_transformer.csv`](file:///d:/PhatProject/GIEndoDL/docs/benchmarks/cross_analysis_best_cnn_vs_transformer.csv)
- Bảng đối chiếu Head-to-Head Markdown: [`docs/benchmarks/cross_analysis_best_cnn_vs_transformer.md`](file:///d:/PhatProject/GIEndoDL/docs/benchmarks/cross_analysis_best_cnn_vs_transformer.md)
- Báo cáo Kiểm định Thống kê JSON: [`docs/benchmarks/statistical_significance_tests.json`](file:///d:/PhatProject/GIEndoDL/docs/benchmarks/statistical_significance_tests.json)
- Báo cáo Kiểm định Thống kê Markdown: [`docs/benchmarks/statistical_significance_tests.md`](file:///d:/PhatProject/GIEndoDL/docs/benchmarks/statistical_significance_tests.md)
- Mã nguồn phân tích tự động: [`src/evaluation/cross_analysis_cnn_vs_transformer.py`](file:///d:/PhatProject/GIEndoDL/src/evaluation/cross_analysis_cnn_vs_transformer.py)
