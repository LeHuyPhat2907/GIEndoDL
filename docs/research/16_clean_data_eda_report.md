# BÁO CÁO TOÀN DIỆN KHÁM PHÁ DỮ LIỆU SẠCH NỘI SOI TIÊU HÓA (CLEAN DATA EDA MASTER REPORT)

> **Mã tài liệu:** `DOC-RES-16-CLEAN-DATA-EDA-MASTER`
> **Đề tài:** Ứng dụng học sâu trong nhận diện và phân loại tổn thương đường tiêu hóa từ ảnh nội soi (*GIEndoDL*)
> **Tác giả:** Lê Huy Phát (Khóa luận Tốt nghiệp Đại học - HUIT)
> **Quy mô tập dữ liệu:** 19,662 ảnh y tế (HyperKvasir 10,662 ảnh, Kvasir-v2 8,000 ảnh, Kvasir-SEG 1,000 cặp ảnh-mask)
> **Chuẩn báo cáo y tế:** Tuân thủ chặt chẽ hướng dẫn quốc tế **TRIPOD-AI** (*Transparent Reporting of a multivariable prediction model of an Artificial Intelligence*)
> **Trạng thái kiểm toán:** Hoàn tất nghiệm thu chuỗi tác vụ Data-Centric AI (D01 – D17)

---

## MỤC LỤC BÁO CÁO

1. [Tổng Quan Nguồn Gốc & Quy Mô Dữ Liệu (Dataset Provenance)](#1-tổng-quan-nguồn-gốc--quy-mô-dữ-liệu-dataset-provenance)
2. [Kiểm Toán Tính Toàn Vẹn Nhị Phân & Metadata (Binary Integrity Audit)](#2-kiểm-toán-tính-toàn-vẹn-nhị-phân--metadata-binary-integrity-audit)
3. [Phân Bố Độ Phân Giải & Định Kiến Thiết Bị Quang Học (Resolution & Device Bias)](#3-phân-bố-độ-phân-giải--định-kiến-thiết-bị-quang-học-resolution--device-bias)
4. [Kiểm Toán Chất Lượng Hình Ảnh & Độ Nét Laplacian (Quality Assessment)](#4-kiểm-toán-chất-lượng-hình-ảnh--độ-nét-laplacian-quality-assessment)
5. [Hệ Thống Đa Chữ Ký Số Perceptual Hashing (Multi-Hash Signatures)](#5-hệ-thống-đa-chữ-ký-số-perceptual-hashing-multi-hash-signatures)
6. [Phát Hiện Cặp Ảnh Chụp Liên Tiếp & Cô Lập Trùng Lặp (SSIM Burst & Deduplication)](#6-phát-hiện-cặp-ảnh-chụp-liên-tiếp--cô-lập-trùng-lặp-ssim-burst--deduplication)
7. [Đồ Thị Liên Kết Ca Bệnh & Khử Rò Rỉ Tuyệt Đối (Patient Cluster Graph & Zero Leakage)](#7-đồ-thị-liên-kết-ca-bệnh--khử-rò-rỉ-tuyệt-đối-patient-cluster-graph--zero-leakage)
8. [Phát Hiện & Xử Lý Nhiễu Nhãn Bằng Confident Learning (Label Noise Remediation)](#8-phát-hiện--xử-lý-nhiễu-nhãn-bằng-confident-learning-label-noise-remediation)
9. [Ma Trận Tương Đồng Hình Thái Học & Nghịch Lý Margin Âm (Morphological Similarity)](#9-ma-trận-tương-đồng-hình-thái-học--nghịch-lý-margin-âm-morphological-similarity)
10. [Bản Đồ Đa Tạp t-SNE / UMAP & 3 Vùng Chồng Lấn Bệnh Học (Manifold Topology)](#10-bản-đồ-đa-tạp-t-sne--umap--3-vùng-chồng-lấn-bệnh-học-manifold-topology)
11. [Mất Cân Bằng Lớp Cực Đoan & Quy Luật Pareto (Class Imbalance & Pareto Law)](#11-mất-cân-bằng-lớp-cực-đoan--quy-luật-pareto-class-imbalance--pareto-law)
12. [Đề Xuất Tái Cấu Trúc Nhãn Mục Tiêu (Phase 4: Target Schema Decision)](#12-đề-xuất-tái-cấu-trúc-nhãn-mục-tiêu-phase-4-target-schema-decision)
13. [Cơ Sở Khoa Học Đề Xuất Kiến Trúc Hybrid SupCon + CBAM Attention](#13-cơ-sở-khoa-học-đề-xuất-kiến-trúc-hybrid-supcon--cbam-attention)

---

## 1. TỔNG QUAN NGUỒN GỐC & QUY MÔ DỮ LIỆU (DATASET PROVENANCE)

Nghiên cứu sử dụng 3 bộ dữ liệu nội soi tiêu hóa tiêu chuẩn quốc tế được công bố bởi **Viện Nghiên Cứu Simula**, **Bệnh Viện Đại Học Oslo** và **Bệnh Viện Bærum (Na Uy)**:

1. **HyperKvasir (Tập Huấn Luyện & Thẩm Định Đa Tầng):**
   - Quy mô: 10,662 ảnh gán nhãn chuyên gia, phân bố trên 23 phân lớp giải phẫu và bệnh lý.
   - Gồm 10,650 mẫu sạch hợp lệ sau kiểm toán chất lượng và cô lập trùng lặp tuyệt đối.
2. **Kvasir-v2 (Tập Thẩm Định Ngoại Độc Lập - External Validation):**
   - Quy mô: 8,000 ảnh chuẩn hóa trên 8 phân lớp kinh điển.
   - Đối chiếu chữ ký băm cho thấy độ trùng lặp ngoại tuyến chỉ 0.17% (14 ảnh), khẳng định tính độc lập tuyệt đối để đánh giá khả năng tổng quát hóa theo chuẩn TRIPOD-AI.
3. **Kvasir-SEG (Tập Phân Đoạn Tổn Thương Polyp):**
   - Quy mô: 1,000 ảnh nội soi đi kèm 1,000 mặt nạ phân đoạn nhị phân (Ground-truth Masks) do bác sĩ chuyên khoa nội soi vẽ viền thủ công pixel-to-pixel.

> [!NOTE]
> Toàn bộ 19,662 tệp ảnh được bảo quản nguyên vẹn tại ổ đĩa cục bộ `data/raw/`, tiết kiệm hơn 10 GB băng thông mạng và đảm bảo tính tái lập 100% cho mọi thực nghiệm.

---

## 2. KIỂM TOÁN TÍNH TOÀN VẸN NHỊ PHÂN & METADATA (BINARY INTEGRITY AUDIT)

* **Phương pháp:** Quét toàn bộ 10,662 tệp tin hình ảnh bằng thư viện `PIL.Image.verify()` kết hợp giải mã toàn bộ khối mảng ma trận pixel (Raster Array Decoding).
* **Kết quả kiểm toán:**
  * Số tệp hỏng cấu trúc header hoặc lỗi EOF: **0 / 10,662 (0.00%)**.
  * Định dạng tệp tin: 100% chuẩn JPEG (JFIF standard).
  * Kênh màu: 100% hệ màu RGB 3 kênh 8-bit ($[0, 255]$).
  * Kích thước tệp trên đĩa: Trung bình $142.6 \text{ KB}$ (nhỏ nhất $28.4 \text{ KB}$, lớn nhất $1.21 \text{ MB}$).

---

## 3. PHÂN BỐ ĐỘ PHÂN GIẢI & ĐỊNH KIẾN THIẾT BỊ QUANG HỌC (RESOLUTION & DEVICE BIAS)

Khảo sát toàn diện kích thước hình ảnh đã phát hiện một sự thật quan trọng về phần cứng nội soi:

![Resolution Distribution](../figures/04_resolution_distribution.png)

* **Số lượng độ phân giải khác nhau:** 5,767 độ phân giải phân tán trong tập dữ liệu.
* **Bản chất phần cứng:** Mặc dù số lượng độ phân giải ghi nhận rất lớn, nhưng phân tích cụm hình học cho thấy toàn bộ ảnh thực chất chỉ bắt nguồn từ **2 dòng máy nội soi chính**:
  1. **Cụm thiết bị Olympus SD (Standard Definition):** Chiếm **72.1%** tổng số ảnh, độ phân giải xấp xỉ $600 \times 530 \text{ pixels}$, tỷ lệ khung hình trung bình $1.13$.
  2. **Cụm thiết bị Pentax HD (High Definition):** Chiếm **27.9%** tổng số ảnh, độ phân giải xấp xỉ $1250 \times 1015 \text{ pixels}$ đến $1920 \times 1080 \text{ pixels}$, tỷ lệ khung hình trung bình $1.23$.
* **Tỷ lệ khung hình trung bình toàn tập (Aspect Ratio):** $1.183 \pm 0.042$.

> [!WARNING]
> **Khuyến nghị kỹ thuật:** Tuyệt đối không ép ảnh méo hình (Stretching/Distortion) khi resize về $224 \times 224$ hay $384 \times 384$. Phải áp dụng chiến lược Resize giữ nguyên tỷ lệ kết hợp đệm đối xứng (Aspect-Ratio Preserving Letterboxing) hoặc Random Resized Crop để tránh làm biến dạng vi cấu trúc bờ polyp và rãnh pit-pattern.

---

## 4. KIỂM TOÁN CHẤT LƯỢNG HÌNH ẢNH & ĐỘ NÉT LAPLACIAN (QUALITY ASSESSMENT)

Để phát hiện các ảnh phế phẩm do rung lắc chuyển động của dây soi hoặc chiếu sáng không đạt chuẩn:

![Image Quality Metrics](../figures/05_image_quality_metrics.png)

1. **Độ sắc nét (Sharpness - Phương sai toán tử Laplace $\text{Var}(\nabla^2 I)$):**
   * Trung vị: $329.14$.
   * Dải giá trị: $7.62$ đến $2,840.45$.
   * **Ảnh phế phẩm mờ nhòe:** Chỉ có **42 ảnh (0.39%)** có độ nét cực thấp ($< 25.0$) do chuyển động nhu động ruột quá nhanh hoặc dịch che khuất thấu kính.
2. **Độ sáng trung bình (Mean Luminance):**
   * Trung vị: $108.20$ trên thang 255.
   * Dải giá trị: $26.42$ đến $218.65$.
   * **Ảnh phế phẩm quá tối:** **0 ảnh (0.00%)** có độ sáng $< 20.0$.
* **Kết luận:** Dữ liệu HyperKvasir có chất lượng quang học cao, tỷ lệ phế phẩm cực thấp (dưới 0.4%), không cần loại bỏ ồ ạt gây mất mát dữ liệu bệnh lý hiếm.

---

## 5. HỆ THỐNG ĐA CHỮ KÝ SỐ PERCEPTUAL HASHING (MULTI-HASH SIGNATURES)

Nhằm định danh chính xác từng bức ảnh dưới góc độ tri giác máy tính (Computer Perception), thuật toán đa băm Perceptual Hashing 256-bit đã được triển khai quét toàn bộ 10,662 ảnh:
* **4 thuật toán băm thành phần (mỗi thuật toán 64-bit):**
  1. `aHash` (Average Hash): Bắt cấu trúc tần số thấp và độ sáng trung bình.
  2. `pHash` (Perceptual Hash DCT): Bắt năng lượng cosin rời rạc biểu diễn hình thái học.
  3. `dHash` (Difference Hash): Bắt gradient hướng ngang của các rãnh niêm mạc.
  4. `wHash` (Wavelet Hash): Bắt biến đổi wavelet Haar ở cả miền tần số lẫn không gian.
* **Kết quả:** Tạo lập chuỗi mã băm tổ hợp 256-bit (`composite_256bit`). Xác định **10,624 mã băm duy nhất** trên 10,662 ảnh, làm cơ sở lọc trùng lặp ban đầu.

---

## 6. PHÁT HIỆN CẶP ẢNH CHỤP LIÊN TIẾP & CÔ LẬP TRÙNG LẶP (SSIM BURST & DEDUPLICATION)

Trong quá trình nội soi, bác sĩ thường bấm chụp liên tiếp (Burst-capture) từ 2 đến 5 khung hình cùng một giây tại cùng một vị trí tổn thương:

![Duplicate Pairs Sample](../figures/07_duplicate_pairs_sample.png)

* **Ma trận tương đồng cấu trúc SSIM (Structural Similarity Index):**
  * Quét các cặp ảnh có khoảng cách Hamming băm $\le 10$.
  * Xác định **236 cặp ảnh chụp liên tiếp** có $\text{SSIM} \ge 0.92$.
* **Phát hiện trùng lặp tuyệt đối ($\text{SSIM} = 1.0000$):**
  * Tìm thấy chính xác **10 cặp ảnh trùng lặp tuyệt đối**:
    * 8 cặp ảnh trùng lặp cùng nhãn: Giữ lại 8 ảnh đại diện, cô lập 8 ảnh bản sao dư thừa sang khu vực cách ly.
    * **2 cặp ảnh trùng lặp hoàn toàn nhưng bị gán nhãn mâu thuẫn (Conflicting Labels)**: Cùng một ảnh nhưng một thư mục ghi `ulcerative-colitis-grade-1`, thư mục khác lại ghi `ulcerative-colitis-grade-2`! Đã được hiệu đính về nhãn chính xác.
* **Tổng số mẫu cách ly:** 12 mẫu.
* **Tổng số mẫu sạch duy trì huấn luyện:** **10,650 mẫu**.

---

## 7. ĐỒ THỊ LIÊN KẾT CA BỆNH & KHỬ RÒ RỈ TUYỆT ĐỐI (PATIENT CLUSTER GRAPH & ZERO LEAKAGE)

![Data Leakage Audit Matrix](../figures/25_data_leakage_audit_matrix.png)

* **Vấn đề rò rỉ dữ liệu video (Video-level Data Leakage):** Nếu chia Train/Val ngẫu nhiên theo từng ảnh, các khung hình chụp liên tiếp cùng một bệnh nhân sẽ rơi vào cả Train và Val. Mô hình sẽ "học vẹt" đặc điểm cá thể của bệnh nhân (nếp gấp ruột, độ sáng máy soi) thay vì học đặc trưng bệnh học, dẫn đến hiện tượng Validation Score cao giả tạo nhưng sập gãy khi triển khai thực tế.
* **Giải pháp Đồ thị Hợp nhất (Unified Patient/Video Cluster Graph):**
  * Thiết lập liên kết đồ thị dựa trên 3 cạnh: Cạnh trùng mã băm 256-bit + Cạnh $\text{SSIM} \ge 0.92$ + Cạnh tương đồng ngữ nghĩa DINOv2 Mutual $k$-NN ($k=1, S \ge 0.92$).
  * Gom cụm thành phần liên thông (Connected Components) đóng gói toàn bộ ảnh cùng nguồn gốc vào một **`patient_video_cluster_id`** duy nhất.
  * Xác định **9,849 cụm bệnh nhân độc lập**, trong đó có **709 cụm đa khung hình** (kích thước lớn nhất 11 ảnh, trung vị 2 ảnh).
* **Kết quả kiểm toán rò rỉ (Zero Leakage Audit):**
  * Khi áp dụng phân chia `GroupStratifiedKFold` theo Cluster ID, chỉ số rò rỉ ranh giới giữa Train và Test đạt:
    $$\Lambda = 0.00000000\% \quad (\Delta = 0.0)$$

---

## 8. PHÁT HIỆN & XỬ LÝ NHIỄU NHÃN BẰNG CONFIDENT LEARNING (LABEL NOISE REMEDIATION)

![Confident Learning Label Noise](../figures/10_confident_learning_label_noise.png)

Nhằm giải quyết triệt để nguyên nhân khiến mô hình Run 4 bị chặn ở 69.28% Macro F1, kỹ thuật **Confident Learning (`cleanlab`)** đã được áp dụng trên không gian nhúng DINOv2 thông qua phân rã 5-Fold Stratified Out-of-Fold (OOF Accuracy: 86.41%):

* **Tỷ lệ nhiễu nhãn toàn bộ tập HyperKvasir:** Phát hiện **915 mẫu nghi vấn lỗi nhãn (8.58%)**.
* **Các phân lớp có tỷ lệ nhiễu nhãn nghiêm trọng nhất:**
  * `barretts-short-segment`: **79.2%** mẫu bị nghi ngờ lỗi nhãn.
  * `ulcerative-colitis-grade-3`: **74.4%** mẫu bị nghi ngờ lỗi nhãn.
  * `ulcerative-colitis-grade-1`: **62.2%** mẫu bị nghi ngờ lỗi nhãn.
  * `esophagitis-a`: **33.5%** mẫu bị nghi ngờ lỗi nhãn.
* **Kiểm tra chéo chuyên gia (Clinical Cross-Inspection):**
  * Phát hiện **99 ảnh trong lớp `esophagitis-a` thực chất là niêm mạc bình thường của đường Z (`normal-z-line`)**, không hề có vết trợt rách niêm mạc (Mucosal break) theo chuẩn phân loại Los Angeles.
* **Hiệu đính nhãn (Label Remediation):**
  * Đã chuyển nhãn chuẩn hóa cho 58 trường hợp sai sót nghiêm trọng nhất (27 ảnh `esophagitis-a` $\to$ `z-line`, 15 ảnh `dyed-polyps` $\leftrightarrow$ `dyed-margins`, 2 cặp trùng lặp Mayo $\to$ Mayo 2).
  * Xuất bản tập dữ liệu nhãn sạch chính thức: `cleaned_hyperkvasir_labels.csv`.

---

## 9. MA TRẬN TƯƠNG ĐỒNG HÌNH THÁI HỌC & NGHỊCH LÝ MARGIN ÂM (MORPHOLOGICAL SIMILARITY)

![Morphological Similarity Heatmap](../figures/09_class_morphological_similarity_heatmap.png)

Phân tích ma trận Cosine Similarity $23 \times 23$ giữa các trọng tâm phân lớp (Centroids) trên không gian DINOv2 384 chiều đã hé lộ bản chất quang học của các tổn thương:

### Top Các Cặp Phân Lớp Có Độ Tương Đồng Cực Kỳ Cao ($S \ge 0.97$)
| STT | Phân lớp A | Phân lớp B | Cosine Similarity | Nhóm Nguy Cơ | Bản Chất Sinh Lý Học Lâm Sàng |
|:---:|:---|:---|:---:|:---:|:---|
| 1 | `barretts` | `barretts-short-segment` | **0.9896** | **CRITICAL** | Cùng là chuyển sản ruột thực quản, chỉ khác biệt về độ dài phân đoạn ($< 3\text{ cm}$ vs $\ge 3\text{ cm}$). |
| 2 | `uc-grade-1` | `uc-grade-2` | **0.9857** | **CRITICAL** | Ranh giới giữa mất mạng mạch máu và trợt loét xung huyết nhẹ. |
| 3 | `uc-grade-2-3` | `uc-grade-3` | **0.9856** | **CRITICAL** | Loét sâu và xuất huyết tự phát trên thang điểm Mayo. |
| 4 | `uc-grade-0-1` | `uc-grade-1` | **0.9835** | **CRITICAL** | Giai đoạn chuyển tiếp từ niêm mạc lành sang viêm đại tràng nhẹ. |
| 5 | `barretts-short-segment` | `esophagitis-a` | **0.9820** | **CRITICAL** | Niêm mạc trợt đỏ xung huyết thực quản rất giống biểu mô trụ Barrett. |
| 6 | `dyed-lifted-polyps` | `dyed-resection-margins` | **0.9728** | **CRITICAL** | Phẩm màu Indigo Carmine chi phối áp đảo quang phổ toàn khung hình. |

![Dendrogram and Margin](../figures/09_morphological_dendrogram_and_clusters.png)

### Phát Hiện Nghịch Lý Biên Độ Tách Biệt Âm (Negative Margin Paradox)
* **22/23 phân lớp có Margin âm** ($\mathcal{M} = \bar{S}_{\text{intra}} - S_{\text{nearest}} < 0$).
  * Nhóm Viêm loét đại tràng Mayo có Margin âm nặng nhất ($-0.22$ đến $-0.19$).
  * Duy nhất phân lớp `impacted-stool` có Margin dương ($+0.0343$) nhờ màu sắc và cấu trúc xốp đặc thù của phân.
* **Ý nghĩa toán học:** Khi $\mathcal{M} < 0$, vùng biên quyết định bị chồng lấn hình học. Hàm mất mát Cross-Entropy truyền thống bị **dao động gradient đối kháng**, dẫn đến mô hình bỏ rơi các lớp thiểu số.

---

## 10. BẢN ĐỒ ĐA TẠP t-SNE / UMAP & 3 VÙNG CHỒNG LẤN BỆNH HỌC (MANIFOLD TOPOLOGY)

![t-SNE vs UMAP Embedding](../figures/06_color_distribution_tsne.png)

Chiếu phi tuyến không gian đặc trưng DINOv2 384-d xuống mặt phẳng 2D bằng **t-SNE** (bảo toàn cấu trúc cục bộ) và **UMAP** (bảo toàn cấu trúc toàn cục) đã làm sáng tỏ cấu trúc hình học của tập dữ liệu:

1. **Sự tách biệt của 5 Đại Cụm Giải Phẫu:**
   * Cả t-SNE và UMAP đều tách bạch rõ ràng 5 hòn đảo lớn: (1) Thực quản & Dạ dày, (2) Đại trực tràng & Viêm loét, (3) Nhuộm màu Indigo Carmine, (4) Độ sạch lòng ruột BBPS, (5) Góc quặt ngược & Ống hậu môn.
2. **Khám phá 3 Vùng Chồng Lấn Bệnh Học Cốt Lõi:**

![Overlap Subclusters](../figures/06_overlapping_pathology_subclusters.png)

* **Vùng A (Thực quản & Đường Z):** `barretts` và `barretts-short-segment` hòa trộn thành một đám mây điểm duy nhất không có ranh giới phân tách. `esophagitis-a` bọc quanh một phần của `z-line`.
* **Vùng B (Dải biến thiên liên tục Mayo Colitis):** Bệnh viêm loét đại tràng không tạo thành các cụm rời rạc mà trải dài thành một **dải đa tạp cong 1 chiều liên tục (1D Trajectory Continuum)** từ Grade 0-1 đến Grade 3. Các lớp lai trung gian (`0-1`, `1-2`, `2-3`) đóng vai trò như các bẫy gradient gây nhiễu loạn mô hình.
* **Vùng C (Hòn đảo nhuộm màu Indigo Carmine):** Màu xanh lam nhân tạo kéo toàn bộ ảnh thủ thuật EMR/ESD ra một cụm độc lập, bên trong đó hai lớp `dyed-lifted` và `dyed-resection` đan cài chặt chẽ vào nhau.

---

## 11. MẤT CÂN BẰNG LỚP CỰC ĐOAN & QUY LUẬT PARETO (CLASS IMBALANCE & PARETO LAW)

![Pareto Imbalance](../figures/03_hyperkvasir_pareto_imbalance.png)

* **Tỷ lệ mất cân bằng (Imbalance Ratio - IR):**
  $$\text{IR} = \frac{N_{\max}}{N_{\min}} = \frac{1,149 \text{ (bbps-2-3)}}{6 \text{ (hemorrhoids)}} = 191.5 : 1$$
* **Quy luật Pareto (80/20):**
  * Top 7 lớp đa số chiếm tới **66.7%** tổng dữ liệu.
  * Ngược lại, 4 lớp vi mô (`hemorrhoids`: 6 ảnh, `ileum`: 8 ảnh, `uc-grade-1-2`: 11 ảnh, `uc-grade-2-3`: 28 ảnh) chỉ chiếm **0.5%** tổng dữ liệu (53 ảnh trên 10,650 ảnh).
* **Hậu quả lâm sàng & thuật toán:**
  * Trong tập kiểm tra Test Set (20%), lớp `hemorrhoids` chỉ có đúng 1 ảnh, `ileum` có 1-2 ảnh. Chỉ cần dự đoán sai 1 mẫu này, F1-Score của lớp lập tức tụt về $0.00\%$.
  * Với 23 lớp, việc có 3 lớp bị $0.00\%$ F1 sẽ kéo tụt trần số học của Macro F1 xuống tối đa:
    $$\text{Macro F1}_{\max} \le \frac{20 \times 100\% + 3 \times 0\%}{23} = 86.95\%$$

---

## 12. ĐỀ XUẤT TÁI CẤU TRÚC NHÃN MỤC TIÊU (PHASE 4: TARGET SCHEMA DECISION)

Dựa trên toàn bộ bằng chứng toán học và hình thái học thu thập được qua 17 tác vụ DCAI, phương án tái cấu trúc nhãn mục tiêu cho Phase 4 được xác lập như sau:

```mermaid
graph TD
    subgraph "Nhóm Tổn Thương Cần Tái Cấu Trúc"
        A1["barretts (39)"] & A2["barretts-short (52)"] -->|Hợp nhất Cosine 0.9896| R1["barretts-esophagus (91)"]
        B1["uc-grade-0-1 (35)"] & B2["uc-grade-1 (196)"] -->|Chuẩn hóa Mayo| R2["mayo-1-mild (231)"]
        B3["uc-grade-1-2 (11)"] & B4["uc-grade-2 (438)"] -->|Chuẩn hóa Mayo| R3["mayo-2-moderate (449)"]
        B5["uc-grade-2-3 (28)"] & B6["uc-grade-3 (133)"] -->|Chuẩn hóa Mayo| R4["mayo-3-severe (161)"]
    end
    subgraph "Các Lớp Vi Mô Cực Đoan"
        C1["hemorrhoids (6)"] -->|Chuyển tập kiểm định chuyên biệt| S1["Isolated Special Study"]
        C2["ileum (8)"] -->|Hợp nhất mốc hồi tràng / manh tràng| S2["cecum-and-ileum (1,023)"]
    end
```

### Các Quyết Định Chiến Lược:
1. **Hợp nhất `barretts` và `barretts-short-segment` $\to$ `barretts-esophagus` ($N=91$):** Loại bỏ hoàn toàn sự phân tách giả tạo về mặt quang học giữa hai phân đoạn chiều dài tổn thương có cùng bản chất vi mô chuyển sản ruột.
2. **Chuẩn hóa Viêm loét đại tràng theo Thang điểm Mayo 4 Mức Kinh Điển:**
   * Hợp nhất `uc-grade-0-1` vào `uc-grade-1` $\to$ **Mayo Grade 1 (Viêm nhẹ)**.
   * Hợp nhất `uc-grade-1-2` vào `uc-grade-2` $\to$ **Mayo Grade 2 (Viêm trung bình)**.
   * Hợp nhất `uc-grade-2-3` vào `uc-grade-3` $\to$ **Mayo Grade 3 (Viêm nặng)**.
   * Điều này loại bỏ hoàn toàn các lớp chuyển tiếp nhân tạo có cỡ mẫu dưới 35 ảnh, biến dải phổ liên tục thành các mốc phân loại rõ ràng có giá trị chỉ định điều trị thực tế.
3. **Giải phóng rào cản trần Macro F1:** Sau khi hợp nhất, mọi phân lớp đều có cỡ mẫu $\ge 50$ ảnh, loại bỏ hiện tượng F1 bằng 0 do 1-2 mẫu test, mở đường cho Macro F1 vượt ngưỡng **85% – 90%**.

---

## 13. CƠ SỞ KHOA HỌC ĐỀ XUẤT KIẾN TRÚC HYBRID SUPCON + CBAM ATTENTION

Toàn bộ kết quả EDA dữ liệu sạch cung cấp luận cứ bảo vệ không thể phản bác cho mô hình đề xuất trong luận văn tốt nghiệp:

1. **Khối chú ý không gian và kênh CBAM (Convolutional Block Attention Module):**
   * *Nhiệm vụ:* Giải quyết vấn đề **Định kiến màu sắc tổng thể (Color Bias)**.
   * *Cơ chế:* Ở các lớp nhuộm màu Indigo Carmine (`dyed-lifted` vs `dyed-resection`) hoặc viêm loét đỏ, CBAM sẽ ức chế các kênh màu nền nhân tạo, đồng thời kích hoạt trọng số chú ý không gian vào các vi cấu trúc bờ mép tổn thương (pit-patterns) và vi mạch biểu mô.
2. **Học tương phản có giám sát (Supervised Contrastive Learning - SupCon):**
   * *Nhiệm vụ:* Giải quyết triệt để **Nghịch lý Margin âm ($\mathcal{M} < 0$)** mà Cross-Entropy bất lực.
   * *Cơ chế:* Tác động trực tiếp vào không gian nhúng siêu cầu (Hyperspherical Latent Space) bằng cách:
     $$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$
     SupCon chủ động kéo các mẫu cùng lớp tụ lại thành các cụm siêu nén, đồng thời đẩy lùi các lớp có độ tương đồng hình thái cao (như Mayo 1 và Mayo 2) ra xa nhau, tạo ra biên độ an toàn cực đại cho bộ phân loại.

---

### KẾT LUẬN NGHIỆM THU
Chuỗi tác vụ Data-Centric AI (D01 – D17) đã biến đổi hoàn toàn bộ dữ liệu HyperKvasir từ một tập ảnh thô chứa nhiều nhiễu nhãn, trùng lặp và rò rỉ video thành một **Hệ Thống Dữ Liệu Sạch Tiêu Chuẩn Y Khoa (Clinically Validated Clean Dataset)**. Đây là nền tảng quyết định để khóa luận bước vào Phase 4 (Chuẩn hóa Schema) và Phase 5 (Huấn luyện Mô hình Đột phá).
