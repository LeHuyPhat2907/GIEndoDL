# BÁO CÁO TOÀN DIỆN KHÁM PHÁ DỮ LIỆU NỘI SOI TIÊU HÓA (COMPREHENSIVE EDA MASTER REPORT)

> **Mã tài liệu:** `DOC-RES-16-EDA-MASTER`
> **Đề tài:** Ứng dụng học sâu trong nhận diện và phân loại tổn thương đường tiêu hóa từ ảnh nội soi.
> **Tác giả:** Lê Huy Phát
> **Quy mô tập dữ liệu:** 20,662 ảnh y tế (HyperKvasir 10,662 ảnh, Kvasir-v2 8,000 ảnh, Kvasir-SEG 1,000 cặp ảnh-mask).
> **Chuẩn báo cáo quốc tế:** Tuân thủ hướng dẫn TRIPOD-AI (Transparent Reporting of a multivariable prediction model of an Artificial Intelligence).

---

## 1. TỔNG QUAN NGUỒN GỐC & TÍNH TOÀN VẸN DỮ LIỆU (DATASET PROVENANCE)

Nghiên cứu sử dụng 3 bộ dữ liệu chuẩn y tế do **Viện nghiên cứu Simula** và **Bệnh viện Bærum (Na Uy)** công bố:
1. **HyperKvasir (Tập Huấn luyện & Đánh giá Đa tầng):** 10,662 ảnh được gán nhãn bởi các chuyên gia tiêu hóa hàng đầu châu Âu, phân bố trên 23 lớp bệnh lý và mốc giải phẫu.
2. **Kvasir-v2 (Tập Thẩm định Ngoài Độc lập - External Validation):** 8,000 ảnh chuẩn 8 lớp. Đối chiếu mã băm MD5 cho thấy chỉ trùng lặp 0.2% (14 ảnh), khẳng định tính độc lập tuyệt đối để đánh giá khả năng tổng quát hóa theo chuẩn TRIPOD-AI.
3. **Kvasir-SEG (Tập Phân đoạn Vùng Tổn thương ROI):** 1,000 ảnh polyp đi kèm 1,000 mặt nạ nhị phân chuyên gia (100% khớp pixel-to-pixel).

---

## 2. BẢN ĐỒ HÌNH THÁI HỌC 23 LỚP (VISUAL ATLAS)

Toàn bộ 23 lớp của HyperKvasir được phân chia theo 4 nhóm chức năng chính trên 2 phân vùng giải phẫu (Upper GI & Lower GI):

![HyperKvasir 23-Class Visual Atlas](../figures/08_hyperkvasir_23_classes_grid.png)

```text
CẤU TRÚC PHÂN CẤP BỘ DỮ LIỆU HYPERKVASIR (10,662 ẢNH)
├── 1. Mốc Giải Phẫu (Anatomical Landmarks): 6 lớp (cecum, ileum, pylorus, retroflex-rectum, retroflex-stomach, z-line)
├── 2. Tổn Thương Bệnh Lý (Pathological Findings): 12 lớp (polyps, trĩ, 6 mức độ viêm đại tràng Mayo, 2 mức viêm thực quản LA, 2 thể Barrett)
├── 3. Can Thiệp Thủ Thuật (Therapeutic Interventions): 2 lớp (dyed-lifted-polyps, dyed-resection-margins)
└── 4. Chất Lượng Quan Sát (Quality of Mucosal Views): 3 lớp (bbps-0-1, bbps-2-3, impacted-stool)
```

## 3. MẤT CÂN BẰNG LỚP CỰC ĐOAN & NGUYÊN LÝ PARETO (CLASS IMBALANCE)
Tỷ lệ Mất cân bằng (Imbalance Ratio - IR): IR = 1,148 (bbps-2-3) / 6 (hemorrhoids) ≈ 191.33 : 1

Quy luật Pareto: Top 7 lớp đa số chiếm tới 66.66% tổng dữ liệu. Ngược lại, 7 lớp thiểu số nhất chỉ chiếm 1.72% (183 ảnh).
--> Cơ sở Kỹ thuật: Việc chẩn đoán sai toàn bộ nhóm thiểu số vẫn có thể cho ra Accuracy 98.28%. Do đó, đề tài bắt buộc:
    Sử dụng Macro F1-Score và Balanced Accuracy làm chỉ số tối ưu chính.
    Tích hợp Weighted Random Sampler và Class-Balanced Focal Loss ở Giai đoạn 4.

## 4. ĐỘ PHÂN GIẢI & TỶ LỆ KHUNG HÌNH (RESOLUTION & ASPECT RATIO)
Phân bố 2 Cụm Thiết bị (Bimodal Clusters): Cụm SD truyền thống (≈ 633 × 532 px) và Cụm HD hiện đại (≈ 1349 × 1071 px).
Tỷ lệ Khung hình (Aspect Ratio): Phân bố hình chuông cực kỳ tập trung quanh mức 1.18 (≈ 5 : 4).
--> Quyết định Chuẩn hóa Đầu vào: Lựa chọn kích thước chuẩn 224 × 224 px (cho Web App real-time) và 384 × 384 px (cho mô hình nghiên cứu độ nét cao) bằng thuật toán Bicubic Interpolation kết hợp Letterbox Padding.

## 5. ĐÁNH GIÁ CHẤT LƯỢNG HÌNH ẢNH (IMAGE QUALITY ASSESSMENT)
79.4% ảnh đạt chuẩn chất lượng cao (độ sắc nét Laplacian Variance > 80, ánh sáng chuẩn Gauss μ ≈ 110).
16.4% ảnh có hiện tượng mờ do chuyển động (Motion Blur) ➔ Giữ lại để huấn luyện tính thích ứng (Robustness) bằng Data Augmentation.
4.2% ảnh có đốm lóa sáng (Specular Highlights > 4%) ➔ Ứng dụng thuật toán CLAHE trên kênh L (không gian LAB) ở Giai đoạn 3 để cân bằng sáng mà không đổi màu mô.

## 6. KHÔNG GIAN MÀU SẮC & CHIẾU t-SNE 2D (COLOR DISTRIBUTION & DEVICE BIAS)
Ưu thế Tuyệt đối của Kênh Đỏ: Kênh Đỏ (R ≈ 141.7) cao vượt trội so với Kênh Xanh dương (B ≈ 82.6) do sắc tố Hemoglobin mao mạch.
Hiện tượng Chồng lấn Sắc tố (Overlap): Trên không gian t-SNE 2D, nhóm Tổn thương bệnh lý và Mốc giải phẫu lành tính chồng lấn nặng nề, chứng minh Màu sắc đơn thuần không đủ để chẩn đoán.
--> Động lực cho Mô hình Lai: Bắt buộc phải tích hợp Khối CBAM (trích xuất hoa văn kết cấu vi mạch) và Vision Transformer (nắm bắt ngữ cảnh không gian toàn cục).

## 7. KIỂM ĐỊNH TRÙNG LẶP & PHÒNG CHỐNG RÒ RỈ DỮ LIỆU (DUPLICATE AUDIT)
Rà soát bằng Perceptual Hash (pHash) và SSIM phát hiện 172 cặp ảnh gần trùng lặp (SSIM ≥ 0.90) sinh ra từ các khung hình video liên tiếp.
--> Giao thức Zero Data Leakage: Bắt buộc cô lập các cặp ảnh sinh đôi vào cùng một phân vùng dữ liệu (hoặc cùng ở Train, hoặc cùng ở Test) khi chia tập ở Task 51.
## 8. ĐỘ TƯƠNG ĐỒNG HÌNH THÁI & HỌC TƯƠNG PHẢN (MORPHOLOGICAL SIMILARITY)
Trích xuất Deep Features bằng ResNet phát hiện 3 cụm bệnh lý có độ tương đồng cực cao (Sim>0.99):
barretts ⟷ barretts-short-segment (Sim=0.9944)
barretts-short-segment ⟷ esophagitis-a (Sim=0.9942)
Các phân độ ulcerative-colitis-grade-0-1 đến grade-3 (Sim=0.990−0.992)
--> Cơ sở Bảo vệ Thuật toán Đề xuất: Khẳng định tính bắt buộc của Supervised Contrastive Learning (SupCon) ở Giai đoạn 9 nhằm tái cấu trúc không gian vector, chủ động kéo tách các lớp dễ nhầm lẫn này.
## 9. TỔNG HỢP CƠ SỞ KHOA HỌC CHO CÁC GIAI ĐOẠN TIẾP THEO
Phát hiện Thực nghiệm	Thách thức Y tế & Kỹ thuật	Giải pháp Thuật toán Đề tài	Giai đoạn Thực hiện
Viền đen & Chữ số ghi hình	Gây nhiễu đặc trưng biên độ	Tự động Cắt ROI & Lọc viền thiết bị	Giai đoạn 3 (Task 37)
Đốm lóa sáng (4.2%)	Mất chi tiết bề mặt polyp	Thuật toán CLAHE trên kênh LAB	Giai đoạn 3 (Task 38)
Mất cân bằng lớp (191:1)	Mô hình bỏ sót bệnh lý hiếm	Weighted Sampler & Class-Balanced Loss	Giai đoạn 4 & 6
172 Cặp trùng lặp pHash	Nguy cơ rò rỉ dữ liệu (Leakage)	Grouped Stratified Split (Zero Leakage)	Giai đoạn 4 (Task 51)
Chồng lấn màu sắc bệnh học	Khó phân biệt tổn thương và niêm mạc lành	Tích hợp Khối chú ý CBAM	Giai đoạn 8
Tương đồng hình thái 0.994	Nhầm lẫn giữa các cấp độ viêm loét & Barrett	Học tương phản SupCon (Contrastive Loss)	Giai đoạn 9
