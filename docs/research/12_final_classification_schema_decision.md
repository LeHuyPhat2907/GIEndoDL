# 🏛️ Quyết định Chiến lược: Schema Phân lớp Chính thức cho Dự án GIEndoDL

> **Mã tài liệu:** `DOC-RES-12-SCHEMA-DECISION`
> **Chủ đề:** Phân tích đánh đổi (Trade-off Analysis), Cơ sở lý luận y khoa & Quyết định Schema phân lớp chính thức cho Mô hình Đề xuất.
> **Căn cứ thực nghiệm:** Báo cáo Phân cấp (`DOC-RES-04`), Báo cáo Pareto 191:1 (`DOC-RES-05`) & Báo cáo Màu sắc t-SNE (`DOC-RES-08`).

---

## 1. Phân tích Bài toán Đánh đổi (Trade-off Analysis)

Khi xây dựng mô hình Trí tuệ Nhân tạo cho Nội soi Tiêu hóa, việc lựa chọn số lượng lớp phân loại luôn phải đối mặt với bài toán tối ưu 2 mặt:

```text
               CHIẾN LƯỢC PHÂN LỚP
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
SCHEMA 8 LỚP LÂM SÀNG             SCHEMA 23 LỚP SIÊU CHI TIẾT
(Clinical Aggregated)              (Fine-Grained 23-Class)
- Dữ liệu cân bằng (~1,000 ảnh/lớp)  - Phản ánh đúng 100% bệnh học y khoa
- Dễ đạt Accuracy/F1 cao (>95%)      - Phân độ chi tiết Mayo & LA grades
- Dễ so sánh với bài báo cũ          - Thách thức lớn: Mất cân bằng 191:1
- Mất đi tính phân độ tổn thương     - Khẳng định tính đột phá của Đề tài
```

## 2. QUYẾT ĐỊNH CHIẾN LƯỢC CHÍNH THỨC: SCHEMA ĐA TẦNG KÉP (DUAL-BENCHMARK SCHEMA)
Để đạt được cả 2 mục tiêu: (1) Đóng góp khoa học đột phá và (2) Đối sánh công bằng với các công bố quốc tế, đề tài quyết định triển khai Chiến lược Phân lớp Kép:

1. SCHEMA CHÍNH (Primary Research Target): 23 LỚP SIÊU CHI TIẾT (FINE-GRAINED)
Quy mô: Toàn bộ 23 lớp gốc của HyperKvasir (10,662 ảnh).
Mục đích: Là nhiệm vụ trọng tâm của Khóa luận. Dùng để chứng minh năng lực vượt trội của kiến trúc đề xuất CNN-CBAM-Transformer + SupCon trong việc giải quyết bài toán phân loại siêu chi tiết và mất cân bằng cực đoan (191:1).
Giải pháp kỹ thuật đi kèm: Sử dụng Weighted Random Sampler + Class-Balanced Focal Loss (GĐ 4) và Supervised Contrastive Learning (GĐ 9).

## 2. SCHEMA ĐỐI SÁNH (Secondary Comparative Benchmark): 8 LỚP LÂM SÀNG
Quy mô: 8 nhóm bệnh lý/mốc giải phẫu chuẩn (polyps, ulcerative-colitis, esophagitis, barretts, dyed-lifted-polyps, dyed-resection-margins, normal-cecum, normal-z-line/pylorus).

Mục đích: Dùng để so sánh đối đầu trực tiếp (Fair Benchmark SOTA) với các công bố quốc tế hiện hành và thực hiện Thẩm định ngoài độc lập (External Validation) trên bộ dữ liệu Kvasir-v2 (8,000 ảnh).

## 3. Bảng Tổng kết Schema 23 Lớp Chính thức & Mã hóa ID
Class ID	Tên lớp chính thức (Label)	Phân vùng (Tract)	Nhóm chức năng	Số ảnh gốc	Tỷ trọng (%)
0	barretts	Upper GI	Pathological Findings	41	0.38%
1	barretts-short-segment	Upper GI	Pathological Findings	53	0.50%
2	bbps-0-1	Lower GI	Quality of Mucosal Views	646	6.06%
3	bbps-2-3	Lower GI	Quality of Mucosal Views	1,148	10.77%
4	cecum	Lower GI	Anatomical Landmarks	1,009	9.46%
5	dyed-lifted-polyps	Lower GI	Therapeutic Interventions	1,002	9.40%
6	dyed-resection-margins	Lower GI	Therapeutic Interventions	989	9.28%
7	esophagitis-a	Upper GI	Pathological Findings	403	3.78%
8	esophagitis-b-d	Upper GI	Pathological Findings	260	2.44%
9	hemorrhoids	Lower GI	Pathological Findings	6	0.06%
10	ileum	Lower GI	Anatomical Landmarks	9	0.08%
11	impacted-stool	Lower GI	Quality of Mucosal Views	131	1.23%
12	polyps	Lower GI	Pathological Findings	1,028	9.64%
13	pylorus	Upper GI	Anatomical Landmarks	999	9.37%
14	retroflex-rectum	Lower GI	Anatomical Landmarks	391	3.67%
15	retroflex-stomach	Upper GI	Anatomical Landmarks	764	7.17%
16	ulcerative-colitis-grade-0-1	Lower GI	Pathological Findings	35	0.33%
17	ulcerative-colitis-grade-1	Lower GI	Pathological Findings	201	1.89%
18	ulcerative-colitis-grade-1-2	Lower GI	Pathological Findings	11	0.10%
19	ulcerative-colitis-grade-2	Lower GI	Pathological Findings	443	4.15%
20	ulcerative-colitis-grade-2-3	Lower GI	Pathological Findings	28	0.26%
21	ulcerative-colitis-grade-3	Lower GI	Pathological Findings	133	1.25%
22	z-line	Upper GI	Anatomical Landmarks	932	8.74%
TỔNG CỘNG			10,662	100.00%

## 4. Đoạn văn Trình bày Cơ sở Khoa học trong Khóa luận
"Nhằm đảm bảo tính thực tiễn lâm sàng và phát huy tối đa năng lực trích xuất đặc trưng của mô hình học sâu, đề tài lựa chọn tiếp cận bài toán phân loại đa tầng với trọng tâm là Schema 23 lớp siêu chi tiết (Fine-grained classification). Quyết định này giúp mô hình không chỉ phát hiện sự hiện diện của tổn thương mà còn phân định chính xác mức độ tiến triển viêm loét đại tràng (Mayo score 0-3) và phân loại Barrett thực quản. Đồng thời, đề tài duy trì Schema 8 lớp lâm sàng làm đối chuẩn so sánh công bằng với các công bố quốc tế trên Kvasir-v2 và thực hiện kiểm định ngoài độc lập theo tiêu chuẩn TRIPOD-AI."
