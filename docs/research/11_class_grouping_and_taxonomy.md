# Báo Cáo Phân Nhóm Nhãn Đa Tầng & Chiến Lược Phân Cấp Y Sinh (Label Taxonomy & Hierarchical Grouping)

> **Mã tài liệu:** `DOC-RES-11-TAXONOMY`
> **Cơ sở thực nghiệm:** Phân tích hình thái DINOv2 (Task D16), Đa tạp t-SNE/UMAP (Task D17), Chuẩn quốc tế TRIPOD-AI
> **Tổng mẫu Master Metadata:** 10,662 mẫu (10,644 mẫu sạch hợp lệ cho huấn luyện)

---

## 1. Bản Chất Khoa Học Của Việc Tái Cấu Trúc Schema

Trong giai đoạn huấn luyện thử nghiệm Run 4 trước đây, mô hình học sâu bị kẹt ở mức 69.28% Macro F1 dù đạt Accuracy 85.93%. Phân tích dữ liệu ở Tasks D01–D17 đã chứng minh hai nguyên nhân căn bản:
1. **Nghịch lý Lớp Vi mô (< 15 mẫu):** Lớp `hemorrhoids` (6 ảnh) và `ileum` (8 ảnh) chỉ có 1–2 ảnh trong tập Test. Khi dự đoán sai 1 ảnh, F1 lập tức rơi về 0.00%, kéo tụt 9–10% tổng Macro F1 trung bình toàn mạng.
2. **Xung đột Hình học Lớp Lai (Negative Margin):** Các lớp chuyển tiếp trung gian của Viêm loét đại tràng (`uc-grade-0-1`, `1-2`, `2-3`) và Barrett thực quản (`barretts-short-segment`) có Cosine Similarity > 0.98 với lớp liền kề, tạo ra các đám mây điểm chồng lấn hoàn toàn khiến hàm mất mát Cross-Entropy bị dao động gradient đối kháng triền miên.

---

## 2. Chi Tiết 4 Quyết Định Chuẩn Hóa Cốt Lõi (Tasks D19 – D22)

### 2.1. Task D19: Xử Lý Các Lớp Cực Vi Mô (< 15 Ảnh)
- **`hemorrhoids` (6 ảnh):** Tách khỏi bài toán phân loại chính 17 lớp, chuyển sang tập **Thẩm định Chuyên biệt Độc lập (Holdout Special Study)**. Điều này bảo toàn giá trị khoa học của dữ liệu mà không làm sai lệch trọng số phân phối của bộ phân loại đa lớp.
- **`ileum` (8 ảnh):** Hợp nhất cùng `cecum` thành mốc giải phẫu **`cecum-and-ileum`** (1,023 ảnh). Trong thực hành lâm sàng, việc quan sát được manh tràng hoặc đẩy ống soi qua van Bauhin vào hồi tràng đều có chung một ý nghĩa quy trình: *Khẳng định thủ thuật nội soi đại tràng đã hoàn thành trọn vẹn (Complete Colonoscopy Indicator)*.

### 2.2. Task D20: Chuẩn Hóa Phân Độ Viêm Loét Đại Tràng Mayo 3 Mức Kinh Điển
Chuẩn y tế quốc tế Mayo Endoscopic Subscore (MES) chỉ công nhận các mức rời rạc: Mayo 1 (Nhẹ), Mayo 2 (Trung bình), Mayo 3 (Nặng). Các lớp lai trung gian là hệ quả của sự phân vân chủ quan giữa các bác sĩ gán nhãn:
- `uc-grade-0-1` (35) + `uc-grade-1` (196) $\to$ **`mayo-grade-1`** (231 mẫu).
- `uc-grade-1-2` (11) + `uc-grade-2` (438) $\to$ **`mayo-grade-2`** (449 mẫu).
- `uc-grade-2-3` (28) + `uc-grade-3` (133) $\to$ **`mayo-grade-3`** (161 mẫu).
> **Kết quả:** Cả 3 mức Mayo đều có cỡ mẫu dồi dào ($N \ge 160$), phản ánh trung thực mức độ nghiêm trọng của bệnh nhân để chỉ định phác đồ thuốc sinh học.

### 2.3. Task D21: Chuẩn Hóa Barrett Thực Quản (`barretts-esophagus`)
- Phân tích DINOv2 ở Task D16 xác nhận `barretts` và `barretts-short-segment` có Cosine Similarity lên đến **0.9896** (cao nhất trong 23 lớp).
- Phân tích UMAP ở Task D17 chứng minh hai lớp này hòa lẫn thành một cụm duy nhất.
- Về giải phẫu bệnh: Cả hai đều là tổn thương chuyển sản ruột (Intestinal Metaplasia), chỉ khác nhau chiều dài tổn thương đo bằng thước đo cơ học của dây soi ($< 3\text{ cm}$ vs $\ge 3\text{ cm}$). Hợp nhất thành **`barretts-esophagus`** ($N=91$) tạo nên lớp bệnh lý thực quản đồng nhất và đủ mẫu kiểm định.

### 2.4. Task D22: Bảo Tồn Phân Độ Viêm Thực Quản Trào Ngược Los Angeles
- Duy trì tách biệt hai phân lớp: **`esophagitis-grade-a`** ($N=376$) và **`esophagitis-grade-b-d`** ($N=261$).
- *Ý nghĩa lâm sàng:* Phân biệt giữa tổn thương trợt xước nhẹ $\le 5\text{ mm}$ (Grade A - điều trị nội khoa bảo tồn) và vết loét lan rộng $> 5\text{ mm}$ hoặc bao quanh chu vi (Grade B-D - nguy cơ hẹp và biến chứng) là yêu cầu tiên quyết trong phác đồ điều trị GERD quốc tế.

---

## 3. Cấu Trúc Phân Bố 5 Nhóm Cơ Quan Giải Phẫu (Organ Groups)

### Upper GI Tract
- **Tổng số mẫu:** 3,450 mẫu (32.4%)
- **Các phân lớp thành phần:** `barretts-esophagus`, `esophagitis-grade-a`, `esophagitis-grade-b-d`, `normal-z-line`, `pylorus`, `retroflex-stomach`

### Lower GI Tract - Colorectum
- **Tổng số mẫu:** 2,441 mẫu (22.9%)
- **Các phân lớp thành phần:** `cecum-and-ileum`, `polyps`, `retroflex-rectum`

### Lower GI Tract - Inflammatory Bowel Disease
- **Tổng số mẫu:** 841 mẫu (7.9%)
- **Các phân lớp thành phần:** `mayo-grade-1`, `mayo-grade-2`, `mayo-grade-3`

### Chromoendoscopy - Interventions
- **Tổng số mẫu:** 1,989 mẫu (18.7%)
- **Các phân lớp thành phần:** `dyed-lifted-polyps`, `dyed-resection-margins`

### Bowel Preparation Quality
- **Tổng số mẫu:** 1,923 mẫu (18.1%)
- **Các phân lớp thành phần:** `bbps-grade-0-1`, `bbps-grade-2-3`, `impacted-stool`
