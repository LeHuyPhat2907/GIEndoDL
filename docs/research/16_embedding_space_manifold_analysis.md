# Báo Cáo Nghiên Cứu: Cấu Trúc Đa Tạp Không Gian Nhúng t-SNE / UMAP & Các Vùng Chồng Lấn Bệnh Lý

> **Mã công việc:** Task D17
> **Phương pháp giảm chiều:** PCA-50 + t-SNE (`perplexity=40`) & UMAP (`n_neighbors=30, min_dist=0.3, metric='cosine'`)
> **Tập dữ liệu:** DINOv2 ViT-S/14 384-d trên 10,650 mẫu sạch (`cleaned_hyperkvasir_labels.csv`)

---

## 1. Mục Đích & Ý Nghĩa Nghiên Cứu

Khi làm việc với các không gian vector biểu diễn đặc trưng sâu (Deep Feature Embeddings) 384 chiều, các phép đo khoảng cách Euclid hay Cosine đơn thuần ở Task D16 mới chỉ cung cấp bức tranh trung bình (Centroid-based). Việc chiếu phi tuyến tính không gian nhúng xuống mặt phẳng 2D thông qua **t-SNE** (bảo toàn cấu trúc lân cận cục bộ) và **UMAP** (bảo toàn hình học đa tạp toàn cục) giúp:

1. Trực quan hóa mật độ phân bố thực tế của từng ca bệnh và từng khung hình nội soi.
2. Chứng minh trực quan sự phân tách rõ rệt giữa 5 đại cụm giải phẫu tự nhiên (Upper GI, Lower GI, Chromoendoscopy, Bowel Cleanliness, Retroflexion).
3. Nhận diện chính xác hình thái hình học của 3 'điểm nóng' chồng lấn bệnh học (Pathological Overlapping Hotspots) - nguyên nhân cốt lõi khiến mô hình học sâu Run 4 dừng bước ở 69.28% Macro F1.

---

## 2. Phân Tích Chi Tiết 3 Điểm Nóng Chồng Lấn Đa Tạp (Topological Overlap Zones)

### 2.1. Vùng Chồng Lấn A: Bệnh Lý Thực Quản & Đường Ranh Giới Z (Esophageal Spectrum)

- **Các lớp liên quan:** `barretts` (39 mẫu), `barretts-short-segment` (52 mẫu), `esophagitis-a` (376 mẫu), `esophagitis-b-d` (261 mẫu), `z-line` (961 mẫu).
- **Hình thái trên UMAP:**
  - Hai lớp `barretts` và `barretts-short-segment` **hòa trộn hoàn toàn thành một đám mây điểm duy nhất**, không tồn tại bất kỳ ranh giới phân tách hình học nào giữa chúng. Điều này hoàn toàn phù hợp với thực tế lâm sàng: sự khác nhau giữa Barrett đoạn ngắn và Barrett điển hình chỉ là chiều dài tổn thương tính bằng centimet theo thang phân loại Prague C&M, trong khi bản chất vi mô mô học (biểu mô trụ chuyển sản ruột màu cá hồi) là giống hệt nhau.
  - Lớp `esophagitis-a` bọc quanh một phần của `z-line` và tiếp giáp chặt chẽ với `barretts`. Sau khi thực hiện Confident Learning ở Task D14/D15, phần lớn các mẫu Z-line gán nhầm đã được đưa về đúng lớp, làm mật độ chồng lấn giữa `esophagitis-a` và `z-line` giảm đáng kể, tạo ra khe hở ranh giới khả dĩ cho mạng nơ-ron nhận diện.

### 2.2. Vùng Chồng Lấn B: Dải Biến Thiên Liên Tục Của Viêm Loét Đại Tràng (Mayo Colitis Continuum)

- **Các lớp liên quan:** `uc-grade-0-1`, `uc-grade-1`, `uc-grade-1-2`, `uc-grade-2`, `uc-grade-2-3`, `uc-grade-3`.
- **Hình thái trên UMAP:**
  - Không gian biểu diễn của bệnh viêm loét đại tràng **không tạo thành các cụm đảo rời rạc (Discrete Clusters)**, mà trải dài thành một **dải đa tạp cong liên tục (1D-Curve / Trajectory Continuum)** đi từ Grade 0-1 qua Grade 1, Grade 2 đến Grade 3.
  - Các lớp lai trung gian (`grade-0-1`, `grade-1-2`, `grade-2-3`) nằm xen kẽ chính giữa các nốt chính và có mật độ mẫu cực thấp ($N \le 35$). Khi huấn luyện phân loại rời rạc 23 lớp, các lớp trung gian này đóng vai trò như 'bẫy gradient', khiến hàm mất mát Cross-Entropy bị nhiễu loạn cực mạnh.
  - **Kết luận giải pháp:** Phải chuẩn hóa hệ thống nhãn Mayo Colitis về 4 mức kinh điển (Mayo 0, 1, 2, 3) hoặc áp dụng bài toán Hồi quy Thứ bậc (Ordinal Regression).

### 2.3. Vùng Chồng Lấn C: Phân Vùng Nhuộm Màu Indigo Carmine (Chromoendoscopy Island)

- **Các lớp liên quan:** `dyed-lifted-polyps` (999 mẫu) và `dyed-resection-margins` (990 mẫu).
- **Hình thái trên UMAP:**
  - Cả hai lớp này tách biệt hoàn toàn thành một 'hòn đảo độc lập' nằm rất xa các lớp niêm mạc tự nhiên khác. Lý do: màu phẩm nhuộm Indigo Carmine (xanh lam đậm) tạo ra phản ứng kích hoạt thị giác áp đảo trên toàn bộ bức ảnh.
  - Tuy nhiên, bên trong hòn đảo nhuộm màu này, `dyed-lifted` (polyp được tiêm phồng dung dịch xanh) và `dyed-resection` (vết loét lòng ruột sau khi cắt bỏ polyp) nằm đan cài chặt chẽ với nhau.
  - **Kết luận giải pháp:** Mô hình mạng bắt buộc phải có cơ chế chú ý kênh và không gian (CBAM Attention) để 'bỏ qua' nền màu xanh lam tổng thể và tập trung vào bờ rìa tổn thương (loét đáy hay khối u gồ).

---

## 3. Tổng Hợp Tệp Tin & Đồ Thị Đã Xuất

- **Tọa độ 2D t-SNE & UMAP:** `data/processed/dinov2_tsne_umap_2d_projections.csv` (10,650 dòng)
- **Biểu đồ So sánh 5 Đại Cụm (t-SNE vs UMAP):** `docs/figures/06_color_distribution_tsne.png`
- **Biểu đồ Chi tiết 23 Phân Lớp:** `docs/figures/embedding_space_tsne_umap_comparison.png`
- **Biểu đồ 3 Vùng Chồng Lấn Bệnh Lý:** `docs/figures/06_overlapping_pathology_subclusters.png`
