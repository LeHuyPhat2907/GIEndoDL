# Báo cáo Phân tích Độ Tương đồng Hình thái & Các Cặp Lớp Dễ Nhầm lẫn (Morphological Similarity)

> **Phương pháp:** Trích xuất không gian nhúng 512 chiều (ResNet Deep Feature Extractor) & Đo lường Cosine Similarity Centroids

---

## 1. Top Các Cặp Bệnh lý / Mốc Giải phẫu Dễ Nhầm lẫn Nhất

| STT | Lớp Bệnh học A | Lớp Bệnh học B | Độ tương đồng Cosine | Nguy cơ Chẩn đoán Lâm sàng |
|:---:|:---|:---|:---:|:---|
| 1 | `barretts` | `barretts-short-segment` | **0.9944** | Khó phân biệt ranh giới tổn thương |
| 2 | `barretts-short-segment` | `esophagitis-a` | **0.9942** | Khó phân biệt ranh giới tổn thương |
| 3 | `ulcerative-colitis-grade-0-1` | `ulcerative-colitis-grade-1` | **0.9920** | Khó phân biệt ranh giới tổn thương |
| 4 | `esophagitis-a` | `z-line` | **0.9915** | Khó phân biệt ranh giới tổn thương |
| 5 | `barretts` | `esophagitis-a` | **0.9909** | Khó phân biệt ranh giới tổn thương |
| 6 | `ulcerative-colitis-grade-2-3` | `ulcerative-colitis-grade-3` | **0.9905** | Khó phân biệt ranh giới tổn thương |
| 7 | `barretts-short-segment` | `z-line` | **0.9902** | Khó phân biệt ranh giới tổn thương |
| 8 | `ulcerative-colitis-grade-1` | `ulcerative-colitis-grade-2` | **0.9899** | Khó phân biệt ranh giới tổn thương |
| 9 | `dyed-lifted-polyps` | `dyed-resection-margins` | **0.9898** | Khó phân biệt ranh giới tổn thương |
| 10 | `ulcerative-colitis-grade-2` | `ulcerative-colitis-grade-2-3` | **0.9896** | Khó phân biệt ranh giới tổn thương |

---

## 2. Kết luận Khoa học & Cơ sở Đề xuất Học Tương Phản (SupCon)

1. **Hiện tượng Ranh giới Mờ nhạt:** Các phân lớp viêm loét đại tràng (Mayo 1 vs Mayo 1-2) và bệnh lý thực quản (Barrett vs Viêm thực quản) có độ tương đồng Cosine $> 0.85$. Điều này giải thích vì sao mạng CNN truyền thống rất dễ nhầm lẫn giữa các cấp độ tổn thương.
2. **Bảo vệ Thuật toán Đề xuất (SupCon + CBAM):**
   - **Khối CBAM (Giai đoạn 8):** Tập trung vào các chi tiết vi mô cục bộ (vi mạch pit-pattern) để phân biệt các lớp có màu nền tương đồng.
   - **Supervised Contrastive Learning (Giai đoạn 9):** Tác động trực tiếp vào không gian nhúng (Embedding Space) bằng cách chủ động **kéo gần các mẫu cùng lớp và đẩy xa các lớp dễ nhầm lẫn**.
