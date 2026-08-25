# 🏷️ Báo cáo Phân nhóm Nhãn Đa tầng & Chiến lược Phân loại (Label Taxonomy)

> **Master Dataset:** HyperKvasir (10,662 ảnh) | **Metadata File:** `data/processed/hyperkvasir_metadata.csv`

---

## 1. Ba Chiến lược Phân loại Đa tầng của Đề tài

| Cấp độ phân loại | Số lượng lớp | Mục tiêu ứng dụng trong Khóa luận & Bài báo |
|:---|:---:|:---|
| **Tầng 1 (Super-Category)** | **4 Nhóm** | Phân loại cấp cao: *Mốc giải phẫu vs Bệnh lý vs Can thiệp vs Chất lượng view*. |
| **Tầng 2 (Clinical Aggregated)** | **8 Lớp** | Đối sánh công bằng (*Fair Benchmark*) trực tiếp với các mô hình trên Kvasir-v2. |
| **Tầng 3 (Fine-Grained 23-Class)** | **23 Lớp** | **Nhiệm vụ trọng tâm của Đề tài:** Đánh giá năng lực của kiến trúc đề xuất *CNN-CBAM-Transformer + SupCon* trên bài toán phân loại siêu chi tiết. |

---

## 2. Bảng Ánh xạ Nhãn Chi tiết (23 Lớp ➔ 4 Nhóm Lớn)

| ID (23) | Tên lớp 23 (Fine-grained) | Nhóm lớn (Super Category) | Số lượng ảnh | Tỷ trọng (%) |
|:---:|:---|:---|:---:|:---:|
| `0` | **barretts** | `Pathological_Findings` | 41 | 0.38% |
| `1` | **barretts-short-segment** | `Pathological_Findings` | 53 | 0.50% |
| `2` | **bbps-0-1** | `Quality_of_Mucosal_Views` | 646 | 6.06% |
| `3` | **bbps-2-3** | `Quality_of_Mucosal_Views` | 1,148 | 10.77% |
| `4` | **cecum** | `Anatomical_Landmarks` | 1,009 | 9.46% |
| `5` | **dyed-lifted-polyps** | `Therapeutic_Interventions` | 1,002 | 9.40% |
| `6` | **dyed-resection-margins** | `Therapeutic_Interventions` | 989 | 9.28% |
| `7` | **esophagitis-a** | `Pathological_Findings` | 403 | 3.78% |
| `8` | **esophagitis-b-d** | `Pathological_Findings` | 260 | 2.44% |
| `9` | **hemorrhoids** | `Pathological_Findings` | 6 | 0.06% |
| `10` | **ileum** | `Anatomical_Landmarks` | 9 | 0.08% |
| `11` | **impacted-stool** | `Quality_of_Mucosal_Views` | 131 | 1.23% |
| `12` | **polyps** | `Pathological_Findings` | 1,028 | 9.64% |
| `13` | **pylorus** | `Anatomical_Landmarks` | 999 | 9.37% |
| `14` | **retroflex-rectum** | `Anatomical_Landmarks` | 391 | 3.67% |
| `15` | **retroflex-stomach** | `Anatomical_Landmarks` | 764 | 7.17% |
| `16` | **ulcerative-colitis-grade-0-1** | `Pathological_Findings` | 35 | 0.33% |
| `17` | **ulcerative-colitis-grade-1** | `Pathological_Findings` | 201 | 1.89% |
| `18` | **ulcerative-colitis-grade-1-2** | `Pathological_Findings` | 11 | 0.10% |
| `19` | **ulcerative-colitis-grade-2** | `Pathological_Findings` | 443 | 4.15% |
| `20` | **ulcerative-colitis-grade-2-3** | `Pathological_Findings` | 28 | 0.26% |
| `21` | **ulcerative-colitis-grade-3** | `Pathological_Findings` | 133 | 1.25% |
| `22` | **z-line** | `Anatomical_Landmarks` | 932 | 8.74% |
