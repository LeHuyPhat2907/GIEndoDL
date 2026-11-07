# Quyết Định Ban Hành Schema Phân Lớp Tối Ưu Cho Đề Tài GIEndoDL (Final Target Schema Decree)

> **Mã tài liệu:** `DOC-RES-12-SCHEMA-DECISION`
> **Quyết định ban hành:** Áp dụng chính thức **Schema 17 Phân Lớp Mục Tiêu (Target Schema V2)** cho toàn bộ quy trình huấn luyện, tối ưu và thẩm định mô hình tốt nghiệp.
> **Tổng số mẫu mục tiêu:** 10,644 mẫu (100% active, 0 rò rỉ ranh giới, 0 mẫu trùng lặp).

---

## 1. Bảng So Sánh Chiến Lược: Schema 23 Lớp Cũ vs Schema 17 Lớp Tối Ưu

| Chỉ Số Kiểm Toán | Schema 23 Lớp Ban Đầu (Raw) | Schema 17 Lớp Tối Ưu (Target V2) | Đánh Giá Tác Động Khoa Học |
|:---|:---:|:---:|:---||
| **Tổng số phân lớp** | 23 lớp | **17 lớp** | Chuẩn hóa theo hướng dẫn y khoa Mayo và Los Angeles quốc tế. |
| **Tổng mẫu huấn luyện** | 10,662 mẫu (chứa nhiễu) | **10,644 mẫu sạch** | Đã loại bỏ 12 mẫu trùng lặp tuyệt đối và cách ly 6 mẫu trĩ cực hiếm. |
| **Cỡ mẫu nhỏ nhất ($N_<built-in function min>$)** | 6 mẫu (`hemorrhoids`) | **91 mẫu (`barretts-esophagus`)** | **Tăng gấp 15.2 lần cỡ mẫu tối thiểu!** Đảm bảo đủ mẫu cho mọi Fold. |
| **Tỷ lệ mất cân bằng (IR)** | **191.5 : 1** | **12.63 : 1** | **Giảm 15 lần độ mất cân bằng**, giải phóng hoàn toàn trần Macro F1. |
| **Chỉ số rò rỉ dữ liệu ($\Lambda$)** | Chưa kiểm soát | **0.00000000%** | Ràng buộc Atomic Patient Video Cluster Graph. |
| **Hiện tượng bẫy gradient** | Nặng nề (3 lớp lai Mayo) | **Triệt tiêu hoàn toàn** | Các lớp đều phân tách hình thái rõ nét trên UMAP. |

---

## 2. Bảng Thống Kê Chi Tiết 17 Phân Lớp Mục Tiêu Chính Thức

| STT | Tên Phân Lớp Mục Tiêu (`final_class`) | Nhóm Cơ Quan (`organ_group`) | Số Lượng Mẫu | Tỷ Trọng (%) | Cỡ Mẫu Test (20%) | Ghi Chú Y Sinh Học |
|:---:|:---|:---|:---:|:---:|:---:|:---||
| 1 | `bbps-grade-2-3` | Bowel Preparation Quality | **1,149** | 10.79% | ~230 mẫu | Phân lớp độc lập đạt chuẩn |
| 2 | `cecum-and-ileum` | Lower GI Tract - Colorectum | **1,023** | 9.61% | ~205 mẫu | Phân lớp độc lập đạt chuẩn |
| 3 | `polyps` | Lower GI Tract - Colorectum | **1,023** | 9.61% | ~205 mẫu | Kèm 1,000 nhãn Bounding Box chuẩn Kvasir-SEG |
| 4 | `dyed-lifted-polyps` | Chromoendoscopy - Interventions | **999** | 9.39% | ~200 mẫu | Phân lớp độc lập đạt chuẩn |
| 5 | `pylorus` | Upper GI Tract | **998** | 9.38% | ~200 mẫu | Phân lớp độc lập đạt chuẩn |
| 6 | `dyed-resection-margins` | Chromoendoscopy - Interventions | **990** | 9.30% | ~198 mẫu | Phân lớp độc lập đạt chuẩn |
| 7 | `normal-z-line` | Upper GI Tract | **961** | 9.03% | ~192 mẫu | Phân lớp độc lập đạt chuẩn |
| 8 | `retroflex-stomach` | Upper GI Tract | **763** | 7.17% | ~153 mẫu | Phân lớp độc lập đạt chuẩn |
| 9 | `bbps-grade-0-1` | Bowel Preparation Quality | **643** | 6.04% | ~129 mẫu | Phân lớp độc lập đạt chuẩn |
| 10 | `mayo-grade-2` | Lower GI Tract - Inflammatory Bowel Disease | **449** | 4.22% | ~90 mẫu | Chuẩn hóa Mayo kinh điển |
| 11 | `retroflex-rectum` | Lower GI Tract - Colorectum | **395** | 3.71% | ~79 mẫu | Phân lớp độc lập đạt chuẩn |
| 12 | `esophagitis-grade-a` | Upper GI Tract | **376** | 3.53% | ~75 mẫu | Phân lớp độc lập đạt chuẩn |
| 13 | `esophagitis-grade-b-d` | Upper GI Tract | **261** | 2.45% | ~52 mẫu | Phân lớp độc lập đạt chuẩn |
| 14 | `mayo-grade-1` | Lower GI Tract - Inflammatory Bowel Disease | **231** | 2.17% | ~46 mẫu | Chuẩn hóa Mayo kinh điển |
| 15 | `mayo-grade-3` | Lower GI Tract - Inflammatory Bowel Disease | **161** | 1.51% | ~32 mẫu | Chuẩn hóa Mayo kinh điển |
| 16 | `impacted-stool` | Bowel Preparation Quality | **131** | 1.23% | ~26 mẫu | Phân lớp độc lập đạt chuẩn |
| 17 | `barretts-esophagus` | Upper GI Tract | **91** | 0.85% | ~18 mẫu | Hợp nhất 2 mức chuyển sản ruột |

---

## 3. Kết Luận Quyết Định

Schema 17 phân lớp mục tiêu là sự cân bằng hoàn hảo giữa **Độ nghiêm ngặt về mặt toán học** (loại bỏ lớp không đủ mẫu, giảm mất cân bằng cực đoan) và **Tính thực tiễn lâm sàng của bác sĩ nội soi tiêu hóa** (chẩn đoán polyp, viêm loét đại tràng 3 mức độ, trào ngược thực quản 2 mức độ, barrett thực quản, và đánh giá độ sạch ruột BBPS).
