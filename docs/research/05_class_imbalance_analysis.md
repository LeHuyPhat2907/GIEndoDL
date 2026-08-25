# Báo cáo Phân tích Mất Cân Bằng Lớp & Phân phối Pareto (HyperKvasir)

> **Imbalance Ratio (IR):** **191.33 : 1** | **Tổng số lớp:** 23 lớp | **Tổng số ảnh:** 10,662 ảnh

---

## 1. Phân tầng Dữ liệu (3-Tier Classification)

| Phân tầng (Tier) | Số lớp | Tổng số ảnh | Tỷ trọng (%) | Danh sách các lớp |
|:---|:---:|:---:|:---:|:---|
| **Đa số (Majority >= 900)** | 7 | 7,107 | 66.66% | `bbps-2-3`, `polyps`, `cecum`, `dyed-lifted-polyps`, `pylorus`, `dyed-resection-margins`, `z-line` |
| **Trung bình (100 - 899)** | 9 | 3,372 | 31.63% | `retroflex-stomach`, `bbps-0-1`, `ulcerative-colitis-grade-2`, `esophagitis-a`, `retroflex-rectum`, `esophagitis-b-d`, `ulcerative-colitis-grade-1`, `ulcerative-colitis-grade-3`, `impacted-stool` |
| **Thiểu số / Cực hiếm (< 100)** | 7 | 183 | 1.72% | `barretts-short-segment`, `barretts`, `ulcerative-colitis-grade-0-1`, `ulcerative-colitis-grade-2-3`, `ulcerative-colitis-grade-1-2`, `ileum`, `hemorrhoids` |

---

## 2. Ý nghĩa Khoa học đối với Thiết kế Mô hình

1. **Hiện tượng Tail-Dominance:** 7 lớp đa số chiếm tới hơn 66% tổng dữ liệu, trong khi 7 lớp hiếm nhất chỉ chiếm dưới 2%. Nếu áp dụng chuẩn đo lường Accuracy thông thường, mô hình có thể đạt 90% Accuracy nhưng hoàn toàn bỏ sót các ca bệnh hiếm (như trĩ, hồi tràng, Barrett).
2. **Chỉ số Đánh giá Tiêu chuẩn:** Đề tài bắt buộc phải sử dụng **F1-Score (Macro)** và **Balanced Accuracy** làm chỉ số tối ưu chính thay cho Accuracy thông thường.
3. **Giải pháp Thuật toán:** Ứng dụng **Weighted Random Sampler** cân bằng xác suất lấy mẫu theo batch và **Class-Balanced Focal Loss** để phạt nặng các lỗi phân loại trên nhóm lớp thiểu số.
