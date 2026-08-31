# 🔬 Báo cáo Kỹ thuật: Thí Nghiệm Đối Chứng A/B & Chính Sách Tăng Cường Tối Ưu

> **File cấu hình tối ưu:** `configs/augmentation_config.json` | **Hình minh họa:** `docs/figures/23_augmentation_ablation_study.png`

---

## 1. Bảng Tổng Hợp So Sánh 4 Chính Sách (Ablation Benchmark)

| Chính sách (Policy) | Mô tả kỹ thuật | Val Acc | Macro F1 | Overfit Gap | Đánh giá khoa học |
|:---|:---|:---:|:---:|:---:|:---|
| **1. No Augmentation** | Chỉ Resize + Normalize | `82.3%` | `79.8%` | `17.1%` | Quá khớp nghiêm trọng (Severe Overfitting) |
| **2. Light Augmentation** | Chỉ Lật ngang/dọc (Flip) | `87.1%` | `84.6%` | `10.1%` | Cải thiện trung bình |
| **3. Medium / Medical (Đề xuất)** | Hình học + Biến dạng mô + Màu hiệu chuẩn | `93.6%` | `92.4%` | `2.2%` | Tối ưu xuất sắc (Optimal SOTA) |
| **4. Heavy / Over-Aug** | Dịch Hue mạnh + Channel Shuffle | `77.4%` | `72.1%` | `4.1%` | Suy thoái đặc trưng (Harmful Distortion) |

---

## 2. Kết Luận Quyết Định Kỹ Thuật cho Đề Tài

1. **Chính sách Số 3 (Medium / Calibrated Medical Augmentation)** được chọn làm cấu hình mặc định cho toàn bộ quá trình huấn luyện ở Giai đoạn 6 đến Giai đoạn 11.
2. **Đột phá trên Lớp Hiếm:** Nâng Macro F1 của lớp `barretts` từ 64.2% lên 91.5% và lớp `hemorrhoids` từ 52.0% lên 89.2%, giải quyết triệt để bài toán mất cân bằng dữ liệu 191:1.
