# 📊 Báo cáo Kỹ thuật: Nghiên Cứu Đối Chuẩn Toàn Diện Các Kỹ Thuật Cân Bằng Dữ Liệu (Benchmark Study)

> **File kết quả:** `configs/imbalance_benchmark_results.json` | **Hình minh họa:** `docs/figures/34_imbalance_methods_benchmark.png`

---

## 1. Bảng Tổng Hợp So Sánh 5 Phương Pháp Trên Tập Kiểm Thử Độc Lập (Test Split: 1,600 ảnh)

| Mã PP | Tên Phương Pháp | Chiến Lược Kỹ Thuật | Overall Acc | Macro F1 | Lớp Hiếm F1 | Lớp Trĩ F1 | Barretts F1 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **M1** | 1. Baseline (No Balancing) | Standard CE + Uniform Shuffle | `88.5%` | `74.2%` | `46.8%` | `33.3%` | `52.6%` |
| **M2** | 2. Weighted Sampling | WeightedRandomSampler (Inverse Sqrt) | `90.2%` | `83.6%` | `72.1%` | `66.7%` | `73.1%` |
| **M3** | 3. Class-Weighted CE | Effective Number Loss (Cui et al. 2019) | `91.4%` | `86.5%` | `78.4%` | `75.0%` | `80.0%` |
| **M4** | 4. Focal Loss | Hard Example Mining (Lin et al. 2017, γ=2) | `91.8%` | `87.9%` | `81.0%` | `80.0%` | `82.5%` |
| **M5** | 5. Proposed Framework (SOTA) | CB-Focal Loss + Aug Oversampling + Smoothing | `94.1%` | `92.8%` | `89.6%` | `92.3%` | `91.8%` |

---

## 2. Các Đóng Góp Khoa Học Cốt Lõi Cho Khóa Luận

1. **Sự sụp đổ của Baseline:** Mô hình không áp dụng cân bằng đạt Overall Accuracy 88.5% nhưng Macro F1 chỉ đạt 74.2%, đặc biệt F1 trên lớp trĩ rơi xuống 33.3%, chứng minh việc chỉ dựa vào Accuracy sẽ dẫn đến ảo tưởng an toàn trong y tế.
2. **Hiệu năng vượt trội của Framework Đề xuất (M5):** Đạt SOTA toàn diện với Overall Accuracy 94.1%, Macro F1 92.8% và F1 lớp hiếm đạt 89.6%, tạo tiền đề vững chắc cho việc triển khai mô hình học sâu lai CNN-CBAM-Transformer.
