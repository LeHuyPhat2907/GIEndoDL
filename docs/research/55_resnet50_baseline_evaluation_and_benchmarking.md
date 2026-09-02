# ⚓ Báo cáo Kỹ thuật: Thiết Lập Hồ Sơ Đánh Giá Mốc Cơ Sở ResNet-50 Baseline

> **File cấu hình:** `configs/resnet50_baseline_report.json` | **Bảng dữ liệu:** `data/processed/resnet50_per_class_baseline.csv` | **Hình minh họa:** `docs/figures/48_resnet50_baseline_benchmark_report.png`

---

## 1. Hồ Sơ Kỹ Thuật Tổng Quan Của Mốc Đối Chứng

| Chỉ số (Metric) | Giá trị ResNet-50 | Nhận xét chuyên môn y tế |
|:---|:---:|:---|
| **Overall Accuracy** | `90.25%` | Độ chính xác tổng thể ổn định |
| **Macro F1-Score** | `88.54%` | **Mốc cơ sở then chốt để đánh giá các cải tiến sau này** |
| **Multi-class OvR AUC** | `97.42%` | Năng lực phân loại ngưỡng tốt |
| **Tham số (Parameters)** | `23.51 Triệu` | Kích thước mô hình tiêu chuẩn ngành |
| **Tốc độ Inference (Tesla T4)** | `178.5 FPS` | Tốc độ đáp ứng thời gian thực cho phòng mổ (>= 30 FPS) |
| **Bộ nhớ VRAM (AMP FP16)** | `2.59 GB` | Vận hành cực êm ái trên Google Colab Free (15.3 GB) |

---

## 2. Luận Cứ Khoa Học Cho Sự Cần Thiết Của Kiến Trúc Đề Xuất

Mặc dù ResNet-50 đạt Macro F1 88.54%, phân tích chi tiết phổ 23 lớp (Panel 3) cho thấy các tổn thương vi thể ranh giới mờ (như Barretts thực quản hay Viêm loét đại tràng phân độ 1-2) chỉ đạt F1 dưới 87%. Hạn chế này xuất phát từ bản chất của tích chập truyền thống không có cơ chế lọc lọc trọng số theo không gian. Đây chính là động lực khoa học chặt chẽ để đề tài tiến hành tích hợp khối chú ý CBAM và khảo sát kiến trúc Transformer trong các giai đoạn tiếp theo.
