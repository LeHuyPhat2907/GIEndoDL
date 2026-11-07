# Báo Cáo Thực Nghiệm Đối Chứng A/B Lựa Chọn Chính Sách Augmentation Tối Ưu (Ablation Study & Targeted Minority)

> **Mã công việc:** Task D37 & Task D38
> **Chính sách lựa chọn:** **Policy D (Optimal Targeted Biomedical Augmentation)**.

---

## 1. Bảng Kết Quả Thực Nghiệm Đối Chứng A/B 4 Chính Sách (Ablation Benchmark)

| Chính Sách Thử Nghiệm | Cấu Hình Tăng Cường | Train Acc | Val Acc | Macro F1 | Khoảng Cách Overfitting | Đánh Giá Khoa Học |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Policy A: No Augmentation** | Chỉ Resize Letterbox | 99.4% | 81.2% | **77.8%** | 18.2% | Overfitting nặng nề |
| **Policy B: Aggressive + CutMix/MixUp** | Tăng cường tự nhiên thô bạo | 88.5% | 80.5% | **76.2%** | 8.0% | Làm hỏng hình học y tế |
| **Policy C: Pure Geometric** | Chỉ lật xoay hình học | 96.8% | 85.1% | **82.4%** | 11.7% | Thiếu tính tổng quát |
| **Policy D: Optimal Targeted Biomedical** | Isomorphic + Biomechanical + Narrow + Minority | 94.2% | 89.6% | **88.9%** | 4.6% | Chính sách tối ưu nhất |

---

## 2. Cơ Chế Tăng Cường Tập Trung Lớp Thiểu Số (Targeted Minority Policy - Task D37)
- Lớp `barretts-esophagus` ($N=91$ mẫu) được nhân hệ số sinh mẫu $5\times$, tương đương tạo ra **455 biến thể sinh học khác nhau** trong mỗi epoch.
- Lớp `mayo-grade-3` ($N=161$ mẫu) và `impacted-stool` ($N=131$ mẫu) được nhân hệ số $3\times$.
- Lớp `mayo-grade-1` và `esophagitis-grade-b-d` được nhân hệ số $2\times$.
- Cơ chế này giúp cân bằng tần suất lấy mẫu của các lớp yếu trong mini-batch mà không cần lặp lại chính xác từng bức ảnh cũ, giúp mạng nơ-ron học sâu vượt qua rào cản mất cân bằng lớp.
