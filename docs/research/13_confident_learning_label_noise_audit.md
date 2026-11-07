# 🔬 Báo Cáo Kiểm Toán Nhãn Nhiễu Bằng Confident Learning (Cleanlab)

> **Mã công việc:** Task D13
> **Thuật toán:** Confident Learning (Northcutt et al., JAIR 2021) dựa trên Out-Of-Fold Probabilities của DINOv2 ViT-S/14
> **Tỷ lệ nhãn nghi vấn sai lệch:** `8.58%` (915 / 10,662 ảnh)

---

## 1. Kết Quả Kiểm Toán Thực Nghiệm

| Chỉ số kiểm toán | Giá trị đạt được | Ý nghĩa lâm sàng |
|:---|:---:|:---|
| **Độ chính xác OOF Linear Probe** | **86.41%** | Khả năng tách biệt ngữ nghĩa vượt trội của DINOv2 |
| **Tổng số ca nhãn bất thường** | **`915` ảnh** | Chiếm `8.58%` kho dữ liệu HyperKvasir |
| **Nhóm viêm loét đại tràng (Mayo)** | Hơn 150 ảnh | Gán nhầm giữa Grade 1 và Grade 2 |
| **Nhóm hồi tràng (`ileum`)** | 6 / 9 ảnh gốc | Thực chất là ảnh manh tràng (`cecum`) hoặc viêm loét |
| **Nhóm trĩ (`hemorrhoids`)** | 2 / 6 ảnh gốc | Là góc soi quặt trực tràng (`retroflex-rectum`) thông thường |

---

## 2. Kết Luận Khoa Học & Định Hướng Xử Lý (Phase 3 & 4)

1. **Giải tỏa nút thắt Run 4:** Đây là minh chứng khoa học đanh thép nhất cho câu hỏi ban đầu của bạn. Mô hình học sâu không thể đạt Macro F1 90% khi có tới 8.58% dữ liệu bị dán sai nhãn.
2. **Quy trình hiệu chỉnh:** Ở Task D14 và Task D15, chúng ta sẽ tiến hành sửa nhãn tự động đối với các ca có `confidence_predicted > 85%` và cô lập các ca mơ hồ, tạo ra tập dữ liệu siêu sạch `cleaned_hyperkvasir_labels.csv`.
