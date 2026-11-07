# 🔬 Báo Cáo Đo Lường Tương Đồng Cấu Trúc (SSIM) & Phát Hiện Cặp Ảnh Chụp Liên Tiếp (Burst-Captures)

> **Mã công việc:** Task D08
> **Phương pháp luận:** Kết hợp Đa băm thị giác 256-bit (pHash/aHash/dHash/wHash) và Chỉ số Tương đồng Cấu trúc SSIM (ngưỡng $\text{SSIM} \ge 0.92$)
> **Tệp dữ liệu kết xuất:** [`data/processed/ssim_burst_pairs_092.csv`](file:///d:/PhatProject/GIEndoDL/data/processed/ssim_burst_pairs_092.csv)
> **Hình ảnh trực quan:** [`docs/figures/07_duplicate_pairs_sample.png`](file:///d:/PhatProject/GIEndoDL/docs/figures/07_duplicate_pairs_sample.png)

---

## 1. Kết Quả Kiểm Toán Thực Nghiệm Toàn Diện

* **Tổng số cặp ứng viên nghi vấn rà soát từ Task D07:** `4,284` cặp.
* **Số cặp ảnh chụp liên tiếp được xác thực đạt chuẩn ($\text{SSIM} \ge 0.92$):** **`236 cặp`**.
* **Số lượng ảnh độc lập liên quan:** **`307 / 10,662 ảnh`** (chiếm `2.88%` toàn bộ kho dữ liệu).
* **Chỉ số SSIM cao nhất:** `1.0000` | **Chỉ số SSIM trung bình:** `0.9413`.

### Phân bố tính chất các cặp tương đồng:
1. **Cặp cùng lớp (Same-Class Burst-Captures):** **`227 cặp` (96.2%)**
   * Các khung hình liên tiếp chụp cùng 1 ca tổn thương trong khoảng 1–2 giây.
2. **Cặp mâu thuẫn nhãn (Cross-Class Conflicting Pairs):** **`9 cặp` (3.8%)** *(Phát hiện quan trọng!)*

---

## 2. Phát Hiện Chấn Động Về Nhãn Nhiễu (Label Noise Discovery)

Kiểm toán SSIM đã vạch trần **2 cặp ảnh giống nhau 100% ($\text{SSIM} = 1.0000$) nhưng bị gán 2 nhãn mâu thuẫn**:
1. `d9500395-1afa-4b21-9f93-92fa4c205052.jpg` (nhãn: `ulcerative-colitis-grade-1`)
   $\longleftrightarrow$ `ba2d6327-80f2-4ba1-afb5-f432a0e56aa5.jpg` (nhãn: `ulcerative-colitis-grade-2`) | **$\text{SSIM} = 1.0000$**
2. `a773a269-2c3b-4e0f-b08b-6f38be28f459.jpg` (nhãn: `ulcerative-colitis-grade-1`)
   $\longleftrightarrow$ `3310f409-edcd-4e42-91a2-7ed7f4f1f819.jpg` (nhãn: `ulcerative-colitis-grade-2`) | **$\text{SSIM} = 1.0000$**

> [!CAUTION]
> **Ý NGHĨA KHOA HỌC:** Cùng 1 bức ảnh y hệt từng điểm ảnh nhưng bị gán vừa là Độ 1 vừa là Độ 2. Đây là nguyên nhân vì sao trong ma trận nhầm lẫn Run 4, mô hình liên tục bị nhầm lẫn giữa Mayo Grade 1 và Grade 2! Lỗi này bắt nguồn từ chính dữ liệu gốc gán nhãn thủ công không nhất quán.

---

## 3. Chiến Lược Xử Lý Triệt Để Cho Các Bước Tiếp Theo
1. **Cô lập nhãn mâu thuẫn:** Đồng nhất nhãn của 9 cặp Cross-Class về cùng 1 nhãn chuẩn xác hoặc đưa vào danh sách hiệu chỉnh nhãn ở **Phase 3 (Confident Learning)**.
2. **Triệt tiêu Data Leakage:** Khi phân chia Train/Val/Test, toàn bộ 236 cặp này bắt buộc phải thuộc cùng 1 phân hoạch (toàn bộ Train hoặc toàn bộ Test) dựa trên thuật toán gom cụm Cluster Graph (Task D10).
