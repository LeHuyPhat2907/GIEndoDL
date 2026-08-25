# 🔍 Báo cáo Kiểm định Trùng lặp & Ngăn chặn Rò rỉ Dữ liệu (Duplicate & Leakage Audit)

> **Số lượng ảnh khảo sát:** 10,662 ảnh | **Phương pháp:** Perceptual Hash (pHash), dHash & Structural Similarity (SSIM)

---

## 1. Kết quả Thực nghiệm Kiểm định

- **Tổng số mã băm pHash độc lập:** `10,327` mã.
- **Số lượng cặp ảnh gần trùng lặp (SSIM $\ge$ 0.90):** `172` cặp.
- **Cặp trùng lặp cùng lớp (Same-class near-duplicates):** `167` cặp.
- **Cặp trùng lặp mâu thuẫn nhãn (Cross-class conflicting):** `5` cặp.

---

## 2. Ý nghĩa Phương pháp luận Y học & Chuẩn TRIPOD-AI

1. **Bản chất của các cặp Near-duplicates:** Trong nội soi tiêu hóa, các cặp ảnh gần trùng lặp sinh ra do camera ghi hình liên tiếp nhiều khung hình ở cùng một vị trí tổn thương (Burst captures / Consecutive frames).
2. **Ngăn chặn Rò rỉ Dữ liệu Tuyệt đối (Zero Data Leakage Protocol):**
   - Khi phân chia tập dữ liệu ở Giai đoạn 4 (Task 51), các cặp ảnh có mã `pHash` trùng khớp **BẮT BUỘC PHẢI ĐƯỢC XẾP CHUNG VÀO CÙNG MỘT TẬP** (Toàn bộ vào Train hoặc toàn bộ vào Test).
   - Tuyệt đối không để 1 ảnh rơi vào Train và ảnh song sinh của nó rơi vào Test, đảm bảo tính khách quan 100% của bài báo khoa học.
