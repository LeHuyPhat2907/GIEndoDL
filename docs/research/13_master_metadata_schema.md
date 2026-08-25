# Cấu trúc Dữ liệu Master Metadata CSV (Ground-Truth Schema)

> **File Master:** `data/processed/hyperkvasir_master_metadata.csv`
> **Quy mô:** 10,662 dòng $\times$ 18 cột thuộc tính chuẩn hóa

---

## 1. Từ điển Dữ liệu (Data Dictionary)

| Tên cột (Column) | Kiểu dữ liệu | Ý nghĩa Y học & Kỹ thuật | Ví dụ mẫu |
|:---|:---:|:---|:---|
| `image_id` | `string` | Mã định danh duy nhất của từng ảnh | `IMG_00001` |
| `filename` | `string` | Tên tệp tin gốc từ Simula Lab | `345a7172...jpg` |
| `relative_path` | `string` | Đường dẫn tương đối từ `data/raw/labeled-images` | `lower-gi-tract/...` |
| `tract` | `string` | Phân vùng đường tiêu hóa | `lower-gi-tract` / `upper-gi-tract` |
| `super_category` | `string` | Nhóm chức năng (4 nhóm) | `Pathological_Findings` |
| `super_category_id` | `int` | Mã số nhóm chức năng (0 - 3) | `1` |
| `class_name` | `string` | Tên lớp bệnh lý chi tiết (23 lớp) | `polyps` |
| `class_id` | `int` | Mã số lớp bệnh lý (0 - 22) | `12` |
| `width` / `height` | `int` | Độ phân giải thực tế của ảnh (px) | `633` / `532` |
| `aspect_ratio` | `float` | Tỷ lệ khung hình ($W / H$) | `1.19` |
| `file_size_kb` | `float` | Dung lượng file trên đĩa (KB) | `85.4` |
| `laplacian_var` | `float` | Độ sắc nét toán tử Laplacian | `542.1` |
| `mean_brightness` | `float` | Độ sáng pixel trung bình (0 - 255) | `112.5` |
| `specular_ratio` | `float` | Tỷ lệ diện tích đốm lóa sáng (%) | `1.42` |
| `quality_tag` | `string` | Thẻ phân loại chất lượng ảnh | `Good_Quality` |
| `phash` | `string` | Mã băm cảm nhận thị giác (Perceptual Hash) | `d2a4e1...` |

---

## 2. Ý nghĩa Phương pháp luận

1. **Tối ưu hóa Tốc độ Huấn luyện:** Giúp PyTorch `DataLoader` truy xuất mọi thuộc tính chỉ trong $O(1)$ mà không cần đọc lại header ảnh nhiều lần.
2. **Hỗ trợ Phân chia Tập dữ liệu Không Rò rỉ (Zero-Leakage Split):** Cột `phash` và `class_id` là chìa khóa để thực hiện thuật toán **Grouped Stratified Split** ở Task 51.
