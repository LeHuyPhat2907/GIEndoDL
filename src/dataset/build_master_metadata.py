"""Script tổng hợp Master Metadata CSV chứa đầy đủ đặc trưng nhãn, kích thước, chất lượng và mã pHash."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image


def process_image_master_record(args):
    """Trích xuất toàn bộ thông tin siêu dữ liệu của 1 ảnh."""
    img_path, raw_root, class_to_idx, super_to_idx = args
    try:
        file_size_kb = round(img_path.stat().st_size / 1024, 2)
        class_name = img_path.parent.name
        category = img_path.parent.parent.name
        tract = img_path.parent.parent.parent.name

        # 1. Nhóm Super Category
        if "therapeutic" in category:
            super_cat = "Therapeutic_Interventions"
        elif "quality" in category:
            super_cat = "Quality_of_Mucosal_Views"
        elif "pathological" in category:
            super_cat = "Pathological_Findings"
        else:
            super_cat = "Anatomical_Landmarks"

        class_id = class_to_idx.get(class_name, -1)
        super_cat_id = super_to_idx.get(super_cat, -1)

        # 2. Đọc ảnh lấy W, H, pHash và các chỉ số chất lượng
        with Image.open(img_path) as pil_img:
            w, h = pil_img.size
            phash_str = str(imagehash.phash(pil_img))

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is not None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            laplacian_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
            mean_brightness = round(float(np.mean(gray)), 2)
            specular_mask = gray > 245
            specular_ratio = round(
                float((np.sum(specular_mask) / (gray.shape[0] * gray.shape[1])) * 100),
                3,
            )
        else:
            laplacian_var, mean_brightness, specular_ratio = 0.0, 0.0, 0.0

        # Phân loại chất lượng
        if laplacian_var < 80:
            quality_tag = "Motion_Blur"
        elif mean_brightness < 45:
            quality_tag = "Under_Exposed"
        elif specular_ratio > 4.0:
            quality_tag = "High_Reflection"
        else:
            quality_tag = "Good_Quality"

        rel_path = str(img_path.relative_to(raw_root)).replace("\\", "/")

        return {
            "filename": img_path.name,
            "relative_path": rel_path,
            "tract": tract,
            "category": category,
            "super_category": super_cat,
            "super_category_id": super_cat_id,
            "class_name": class_name,
            "class_id": class_id,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3),
            "file_size_kb": file_size_kb,
            "laplacian_var": laplacian_var,
            "mean_brightness": mean_brightness,
            "specular_ratio": specular_ratio,
            "quality_tag": quality_tag,
            "phash": phash_str,
        }
    except Exception:
        return None


def generate_master_metadata(
    raw_dir: str, config_dir: str, proc_dir: str, doc_dir: str
):
    raw_path = Path(raw_dir) / "labeled-images"
    cfg_path = Path(config_dir)
    proc_path = Path(proc_dir)
    doc_path = Path(doc_dir)

    proc_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"Không tìm thấy thư mục: {raw_path}")
        return

    # Đọc cấu hình ánh xạ ID
    with open(cfg_path / "classes_23.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)["class_to_idx"]

    with open(cfg_path / "super_categories_4.json", "r", encoding="utf-8") as f:
        super_to_idx = json.load(f)["class_to_idx"]

    print("=" * 75)
    print("ĐANG TỔNG HỢP MASTER METADATA CSV CHO 10,662 ẢNH...")
    print("=" * 75)

    image_exts = {".jpg", ".jpeg", ".png"}
    all_imgs = [f for f in raw_path.rglob("*") if f.suffix.lower() in image_exts]

    tasks = [(p, raw_path, class_to_idx, super_to_idx) for p in sorted(all_imgs)]

    records = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = executor.map(process_image_master_record, tasks)
        for res in results:
            if res:
                records.append(res)

    df = pd.DataFrame(records)
    total_imgs = len(df)

    # Đánh số thứ tự ID mẫu
    df.insert(0, "image_id", [f"IMG_{i+1:05d}" for i in range(total_imgs)])

    # Xuất file Master CSV chính thức
    master_csv_path = proc_path / "hyperkvasir_master_metadata.csv"
    df.to_csv(master_csv_path, index=False, encoding="utf-8")

    print(f"Tổng số mẫu đã hợp nhất:        {total_imgs:,} ảnh")
    print(f"Số lượng cột thuộc tính (Columns): {len(df.columns)} cột")
    print(f"ĐÃ TẠO FILE MASTER METADATA TẠI:   {master_csv_path}")
    print("=" * 75)

    # Xuất tài liệu mô tả cấu trúc Schema
    md_file = doc_path / "13_master_metadata_schema.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Cấu trúc Dữ liệu Master Metadata CSV (Ground-Truth Schema)\n\n")
        f.write(
            "> **File Master:** `data/processed/hyperkvasir_master_metadata.csv`  \n"
        )
        f.write(
            f"> **Quy mô:** {total_imgs:,} dòng $\\times$ {len(df.columns)} cột thuộc tính chuẩn hóa\n\n---\n\n"
        )

        f.write("## 1. Từ điển Dữ liệu (Data Dictionary)\n\n")
        f.write(
            "| Tên cột (Column) | Kiểu dữ liệu | Ý nghĩa Y học & Kỹ thuật | Ví dụ mẫu |\n"
        )
        f.write("|:---|:---:|:---|:---|\n")
        f.write(
            "| `image_id` | `string` | Mã định danh duy nhất của từng ảnh | `IMG_00001` |\n"
        )
        f.write(
            "| `filename` | `string` | Tên tệp tin gốc từ Simula Lab | `345a7172...jpg` |\n"
        )
        f.write(
            "| `relative_path` | `string` | Đường dẫn tương đối từ `data/raw/labeled-images` | `lower-gi-tract/...` |\n"
        )
        f.write(
            "| `tract` | `string` | Phân vùng đường tiêu hóa | `lower-gi-tract` / `upper-gi-tract` |\n"
        )
        f.write(
            "| `super_category` | `string` | Nhóm chức năng (4 nhóm) | `Pathological_Findings` |\n"
        )
        f.write(
            "| `super_category_id` | `int` | Mã số nhóm chức năng (0 - 3) | `1` |\n"
        )
        f.write(
            "| `class_name` | `string` | Tên lớp bệnh lý chi tiết (23 lớp) | `polyps` |\n"
        )
        f.write("| `class_id` | `int` | Mã số lớp bệnh lý (0 - 22) | `12` |\n")
        f.write(
            "| `width` / `height` | `int` | Độ phân giải thực tế của ảnh (px) | `633` / `532` |\n"
        )
        f.write("| `aspect_ratio` | `float` | Tỷ lệ khung hình ($W / H$) | `1.19` |\n")
        f.write(
            "| `file_size_kb` | `float` | Dung lượng file trên đĩa (KB) | `85.4` |\n"
        )
        f.write(
            "| `laplacian_var` | `float` | Độ sắc nét toán tử Laplacian | `542.1` |\n"
        )
        f.write(
            "| `mean_brightness` | `float` | Độ sáng pixel trung bình (0 - 255) | `112.5` |\n"
        )
        f.write(
            "| `specular_ratio` | `float` | Tỷ lệ diện tích đốm lóa sáng (%) | `1.42` |\n"
        )
        f.write(
            "| `quality_tag` | `string` | Thẻ phân loại chất lượng ảnh | `Good_Quality` |\n"
        )
        f.write(
            "| `phash` | `string` | Mã băm cảm nhận thị giác (Perceptual Hash) | `d2a4e1...` |\n"
        )

        f.write("\n---\n\n## 2. Ý nghĩa Phương pháp luận\n\n")
        f.write(
            "1. **Tối ưu hóa Tốc độ Huấn luyện:** Giúp PyTorch `DataLoader` truy xuất mọi thuộc tính chỉ trong $O(1)$ mà không cần đọc lại header ảnh nhiều lần.\n"
        )
        f.write(
            "2. **Hỗ trợ Phân chia Tập dữ liệu Không Rò rỉ (Zero-Leakage Split):** Cột `phash` và `class_id` là chìa khóa để thực hiện thuật toán **Grouped Stratified Split** ở Task 51.\n"
        )

    print(f"Đã lưu tài liệu mô tả Schema tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_path = os.path.join(project_root, "data", "raw")
    config_path = os.path.join(project_root, "configs", "class_mappings")
    processed_path = os.path.join(project_root, "data", "processed")
    doc_path = os.path.join(project_root, "docs", "research")

    generate_master_metadata(raw_path, config_path, processed_path, doc_path)
