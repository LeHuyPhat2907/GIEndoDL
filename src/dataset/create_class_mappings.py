"""Script xây dựng cấu trúc phân nhóm nhãn (23 lớp, 8 lớp lâm sàng, 4 nhóm lớn) và tạo metadata CSV."""

import json
import os
from pathlib import Path
import pandas as pd

# 1. Định nghĩa quy tắc ánh xạ 8 lớp lâm sàng (Gộp các sub-grades)
MAP_23_TO_8 = {
    # Nhóm polyp
    "polyps": "polyps",
    "dyed-lifted-polyps": "dyed-lifted-polyps",
    "dyed-resection-margins": "dyed-resection-margins",
    # Nhóm viêm loét đại tràng (Gộp 6 sub-grades)
    "ulcerative-colitis-grade-0-1": "ulcerative-colitis",
    "ulcerative-colitis-grade-1": "ulcerative-colitis",
    "ulcerative-colitis-grade-1-2": "ulcerative-colitis",
    "ulcerative-colitis-grade-2": "ulcerative-colitis",
    "ulcerative-colitis-grade-2-3": "ulcerative-colitis",
    "ulcerative-colitis-grade-3": "ulcerative-colitis",
    # Nhóm thực quản (Gộp LA grades & Barretts)
    "esophagitis-a": "esophagitis",
    "esophagitis-b-d": "esophagitis",
    "barretts": "barretts",
    "barretts-short-segment": "barretts",
    # Nhóm mốc giải phẫu
    "cecum": "normal-cecum",
    "pylorus": "normal-pylorus",
    "z-line": "normal-z-line",
    "retroflex-rectum": "other-landmarks",
    "retroflex-stomach": "other-landmarks",
    "ileum": "other-landmarks",
    # Nhóm chất lượng / khác
    "bbps-0-1": "quality-views",
    "bbps-2-3": "quality-views",
    "impacted-stool": "quality-views",
    "hemorrhoids": "hemorrhoids",
}


def build_taxonomy_and_metadata(
    raw_dir: str, config_dir: str, processed_dir: str, doc_dir: str
):
    raw_path = Path(raw_dir) / "labeled-images"
    cfg_path = Path(config_dir)
    proc_path = Path(processed_dir)
    doc_path = Path(doc_dir)

    cfg_path.mkdir(parents=True, exist_ok=True)
    proc_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"❌ Không tìm thấy thư mục: {raw_path}")
        return

    print("=" * 75)
    print("🏷️ ĐANG XÂY DỰNG BẢNG ÁNH XẠ NHÃN ĐA TẦNG & TẠO METADATA CSV...")
    print("=" * 75)

    image_exts = {".jpg", ".jpeg", ".png"}
    records = []

    # Duyệt toàn bộ 10,662 ảnh
    for img_file in sorted(raw_path.rglob("*")):
        if img_file.is_file() and img_file.suffix.lower() in image_exts:
            class_23 = img_file.parent.name
            category = img_file.parent.parent.name
            tract = img_file.parent.parent.parent.name

            # Phân loại 4 nhóm lớn (Super Category)
            if "therapeutic" in category:
                super_cat = "Therapeutic_Interventions"
            elif "quality" in category:
                super_cat = "Quality_of_Mucosal_Views"
            elif "pathological" in category:
                super_cat = "Pathological_Findings"
            else:
                super_cat = "Anatomical_Landmarks"

            class_8 = MAP_23_TO_8.get(class_23, "other")

            records.append(
                {
                    "filename": img_file.name,
                    "rel_path": str(img_file.relative_to(raw_path)).replace("\\", "/"),
                    "tract": tract,
                    "category": category,
                    "super_category": super_cat,
                    "label_23": class_23,
                    "label_8": class_8,
                }
            )

    df = pd.DataFrame(records)
    total_imgs = len(df)

    # 1. Tạo từ điển mã hóa nhãn (Label to ID mappings)
    # Mapping 23 lớp
    classes_23 = sorted(df["label_23"].unique())
    class_to_idx_23 = {cls: idx for idx, cls in enumerate(classes_23)}
    idx_to_class_23 = {idx: cls for idx, cls in enumerate(classes_23)}

    with open(cfg_path / "classes_23.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_classes": len(classes_23),
                "class_to_idx": class_to_idx_23,
                "idx_to_class": idx_to_class_23,
            },
            f,
            indent=4,
        )

    # Mapping 4 nhóm lớn
    super_cats = sorted(df["super_category"].unique())
    super_to_idx = {cat: idx for idx, cat in enumerate(super_cats)}

    with open(cfg_path / "super_categories_4.json", "w", encoding="utf-8") as f:
        json.dump(
            {"num_classes": len(super_cats), "class_to_idx": super_to_idx},
            f,
            indent=4,
        )

    # Gán ID số nguyên vào DataFrame
    df["label_23_id"] = df["label_23"].map(class_to_idx_23)
    df["super_category_id"] = df["super_category"].map(super_to_idx)

    # 2. Lưu file Metadata CSV chính thức của dự án
    csv_file = proc_path / "hyperkvasir_metadata.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8")

    print(f"🖼️ Tổng số ảnh đã gắn nhãn đa tầng: {total_imgs:,} ảnh")
    print(f"📁 Đã tạo cấu hình 23 lớp tại:       {cfg_path / 'classes_23.json'}")
    print(
        f"📁 Đã tạo cấu hình 4 nhóm lớn tại:    {cfg_path / 'super_categories_4.json'}"
    )
    print(f"📊 Đã tạo file Master Metadata tại:   {csv_file}")
    print("=" * 75)

    # 3. Xuất tài liệu Markdown
    md_file = doc_path / "11_class_grouping_and_taxonomy.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# 🏷️ Báo cáo Phân nhóm Nhãn Đa tầng & Chiến lược Phân loại (Label Taxonomy)\n\n"
        )
        f.write(
            "> **Master Dataset:** HyperKvasir (10,662 ảnh) | **Metadata File:** `data/processed/hyperkvasir_metadata.csv`\n\n---\n\n"
        )

        f.write("## 1. Ba Chiến lược Phân loại Đa tầng của Đề tài\n\n")
        f.write(
            "| Cấp độ phân loại | Số lượng lớp | Mục tiêu ứng dụng trong Khóa luận & Bài báo |\n"
        )
        f.write("|:---|:---:|:---|\n")
        f.write(
            "| **Tầng 1 (Super-Category)** | **4 Nhóm** | Phân loại cấp cao: *Mốc giải phẫu vs Bệnh lý vs Can thiệp vs Chất lượng view*. |\n"
        )
        f.write(
            "| **Tầng 2 (Clinical Aggregated)** | **8 Lớp** | Đối sánh công bằng (*Fair Benchmark*) trực tiếp với các mô hình trên Kvasir-v2. |\n"
        )
        f.write(
            "| **Tầng 3 (Fine-Grained 23-Class)** | **23 Lớp** | **Nhiệm vụ trọng tâm của Đề tài:** Đánh giá năng lực của kiến trúc đề xuất *CNN-CBAM-Transformer + SupCon* trên bài toán phân loại siêu chi tiết. |\n\n---\n\n"
        )

        f.write("## 2. Bảng Ánh xạ Nhãn Chi tiết (23 Lớp ➔ 4 Nhóm Lớn)\n\n")
        f.write(
            "| ID (23) | Tên lớp 23 (Fine-grained) | Nhóm lớn (Super Category) | Số lượng ảnh | Tỷ trọng (%) |\n"
        )
        f.write("|:---:|:---|:---|:---:|:---:|\n")

        for idx, cls in enumerate(classes_23):
            sub_df = df[df["label_23"] == cls]
            count = len(sub_df)
            pct = (count / total_imgs) * 100
            s_cat = sub_df.iloc[0]["super_category"]
            f.write(f"| `{idx}` | **{cls}** | `{s_cat}` | {count:,} | {pct:.2f}% |\n")

    print(f"✅ Đã lưu tài liệu phân nhóm tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_path = os.path.join(project_root, "data", "raw")
    config_path = os.path.join(project_root, "configs", "class_mappings")
    processed_path = os.path.join(project_root, "data", "processed")
    doc_path = os.path.join(project_root, "docs", "research")

    build_taxonomy_and_metadata(raw_path, config_path, processed_path, doc_path)
