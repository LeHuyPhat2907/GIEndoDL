"""Script phân tích cấu trúc thư mục phân cấp và xuất báo cáo 23 lớp HyperKvasir."""

import os
from pathlib import Path
import pandas as pd


def analyze_hyperkvasir_hierarchy(raw_data_dir: str, output_doc_path: str):
    raw_path = Path(raw_data_dir)
    labeled_dir = raw_path / "labeled-images"

    print("=" * 75)
    print("PHÂN TÍCH CẤU TRÚC PHÂN CẤP THƯ MỤC BỘ DỮ LIỆU HYPERKVASIR")
    print("=" * 75)

    if not labeled_dir.exists():
        print(f"Không tìm thấy thư mục: {labeled_dir}")
        return

    records = []
    image_exts = {".jpg", ".jpeg", ".png"}

    # Duyệt qua các tầng thư mục
    for tract_dir in sorted(labeled_dir.iterdir()):
        if not tract_dir.is_dir():
            continue

        for cat_dir in sorted(tract_dir.iterdir()):
            if not cat_dir.is_dir():
                continue

            for class_dir in sorted(cat_dir.iterdir()):
                if not class_dir.is_dir():
                    continue

                # Kiểm tra nếu có phân lớp sâu hơn (như ulcerative-colitis, esophagitis)
                sub_classes = [d for d in class_dir.iterdir() if d.is_dir()]
                if sub_classes:
                    for sub_dir in sorted(sub_classes):
                        imgs = [
                            f
                            for f in sub_dir.glob("*")
                            if f.suffix.lower() in image_exts
                        ]
                        records.append(
                            {
                                "Tract": tract_dir.name,
                                "Category": cat_dir.name,
                                "Parent_Class": class_dir.name,
                                "Class_Name": sub_dir.name,
                                "Path": str(sub_dir.relative_to(labeled_dir)).replace(
                                    "\\", "/"
                                ),
                                "Count": len(imgs),
                            }
                        )
                else:
                    imgs = [
                        f for f in class_dir.glob("*") if f.suffix.lower() in image_exts
                    ]
                    records.append(
                        {
                            "Tract": tract_dir.name,
                            "Category": cat_dir.name,
                            "Parent_Class": "-",
                            "Class_Name": class_dir.name,
                            "Path": str(class_dir.relative_to(labeled_dir)).replace(
                                "\\", "/"
                            ),
                            "Count": len(imgs),
                        }
                    )

    df = pd.DataFrame(records)
    total_imgs = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total_imgs) * 100

    print(f"Tổng số lớp bệnh lý/mốc giải phẫu: {len(df)} lớp")
    print(f"Tổng số lượng ảnh: {total_imgs:,} ảnh\n")

    # In cây phân cấp ra console
    current_tract = ""
    current_cat = ""
    for _, row in df.iterrows():
        if row["Tract"] != current_tract:
            current_tract = row["Tract"]
            print(f"\n[{current_tract.upper()}]")
        if row["Category"] != current_cat:
            current_cat = row["Category"]
            print(f"  └── {current_cat}")
        print(
            f"       ├── {row['Class_Name']:<30} : {row['Count']:>5,} ảnh ({row['Percentage']:>5.2f}%)"
        )

    # Xuất tài liệu Markdown
    doc_file = Path(output_doc_path)
    doc_file.parent.mkdir(parents=True, exist_ok=True)

    with open(doc_file, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Cấu trúc Phân cấp Thư mục Bộ dữ liệu HyperKvasir\n\n")
        f.write(
            f"> **Tổng số ảnh:** {total_imgs:,} ảnh | **Tổng số lớp:** {len(df)} lớp  \n"
        )
        f.write(
            "> **Nguồn:** Simula Research Laboratory & Bærum Hospital (Norway)\n\n---\n\n"
        )

        f.write("## 1. Bảng Thống kê Phân cấp Chi tiết 23 Lớp\n\n")
        f.write(
            "| STT | Phân vùng (Tract) | Nhóm (Category) | Tên lớp (Class Name) | Số lượng ảnh | Tỷ lệ (%) |\n"
        )
        f.write("|:---:|:---|:---|:---|:---:|:---:|\n")
        for idx, row in df.iterrows():
            f.write(
                f"| {idx+1} | `{row['Tract']}` | `{row['Category']}` | **{row['Class_Name']}** | {row['Count']:,} | {row['Percentage']:.2f}% |\n"
            )

        f.write(
            f"\n| | **TỔNG CỘNG** | | | **{total_imgs:,}** | **100.00%** |\n\n---\n\n"
        )

        f.write("## 2. Sơ đồ Cây Phân cấp (Mermaid Hierarchy Diagram)\n\n")
        f.write("```mermaid\ngraph TD\n")
        f.write("    Root[HyperKvasir 10,662 ảnh] --> Upper[Upper GI Tract]\n")
        f.write("    Root --> Lower[Lower GI Tract]\n\n")
        f.write("    Upper --> U_Anat[Anatomical Landmarks]\n")
        f.write("    Upper --> U_Path[Pathological Findings]\n\n")
        f.write("    Lower --> L_Anat[Anatomical Landmarks]\n")
        f.write("    Lower --> L_Path[Pathological Findings]\n")
        f.write("    Lower --> L_Ther[Therapeutic Interventions]\n")
        f.write("    Lower --> L_Qual[Quality of Mucosal Views]\n")
        f.write("```\n")

    print("\n" + "=" * 75)
    print(f"Đã tự động xuất báo cáo cấu trúc tại: {doc_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_path = os.path.join(project_root, "data", "raw")
    doc_path = os.path.join(
        project_root, "docs", "research", "04_hyperkvasir_hierarchy.md"
    )

    analyze_hyperkvasir_hierarchy(raw_path, doc_path)
