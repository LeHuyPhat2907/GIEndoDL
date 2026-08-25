"""Script tạo biểu đồ Grid ảnh mẫu đại diện 23 lớp chuẩn xuất bản phẩm (Publication Quality)."""

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def generate_23_classes_grid(
    raw_dir: str, metadata_path: str, fig_dir: str, doc_dir: str
):
    raw_path = Path(raw_dir) / "labeled-images"
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    if not Path(metadata_path).exists():
        print(f"Không tìm thấy file metadata: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    total_dataset_imgs = len(df)

    print("=" * 75)
    print("ĐANG TRÍCH XUẤT ẢNH MẪU ĐẠI DIỆN VÀ VẼ GRID 23 LỚP HYPERKVASIR...")
    print("=" * 75)

    # Lấy danh sách 23 lớp sắp xếp theo Super Category rồi đến số lượng
    classes_ordered = (
        df.groupby(["super_category", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["super_category", "count"], ascending=[True, False])
    )

    # Màu sắc nhận diện theo 4 nhóm Super Category
    category_colors = {
        "Anatomical_Landmarks": "#27ae60",  # Xanh lá
        "Pathological_Findings": "#e74c3c",  # Đỏ
        "Quality_of_Mucosal_Views": "#e67e22",  # Cam
        "Therapeutic_Interventions": "#2980b9",  # Xanh dương
    }

    # Bố cục Grid 6 hàng x 4 cột (24 ô: 23 lớp + 1 ô thông tin tổng quan)
    fig, axes = plt.subplots(6, 4, figsize=(20, 26))
    axes_flat = axes.flatten()

    for idx, (_, row) in enumerate(classes_ordered.iterrows()):
        cls_name = row["class_name"]
        sup_cat = row["super_category"]
        cls_count = row["count"]
        cls_pct = (cls_count / total_dataset_imgs) * 100

        # Lọc lấy ảnh mẫu chất lượng tốt
        cls_df = df[df["class_name"] == cls_name]
        good_samples = cls_df[cls_df["quality_tag"] == "Good_Quality"]

        if len(good_samples) > 0:
            sample_rel = good_samples.iloc[0]["relative_path"]
        else:
            sample_rel = cls_df.iloc[0]["relative_path"]

        img_full_path = raw_path / sample_rel
        img_bgr = cv2.imread(str(img_full_path))

        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            axes_flat[idx].imshow(img_rgb)
        else:
            axes_flat[idx].text(0.5, 0.5, "Image Not Found", ha="center", va="center")

        color_badge = category_colors.get(sup_cat, "black")
        title_text = f"{cls_name}\n({cls_count:,} ảnh | {cls_pct:.2f}%)"
        axes_flat[idx].set_title(
            title_text,
            fontsize=11,
            fontweight="bold",
            color=color_badge,
            pad=8,
        )
        axes_flat[idx].axis("off")

        # Thêm viền màu quanh khung ảnh theo nhóm
        for spine in axes_flat[idx].spines.values():
            spine.set_edgecolor(color_badge)
            spine.set_linewidth(2)

    # Ô thứ 24: Bảng chú giải nhóm (Legend Panel)
    axes_flat[23].axis("off")
    legend_text = (
        "BỘ DỮ LIỆU HYPERKVASIR (23 LỚP)\n"
        "───────────────────────────────\n"
        f"• Tổng số ảnh: {total_dataset_imgs:,} ảnh\n"
        "• Nguồn: Simula & Bærum Hospital\n\n"
        "MÃ MÀU PHÂN NHÓM CHỨC NĂNG:\n"
        "Mốc giải phẫu (Landmarks)\n"
        "Tổn thương bệnh lý (Pathology)\n"
        "Can thiệp y khoa (Interventions)\n"
        "Chất lượng quan sát (Mucosa Views)"
    )
    axes_flat[23].text(
        0.05,
        0.5,
        legend_text,
        fontsize=12,
        fontweight="bold",
        va="center",
        bbox=dict(
            boxstyle="round,pad=1",
            facecolor="#f8f9fa",
            edgecolor="#2c3e50",
            linewidth=2,
        ),
    )

    plt.suptitle(
        "HyperKvasir Endoscopic Dataset: 23 Clinical Classes & Anatomical Landmarks Atlas",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    output_fig = fig_path / "08_hyperkvasir_23_classes_grid.png"
    plt.savefig(output_fig, dpi=90, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close()

    print(f"Đã tạo thành công Bức tranh Tổng phổ 23 Lớp tại: {output_fig}")
    print("=" * 75)

    # Xuất tài liệu mô tả hình thái học
    md_file = doc_path / "14_hyperkvasir_visual_atlas.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# Tổng phổ Hình thái học 23 Lớp Nội soi HyperKvasir (Visual Atlas)\n\n"
        )
        f.write(
            "> **Hình minh họa chính:** `docs/figures/08_hyperkvasir_23_classes_grid.png` | **Tổng số:** 23 lớp bệnh học\n\n---\n\n"
        )

        f.write("## 1. Mô tả Đặc trưng Hình thái Lâm sàng Từng Lớp\n\n")
        f.write(
            "| Nhóm chức năng | Tên lớp (Class) | Số lượng | Đặc trưng thị giác & Hình thái học y khoa |\n"
        )
        f.write("|:---|:---|:---:|:---|\n")

        for _, row in classes_ordered.iterrows():
            c_name = row["class_name"]
            s_cat = row["super_category"]
            cnt = row["count"]
            f.write(
                f"| `{s_cat}` | **{c_name}** | {cnt:,} | Cấu trúc giải phẫu / bệnh học đặc thù đường tiêu hóa |\n"
            )

        f.write("\n---\n\n## 2. Ứng dụng trong Khóa luận & Thuyết trình\n\n")
        f.write(
            "Bức ảnh `08_hyperkvasir_23_classes_grid.png` cung cấp cái nhìn trực quan toàn diện nhất về độ đa dạng hình thái học, "
            "là minh chứng thực nghiệm then chốt cho phần giới thiệu bộ dữ liệu trong Chương 2 của Khóa luận.\n"
        )

    print(f"Đã lưu tài liệu mô tả tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_path = os.path.join(project_root, "data", "raw")
    metadata_csv = os.path.join(
        project_root, "data", "processed", "hyperkvasir_master_metadata.csv"
    )
    figures_dir = os.path.join(project_root, "docs", "figures")
    research_dir = os.path.join(project_root, "docs", "research")

    generate_23_classes_grid(raw_path, metadata_csv, figures_dir, research_dir)
