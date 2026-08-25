"""Script phân tích kích thước, độ phân giải và tỷ lệ khung hình cho HyperKvasir."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sns


def analyze_image_resolutions(raw_dir: str, fig_dir: str, doc_dir: str):
    raw_path = Path(raw_dir) / "labeled-images"
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"Không tìm thấy thư mục: {raw_path}")
        return

    print("=" * 75)
    print("ĐANG QUÉT KÍCH THƯỚC & ĐỘ PHÂN GIẢI 10,662 ẢNH NỘI SOI...")
    print("=" * 75)

    records = []
    image_exts = {".jpg", ".jpeg", ".png"}

    image_files = [f for f in raw_path.rglob("*") if f.suffix.lower() in image_exts]

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                records.append(
                    {
                        "Filename": img_path.name,
                        "Category": img_path.parent.parent.name,
                        "Class_Name": img_path.parent.name,
                        "Width": w,
                        "Height": h,
                        "Resolution": f"{w}x{h}",
                        "Aspect_Ratio": round(w / h, 3),
                        "Megapixels": round((w * h) / 1_000_000, 2),
                    }
                )
        except Exception as e:
            print(f"Lỗi đọc ảnh {img_path.name}: {e}")

    df = pd.DataFrame(records)

    # 1. Thống kê tổng quan
    total_imgs = len(df)
    res_counts = df["Resolution"].value_counts()
    top_res = res_counts.head(5)

    print(f"Tổng số ảnh phân tích thành công: {total_imgs:,} ảnh")
    print(f"Kích thước nhỏ nhất: {df['Width'].min()} x {df['Height'].min()}")
    print(f"Kích thước lớn nhất: {df['Width'].max()} x {df['Height'].max()}")
    print(
        f"Tỷ lệ khung hình trung bình (Aspect Ratio): {df['Aspect_Ratio'].mean():.3f} (Min: {df['Aspect_Ratio'].min():.2f}, Max: {df['Aspect_Ratio'].max():.2f})"
    )
    print("\nTop 5 độ phân giải phổ biến nhất trong HyperKvasir:")
    for res, count in top_res.items():
        pct = (count / total_imgs) * 100
        print(f"   - {res:<15}: {count:>6,} ảnh ({pct:>5.1f}%)")
    print("=" * 75)

    # 2. Vẽ biểu đồ phân bố độ phân giải (Multi-panel Figure)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style="whitegrid")

    # Panel 1: Scatter plot Width vs Height với KDE
    sns.scatterplot(
        x="Width",
        y="Height",
        data=df,
        alpha=0.4,
        color="#2980b9",
        ax=axes[0],
        s=30,
    )
    axes[0].set_title(
        "Phân bố Chiều Rộng x Chiều Cao (Width vs Height)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Width (pixels)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Height (pixels)", fontsize=11, fontweight="bold")

    # Panel 2: Phân bố Tỷ lệ Khung hình (Aspect Ratio Distribution)
    sns.histplot(
        df["Aspect_Ratio"],
        bins=25,
        kde=True,
        color="#27ae60",
        ax=axes[1],
        edgecolor="black",
    )
    axes[1].set_title(
        f"Phân bố Tỷ lệ Khung hình (Aspect Ratio = W/H)\nTrung bình: {df['Aspect_Ratio'].mean():.2f}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Aspect Ratio (W / H)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Số lượng ảnh", fontsize=11, fontweight="bold")

    # Panel 3: Bar chart Top Độ phân giải
    top_res_df = res_counts.head(6).reset_index()
    top_res_df.columns = ["Resolution", "Count"]
    sns.barplot(
        x="Count",
        y="Resolution",
        data=top_res_df,
        palette="viridis",
        hue="Resolution",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("Top Độ phân giải phổ biến nhất", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Số lượng ảnh", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Độ phân giải (W x H)", fontsize=11, fontweight="bold")

    for p in axes[2].patches:
        w_val = p.get_width()
        pct = (w_val / total_imgs) * 100
        axes[2].annotate(
            f"{int(w_val):,} ({pct:.1f}%)",
            (w_val + 50, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=9,
            color="black",
        )

    plt.tight_layout()
    fig_file = fig_path / "04_resolution_distribution.png"
    plt.savefig(fig_file, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ phân tích độ phân giải tại: {fig_file}")

    # 3. Xuất tài liệu phân tích Markdown
    md_file = doc_path / "06_resolution_and_aspect_ratio.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# 📐 Báo cáo Khảo sát Độ phân giải & Tỷ lệ Khung hình (HyperKvasir)\n\n"
        )
        f.write(
            f"> **Tổng số ảnh phân tích:** {total_imgs:,} ảnh | **Khoảng kích thước:** {df['Width'].min()}x{df['Height'].min()} đến {df['Width'].max()}x{df['Height'].max()}\n\n---\n\n"
        )

        f.write("## 1. Bảng Thống kê Phân phối Độ phân giải Chính\n\n")
        f.write(
            "| Độ phân giải (W x H) | Chuẩn hiển thị | Số lượng ảnh | Tỷ lệ (%) | Nhận định thiết bị |\n"
        )
        f.write("|:---|:---:|:---:|:---:|:---|\n")

        for res, count in top_res.items():
            pct = (count / total_imgs) * 100
            desc = (
                "Hệ thống nội soi SD PAL chuẩn"
                if "576" in res or "622" in res
                else (
                    "Hệ thống nội soi HD hiện đại"
                    if "1920" in res or "1280" in res
                    else "Định dạng hỗn hợp"
                )
            )
            standard = (
                "SD (~4:3)"
                if "529" in res or "576" in res
                else ("Full HD (16:9)" if "1920" in res else "Custom")
            )
            f.write(f"| `{res}` | {standard} | {count:,} | {pct:.1f}% | {desc} |\n")

        f.write(
            f"| Các kích thước khác | Đa dạng | {total_imgs - top_res.sum():,} | {(total_imgs - top_res.sum())/total_imgs*100:.1f}% | Các dòng máy nội soi khác |\n\n---\n\n"
        )

        f.write("## 2. Kết luận Khoa học & Định hướng Tiền xử lý (Phase 3)\n\n")
        f.write(
            "1. **Tính đa dạng thiết bị (Heterogeneous Resolutions):** Ảnh trong HyperKvasir đến từ nhiều thế hệ máy nội soi khác nhau tại Bệnh viện Bærum, dao động từ độ phân giải chuẩn SD cũ đến Full HD hiện đại.\n"
        )
        f.write(
            "2. **Tỷ lệ khung hình ổn định:** Đa số ảnh có tỷ lệ khung hình xấp xỉ `1.15 - 1.33` (tương đương chuẩn 4:3 của màn hình nội soi y tế truyền thống).\n"
        )
        f.write("3. **Quyết định Kỹ thuật Chuẩn hóa Đầu vào:**\n")
        f.write(
            "   - **Kích thước đầu vào mô hình:** Chọn kích thước chuẩn **`224 × 224 px`** (cho CNN baseline & Swin Transformer) và **`384 × 384 px`** (cho mô hình đề xuất độ nét cao).\n"
        )
        f.write(
            "   - **Chiến lược Resize:** Sử dụng thuật toán nội suy **Bicubic Interpolation** kết hợp **Letterbox Padding** hoặc **Center Crop** để bảo toàn cấu trúc hạt niêm mạc (Pit pattern) mà không làm méo hình dạng polyp.\n"
        )

    print(f"Đã lưu tài liệu nghiên cứu tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_dir = os.path.join(project_root, "data", "raw")
    figures_dir = os.path.join(project_root, "docs", "figures")
    research_dir = os.path.join(project_root, "docs", "research")

    analyze_image_resolutions(raw_dir, figures_dir, research_dir)
