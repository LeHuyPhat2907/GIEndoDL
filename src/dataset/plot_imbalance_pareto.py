"""Script phân tích chuyên sâu mất cân bằng lớp và vẽ biểu đồ Pareto cho HyperKvasir."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_imbalance_and_pareto(raw_dir: str, fig_dir: str, doc_dir: str):
    raw_path = Path(raw_dir) / "labeled-images"
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"Không tìm thấy thư mục: {raw_path}")
        return

    # 1. Thu thập dữ liệu 23 lớp
    records = []
    image_exts = {".jpg", ".jpeg", ".png"}

    for class_folder in raw_path.rglob("*"):
        if class_folder.is_dir() and not any(
            d.is_dir() for d in class_folder.iterdir()
        ):
            imgs = [f for f in class_folder.glob("*") if f.suffix.lower() in image_exts]
            if imgs:
                records.append(
                    {
                        "Category": class_folder.parent.name,
                        "Class_Name": class_folder.name,
                        "Count": len(imgs),
                    }
                )

    df = pd.DataFrame(records)
    df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)

    total_images = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total_images) * 100
    df["Cumulative_Pct"] = df["Percentage"].cumsum()

    # Phân nhóm 3 tầng: Majority, Moderate, Minority
    def get_tier(count):
        if count >= 900:
            return "Đa số (Majority >= 900)"
        elif count >= 100:
            return "Trung bình (100 - 899)"
        else:
            return "Thiểu số / Cực hiếm (< 100)"

    df["Tier"] = df["Count"].apply(get_tier)

    max_c = df["Count"].max()
    min_c = df["Count"].min()
    ir = max_c / min_c

    print("=" * 75)
    print("BÁO CÁO PHÂN TÍCH MẤT CÂN BẰNG LỚP & NGUYÊN LÝ PARETO (HYPERKVASIR)")
    print("=" * 75)
    print(f"Tổng số lượng ảnh: {total_images:,} ảnh")
    print(f"Tổng số lớp: {len(df)} lớp")
    print(f"Lớp nhiều nhất: {df.iloc[0]['Class_Name']} ({max_c:,} ảnh)")
    print(f"Lớp ít nhất:    {df.iloc[-1]['Class_Name']} ({min_c:,} ảnh)")
    print(f"TỶ LỆ MẤT CÂN BẰNG (Imbalance Ratio): {ir:.2f} : 1")
    print("-" * 75)
    print(
        f"Nhận xét Pareto: Top 7 lớp đầu tiên đã chiếm {df.iloc[6]['Cumulative_Pct']:.1f}% toàn bộ dữ liệu!"
    )
    print(
        f"   7 lớp hiếm nhất chỉ chiếm vỏn vẹn {df.tail(7)['Percentage'].sum():.2f}% toàn bộ dữ liệu."
    )
    print("=" * 75)

    # 2. Vẽ Biểu đồ Pareto chất lượng cao (2 trục Y)
    fig, ax1 = plt.subplots(figsize=(15, 8))
    sns.set_theme(style="whitegrid")

    # Trục 1: Bar chart số lượng
    colors = [
        (
            "#2ecc71"
            if "Majority" in t
            else "#f39c12"
            if "Trung bình" in t
            else "#e74c3c"
        )
        for t in df["Tier"]
    ]
    ax1.bar(
        df["Class_Name"],
        df["Count"],
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Số lượng ảnh (Image Count)", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(df["Class_Name"], rotation=45, ha="right", fontsize=10)
    ax1.set_ylim(0, max_c * 1.15)

    # Trục 2: Đường phần trăm tích lũy (Cumulative Percentage Line)
    ax2 = ax1.twinx()
    ax2.plot(
        df["Class_Name"],
        df["Cumulative_Pct"],
        color="#2c3e50",
        marker="o",
        linewidth=2.5,
        markersize=6,
    )
    ax2.set_ylabel(
        "Phần trăm tích lũy (%) - Cumulative %",
        fontsize=12,
        fontweight="bold",
        color="#2c3e50",
    )
    ax2.set_ylim(0, 105)
    ax2.grid(False)

    # Thêm đường tham chiếu 80% Pareto
    ax2.axhline(80, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.text(
        len(df) - 3,
        81.5,
        "Ngưỡng Pareto 80%",
        color="red",
        fontweight="bold",
        fontsize=10,
    )

    plt.title(
        f"HyperKvasir Class Imbalance & Pareto Distribution (Total: {total_images:,} Images, 23 Classes)\n"
        f"Imbalance Ratio: {ir:.2f}:1 (Max: {max_c:,} vs Min: {min_c:,})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    chart_file = fig_path / "03_hyperkvasir_pareto_imbalance.png"
    plt.savefig(chart_file, dpi=200)
    plt.close()
    print(f"\nĐã lưu biểu đồ Pareto tại: {chart_file}")

    # 3. Xuất tài liệu phân tích Markdown
    md_file = doc_path / "05_class_imbalance_analysis.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# Báo cáo Phân tích Mất Cân Bằng Lớp & Phân phối Pareto (HyperKvasir)\n\n"
        )
        f.write(
            f"> **Imbalance Ratio (IR):** **{ir:.2f} : 1** | **Tổng số lớp:** 23 lớp | **Tổng số ảnh:** {total_images:,} ảnh\n\n---\n\n"
        )

        f.write("## 1. Phân tầng Dữ liệu (3-Tier Classification)\n\n")
        f.write(
            "| Phân tầng (Tier) | Số lớp | Tổng số ảnh | Tỷ trọng (%) | Danh sách các lớp |\n"
        )
        f.write("|:---|:---:|:---:|:---:|:---|\n")

        for tier_name in [
            "Đa số (Majority >= 900)",
            "Trung bình (100 - 899)",
            "Thiểu số / Cực hiếm (< 100)",
        ]:
            tier_df = df[df["Tier"] == tier_name]
            classes_str = ", ".join([f"`{c}`" for c in tier_df["Class_Name"]])
            f.write(
                f"| **{tier_name}** | {len(tier_df)} | {tier_df['Count'].sum():,} | {tier_df['Percentage'].sum():.2f}% | {classes_str} |\n"
            )

        f.write("\n---\n\n## 2. Ý nghĩa Khoa học đối với Thiết kế Mô hình\n\n")
        f.write(
            "1. **Hiện tượng Tail-Dominance:** 7 lớp đa số chiếm tới hơn 66% tổng dữ liệu, trong khi 7 lớp hiếm nhất chỉ chiếm dưới 2%. "
        )
        f.write(
            "Nếu áp dụng chuẩn đo lường Accuracy thông thường, mô hình có thể đạt 90% Accuracy nhưng hoàn toàn bỏ sót các ca bệnh hiếm (như trĩ, hồi tràng, Barrett).\n"
        )
        f.write(
            "2. **Chỉ số Đánh giá Tiêu chuẩn:** Đề tài bắt buộc phải sử dụng **F1-Score (Macro)** và **Balanced Accuracy** làm chỉ số tối ưu chính thay cho Accuracy thông thường.\n"
        )
        f.write(
            "3. **Giải pháp Thuật toán:** Ứng dụng **Weighted Random Sampler** cân bằng xác suất lấy mẫu theo batch và **Class-Balanced Focal Loss** để phạt nặng các lỗi phân loại trên nhóm lớp thiểu số.\n"
        )

    print(f"Đã lưu tài liệu phân tích tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_dir = os.path.join(project_root, "data", "raw")
    figures_dir = os.path.join(project_root, "docs", "figures")
    research_dir = os.path.join(project_root, "docs", "research")

    analyze_imbalance_and_pareto(raw_dir, figures_dir, research_dir)
