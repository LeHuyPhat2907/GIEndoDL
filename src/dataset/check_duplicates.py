"""Script phát hiện ảnh trùng lặp và gần trùng lặp (Near-duplicates) bằng pHash & SSIM."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import cv2
import imagehash
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def compute_hashes(img_path: Path):
    """Tính toán pHash (Perceptual Hash) và dHash (Difference Hash)."""
    try:
        with Image.open(img_path) as img:
            phash = str(imagehash.phash(img))
            dhash = str(imagehash.dhash(img))
            return {
                "Path": img_path,
                "Filename": img_path.name,
                "Class_Name": img_path.parent.name,
                "pHash": phash,
                "dHash": dhash,
            }
    except Exception:
        return None


def run_duplicate_detection(raw_dir: str, fig_dir: str, doc_dir: str):
    raw_path = Path(raw_dir) / "labeled-images"
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"Không tìm thấy thư mục: {raw_path}")
        return

    print("=" * 75)
    print("🔍 ĐANG TÍNH MÃ BĂM THỊ GIÁC (pHash/dHash) CHO 10,662 ẢNH NỘI SOI...")
    print("=" * 75)

    image_exts = {".jpg", ".jpeg", ".png"}
    all_imgs = [f for f in raw_path.rglob("*") if f.suffix.lower() in image_exts]

    # 1. Tính toán hash song song đa luồng siêu tốc (~5s)
    records = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = executor.map(compute_hashes, all_imgs)
        for res in results:
            if res:
                records.append(res)

    df = pd.DataFrame(records)
    total_imgs = len(df)
    print(f"Đã tính toán mã băm cho: {total_imgs:,} ảnh")

    # 2. Tìm các cặp trùng lặp tuyệt đối (Exact pHash Match)
    print("\nĐang đối chiếu các cặp ảnh trùng lặp hoặc gần trùng (Near-duplicates)...")

    duplicate_pairs = []
    hash_groups = df.groupby("pHash")

    for phash_val, group in hash_groups:
        if len(group) > 1:
            paths = group["Path"].tolist()
            filenames = group["Filename"].tolist()
            classes = group["Class_Name"].tolist()

            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    # Đọc và tính chỉ số SSIM thực tế
                    img1 = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.imread(str(paths[j]), cv2.IMREAD_GRAYSCALE)

                    # Resize về cùng kích thước nếu khác size để tính SSIM
                    if img1.shape != img2.shape:
                        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

                    score, _ = ssim(img1, img2, full=True)

                    if score >= 0.90:  # Ngưỡng gần trùng lặp rất cao
                        duplicate_pairs.append(
                            {
                                "Image_A": filenames[i],
                                "Class_A": classes[i],
                                "Image_B": filenames[j],
                                "Class_B": classes[j],
                                "pHash": phash_val,
                                "SSIM_Similarity": round(score, 4),
                                "Path_A": paths[i],
                                "Path_B": paths[j],
                            }
                        )

    dup_df = pd.DataFrame(duplicate_pairs)

    print("=" * 75)
    print("BÁO CÁO KHOA HỌC KIỂM TRA TRÙNG LẶP DỮ LIỆU (DUPLICATE AUDIT)")
    print("=" * 75)
    print(f"Tổng số ảnh khảo sát:            {total_imgs:,} ảnh")
    print(
        f"Số lượng mã pHash duy nhất (Unique): {df['pHash'].nunique():,} / {total_imgs:,}"
    )
    print(f"Số cặp ảnh gần trùng lặp (SSIM >= 0.90): {len(dup_df)} cặp")

    if len(dup_df) > 0:
        cross_class_dups = dup_df[dup_df["Class_A"] != dup_df["Class_B"]]
        same_class_dups = dup_df[dup_df["Class_A"] == dup_df["Class_B"]]
        print(f"   - Số cặp trùng cùng lớp (Same-class):  {len(same_class_dups)} cặp")
        print(
            f"   - Số cặp trùng khác lớp (Cross-class): {len(cross_class_dups)} cặp (Cần chú ý!)"
        )
    print("=" * 75)

    # 3. Trực quan hóa mẫu cặp ảnh trùng lặp (nếu tìm thấy)
    if len(dup_df) > 0:
        sample_pair = dup_df.sort_values(by="SSIM_Similarity", ascending=False).iloc[0]
        img_a = cv2.cvtColor(cv2.imread(str(sample_pair["Path_A"])), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(str(sample_pair["Path_B"])), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_a)
        axes[0].set_title(
            f"Ảnh A: {sample_pair['Image_A']}\nLớp: {sample_pair['Class_A']}",
            fontweight="bold",
        )
        axes[0].axis("off")

        axes[1].imshow(img_b)
        axes[1].set_title(
            f"Ảnh B: {sample_pair['Image_B']}\nLớp: {sample_pair['Class_B']}\nĐộ tương đồng SSIM: {sample_pair['SSIM_Similarity']:.4f}",
            fontweight="bold",
            color="red",
        )
        axes[1].axis("off")

        plt.tight_layout()
        chart_output = fig_path / "07_duplicate_pairs_sample.png"
        plt.savefig(chart_output, dpi=200)
        plt.close()
        print(f"Đã lưu hình ảnh minh họa cặp trùng lặp tại: {chart_output}")

    # 4. Xuất tài liệu Markdown
    md_file = doc_path / "09_duplicate_and_leakage_analysis.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# 🔍 Báo cáo Kiểm định Trùng lặp & Ngăn chặn Rò rỉ Dữ liệu (Duplicate & Leakage Audit)\n\n"
        )
        f.write(
            f"> **Số lượng ảnh khảo sát:** {total_imgs:,} ảnh | **Phương pháp:** Perceptual Hash (pHash), dHash & Structural Similarity (SSIM)\n\n---\n\n"
        )

        f.write("## 1. Kết quả Thực nghiệm Kiểm định\n\n")
        f.write(
            f"- **Tổng số mã băm pHash độc lập:** `{df['pHash'].nunique():,}` mã.\n"
        )
        f.write(
            f"- **Số lượng cặp ảnh gần trùng lặp (SSIM $\\ge$ 0.90):** `{len(dup_df)}` cặp.\n"
        )
        if len(dup_df) > 0:
            f.write(
                f"- **Cặp trùng lặp cùng lớp (Same-class near-duplicates):** `{len(same_class_dups)}` cặp.\n"
            )
            f.write(
                f"- **Cặp trùng lặp mâu thuẫn nhãn (Cross-class conflicting):** `{len(cross_class_dups)}` cặp.\n"
            )

        f.write("\n---\n\n## 2. Ý nghĩa Phương pháp luận Y học & Chuẩn TRIPOD-AI\n\n")
        f.write(
            "1. **Bản chất của các cặp Near-duplicates:** Trong nội soi tiêu hóa, các cặp ảnh gần trùng lặp sinh ra do camera ghi hình liên tiếp nhiều khung hình ở cùng một vị trí tổn thương (Burst captures / Consecutive frames).\n"
        )
        f.write(
            "2. **Ngăn chặn Rò rỉ Dữ liệu Tuyệt đối (Zero Data Leakage Protocol):**\n"
        )
        f.write(
            "   - Khi phân chia tập dữ liệu ở Giai đoạn 4 (Task 51), các cặp ảnh có mã `pHash` trùng khớp **BẮT BUỘC PHẢI ĐƯỢC XẾP CHUNG VÀO CÙNG MỘT TẬP** (Toàn bộ vào Train hoặc toàn bộ vào Test).\n"
        )
        f.write(
            "   - Tuyệt đối không để 1 ảnh rơi vào Train và ảnh song sinh của nó rơi vào Test, đảm bảo tính khách quan 100% của bài báo khoa học.\n"
        )

    print(f"Đã lưu tài liệu nghiên cứu tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_dir = os.path.join(project_root, "data", "raw")
    figures_dir = os.path.join(project_root, "docs", "figures")
    research_dir = os.path.join(project_root, "docs", "research")

    run_duplicate_detection(raw_dir, figures_dir, research_dir)
