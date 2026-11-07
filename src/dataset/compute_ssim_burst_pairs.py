"""Module Tính Toán Ma Trận Tương Đồng Cấu Trúc (SSIM) Cho Các Cặp Ảnh Burst-Captures.

Task D08: Quét toàn bộ các cặp ứng viên từ Multi-Hash và đo đạc chỉ số SSIM
với ngưỡng >= 0.92 để nhận diện triệt để các khung hình chụp liên tiếp trong nội soi.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim


def compute_pair_ssim(args: Tuple[int, int, Dict, Dict, float]) -> Optional[Dict]:
    """Tính toán chỉ số SSIM thực tế giữa 2 ảnh."""
    idx1, idx2, row1, row2, threshold = args
    path1 = Path(row1["relative_path"])
    path2 = Path(row2["relative_path"])

    try:
        im1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(str(path2), cv2.IMREAD_GRAYSCALE)

        if im1 is None or im2 is None:
            return None

        # Đưa về cùng kích thước nếu có lệch 1-2 pixel
        if im1.shape != im2.shape:
            im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

        score, _ = ssim(im1, im2, full=True)

        if score >= threshold:
            # Tính khoảng cách Hamming của pHash
            h1 = int(row1["phash"], 16)
            h2 = int(row2["phash"], 16)
            hamming_phash = bin(h1 ^ h2).count("1")

            same_class = row1["class_name"] == row2["class_name"]

            return {
                "idx_a": idx1,
                "file_a": row1["file_name"],
                "path_a": str(path1).replace("\\", "/"),
                "class_a": row1["class_name"],
                "idx_b": idx2,
                "file_b": row2["file_name"],
                "path_b": str(path2).replace("\\", "/"),
                "class_b": row2["class_name"],
                "ssim_score": round(float(score), 4),
                "hamming_phash": hamming_phash,
                "same_class": same_class,
                "is_burst_capture": True,
            }
    except Exception:
        return None

    return None


def run_ssim_burst_detection(
    multi_hash_csv: str,
    output_csv: str,
    fig_output: str,
    doc_output: str,
    ssim_threshold: float = 0.92,
    max_workers: int = 8,
):
    """Thực thi quét SSIM trên các cặp ứng viên băm."""
    print("=" * 80)
    print(
        f"🔬 BẮT ĐẦU TÍNH TOÁN MA TRẬN SSIM (NGƯỠNG >= {ssim_threshold}) PHÁT HIỆN BURST-CAPTURES..."
    )
    print("=" * 80)

    start_time = time.time()
    df_hashes = pd.read_csv(multi_hash_csv)
    print(f"[*] Đã nạp {len(df_hashes):,} bản ghi từ cơ sở dữ liệu đa băm.")

    # 1. Tìm tập hợp tất cả các cặp ứng viên nghi vấn
    phash_ints = np.array([int(h, 16) for h in df_hashes["phash"]], dtype=np.uint64)
    all_candidates = set()

    # Exact matches trên bất kỳ mã băm nào (aHash, pHash, dHash, wHash)
    for h_col in ["phash", "ahash", "whash", "dhash"]:
        for _, grp in df_hashes.groupby(h_col):
            if len(grp) > 1:
                idxs = grp.index.tolist()
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        all_candidates.add(
                            (min(idxs[i], idxs[j]), max(idxs[i], idxs[j]))
                        )

    # Within-class pairs có khoảng cách Hamming pHash <= 4
    for _, grp in df_hashes.groupby("class_name"):
        idxs = grp.index.to_numpy()
        hashes = phash_ints[idxs]
        n = len(idxs)
        for i in range(n):
            h_i = hashes[i]
            for j in range(i + 1, n):
                xor = int(h_i ^ hashes[j])
                if bin(xor).count("1") <= 4:
                    all_candidates.add((idxs[i], idxs[j]))

    candidate_list = list(all_candidates)
    print(
        f"[*] Đã lọc được {len(candidate_list):,} cặp ảnh ứng viên nghi vấn có độ tương đồng cao."
    )

    # Chuẩn bị tham số cho ThreadPoolExecutor
    tasks = []
    for idx1, idx2 in candidate_list:
        row1 = df_hashes.iloc[idx1].to_dict()
        row2 = df_hashes.iloc[idx2].to_dict()
        tasks.append((idx1, idx2, row1, row2, ssim_threshold))

    print(f"[*] Đang tính toán chỉ số SSIM trên {max_workers} luồng xử lý...")
    burst_pairs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(compute_pair_ssim, tasks):
            if res:
                burst_pairs.append(res)

    df_burst = pd.DataFrame(burst_pairs)
    if len(df_burst) > 0:
        df_burst = df_burst.sort_values(by="ssim_score", ascending=False).reset_index(
            drop=True
        )

    # Lưu kết quả CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_burst.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"✅ HOÀN TẤT ĐO ĐẠC SSIM TRONG: {elapsed:.2f} giây!")
    print(f"📁 Đã lưu danh sách các cặp Burst-Captures tại: {out_path}")
    print("=" * 80)

    # Thống kê kết quả
    total_burst = len(df_burst)
    if total_burst > 0:
        same_class = df_burst[df_burst["same_class"] == True]
        cross_class = df_burst[df_burst["same_class"] == False]
        unique_imgs = set(df_burst["file_a"]).union(set(df_burst["file_b"]))

        print("📊 BẢNG TỔNG HỢP KIỂM TOÁN CÁC CẶP ẢNH LIÊN TIẾP (BURST-CAPTURES):")
        print(
            f"  • Tổng số cặp đạt chuẩn SSIM >= {ssim_threshold}: {total_burst:,} cặp"
        )
        print(
            f"  • Số ảnh độc lập liên quan:               {len(unique_imgs):,} / {len(df_hashes):,} ảnh ({len(unique_imgs)/len(df_hashes)*100:.2f}%)"
        )
        print(
            f"  • Cặp trùng cùng một lớp (Same-Class):    {len(same_class):,} cặp ({len(same_class)/total_burst*100:.1f}%)"
        )
        print(
            f"  • Cặp trùng khác lớp (Cross-Class):       {len(cross_class):,} cặp ({len(cross_class)/total_burst*100:.1f}%) [Cần sửa nhãn!]"
        )
        print(
            f"  • Chỉ số SSIM cao nhất:                   {df_burst['ssim_score'].max():.4f}"
        )
        print(
            f"  • Chỉ số SSIM trung bình:                 {df_burst['ssim_score'].mean():.4f}"
        )
        print("=" * 80)

        # 2. Vẽ biểu đồ trực quan hóa cặp ảnh mẫu có SSIM cao nhất
        fig_path = Path(fig_output)
        fig_path.parent.mkdir(parents=True, exist_ok=True)

        sample = df_burst.iloc[0]
        im_a = cv2.cvtColor(cv2.imread(sample["path_a"]), cv2.COLOR_BGR2RGB)
        im_b = cv2.cvtColor(cv2.imread(sample["path_b"]), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(im_a)
        axes[0].set_title(
            f"Ảnh A: {sample['file_a']}\nLớp: {sample['class_a']}",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].axis("off")

        axes[1].imshow(im_b)
        axes[1].set_title(
            f"Ảnh B: {sample['file_b']}\nLớp: {sample['class_b']}\nĐộ tương đồng cấu trúc SSIM: {sample['ssim_score']:.4f} (>= 0.92)",
            fontsize=12,
            fontweight="bold",
            color="darkred" if not sample["same_class"] else "green",
        )
        axes[1].axis("off")

        plt.suptitle(
            "Minh Họa Cặp Ảnh Nội Soi Chụp Liên Tiếp (Burst-Capture Pair) Trong HyperKvasir",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"🖼️ Đã lưu hình ảnh trực quan cặp Burst-Capture tại: {fig_path}")

        # 3. Xuất tài liệu nghiên cứu Markdown
        doc_path = Path(doc_output)
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(
                "# 🔬 Báo cáo Đo lường Tương đồng Cấu trúc (SSIM) & Phát hiện Burst-Captures\n\n"
            )
            f.write(
                f"> **Phương pháp:** Kết hợp Đa băm thị giác (pHash/aHash/dHash/wHash) & Chỉ số SSIM (Structural Similarity Index) với ngưỡng $\\text{{SSIM}} \\ge {ssim_threshold}$\n\n---\n\n"
            )
            f.write("## 1. Kết Quả Kiểm Toán Thực Nghiệm\n\n")
            f.write(
                f"- **Tổng số cặp ứng viên rà soát:** `{len(candidate_list):,}` cặp.\n"
            )
            f.write(
                f"- **Số cặp ảnh chụp liên tiếp được xác nhận (SSIM $\\ge$ {ssim_threshold}):** `{total_burst:,}` cặp.\n"
            )
            f.write(
                f"- **Số lượng ảnh liên quan:** `{len(unique_imgs):,}` ảnh ({len(unique_imgs)/len(df_hashes)*100:.2f}% toàn bộ kho dữ liệu).\n"
            )
            f.write(f"- **Cặp cùng lớp (Same-Class):** `{len(same_class):,}` cặp.\n")
            f.write(
                f"- **Cặp khác lớp (Cross-Class mâu thuẫn nhãn):** `{len(cross_class):,}` cặp.\n\n---\n\n"
            )
            f.write("## 2. Ý Nghĩa Lâm Sàng & Khắc Phục Rò Rỉ Dữ Liệu\n\n")
            f.write(
                "1. **Nguồn gốc Burst-Captures:** Các bác sĩ nội soi khi phát hiện một tổn thương thường bấm máy liên tục nhiều lần trong vài giây để ghi nhận ở nhiều góc vi chỉnh. Điều này tạo ra hàng trăm cặp ảnh gần như giống hệt nhau về cấu trúc niêm mạc.\n"
            )
            f.write(
                "2. **Nguy cơ Data Leakage:** Nếu chia ngẫu nhiên thông thường (Stratified Random Split), một ảnh của cặp rơi vào Train và ảnh kia rơi vào Test. Mô hình sẽ 'học vẹt' chính xác từng điểm ảnh và cho độ chính xác ảo (Overly optimistic performance) nhưng thất bại khi thử nghiệm ngoài thực tế.\n"
            )
            f.write(
                "3. **Chiến lược Zero Data Leakage (Giai đoạn tiếp theo):** Toàn bộ các ảnh thuộc cùng một cụm Burst-Capture bắt buộc phải được đóng gói vào cùng một **Patient/Video Cluster ID** duy nhất để chuyển trọn gói sang Train hoặc Test.\n"
            )
        print(f"📝 Đã cập nhật tài liệu nghiên cứu tại: {doc_path}")
        print("=" * 80)

    return df_burst


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    hash_csv = project_root / "data" / "processed" / "multi_hash_10662.csv"
    out_csv = project_root / "data" / "processed" / "ssim_burst_pairs_092.csv"
    fig_file = project_root / "docs" / "figures" / "07_duplicate_pairs_sample.png"
    doc_file = (
        project_root / "docs" / "research" / "09_duplicate_and_leakage_analysis.md"
    )

    run_ssim_burst_detection(
        str(hash_csv), str(out_csv), str(fig_file), str(doc_file), ssim_threshold=0.92
    )
