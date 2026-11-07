"""Module Đa Băm Thị Giác (Multi-Perceptual Hashing) cho Toàn Bộ Kho Dữ Liệu Nội Soi.

Triển khai đồng thời 4 thuật toán băm cảm nhận thị giác:
1. aHash (Average Hash): Bắt cấu trúc phân bố ánh sáng mức thấp.
2. pHash (Perceptual DCT Hash): Kháng biến đổi độ tương phản, nén JPEG và ánh sáng y tế.
3. dHash (Difference Gradient Hash): Theo dõi độ dốc biên và viền mô học niêm mạc.
4. wHash (Wavelet DWT Hash): Phân tích hoa văn vi mô đa tỷ lệ không gian - tần số.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Dict, Optional
import imagehash
import pandas as pd
from PIL import Image


def compute_multi_hashes(img_path: Path) -> Optional[Dict]:
    """Tính 4 mã băm thị giác cho một tệp ảnh."""
    try:
        with Image.open(img_path) as img:
            # Chuyển RGB nếu cần
            if img.mode != "RGB":
                img = img.convert("RGB")

            ah = str(imagehash.average_hash(img))
            ph = str(imagehash.phash(img))
            dh = str(imagehash.dhash(img))
            wh = str(imagehash.whash(img))

            return {
                "file_name": img_path.name,
                "relative_path": str(img_path).replace("\\", "/"),
                "class_name": img_path.parent.name,
                "ahash": ah,
                "phash": ph,
                "dhash": dh,
                "whash": wh,
                "composite_256bit": f"{ah}_{ph}_{dh}_{wh}",
            }
    except Exception:
        return None


def run_multi_hash_scan(
    raw_dir: str,
    output_csv: str,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Quét và tính toán đa băm trên toàn bộ kho ảnh."""
    raw_path = Path(raw_dir)
    out_file = Path(output_csv)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(
        "🚀 BẮT ĐẦU QUÉT ĐA BĂM THỊ GIÁC (aHash + pHash + dHash + wHash) TRÊN 10,662 ẢNH..."
    )
    print("=" * 80)

    start_time = time.time()
    image_exts = {".jpg", ".jpeg", ".png"}
    all_imgs = [p for p in raw_path.rglob("*") if p.suffix.lower() in image_exts]
    total_imgs = len(all_imgs)
    print(f"[*] Tìm thấy {total_imgs:,} ảnh trong thư mục: {raw_path}")

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(compute_multi_hashes, all_imgs)
        for idx, res in enumerate(results, 1):
            if res:
                records.append(res)
            if idx % 2000 == 0 or idx == total_imgs:
                elapsed = time.time() - start_time
                print(
                    f"  [>] Đã xử lý {idx:,}/{total_imgs:,} ảnh ({idx/total_imgs*100:.1f}%) - Thời gian: {elapsed:.1f}s"
                )

    df = pd.DataFrame(records)
    df.to_csv(out_file, index=False, encoding="utf-8")

    elapsed_total = time.time() - start_time
    print("=" * 80)
    print(f"✅ HOÀN TẤT QUÉT ĐA BĂM TRONG: {elapsed_total:.2f} giây!")
    print(f"📁 Đã lưu cơ sở dữ liệu mã băm tại: {out_file}")
    print("=" * 80)

    # Thống kê số lượng mã duy nhất
    print("📊 BẢNG KIỂM TOÁN TÍNH ĐỘC LẬP CỦA CÁC MÃ BĂM:")
    print(
        f"  • aHash (Average Hash):        {df['ahash'].nunique():,} unique codes / {len(df):,} images"
    )
    print(
        f"  • pHash (Perceptual DCT Hash): {df['phash'].nunique():,} unique codes / {len(df):,} images"
    )
    print(
        f"  • dHash (Difference Hash):     {df['dhash'].nunique():,} unique codes / {len(df):,} images"
    )
    print(
        f"  • wHash (Wavelet Hash):        {df['whash'].nunique():,} unique codes / {len(df):,} images"
    )
    print(
        f"  • Composite (256-bit Fusion):  {df['composite_256bit'].nunique():,} unique codes / {len(df):,} images"
    )
    print("=" * 80)

    return df


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    raw_dir = project_root / "data" / "raw" / "labeled-images"
    output_csv = project_root / "data" / "processed" / "multi_hash_10662.csv"

    run_multi_hash_scan(str(raw_dir), str(output_csv))
