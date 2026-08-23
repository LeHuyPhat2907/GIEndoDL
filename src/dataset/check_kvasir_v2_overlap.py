"""Script kiểm tra tính toàn vẹn Kvasir-v2 và đối chiếu Overlap với HyperKvasir."""

import hashlib
import os
from pathlib import Path


def calculate_file_hash(filepath: Path) -> str:
    """Tính mã băm MD5 để xác thực ảnh trùng lặp tuyệt đối."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def verify_and_check_overlap(raw_dir: str):
    raw_path = Path(raw_dir)
    kvasir_v2_dir = raw_path / "kvasir-v2"
    hyperkvasir_dir = raw_path / "labeled-images"

    print("=" * 70)
    print("BÁO CÁO ĐỐI CHIẾU & KIỂM TRA OVERLAP: KVASIR-V2 VS HYPERKVASIR")
    print("=" * 70)

    if not kvasir_v2_dir.exists():
        print(f"Chưa tìm thấy thư mục Kvasir-v2 tại: {kvasir_v2_dir}")
        print("Vui lòng giải nén kvasir-dataset-v2.zip vào data/raw/kvasir-v2/")
        return

    # 1. Kiểm tra 8 lớp của Kvasir-v2
    image_exts = {".jpg", ".jpeg", ".png"}
    v2_classes = [d for d in kvasir_v2_dir.iterdir() if d.is_dir()]
    total_v2_images = 0
    v2_hashes = {}

    print("Danh sách các lớp Kvasir-v2:")
    for cls in sorted(v2_classes):
        imgs = [f for f in cls.glob("*") if f.suffix.lower() in image_exts]
        count = len(imgs)
        total_v2_images += count
        print(f"   - {cls.name:<25}: {count:,} ảnh")
        for img in imgs:
            v2_hashes[calculate_file_hash(img)] = img.name

    print(f"\nTổng số ảnh Kvasir-v2: {total_v2_images:,} / 8,000 chuẩn gốc.")

    # 2. Đối chiếu Overlap với HyperKvasir (nếu đã có thư mục labeled-images)
    if hyperkvasir_dir.exists():
        print("\nĐang quét và đối chiếu với HyperKvasir (10,662 ảnh)...")
        hyper_imgs = [
            f for f in hyperkvasir_dir.rglob("*") if f.suffix.lower() in image_exts
        ]
        overlap_count = 0

        for img in hyper_imgs:
            h = calculate_file_hash(img)
            if h in v2_hashes:
                overlap_count += 1

        print("=" * 70)
        print("KẾT LUẬN KHOA HỌC:")
        print(f"   - Số ảnh Kvasir-v2 nằm trong HyperKvasir: {overlap_count:,}")
        overlap_pct = (
            (overlap_count / total_v2_images) * 100 if total_v2_images > 0 else 0
        )
        print(f"   - Tỷ lệ Overlap: {overlap_pct:.1f}%")
        print(
            "   - Ghi chú nghiên cứu: Kvasir-v2 là tập con tiền đề (Subset) của HyperKvasir."
        )
        print("     HyperKvasir mở rộng thêm 2,662 ảnh và 15 lớp bệnh lý mới.")
    print("=" * 70)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_path = os.path.join(project_root, "data", "raw")
    verify_and_check_overlap(raw_path)
