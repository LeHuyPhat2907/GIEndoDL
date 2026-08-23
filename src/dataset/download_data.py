"""Script kiểm tra tính toàn vẹn và phân loại dữ liệu HyperKvasir."""

import os
from pathlib import Path


def check_dataset_integrity(raw_data_dir: str):
    """Kiểm tra số lượng ảnh sau khi giải nén vào data/raw/."""
    raw_path = Path(raw_data_dir)
    labeled_dir = raw_path / "labeled-images"
    segmented_dir = raw_path / "segmented-images"

    print("=" * 65)
    print("BÁO CÁO KIỂM TRA TÍNH TOÀN VẸN BỘ DỮ LIỆU HYPERKVASIR")
    print("=" * 65)

    # 1. Kiểm tra tập labeled-images
    if labeled_dir.exists():
        image_extensions = {".jpg", ".jpeg", ".png"}
        labeled_images = [
            f for f in labeled_dir.rglob("*") if f.suffix.lower() in image_extensions
        ]
        print(f"Thư mục Labeled:   {labeled_dir}")
        print(f"Số lượng ảnh đếm được: {len(labeled_images)} / 10,662 ảnh gốc")
        if len(labeled_images) == 10662:
            print("   -> Tập Labeled hoàn toàn đầy đủ 100%!")
        else:
            print(
                f"   -> Tìm thấy {len(labeled_images)} ảnh (hãy kiểm tra xem đã giải nén hết chưa)."
            )
    else:
        print(f"Chưa tìm thấy thư mục: {labeled_dir}")

    # 2. Kiểm tra tập segmented-images
    if segmented_dir.exists():
        seg_images = list((segmented_dir / "images").glob("*.jpg")) + list(
            (segmented_dir / "images").glob("*.png")
        )
        seg_masks = list((segmented_dir / "masks").glob("*.jpg")) + list(
            (segmented_dir / "masks").glob("*.png")
        )
        print(
            f"\nThư mục Segmented: {len(seg_images)} ảnh + {len(seg_masks)} mặt nạ masks"
        )
        if len(seg_images) == 1000 and len(seg_masks) == 1000:
            print("   -> Tập Segmented hoàn toàn đầy đủ 1,000 cặp ảnh-mask!")
    else:
        print(f"Chưa tìm thấy thư mục: {segmented_dir} (có thể giải nén sau)")

    print("=" * 65)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_dir = os.path.join(project_root, "data", "raw")

    print(f"Đang kiểm tra thư mục: {raw_dir}\n")
    check_dataset_integrity(raw_dir)
