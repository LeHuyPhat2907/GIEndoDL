"""Script kiểm tra tính toàn vẹn và trực quan hóa Kvasir-SEG Ground-Truth Masks."""

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def verify_kvasir_seg(raw_dir: str, output_fig_dir: str):
    raw_path = Path(raw_dir)
    seg_dir = raw_path / "segmented-images"
    if not seg_dir.exists():
        seg_dir = raw_path / "kvasir-seg"

    fig_path = Path(output_fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BÁO CÁO KIỂM TRA MẶT NẠ PHÂN ĐOẠN (KVASIR-SEG MASKS)")
    print("=" * 70)

    if not seg_dir.exists():
        print(f"Chưa tìm thấy thư mục Segmented tại: {seg_dir}")
        return

    images_dir = seg_dir / "images"
    masks_dir = seg_dir / "masks"

    images = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in [".jpg", ".png"]]
    )
    masks = sorted(
        [f for f in masks_dir.iterdir() if f.suffix.lower() in [".jpg", ".png"]]
    )

    print(f"Tổng số ảnh nội soi: {len(images):,} ảnh")
    print(f"Tổng số mặt nạ mask: {len(masks):,} masks")

    if len(images) == 1000 and len(masks) == 1000:
        print("Bộ dữ liệu Kvasir-SEG hoàn toàn đầy đủ 1,000 cặp ảnh-mask!")

    # 1. Kiểm tra ngẫu nhiên tính tương thích kích thước và giá trị nhị phân
    sample_img_path = images[0]
    sample_mask_path = masks_dir / sample_img_path.name
    if not sample_mask_path.exists():
        sample_mask_path = masks[0]

    img = cv2.imread(str(sample_img_path))
    mask = cv2.imread(str(sample_mask_path), cv2.IMREAD_GRAYSCALE)

    print(f"\nThông số mẫu ({sample_img_path.name}):")
    print(f"   - Kích thước ảnh gốc: {img.shape[1]}x{img.shape[0]} (W x H)")
    print(f"   - Kích thước mask:    {mask.shape[1]}x{mask.shape[0]} (W x H)")
    print(
        f"   - Giá trị pixel mask: Min={mask.min()}, Max={mask.max()} (Chuẩn nhị phân 0-255)"
    )

    # 2. Tạo hình ảnh minh họa Mask Overlay (Ảnh gốc + Mask + Vùng tổn thương)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colored_mask = np.zeros_like(img_rgb)
    colored_mask[:, :, 0] = mask  # Tô màu đỏ cho vùng polyp phát hiện trong mask

    overlay = cv2.addWeighted(img_rgb, 0.7, colored_mask, 0.3, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(
        f"Ảnh Nội soi Gốc\n({sample_img_path.name})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(
        "Ground-Truth Mask (Binary)\n(Vùng tổn thương màu trắng)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        "Lớp phủ Tổn thương (ROI Overlay)\n(Vùng polyp được khoanh đỏ)",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.tight_layout()
    output_img_path = fig_path / "02_kvasir_seg_sample.png"
    plt.savefig(output_img_path, dpi=300)
    plt.close()

    print(f"\n✅ Đã lưu hình ảnh trực quan mẫu tại: {output_img_path}")
    print("=" * 70)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_path = os.path.join(project_root, "data", "raw")
    figures_path = os.path.join(project_root, "docs", "figures")
    verify_kvasir_seg(raw_path, figures_path)
