"""Script kiểm thử trực quan hóa 6 biến thể Data Augmentation trên ảnh nội soi."""

import os
from pathlib import Path
import sys
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Thiết lập đường dẫn root an toàn
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.preprocessing.crop_roi import EndoscopeROIExtractor
except ImportError:
    from crop_roi import EndoscopeROIExtractor


def run_augmentation_demo(raw_dir: str, metadata_path: str, fig_dir: str, doc_dir: str):
    raw_path = Path(raw_dir) / "labeled-images"
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    if not Path(metadata_path).exists():
        print(f"❌ Không tìm thấy metadata: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    roi_extractor = EndoscopeROIExtractor()

    print("=" * 75)
    print("🔄 ĐANG KIỂM THỬ PIPELINE DATA AUGMENTATION HÌNH HỌC (ALBUMENTATIONS)...")
    print("=" * 75)

    # Chọn 2 ca bệnh điển hình: 1 polyp và 1 viêm đại tràng
    sample_classes = ["polyps", "ulcerative-colitis-grade-2"]
    sample_rows = []
    for cls in sample_classes:
        sub = df[df["class_name"] == cls]
        if len(sub) > 0:
            sample_rows.append(sub.iloc[0])

    # Định nghĩa 5 phép biến đổi đơn lẻ để trực quan hóa
    t_hflip = A.HorizontalFlip(p=1.0)
    t_vflip = A.VerticalFlip(p=1.0)
    t_rotate = A.Rotate(limit=(35, 35), border_mode=cv2.BORDER_REFLECT, p=1.0)
    t_crop = A.RandomResizedCrop(
        size=(224, 224), scale=(0.75, 0.75), interpolation=cv2.INTER_CUBIC, p=1.0
    )
    t_pipeline = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.06,
                scale_limit=0.1,
                rotate_limit=25,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0,
            ),
            A.Resize(height=224, width=224, interpolation=cv2.INTER_CUBIC),
        ]
    )

    fig, axes = plt.subplots(2, 6, figsize=(22, 8))

    for idx, row in enumerate(sample_rows):
        img_p = raw_path / row["relative_path"]
        raw_bgr = cv2.imread(str(img_p))

        if raw_bgr is None:
            continue

        cropped_bgr = roi_extractor.crop_roi(raw_bgr)
        base_rgb = cv2.cvtColor(cv2.resize(cropped_bgr, (224, 224)), cv2.COLOR_BGR2RGB)

        # Tạo 5 biến thể
        aug_hflip = t_hflip(image=base_rgb)["image"]
        aug_vflip = t_vflip(image=base_rgb)["image"]
        aug_rotate = t_rotate(image=base_rgb)["image"]
        aug_crop = t_crop(image=base_rgb)["image"]
        aug_full = t_pipeline(image=base_rgb)["image"]

        images_to_show = [
            (base_rgb, f"Gốc: {row['class_name']}", "black"),
            (aug_hflip, "1. Lật ngang (Horizontal Flip)", "darkgreen"),
            (aug_vflip, "2. Lật dọc (Vertical Flip)", "darkgreen"),
            (aug_rotate, "3. Xoay tự do (Rotate +35 deg)", "darkblue"),
            (aug_crop, "4. Zoom sâu rãnh u (Random Crop)", "purple"),
            (aug_full, "5. Pipeline tổng hợp (Full Aug)", "darkred"),
        ]

        for col_idx, (im, title, color_txt) in enumerate(images_to_show):
            axes[idx, col_idx].imshow(im)
            axes[idx, col_idx].set_title(
                title, fontsize=10.5, fontweight="bold", color=color_txt
            )
            axes[idx, col_idx].axis("off")

        print(f"✅ Đã tạo 6 biến thể Augmentation cho mẫu {idx+1}: {row['class_name']}")

    plt.suptitle(
        "Medical Geometric Data Augmentation Benchmark: Rotation & Reflection Invariance (Albumentations)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    out_fig = fig_path / "18_basic_data_augmentations.png"
    plt.savefig(out_fig, dpi=120, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 75)
    print(f"✅ Đã lưu bức ảnh so sánh Data Augmentation tại: {out_fig}")
    print("=" * 75)

    # Xuất tài liệu kỹ thuật
    md_file = doc_path / "25_basic_data_augmentation_pipeline.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# 🔄 Báo cáo Kỹ thuật: Pipeline Tăng Cường Dữ Liệu Hình Học Y Khoa (Albumentations)\n\n"
        )
        f.write(
            "> **Module chính:** `src/preprocessing/augmentations.py` | **Hình minh họa:** `docs/figures/18_basic_data_augmentations.png`\n\n---\n\n"
        )

        f.write("## 1. Cơ sở Lý luận Y học (Domain Knowledge Rationale)\n\n")
        f.write(
            "1. **Tính Bất biến Không gian 3D (Spatial Invariance):** Ruột người có dạng hình trụ uốn lượn tự do. Khối polyp ở vị trí 12 giờ hay vị trí 6 giờ trong lòng ruột đều có chung bản chất bệnh lý. "
            "Do đó, các phép lật (Flip) và xoay (Rotate) hoàn toàn an toàn và phản ánh đúng thao tác xoay ống soi thực tế của bác sĩ.\n"
        )
        f.write(
            "2. **Tăng cường Khả năng Hội tụ cho Lớp Hiếm:** Giúp các lớp thiểu số (như Barretts chỉ có 41 ảnh, Trĩ chỉ có 6 ảnh) được nhân bản góc nhìn, chống lại hiện tượng Overfitting khi train 100 epochs.\n"
        )

    print(f"✅ Đã lưu tài liệu nghiên cứu tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    raw_dir_path = os.path.join(project_root, "data", "raw")
    metadata_csv_path = os.path.join(
        project_root, "data", "processed", "hyperkvasir_master_metadata.csv"
    )
    figures_dir_path = os.path.join(project_root, "docs", "figures")
    research_dir_path = os.path.join(project_root, "docs", "research")

    run_augmentation_demo(
        raw_dir_path, metadata_csv_path, figures_dir_path, research_dir_path
    )
