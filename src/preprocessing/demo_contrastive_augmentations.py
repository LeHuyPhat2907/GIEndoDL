"""Script kiểm thử và trực quan hóa 2 Augmented Views (v1, v2) cho SupCon trên các ca bệnh nội soi."""

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


def run_contrastive_demo(raw_dir: str, metadata_path: str, fig_dir: str, doc_dir: str):
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
    print(
        "🔁 ĐANG KIỂM THỬ PIPELINE TẠO 2 GÓC NHÌN (TWO VIEWS) CHO CONTRASTIVE LEARNING (SUPCON)..."
    )
    print("=" * 75)

    # Chọn 3 mẫu bệnh lý điển hình: Polyp, Viêm loét đại tràng, Thực quản Barretts
    test_classes = ["polyps", "ulcerative-colitis-grade-2", "barretts"]
    test_rows = []
    for cls in test_classes:
        sub = df[df["class_name"] == cls]
        if len(sub) > 0:
            test_rows.append(sub.iloc[0])

    # Định nghĩa pipeline biến đổi thị giác mạnh (Strong Augmentation)
    strong_transform = A.Compose(
        [
            A.RandomResizedCrop(
                size=(224, 224),
                scale=(0.55, 0.95),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1.0,
                        sigma=40,
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0,
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.15,
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.20, contrast=0.20, saturation=0.20, hue=0.04, p=0.8
            ),
            A.ToGray(p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ]
    )

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for idx, row in enumerate(test_rows):
        img_p = raw_path / row["relative_path"]
        raw_bgr = cv2.imread(str(img_p))

        if raw_bgr is None:
            continue

        cropped_bgr = roi_extractor.crop_roi(raw_bgr)
        base_rgb = cv2.cvtColor(cv2.resize(cropped_bgr, (224, 224)), cv2.COLOR_BGR2RGB)

        # Tạo 2 góc nhìn độc lập (View 1 & View 2)
        view1 = strong_transform(image=base_rgb)["image"]
        view2 = strong_transform(image=base_rgb)["image"]

        # Cột 1: Ảnh gốc
        axes[idx, 0].imshow(base_rgb)
        axes[idx, 0].set_title(
            f"Ảnh Gốc {idx+1}: {row['class_name']}\n(Original Anchor Image)",
            fontsize=11,
            fontweight="bold",
        )
        axes[idx, 0].axis("off")

        # Cột 2: View 1
        axes[idx, 1].imshow(view1)
        axes[idx, 1].set_title(
            "Góc Nhìn 1 (Augmented View 1)\n✅ Zoom rãnh u / Biến dạng mô",
            fontsize=10.5,
            fontweight="bold",
            color="darkblue",
        )
        axes[idx, 1].axis("off")

        # Cột 3: View 2
        axes[idx, 2].imshow(view2)
        axes[idx, 2].set_title(
            "Góc Nhìn 2 (Augmented View 2)\n✅ Xoay / Lệch màu / Mờ Gaussian / Gray",
            fontsize=10.5,
            fontweight="bold",
            color="darkgreen",
        )
        axes[idx, 2].axis("off")

        # Cột 4: Mô tả mục tiêu tối ưu Contrastive Loss
        axes[idx, 3].text(
            0.5,
            0.5,
            "🎯 MỤC TIÊU SUPCON:\n\n"
            "1. Kéo gần Vector Embedding\n   của View 1 & View 2 lại gần nhau.\n\n"
            "2. Đẩy xa các ảnh thuộc\n   lớp bệnh lý khác.\n\n"
            "👉 Giúp AI học bất biến với\n   góc nhìn và độ mờ!",
            fontsize=11,
            va="center",
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#ecf0f1",
                edgecolor="#bdc3c7",
                lw=1.5,
            ),
        )
        axes[idx, 3].axis("off")

        print(
            f"✅ Đã tạo thành công cặp View 1 & View 2 cho mẫu {idx+1}: {row['class_name']}"
        )

    plt.suptitle(
        "Supervised Contrastive Learning (SupCon) Two-View Strong Augmentation Pipeline",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    out_fig = fig_path / "22_contrastive_learning_augmentations.png"
    plt.savefig(out_fig, dpi=120, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 75)
    print(f"✅ Đã lưu bức ảnh so sánh Contrastive Augmentation tại: {out_fig}")
    print("=" * 75)

    # Xuất tài liệu kỹ thuật
    md_file = doc_path / "29_contrastive_augmentation_two_view_pipeline.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# 🔁 Báo cáo Kỹ thuật: Pipeline Tăng Cường Hai Góc Nhìn Cho Học Tương Phản (SupCon Two-View Augmentation)\n\n"
        )
        f.write(
            "> **Module chính:** `src/preprocessing/contrastive_augmenter.py` | **Hình minh họa:** `docs/figures/22_contrastive_learning_augmentations.png`\n\n---\n\n"
        )

        f.write("## 1. Cơ sở Phương pháp Luận của Contrastive Learning\n\n")
        f.write(
            "Trong kiến trúc **Supervised Contrastive Learning (SupCon)** ở Giai đoạn 9, mục tiêu của hàm mất mát là kéo gần biểu diễn vector không gian (Embedding Vectors) "
            "của 2 góc nhìn $(v_1, v_2)$ cùng nhãn bệnh lý và đẩy xa các góc nhìn khác nhãn.\n\n"
        )
        f.write("## 2. Vì sao Cần Kỹ thuật Random Grayscale & Gaussian Blur?\n\n")
        f.write(
            "1. **Random Grayscale ($p=0.2$):** Vì ảnh nội soi có kênh Đỏ chiếm tới 57%, nếu không có Grayscale, bộ mã hóa (Encoder) sẽ có xu hướng học mẹo bằng cách so khớp màu đỏ thay vì học hoa văn vi mạch. "
            "Grayscale ép mạng phải trích xuất các đặc trưng hình thái học sâu sắc.\n"
        )
        f.write(
            "2. **Gaussian Blur ($p=0.5$):** Loại bỏ bẫy so khớp tần số cao của các đốm lóa sáng, giúp vector nhúng tập trung vào bản chất rãnh u.\n"
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

    run_contrastive_demo(
        raw_dir_path, metadata_csv_path, figures_dir_path, research_dir_path
    )
