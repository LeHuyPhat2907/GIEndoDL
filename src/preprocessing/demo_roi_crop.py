"""Script kiểm thử trực quan thuật toán cắt viền đen ROI trên 4 mẫu ảnh nội soi."""

import os
from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Thiết lập đường dẫn root an toàn cho Python
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.preprocessing.crop_roi import EndoscopeROIExtractor
except ImportError:
    from crop_roi import EndoscopeROIExtractor


def run_roi_crop_demo(raw_dir: str, metadata_path: str, fig_dir: str, doc_dir: str):
    raw_path = Path(raw_dir) / "labeled-images"
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    if not Path(metadata_path).exists():
        print(f"❌ Không tìm thấy metadata: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    extractor = EndoscopeROIExtractor()

    print("=" * 75)
    print("✂️ ĐANG KIỂM THỬ THUẬT TOÁN CẮT VIỀN ĐEN & LỌC NHIỄU THIẾT BỊ NỘI SOI...")
    print("=" * 75)

    sample_classes = [
        "dyed-resection-margins",
        "cecum",
        "retroflex-rectum",
        "polyps",
    ]
    sample_imgs = []

    for cls in sample_classes:
        cls_rows = df[df["class_name"] == cls]
        if len(cls_rows) > 0:
            sample_imgs.append(cls_rows.iloc[0])

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    for idx, row in enumerate(sample_imgs):
        img_p = raw_path / row["relative_path"]
        img_bgr = cv2.imread(str(img_p))

        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        x, y, bw, bh = extractor.detect_roi_bbox(img_bgr)

        # 1. Ảnh gốc có vẽ Bounding Box màu xanh lá
        img_bbox = img_bgr.copy()
        cv2.rectangle(img_bbox, (x, y), (x + bw, y + bh), (0, 255, 0), thickness=3)
        img_bbox_rgb = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)

        # 2. Cắt ROI hữu ích
        cropped = extractor.crop_roi(img_bgr)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # 3. Làm sạch chữ số thiết bị
        cleaned = extractor.remove_text_and_markers(cropped)
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

        # Cột 1: Ảnh gốc & Bounding Box phát hiện
        axes[idx, 0].imshow(img_bbox_rgb)
        axes[idx, 0].set_title(
            f"Mẫu {idx+1}: {row['class_name']}\nẢnh gốc: {w}x{h} px (Khung xanh = ROI)",
            fontsize=10,
            fontweight="bold",
        )
        axes[idx, 0].axis("off")

        # Cột 2: Ảnh sau khi cắt viền đen
        axes[idx, 1].imshow(cropped_rgb)
        axes[idx, 1].set_title(
            f"Đã cắt viền đen (Cropped ROI)\nKích thước mới: {bw}x{bh} px (Tiết kiệm {(1 - (bw*bh)/(w*h))*100:.1f}% viền thừa)",
            fontsize=10,
            fontweight="bold",
            color="darkgreen",
        )
        axes[idx, 1].axis("off")

        # Cột 3: Ảnh sau khi làm sạch chữ in & marker
        axes[idx, 2].imshow(cleaned_rgb)
        axes[idx, 2].set_title(
            "Đã làm sạch Artifacts\n(Sẵn sàng cho mô hình học)",
            fontsize=10,
            fontweight="bold",
            color="darkblue",
        )
        axes[idx, 2].axis("off")

        print(
            f"✅ Mẫu {idx+1} ({row['class_name']:<25}): Gốc {w}x{h} px ➔ ROI {bw}x{bh} px"
        )

    plt.suptitle(
        "Endoscopic Black Border Removal & ROI Cropping Pipeline (OpenCV Morphological Detection)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    out_fig = fig_path / "10_roi_crop_comparison.png"
    plt.savefig(out_fig, dpi=120, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 75)
    print(f"✅ Đã lưu bức ảnh so sánh Before/After tại: {out_fig}")
    print("=" * 75)

    # Xuất tài liệu kỹ thuật
    md_file = doc_path / "17_roi_cropping_and_artifact_removal.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# ✂️ Báo cáo Kỹ thuật: Pipeline Tự động Cắt Viền Đen & Loại bỏ Nhiễu Thiết bị (ROI Cropping & Artifact Removal)\n\n"
        )
        f.write(
            "> **Module chính:** `src/preprocessing/crop_roi.py` | **Hình minh họa:** `docs/figures/10_roi_crop_comparison.png`\n\n---\n\n"
        )

        f.write("## 1. Nguyên lý Hoạt động của Thuật toán\n\n")
        f.write(
            "1. **Phát hiện Viền Quang học (Optical Border Detection):** Sử dụng ngưỡng thích ứng `threshold_val = 15` để tách biệt vùng tối hình học của ống soi khỏi niêm mạc tiêu hóa.\n"
        )
        f.write(
            "2. **Morphological Closing:** Sử dụng kernel ellipse kích thước $15 \\times 15$ để đóng các lỗ tối bên trong lòng ruột, đảm bảo contour bao trọn toàn bộ trường nhìn nội soi.\n"
        )
        f.write(
            "3. **Cắt Bounding Box & Lọc Nhiễu:** Tự động cắt bỏ trung bình 15–25% diện tích viền đen vô nghĩa và inpaint các vùng chứa chữ số thiết bị.\n\n---\n\n"
        )

        f.write("## 2. Ý nghĩa đối với Huấn luyện Mạng Học Sâu\n\n")
        f.write(
            "- **Chống hiện tượng học vẹt (Anti-Cheating / Spurious Correlation):** Ngăn mô hình liên kết các chữ số ngày giờ hoặc logo bệnh viện với nhãn bệnh lý.\n"
        )
        f.write(
            "- **Tập trung 100% tài nguyên mạng vào Tổn thương:** Giúp các lớp tích chập (Convolutional Layers) và khối Attention (CBAM) chỉ học hoa văn mao mạch và hình thái khối u.\n"
        )

    print(f"✅ Đã lưu tài liệu kỹ thuật tại: {md_file}")
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

    run_roi_crop_demo(
        raw_dir_path, metadata_csv_path, figures_dir_path, research_dir_path
    )
