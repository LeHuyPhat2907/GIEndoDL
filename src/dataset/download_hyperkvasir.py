"""Script tự động tải và giải nén bộ dữ liệu HyperKvasir (labeled-images 10,662 ảnh).

Dành riêng cho môi trường Google Colab / Kaggle / Máy cá nhân:
- Tải trực tiếp từ máy chủ chính thức của Simula Research Laboratory (3.9 GB).
- Tự động hiển thị thanh tiến trình tải và giải nén.
- Tự động xóa file zip sau khi giải nén để tiết kiệm tối đa dung lượng ổ đĩa.
"""

from pathlib import Path
import sys
import urllib.request
import zipfile
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATASET_URL = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_hyperkvasir(dest_dir: Path = DATA_RAW_DIR):
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_labeled_dir = dest_dir / "labeled-images"

    # Kiểm tra nếu dữ liệu đã tồn tại
    if target_labeled_dir.exists():
        sub_dirs = [d for d in target_labeled_dir.iterdir() if d.is_dir()]
        if len(sub_dirs) >= 2:
            print(f"✅ Dữ liệu HyperKvasir đã tồn tại sẵn sàng tại: {target_labeled_dir}")
            return target_labeled_dir

    zip_path = dest_dir / "hyper-kvasir-labeled-images.zip"
    print("=" * 80)
    print("🌐 BẮT ĐẦU TẢI BỘ DỮ LIỆU HYPERKVASIR CHÍNH THỨC TỪ SIMULA (3.9 GB)...")
    print(f"🔗 URL: {DATASET_URL}")
    print(f"💾 Nơi lưu tạm: {zip_path}")
    print("=" * 80)

    try:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc="📥 Đang tải HyperKvasir"
        ) as t:
            urllib.request.urlretrieve(
                DATASET_URL, filename=str(zip_path), reporthook=t.update_to
            )

        print("\n📦 Đang giải nén bộ dữ liệu ảnh (labeled-images)...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            for member in tqdm(members, desc="📂 Đang giải nén", unit="file"):
                zip_ref.extract(member, dest_dir)

        print(f"✅ Giải nén hoàn tất vào: {dest_dir}")
    finally:
        # Xóa file zip để giải phóng 3.9 GB dung lượng đĩa
        if zip_path.exists():
            print("🧹 Đang dọn dẹp file zip tạm thời để tiết kiệm bộ nhớ...")
            zip_path.unlink()
            print("✨ Đã dọn dẹp xong!")

    print("=" * 80)
    print(f"🎉 BỘ DỮ LIỆU HYPERKVASIR ĐÃ SẴN SÀNG TẠI: {target_labeled_dir}")
    print("=" * 80 + "\n")
    return target_labeled_dir


if __name__ == "__main__":
    download_and_extract_hyperkvasir()
