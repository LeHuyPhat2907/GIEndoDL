"""Script nén tập dữ liệu HyperKvasir thành hyperkvasir_data.zip phục vụ Google Colab."""

from pathlib import Path
import shutil
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_ZIP = DATA_DIR / "hyperkvasir_data"

print("=" * 75)
print(f"📦 ĐANG NÉN DỮ LIỆU TỪ: {RAW_DIR / 'labeled-images'}")
print("⏳ Vui lòng chờ 1 - 2 phút (tùy tốc độ ổ cứng của bạn)...")
print("=" * 75)

start_time = time.time()
shutil.make_archive(
    base_name=str(OUTPUT_ZIP),
    format="zip",
    root_dir=str(RAW_DIR),
    base_dir="labeled-images",
)
elapsed = time.time() - start_time

zip_file = DATA_DIR / "hyperkvasir_data.zip"
size_mb = zip_file.stat().st_size / (1024 * 1024)

print(f"✅ ĐÃ NÉN THÀNH CÔNG: {zip_file}")
print(f"📊 Dung lượng file nén: {size_mb:.2f} MB (~{size_mb / 1024:.2f} GB)")
print(f"⏱️ Thời gian nén: {elapsed:.1f} giây")
print("=" * 75)
