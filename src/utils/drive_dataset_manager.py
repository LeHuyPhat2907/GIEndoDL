"""Module quản lý nén, đồng bộ Google Drive và giải nén đệm siêu tốc trên ổ cứng NVMe Colab."""

from pathlib import Path
import shutil
import zipfile


class DriveDatasetManager:
    """Quản lý đóng gói và giải nén dữ liệu cho Google Colab."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def pack_for_drive(self, output_zip_path: str) -> str:
        """Đóng gói thư mục ảnh và metadata CSV thành file zip duy nhất để tải lên Drive."""
        out_zip = Path(output_zip_path)
        out_zip.parent.mkdir(parents=True, exist_ok=True)

        labeled_dir = self.data_root / "raw" / "labeled-images"
        proc_dir = self.data_root / "processed"

        print(f"📦 Đang đóng gói dữ liệu vào file zip: {out_zip}...")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Nén các file CSV đã xử lý
            if proc_dir.exists():
                for csv_file in proc_dir.glob("*.csv"):
                    zipf.write(csv_file, arcname=f"processed/{csv_file.name}")

            # Nén thư mục ảnh có nhãn
            if labeled_dir.exists():
                for img_file in labeled_dir.rglob("*.jpg"):
                    rel_p = img_file.relative_to(self.data_root / "raw")
                    zipf.write(img_file, arcname=f"raw/{rel_p}")

        print(f"✅ Đóng gói hoàn tất: {out_zip.stat().st_size / (1024**2):.1f} MB")
        return str(out_zip)

    @staticmethod
    def extract_to_colab_nvme(
        drive_zip_path: str, colab_dest: str = "/content/data"
    ) -> bool:
        """Sao chép và giải nén dữ liệu vào ổ cứng NVMe cục bộ của Colab trong vài giây."""
        zip_p = Path(drive_zip_path)
        dest_p = Path(colab_dest)

        if not zip_p.exists():
            print(f"❌ Không tìm thấy file zip tại: {zip_p}")
            return False

        dest_p.mkdir(parents=True, exist_ok=True)
        local_zip = Path("/content") / zip_p.name

        print("⚡ Đang sao chép file zip từ Google Drive vào NVMe SSD Colab...")
        shutil.copyfile(zip_p, local_zip)

        print("🚀 Đang giải nén dữ liệu siêu tốc...")
        with zipfile.ZipFile(local_zip, "r") as zipf:
            zipf.extractall(dest_p)

        if local_zip.exists():
            local_zip.unlink()  # Xóa file zip để giải phóng RAM/ổ cứng

        print(f"✅ Dữ liệu đã sẵn sàng trên ổ SSD cục bộ tại: {dest_p}")
        return True
