# -*- coding: utf-8 -*-
"""Script tự động nén toàn bộ kết quả huấn luyện (Task #127) thành file zip duy nhất để tải về máy tính.

Bao gồm:
1. Trọng số mô hình tối ưu (checkpoints/best_model.pth & last_model.pth)
2. Báo cáo định lượng (training_history.csv, per_class_metrics.csv, train_optimal_model.json, .md)
3. Toàn bộ biểu đồ khoa học 300 DPI (đồ thị hội tụ, ma trận nhầm lẫn 23x23, bar chart 23 lớp)
"""

from pathlib import Path
import sys
import time
import zipfile

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def zip_training_artifacts(output_zip: Path = None) -> Path:
    if output_zip is None:
        if Path("/kaggle/working").exists():
            output_zip = Path("/kaggle/working/GIEndoDL_optimal_results.zip")
        else:
            output_zip = PROJECT_ROOT / "GIEndoDL_optimal_results.zip"

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("📦 BẮT ĐẦU ĐÓNG GÓI TOÀN BỘ KẾT QUẢ HUẤN LUYỆN VÀO FILE ZIP...")
    print(f"🎯 File đích: {output_zip}")
    print("=" * 85)

    start_time = time.time()
    collected_files = []

    # 1. Thu thập Checkpoints
    chk_candidates = [
        PROJECT_ROOT / "checkpoints" / "optimal_production",
        PROJECT_ROOT / "checkpoints",
        Path("/kaggle/working/GIEndoDL/checkpoints/optimal_production"),
        Path("/kaggle/working/GIEndoDL/checkpoints"),
        Path("/kaggle/working/checkpoints"),
        Path("/kaggle/working"),
    ]
    seen_names = set()
    for c_dir in chk_candidates:
        if c_dir.exists():
            for pth in c_dir.glob("*.pth"):
                if pth.name not in seen_names:
                    seen_names.add(pth.name)
                    collected_files.append((pth, f"checkpoints/{pth.name}"))

    # 2. Thu thập Benchmarks CSV / JSON / MD
    bench_candidates = [
        PROJECT_ROOT / "docs" / "benchmarks",
        Path("/kaggle/working/GIEndoDL/docs/benchmarks"),
        Path("/kaggle/working/metrics"),
    ]
    for b_dir in bench_candidates:
        if b_dir.exists():
            for f in b_dir.iterdir():
                if f.is_file() and f.suffix in [".csv", ".json", ".md"]:
                    arc = f"benchmarks/{f.name}"
                    if not any(x[1] == arc for x in collected_files):
                        collected_files.append((f, arc))

    # 3. Thu thập Figures PNG (300 DPI)
    fig_candidates = [
        PROJECT_ROOT / "docs" / "figures",
        Path("/kaggle/working/GIEndoDL/docs/figures"),
        Path("/kaggle/working/figures"),
    ]
    for f_dir in fig_candidates:
        if f_dir.exists():
            for f in f_dir.glob("*.png"):
                arc = f"figures/{f.name}"
                if not any(x[1] == arc for x in collected_files):
                    collected_files.append((f, arc))

    # 4. Tạo file zip
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for src_path, arc_path in collected_files:
            zipf.write(src_path, arcname=arc_path)
            f_size_kb = src_path.stat().st_size / 1024
            if f_size_kb > 1024:
                print(f"  • [Đã nén] {arc_path} ({f_size_kb / 1024:.2f} MB)")
            else:
                print(f"  • [Đã nén] {arc_path} ({f_size_kb:.1f} KB)")

    elapsed = time.time() - start_time
    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)

    print("=" * 85)
    print("🎉 ĐÓNG GÓI HOÀN TẤT THÀNH CÔNG!")
    print(f"📁 Tổng số file đóng gói: {len(collected_files)} files")
    print(f"💾 Dung lượng file zip:    {zip_size_mb:.2f} MB")
    print(f"⏱️ Thời gian nén:          {elapsed:.2f} giây")
    print(f"📍 Đường dẫn file:         {output_zip}")
    print("=" * 85)

    try:
        from IPython.display import FileLink, display

        print("\n👉 Bấm vào đường link dưới đây để tải trực tiếp file zip về máy tính:")
        display(FileLink(str(output_zip.name)))
    except Exception:
        pass

    return output_zip


if __name__ == "__main__":
    zip_training_artifacts()
