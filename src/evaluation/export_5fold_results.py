"""Script đọc kết quả 5-Fold đã huấn luyện từ ổ cứng để xuất bảng báo cáo Mean ± Std và vẽ Box Plot ngay lập tức."""

import json
from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.cross_validation_reporter import CrossValidationReporter


def export_existing_5fold_results():
    chk_base = ROOT_DIR / "models" / "checkpoints" / "resnet50_5folds"
    assert chk_base.exists(), f"❌ Không tìm thấy thư mục: {chk_base}"

    all_fold_metrics = []
    all_fold_histories = []

    print("=" * 80)
    print("📊 ĐANG THU THẬP KẾT QUẢ CỦA 5 FOLDS TỪ Ổ CỨNG...")
    print("=" * 80)

    for fold_idx in range(5):
        fold_dir = chk_base / f"fold_{fold_idx}"
        metrics_file = fold_dir / "fold_metrics.json"
        history_file = fold_dir / "history.csv"

        assert metrics_file.exists(), f"❌ Thiếu file: {metrics_file}"
        assert history_file.exists(), f"❌ Thiếu file: {history_file}"

        with open(metrics_file, "r", encoding="utf-8") as f:
            m = json.load(f)
            all_fold_metrics.append(m)

        h_df = pd.read_csv(history_file)
        all_fold_histories.append(h_df)
        print(f"  ✅ Fold {fold_idx}: Accuracy = {m['accuracy']}% | Macro F1 = {m['macro_f1']}% | Best Epoch = {m.get('best_epoch', 'N/A')}")

    print("=" * 80)
    print("🏆 TỔNG HỢP VÀ XUẤT BẢN BÁO CÁO KHOA HỌC CHUẨN MỰC...")
    print("=" * 80)

    reporter = CrossValidationReporter(
        model_name="ResNet-50 Run 4 (5-Fold CV)",
        cv_result_dir=str(chk_base),
    )
    summary_df = reporter.aggregate_metrics(all_fold_metrics)

    print("\n📋 BẢNG BÁO CÁO KHOA HỌC (NỘP GIẢNG VIÊN):")
    try:
        print(summary_df[["Chỉ số lâm sàng", "Mean ± Std", "Khoảng dao động [Min, Max]"]].to_markdown(index=False))
    except (ImportError, ModuleNotFoundError):
        print(summary_df[["Chỉ số lâm sàng", "Mean ± Std", "Khoảng dao động [Min, Max]"]].to_string(index=False))

    # Xuất biểu đồ hộp Box Plot và biểu đồ đường 5 Folds
    reporter.plot_boxplots(all_fold_metrics, "54_resnet50_5fold_boxplots.png")
    reporter.plot_5fold_learning_curves(all_fold_histories, "55_resnet50_5fold_learning_curves.png")

    out_csv = ROOT_DIR / "data" / "processed" / "resnet50_5fold_summary_report.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\n💾 Đã lưu bảng báo cáo CSV tại: {out_csv}")
    print("🎉 TẤT CẢ KẾT QUẢ ĐÃ HOÀN TẤT TRỌN VẸN!")


if __name__ == "__main__":
    export_existing_5fold_results()
