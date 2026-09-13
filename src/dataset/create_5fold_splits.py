"""Script tự động tạo bộ chia Stratified 5-Fold Cross Validation chuẩn Y khoa cho HyperKvasir."""

import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Thiết lập đường dẫn root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def generate_stratified_5folds(seed: int = 42, n_splits: int = 5):
    print("=" * 80)
    print(
        f"🔄 KHỞI TẠO BỘ CHIA STRATIFIED {n_splits}-FOLD CROSS VALIDATION (SEED {seed})..."
    )
    print("=" * 80)

    proc_dir = ROOT_DIR / "data" / "processed"
    out_dir = proc_dir / "5folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tập hợp toàn bộ 10,662 ảnh có nhãn từ 3 file split trước đó
    split_files = ["train_split.csv", "val_split.csv", "test_split.csv"]
    dfs = []
    for f in split_files:
        p = proc_dir / f
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        # Nếu không có split files, đọc file metadata tổng
        meta_p = proc_dir / "hyperkvasir_metadata.csv"
        assert meta_p.exists(), f"❌ Không tìm thấy metadata tại {meta_p}"
        full_df = pd.read_csv(meta_p)
        if "class_name" not in full_df.columns and "label_23" in full_df.columns:
            full_df["class_name"] = full_df["label_23"]
    else:
        full_df = pd.concat(dfs, ignore_index=True)
        # Loại bỏ trùng lặp nếu có
        full_df = full_df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    total_samples = len(full_df)
    unique_classes = sorted(full_df["class_name"].unique())
    num_classes = len(unique_classes)

    print(f"📦 Tổng số mẫu dữ liệu: {total_samples} ảnh")
    print(f"🏷️ Tổng số lớp bệnh học: {num_classes} lớp")

    # 2. Khởi tạo Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    audit_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["class_name"])
    ):
        fold_folder = out_dir / f"fold_{fold_idx}"
        fold_folder.mkdir(parents=True, exist_ok=True)

        train_df = full_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = full_df.iloc[val_idx].copy().reset_index(drop=True)

        # Gán nhãn split cho rõ ràng
        train_df["fold"] = fold_idx
        train_df["split"] = "train"
        val_df["fold"] = fold_idx
        val_df["split"] = "val"

        # Lưu file CSV
        train_csv_path = fold_folder / "train.csv"
        val_csv_path = fold_folder / "val.csv"
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        print(
            f"  📁 Fold {fold_idx}: Train = {len(train_df)} ảnh ({len(train_df)/total_samples*100:.1f}%) | Val = {len(val_df)} ảnh ({len(val_df)/total_samples*100:.1f}%)"
        )

        # Kiểm toán phân bố lớp trong Fold
        train_counts = train_df["class_name"].value_counts()
        val_counts = val_df["class_name"].value_counts()

        for c in unique_classes:
            audit_records.append(
                {
                    "fold": fold_idx,
                    "class_name": c,
                    "train_count": int(train_counts.get(c, 0)),
                    "val_count": int(val_counts.get(c, 0)),
                    "total_class_samples": int(
                        train_counts.get(c, 0) + val_counts.get(c, 0)
                    ),
                    "val_ratio": round(
                        float(
                            val_counts.get(c, 0)
                            / max(1, (train_counts.get(c, 0) + val_counts.get(c, 0)))
                        ),
                        3,
                    ),
                }
            )

    # 3. Xuất file báo cáo kiểm toán phân phối Stratification Audit
    audit_df = pd.DataFrame(audit_records)
    audit_path = out_dir / "5fold_distribution_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    summary_info = {
        "n_splits": n_splits,
        "seed": seed,
        "total_samples": total_samples,
        "num_classes": num_classes,
        "classes": unique_classes,
        "folds": {
            f"fold_{i}": {
                "train_samples": len(pd.read_csv(out_dir / f"fold_{i}" / "train.csv")),
                "val_samples": len(pd.read_csv(out_dir / f"fold_{i}" / "val.csv")),
            }
            for i in range(n_splits)
        },
    }
    with open(out_dir / "5fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_info, f, indent=4)

    print("=" * 80)
    print(f"✅ ĐÃ TẠO THÀNH CÔNG TRỌN BỘ 5-FOLD CROSS VALIDATION TẠI:\n   👉 {out_dir}")
    print(f"📊 Đã xuất bảng kiểm toán phân bố lớp tại: {audit_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_stratified_5folds(seed=42, n_splits=5)
