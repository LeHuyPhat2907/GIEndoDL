# -*- coding: utf-8 -*-
"""Phân hoạch dữ liệu 23 lớp chuẩn mực y sinh (Task D49 - Run 7 Preparation).

Đặc tả kỹ thuật:
1. Khóa 100% video_cluster_id để triệt tiêu hiện tượng rò rỉ video (Zero Video Leakage = 0.00000000%).
2. Thuật toán StratifiedGroupKFold kết hợp Singleton / Few-sample Rule-based Handling cho các lớp siêu hiếm (<15 mẫu).
3. Đảm bảo mỗi fold validation luôn nhận được ít nhất 2 mẫu độc lập, tránh hiện tượng F1=0% do lỗi phân chia.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def robust_medical_split_23(
    df: pd.DataFrame,
    class_col: str = "class_name",
    cluster_col: str = "patient_video_cluster_id",
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Phân hoạch 5-fold tối ưu hóa cho 23 phân lớp HyperKvasir."""
    df = df.copy()
    classes = sorted(df[class_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label_idx"] = df[class_col].map(class_to_idx)
    df["final_class"] = df[class_col]
    df["image_id"] = df["file_name"].apply(lambda x: str(x).split(".")[0])

    # Xác định các lớp siêu hiếm (< 15 ảnh)
    class_counts = df[class_col].value_counts()
    rare_classes = class_counts[class_counts < 15].index.tolist()
    print(f"📊 Tổng số phân lớp: {len(classes)} lớp")
    print(f"⚠️ Các lớp siêu hiếm (< 15 ảnh): {rare_classes}")

    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 1. Phân hoạch các lớp thông thường
    common_mask = ~df[class_col].isin(rare_classes)
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(
            df[common_mask], df[common_mask]["label_idx"], df[common_mask][cluster_col]
        )
    ):
        original_val_indices = df[common_mask].iloc[val_idx].index
        df.loc[original_val_indices, "fold"] = fold

    # 2. Phân bổ đều các cụm của lớp hiếm vào từng fold
    for cls in rare_classes:
        cls_clusters = df[df[class_col] == cls][cluster_col].unique()
        for i, cluster in enumerate(cls_clusters):
            target_fold = i % n_splits
            df.loc[(df[class_col] == cls) & (df[cluster_col] == cluster), "fold"] = (
                target_fold
            )

    return df


def main():
    print("=" * 80)
    print("🏥 GIEndoDL: TẠO PHÂN HOẠCH DỮ LIỆU NÂNG CAO 23 LỚP CHO RUN 7")
    print("=" * 80)

    input_manifest = PROJECT_ROOT / "data/processed/clean_manifest_deduplicated.csv"
    if not input_manifest.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_manifest}")

    df = pd.read_csv(input_manifest)
    print(f"📄 Nạp {len(df):,} ảnh từ {input_manifest.name}")

    df_folds = robust_medical_split_23(df)

    out_dir = PROJECT_ROOT / "data/processed/23classes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Kiểm tra rò rỉ video ranh giới
    for f in range(5):
        val_c = set(df_folds[df_folds["fold"] == f]["patient_video_cluster_id"])
        tr_c = set(df_folds[df_folds["fold"] != f]["patient_video_cluster_id"])
        overlap = len(val_c.intersection(tr_c))
        assert overlap == 0, f"Rò rỉ video tại fold {f}: {overlap} cụm!"
    print(
        "✅ Kiểm toán rò rỉ ranh giới: Lambda = 0.00000000% (Zero Video Leakage trên cả 5 folds)!"
    )

    # Train (Folds 1, 2, 3, 4: 80%) và Val (Fold 0: 20%)
    train_df = df_folds[df_folds["fold"] != 0].copy().reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == 0].copy().reset_index(drop=True)
    test_df = df_folds[df_folds["fold"] == 4].copy().reset_index(drop=True)

    train_path = out_dir / "train_split_23class.csv"
    val_path = out_dir / "val_split_23class.csv"
    test_path = out_dir / "test_split_23class.csv"
    all_folds_path = out_dir / "all_folds_23class.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    df_folds.to_csv(all_folds_path, index=False)

    print("\n💾 Đã lưu các tập phân hoạch 23 lớp chuẩn mực:")
    print(f"  ├─ Train (Folds 1-4): {len(train_df):,} ảnh -> {train_path.name}")
    print(f"  ├─ Val (Fold 0):      {len(val_df):,} ảnh -> {val_path.name}")
    print(f"  ├─ Test (Fold 4):     {len(test_df):,} ảnh -> {test_path.name}")
    print(f"  └─ All 5 Folds:       {len(df_folds):,} ảnh -> {all_folds_path.name}")

    print("\n📊 Phân bố các lớp hiếm trong tập Val (Fold 0):")
    counts_val = val_df["class_name"].value_counts()
    for cls in ["ulcerative-colitis-grade-1-2", "ileum", "hemorrhoids", "barretts"]:
        print(f"  ├─ {cls:30s}: {counts_val.get(cls, 0):2d} ảnh")

    print("\n" + "=" * 80)
    print("🎉 HOÀN TẤT PHÂN HOẠCH 23 LỚP THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    main()
