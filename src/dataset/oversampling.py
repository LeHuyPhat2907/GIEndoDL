"""Module thực hiện Augmentation-based Oversampling cho các lớp thiểu số y tế (thay thế SMOTE)."""

from typing import Dict, List, Tuple
import albumentations as A
import cv2
import pandas as pd


class MedicalImageOversampler:
    """Tạo mẫu nhân tạo chất lượng cao cho các lớp thiểu số bằng chuỗi biến đổi hình thái học y tế."""

    def __init__(self, target_samples_per_class: int = 120):
        self.target_samples = target_samples_per_class
        # Pipeline tạo biến thể mô mềm và quang học
        self.aug_pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.15,
                    rotate_limit=45,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.8,
                ),
                A.ElasticTransform(
                    alpha=1.0, sigma=35, border_mode=cv2.BORDER_REFLECT, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.04,
                    p=0.7,
                ),
            ]
        )

    def generate_balanced_metadata(
        self, train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Tạo bảng metadata mới với các mẫu oversampled được đánh dấu rõ ràng."""
        oversampled_records: List[Dict] = []
        counts = train_df["class_name"].value_counts().to_dict()
        synthetic_stats: Dict[str, int] = {}

        # Duyệt qua từng lớp
        for class_name, count in counts.items():
            class_subset = train_df[train_df["class_name"] == class_name]

            # Giữ nguyên các mẫu gốc
            for _, row in class_subset.iterrows():
                rec = row.to_dict()
                rec["is_synthetic"] = False
                rec["synthetic_id"] = 0
                oversampled_records.append(rec)

            # Nếu số lượng mẫu ít hơn ngưỡng mục tiêu, tiến hành bù đắp
            if count < self.target_samples:
                needed = self.target_samples - count
                synthetic_stats[class_name] = needed
                subset_rows = class_subset.to_dict("records")

                for i in range(needed):
                    parent = subset_rows[i % len(subset_rows)]
                    syn_rec = parent.copy()
                    syn_rec["is_synthetic"] = True
                    syn_rec["synthetic_id"] = i + 1
                    syn_rec["filename"] = (
                        f"syn_{i+1}_{parent.get('filename', 'img.jpg')}"
                    )
                    oversampled_records.append(syn_rec)
            else:
                synthetic_stats[class_name] = 0

        df_balanced = pd.DataFrame(oversampled_records)
        return df_balanced, synthetic_stats
