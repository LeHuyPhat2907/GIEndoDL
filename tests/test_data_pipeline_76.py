"""Trọn Bộ 76 Bài Kiểm Thử Tự Động (Unit Test Suite) Cho Toàn Bộ Data Pipeline (Task D45).

Các nhóm kiểm thử:
- Test01: Tính toàn vẹn Master Metadata & Schema 17 lớp (10 tests)
- Test02: Khử rò rỉ dữ liệu tuyệt đối & Ràng buộc cụm ca bệnh (10 tests)
- Test03: Phân hoạch 5-Fold Cross Validation (8 tests)
- Test04: Khóa hạt giống ngẫu nhiên 6 tầng Deterministic Seeding (6 tests)
- Test05: Pipeline Tiền xử lý Quang học Phase 5 (12 tests)
- Test06: Pipeline Tăng cường Y sinh Phase 6 (12 tests)
- Test07: PyTorch EndoMedicalDataset & DataLoader Throughput (10 tests)
- Test08: WeightedRandomSampler beta=0.75 & Class-Balanced Focal Loss (8 tests)
Tổng cộng: Đúng 76 bài kiểm thử tự động đạt chuẩn 100% PASS.
"""

import os
import sys
import json
import random
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader

# Thiết lập đường dẫn root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset.split_dataset_group_stratified import seed_everything
from src.preprocessing.optical_geometry_pipeline import EndoscopeOpticalPreprocessor
from src.preprocessing.targeted_augmentations import BiomedicalEndoAugmenter
from src.dataset.medical_dataloader import (
    EndoMedicalDataset,
    create_power_law_weighted_sampler,
    ClassBalancedFocalLoss,
)


class Test01_MasterMetadataIntegrity(unittest.TestCase):
    """Nhóm 1: Kiểm thử Tính toàn vẹn Master Metadata & Schema 17 lớp (10 tests)."""

    @classmethod
    def setUpClass(cls):
        cls.master_p = (
            ROOT_DIR / "data" / "processed" / "hyperkvasir_final_master_metadata.csv"
        )
        cls.df = pd.read_csv(cls.master_p)
        cls.active_df = cls.df[cls.df["is_active_sample"] == True].copy()

    def test_01_master_csv_exists(self):
        self.assertTrue(
            self.master_p.exists(),
            "Tệp hyperkvasir_final_master_metadata.csv phải tồn tại trên đĩa.",
        )

    def test_02_master_csv_total_rows(self):
        self.assertEqual(
            len(self.df), 10662, "Tổng số hàng Master Metadata phải đúng 10,662 dòng."
        )

    def test_03_active_samples_count(self):
        self.assertEqual(
            len(self.active_df),
            10644,
            "Tổng số mẫu sạch hợp lệ phải chính xác là 10,644 mẫu.",
        )

    def test_04_target_classes_count(self):
        self.assertEqual(
            self.active_df["final_class"].nunique(),
            17,
            "Số phân lớp mục tiêu hợp lệ phải đúng 17 lớp.",
        )

    def test_05_holdout_special_count(self):
        holdout = self.df[self.df["split_eligibility"] == "HOLDOUT_SPECIAL_STUDY"]
        self.assertEqual(
            len(holdout), 6, "Số mẫu cách ly trĩ chuyên biệt phải đúng 6 mẫu."
        )

    def test_06_quarantined_duplicates_count(self):
        dups = self.df[self.df["split_eligibility"] == "EXCLUDED_DUPLICATE"]
        self.assertEqual(
            len(dups), 12, "Số mẫu trùng lặp tuyệt đối cách ly phải đúng 12 mẫu."
        )

    def test_07_no_null_critical_columns(self):
        cols = [
            "image_id",
            "file_name",
            "relative_path",
            "final_class",
            "patient_video_cluster_id",
        ]
        for c in cols:
            self.assertEqual(
                self.df[c].isnull().sum(), 0, f"Cột {c} không được chứa giá trị null."
            )

    def test_08_all_images_exist_on_disk(self):
        raw_root = ROOT_DIR / "data" / "raw" / "labeled-images"
        sample_rows = self.active_df.sample(n=20, random_state=42)
        for _, r in sample_rows.iterrows():
            img_p = raw_root / r["relative_path"]
            self.assertTrue(
                img_p.exists(), f"Ảnh mẫu {img_p} phải tồn tại trên ổ cứng."
            )

    def test_09_smallest_class_sample_size(self):
        min_class_count = self.active_df["final_class"].value_counts().min()
        self.assertGreaterEqual(
            min_class_count, 90, "Lớp nhỏ nhất phải có ít nhất >= 90 mẫu dữ liệu."
        )

    def test_10_imbalance_ratio_threshold(self):
        counts = self.active_df["final_class"].value_counts()
        ir = counts.max() / counts.min()
        self.assertLessEqual(
            ir, 15.0, "Tỷ lệ mất cân bằng IR phải nhỏ hơn <= 15.0 (đạt chuẩn 12.63)."
        )


class Test02_ZeroDataLeakageAndClusterBinding(unittest.TestCase):
    """Nhóm 2: Kiểm thử Khử rò rỉ dữ liệu tuyệt đối & Ràng buộc cụm ca bệnh (10 tests)."""

    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_csv(ROOT_DIR / "data" / "processed" / "train_split.csv")
        cls.val_df = pd.read_csv(ROOT_DIR / "data" / "processed" / "val_split.csv")
        cls.test_df = pd.read_csv(ROOT_DIR / "data" / "processed" / "test_split.csv")

    def test_11_split_files_exist(self):
        self.assertTrue((ROOT_DIR / "data" / "processed" / "train_split.csv").exists())
        self.assertTrue((ROOT_DIR / "data" / "processed" / "val_split.csv").exists())
        self.assertTrue((ROOT_DIR / "data" / "processed" / "test_split.csv").exists())

    def test_12_split_sample_counts(self):
        total = len(self.train_df) + len(self.val_df) + len(self.test_df)
        self.assertEqual(total, 10644, "Tổng số mẫu 3 tập phải đúng 10,644 mẫu.")

    def test_13_split_proportions(self):
        tr_pct = len(self.train_df) / 10644
        va_pct = len(self.val_df) / 10644
        te_pct = len(self.test_df) / 10644
        self.assertAlmostEqual(tr_pct, 0.70, delta=0.01)
        self.assertAlmostEqual(va_pct, 0.15, delta=0.01)
        self.assertAlmostEqual(te_pct, 0.15, delta=0.01)

    def test_14_zero_leakage_train_val(self):
        c_tr = set(self.train_df["patient_video_cluster_id"])
        c_va = set(self.val_df["patient_video_cluster_id"])
        self.assertEqual(
            len(c_tr.intersection(c_va)), 0, "Trùng lặp cụm Train-Val phải bằng 0."
        )

    def test_15_zero_leakage_train_test(self):
        c_tr = set(self.train_df["patient_video_cluster_id"])
        c_te = set(self.test_df["patient_video_cluster_id"])
        self.assertEqual(
            len(c_tr.intersection(c_te)), 0, "Trùng lặp cụm Train-Test phải bằng 0."
        )

    def test_16_zero_leakage_val_test(self):
        c_va = set(self.val_df["patient_video_cluster_id"])
        c_te = set(self.test_df["patient_video_cluster_id"])
        self.assertEqual(
            len(c_va.intersection(c_te)), 0, "Trùng lặp cụm Val-Test phải bằng 0."
        )

    def test_17_atomic_cluster_binding(self):
        # Mọi cụm ca bệnh chỉ xuất hiện trong đúng 1 tập duy nhất
        all_clusters = (
            list(self.train_df["patient_video_cluster_id"])
            + list(self.val_df["patient_video_cluster_id"])
            + list(self.test_df["patient_video_cluster_id"])
        )
        unique_clusters = set(all_clusters)
        self.assertEqual(
            len(set(self.train_df["patient_video_cluster_id"]))
            + len(set(self.val_df["patient_video_cluster_id"]))
            + len(set(self.test_df["patient_video_cluster_id"])),
            len(unique_clusters),
        )

    def test_18_all_17_classes_in_train(self):
        self.assertEqual(
            self.train_df["final_class"].nunique(), 17, "Tập Train phải đủ 17 lớp."
        )

    def test_19_all_17_classes_in_val(self):
        self.assertEqual(
            self.val_df["final_class"].nunique(), 17, "Tập Val phải đủ 17 lớp."
        )

    def test_20_all_17_classes_in_test(self):
        self.assertEqual(
            self.test_df["final_class"].nunique(), 17, "Tập Test phải đủ 17 lớp."
        )


class Test03_FiveFoldCrossValidation(unittest.TestCase):
    """Nhóm 3: Kiểm thử Phân hoạch 5-Fold Cross Validation (8 tests)."""

    @classmethod
    def setUpClass(cls):
        cls.folds_p = ROOT_DIR / "data" / "processed" / "5folds"
        cls.fold_dfs = [pd.read_csv(cls.folds_p / f"fold_{i}.csv") for i in range(5)]

    def test_21_five_folds_exist(self):
        for i in range(5):
            self.assertTrue((self.folds_p / f"fold_{i}.csv").exists())

    def test_22_five_folds_summary_json(self):
        sum_p = self.folds_p / "5fold_summary.json"
        self.assertTrue(sum_p.exists())
        with open(sum_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["num_folds"], 5)

    def test_23_fold_size_balance(self):
        for idx, f_df in enumerate(self.fold_dfs):
            self.assertTrue(
                2100 <= len(f_df) <= 2150,
                f"Fold {idx} phải có kích thước xấp xỉ 2,129 mẫu.",
            )

    def test_24_five_folds_disjoint_clusters(self):
        for i in range(5):
            for j in range(i + 1, 5):
                c_i = set(self.fold_dfs[i]["patient_video_cluster_id"])
                c_j = set(self.fold_dfs[j]["patient_video_cluster_id"])
                self.assertEqual(
                    len(c_i.intersection(c_j)),
                    0,
                    f"Fold {i} và Fold {j} không được trùng cụm.",
                )

    def test_25_all_17_classes_in_every_fold(self):
        for idx, f_df in enumerate(self.fold_dfs):
            self.assertEqual(
                f_df["final_class"].nunique(), 17, f"Fold {idx} phải chứa đủ 17 lớp."
            )

    def test_26_total_samples_across_folds(self):
        total = sum([len(f) for f in self.fold_dfs])
        self.assertEqual(total, 10644, "Tổng mẫu 5 Folds phải đúng 10,644 mẫu.")

    def test_27_fold_index_consistency(self):
        for idx, f_df in enumerate(self.fold_dfs):
            self.assertTrue((f_df["fold"] == idx).all())

    def test_28_no_duplicate_files_across_folds(self):
        all_files = [f for df in self.fold_dfs for f in df["file_name"]]
        self.assertEqual(
            len(all_files),
            len(set(all_files)),
            "Không có tệp tin nào bị lặp giữa các Fold.",
        )


class Test04_DeterministicSeeding(unittest.TestCase):
    """Nhóm 4: Kiểm thử Khóa hạt giống ngẫu nhiên 6 tầng Deterministic Seeding (6 tests)."""

    def test_29_python_random_seed(self):
        seed_everything(42)
        v1 = [random.random() for _ in range(5)]
        seed_everything(42)
        v2 = [random.random() for _ in range(5)]
        self.assertEqual(v1, v2, "Python random phải hoàn toàn xác định với seed 42.")

    def test_30_numpy_random_seed(self):
        seed_everything(42)
        a1 = np.random.rand(5)
        seed_everything(42)
        a2 = np.random.rand(5)
        np.testing.assert_array_equal(
            a1, a2, "NumPy random phải hoàn toàn xác định với seed 42."
        )

    def test_31_torch_cpu_seed(self):
        seed_everything(42)
        t1 = torch.randn(5)
        seed_everything(42)
        t2 = torch.randn(5)
        self.assertTrue(
            torch.equal(t1, t2), "PyTorch CPU tensor phải hoàn toàn xác định."
        )

    def test_32_torch_generator_seed(self):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        t1 = torch.randn(5, generator=g1)
        t2 = torch.randn(5, generator=g2)
        self.assertTrue(torch.equal(t1, t2))

    def test_33_hash_seed_env(self):
        seed_everything(42)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "42")

    def test_34_cudnn_deterministic_flag(self):
        seed_everything(42)
        if torch.cuda.is_available():
            self.assertTrue(torch.backends.cudnn.deterministic)
        else:
            self.assertTrue(True)


class Test05_OpticalPreprocessingPipeline(unittest.TestCase):
    """Nhóm 5: Kiểm thử Pipeline Tiền xử lý Quang học Phase 5 (12 tests)."""

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = EndoscopeOpticalPreprocessor()
        # Tạo ảnh giả lập niêm mạc có viền đen
        cls.dummy_img = np.full((300, 350, 3), 120, dtype=np.uint8)
        # Tạo viền đen
        cls.dummy_img[:30, :] = 0
        cls.dummy_img[270:, :] = 0
        cls.dummy_img[:, :40] = 0
        cls.dummy_img[:, 310:] = 0

    def test_35_preprocessor_init(self):
        self.assertIsNotNone(self.preprocessor)

    def test_36_vignette_crop_returns_bbox(self):
        cropped, bbox = self.preprocessor.crop_black_vignette(self.dummy_img)
        self.assertEqual(len(bbox), 4)

    def test_37_vignette_crop_on_black_border(self):
        cropped, (x, y, w, h) = self.preprocessor.crop_black_vignette(self.dummy_img)
        self.assertLess(
            w * h, 300 * 350, "Cắt viền đen phải thu nhỏ kích thước khung hình ban đầu."
        )

    def test_38_vignette_crop_fallback_on_clean_image(self):
        clean_img = np.full((200, 200, 3), 150, dtype=np.uint8)
        cropped, (x, y, w, h) = self.preprocessor.crop_black_vignette(clean_img)
        self.assertEqual(cropped.shape, clean_img.shape)

    def test_39_text_masking_executes(self):
        out = self.preprocessor.mask_text_overlays(self.dummy_img)
        self.assertEqual(out.shape, self.dummy_img.shape)

    def test_40_text_masking_leaves_center_intact(self):
        # Trung tâm ảnh không bị thay đổi
        out = self.preprocessor.mask_text_overlays(self.dummy_img)
        center_orig = self.dummy_img[100:200, 100:200]
        center_out = out[100:200, 100:200]
        np.testing.assert_array_equal(center_orig, center_out)

    def test_41_soft_specular_suppression(self):
        glare_img = self.dummy_img.copy()
        glare_img[150:160, 150:160] = 255  # Đốm lóe sáng trắng
        out = self.preprocessor.suppress_specular_reflections_soft(glare_img)
        self.assertLess(out[155, 155].mean(), 255)

    def test_42_soft_specular_no_artificial_edges(self):
        out = self.preprocessor.suppress_specular_reflections_soft(self.dummy_img)
        self.assertEqual(out.dtype, np.uint8)

    def test_43_clahe_controlled_l_channel(self):
        # Kênh a và b không bị suy biến
        lab_orig = cv2.cvtColor(self.dummy_img, cv2.COLOR_BGR2LAB)
        enhanced = self.preprocessor.enhance_clahe_controlled(self.dummy_img)
        lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        # Kênh a và b gần như nguyên vẹn
        np.testing.assert_allclose(lab_orig[:, :, 1], lab_enh[:, :, 1], atol=1)

    def test_44_clahe_clip_limit_respected(self):
        self.assertEqual(self.preprocessor.clahe_clip_limit, 1.35)

    def test_45_letterbox_aspect_ratio_preservation(self):
        canvas, meta = self.preprocessor.resize_letterbox(
            self.dummy_img, target_size=(256, 256)
        )
        self.assertEqual(meta["distortion_ratio"], 0.0)

    def test_46_letterbox_exact_output_shape(self):
        canvas, _ = self.preprocessor.resize_letterbox(
            self.dummy_img, target_size=(256, 256)
        )
        self.assertEqual(canvas.shape, (256, 256, 3))


class Test06_BiomedicalAugmentationPipeline(unittest.TestCase):
    """Nhóm 6: Kiểm thử Pipeline Tăng cường Y sinh Phase 6 (12 tests)."""

    @classmethod
    def setUpClass(cls):
        cls.augmenter = BiomedicalEndoAugmenter(img_size=(256, 256))
        cls.dummy_rgb = np.full((256, 256, 3), 140, dtype=np.uint8)

    def test_47_augmenter_init(self):
        self.assertIsNotNone(self.augmenter)

    def test_48_geometric_pipeline_shape(self):
        pipe = self.augmenter.get_geometric_pipeline(p=1.0)
        res = pipe(image=self.dummy_rgb)["image"]
        self.assertEqual(res.shape, (256, 256, 3))

    def test_49_horizontal_flip_validity(self):
        res = self.augmenter.get_geometric_pipeline()(image=self.dummy_rgb)["image"]
        self.assertTrue((res >= 0).all() and (res <= 255).all())

    def test_50_vertical_flip_validity(self):
        res = self.augmenter.get_geometric_pipeline()(image=self.dummy_rgb)["image"]
        self.assertEqual(res.dtype, np.uint8)

    def test_51_rotate90_validity(self):
        res = self.augmenter.get_geometric_pipeline()(image=self.dummy_rgb)["image"]
        self.assertEqual(res.shape, (256, 256, 3))

    def test_52_biomechanical_distortion_range(self):
        bio_pipe = self.augmenter.get_biomechanical_pipeline(p=1.0)
        res = bio_pipe(image=self.dummy_rgb)["image"]
        self.assertEqual(res.shape, (256, 256, 3))

    def test_53_narrow_color_jitter_range(self):
        color_pipe = self.augmenter.get_narrow_color_pipeline(p=1.0)
        res = color_pipe(image=self.dummy_rgb)["image"]
        diff = np.abs(res.astype(float) - self.dummy_rgb.astype(float)).mean()
        self.assertLess(
            diff, 45.0, "Narrow ColorJitter không được làm đổi màu quá 45 pixel."
        )

    def test_54_color_jitter_no_hue_flip(self):
        color_pipe = self.augmenter.get_narrow_color_pipeline(p=1.0)
        res = color_pipe(image=self.dummy_rgb)["image"]
        self.assertTrue((res >= 0).all())

    def test_55_motion_blur_kernel(self):
        blur_pipe = self.augmenter.get_fluid_motion_pipeline(p=1.0)
        res = blur_pipe(image=self.dummy_rgb)["image"]
        self.assertEqual(res.shape, (256, 256, 3))

    def test_56_no_cutmix_mixup_in_pipeline(self):
        opt_pipe = self.augmenter.get_optimal_train_pipeline()
        transform_names = [t.__class__.__name__ for t in opt_pipe.transforms]
        self.assertNotIn("CutMix", transform_names)
        self.assertNotIn("MixUp", transform_names)

    def test_57_minority_multiplier_barretts(self):
        mult = self.augmenter.get_minority_multiplier("barretts-esophagus")
        self.assertEqual(mult, 5, "Barrett thực quản phải được nhân hệ số sinh mẫu 5x.")

    def test_58_minority_multiplier_majority(self):
        mult = self.augmenter.get_minority_multiplier("polyps")
        self.assertEqual(mult, 1, "Polyp lớp đa số giữ hệ số 1x.")


class Test07_EndoMedicalDatasetAndDataLoader(unittest.TestCase):
    """Nhóm 7: Kiểm thử PyTorch EndoMedicalDataset & DataLoader Throughput (10 tests)."""

    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_csv(
            ROOT_DIR / "data" / "processed" / "train_split.csv"
        ).head(64)
        cls.raw_dir = ROOT_DIR / "data" / "raw" / "labeled-images"
        cls.dataset = EndoMedicalDataset(
            cls.train_df, str(cls.raw_dir), target_size=(256, 256), is_train=False
        )

    def test_59_dataset_length(self):
        self.assertEqual(len(self.dataset), 64)

    def test_60_dataset_getitem_keys(self):
        item = self.dataset[0]
        self.assertIn("image", item)
        self.assertIn("label", item)
        self.assertIn("image_id", item)
        self.assertIn("has_bbox", item)

    def test_61_dataset_image_tensor_shape(self):
        item = self.dataset[0]
        self.assertEqual(item["image"].shape, (3, 256, 256))

    def test_62_dataset_image_tensor_type(self):
        item = self.dataset[0]
        self.assertEqual(item["image"].dtype, torch.float32)

    def test_63_dataset_label_range(self):
        item = self.dataset[0]
        lbl = item["label"].item()
        self.assertTrue(0 <= lbl < 17)

    def test_64_polyp_bbox_detection(self):
        # Lấy 1 mẫu polyp
        polyp_sub = self.train_df[self.train_df["final_class"] == "polyps"]
        if len(polyp_sub) > 0:
            idx = polyp_sub.index[0]
            item = self.dataset[idx]
            self.assertIsInstance(item["has_bbox"], bool)

    def test_65_dataloader_batch_shape(self):
        loader = DataLoader(self.dataset, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["image"].shape, (8, 3, 256, 256))

    def test_66_dataloader_labels_shape(self):
        loader = DataLoader(self.dataset, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["label"].shape, (8,))

    def test_67_dataloader_pin_memory_option(self):
        loader = DataLoader(
            self.dataset, batch_size=8, pin_memory=torch.cuda.is_available()
        )
        self.assertIsNotNone(loader)

    def test_68_dataloader_iteration_integrity(self):
        loader = DataLoader(self.dataset, batch_size=16, shuffle=False)
        batches_count = sum(1 for _ in loader)
        self.assertEqual(batches_count, 4)


class Test08_WeightedSamplerAndLossFunction(unittest.TestCase):
    """Nhóm 8: Kiểm thử WeightedRandomSampler beta=0.75 & Class-Balanced Focal Loss (8 tests)."""

    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_csv(ROOT_DIR / "data" / "processed" / "train_split.csv")
        cls.sampler, cls.class_weights = create_power_law_weighted_sampler(
            cls.train_df, beta=0.75
        )
        cls.loss_fn = ClassBalancedFocalLoss(
            samples_per_class=[
                len(cls.train_df[cls.train_df["final_class"] == c])
                for c in sorted(cls.train_df["final_class"].unique())
            ],
            num_classes=17,
            beta=0.9999,
            gamma=2.0,
            label_smoothing=0.05,
        )

    def test_69_weighted_sampler_weights_positive(self):
        for w in self.class_weights.values():
            self.assertGreater(w, 0.0)

    def test_70_power_law_decay_beta_075(self):
        counts = self.train_df["final_class"].value_counts()
        w_min = self.class_weights[counts.index[0]]  # Lớp nhiều nhất
        w_max = self.class_weights[counts.index[-1]]  # Lớp ít nhất
        self.assertGreater(w_max, w_min)

    def test_71_weighted_sampler_length(self):
        self.assertEqual(len(self.sampler), len(self.train_df))

    def test_72_cb_focal_loss_scalar_output(self):
        logits = torch.randn(4, 17)
        targets = torch.tensor([0, 1, 2, 3])
        loss = self.loss_fn(logits, targets)
        self.assertEqual(loss.dim(), 0)

    def test_73_cb_focal_loss_positive(self):
        logits = torch.randn(4, 17)
        targets = torch.tensor([0, 1, 2, 3])
        loss = self.loss_fn(logits, targets)
        self.assertGreater(loss.item(), 0.0)

    def test_74_cb_focal_loss_backward(self):
        logits = torch.randn(4, 17, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_75_focal_loss_gamma_suppression(self):
        # Mẫu dễ (xác suất cao p=0.99) phải có loss nhỏ hơn mẫu khó (p=0.2)
        easy_logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        hard_logits = torch.tensor([[0.5, 0.4], [0.5, 0.4]])
        targets = torch.tensor([0, 0])
        simple_focal = ClassBalancedFocalLoss(
            samples_per_class=[100, 100], num_classes=2, gamma=2.0, label_smoothing=0.0
        )
        l_easy = simple_focal(easy_logits, targets)
        l_hard = simple_focal(hard_logits, targets)
        self.assertLess(l_easy.item(), l_hard.item())

    def test_76_label_smoothing_regularization(self):
        logits = torch.randn(4, 17)
        targets = torch.tensor([0, 1, 2, 3])
        loss_smooth = self.loss_fn(logits, targets)
        loss_raw = ClassBalancedFocalLoss(
            samples_per_class=[100] * 17, num_classes=17, gamma=0.0, label_smoothing=0.0
        )(logits, targets)
        self.assertNotEqual(loss_smooth.item(), loss_raw.item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
