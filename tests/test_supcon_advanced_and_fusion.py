"""Unit tests cho các Mô-đun Nâng Cao: SupCon Pretraining, Fine-tuning, Joint Training, t-SNE/Clustering và Handcrafted Fusion (Tasks #112-#117)."""

# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.evaluation.evaluate_clustering_metrics import compute_geometric_distances
from src.models.handcrafted_feature_extractor import HandcraftedFeatureExtractor
from src.models.hybrid_handcrafted_fusion import HybridHandcraftedDeepFusionModel
from src.training.joint_supcon_ce_training import (
    JointHybridSupConModel,
    MultiTaskJointLoss,
)
from src.training.stage1_supcon_pretraining import (
    Stage1SupConTrainer,
    Stage1TrainingArgs,
)
from src.training.stage2_classifier_finetuning import (
    Stage2ClassifierFineTuner,
    Stage2FineTuneArgs,
)
from src.visualization.visualize_embedding_space import generate_synthetic_embeddings


class TestStage1AndStage2Training:
    """Kiểm thử chu trình tiền huấn luyện Stage 1 và tinh chỉnh Stage 2 (Tasks #112, #113)."""

    def test_stage1_pretraining_step(self):
        """Kiểm tra hoạt động của Stage1SupConTrainer và tính lũy gradient."""
        args = Stage1TrainingArgs(
            epochs=2, physical_batch_size=2, gradient_accumulation_steps=1
        )
        trainer = Stage1SupConTrainer(args=args)

        loss = trainer.train_epoch_simulated(epoch=1, total_steps_per_epoch=1)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_stage2_finetuning_step(self):
        """Kiểm tra hoạt động của Stage2ClassifierFineTuner với Focal Cross-Entropy."""
        args = Stage2FineTuneArgs(epochs=2, batch_size=2)
        tuner = Stage2ClassifierFineTuner(args=args)

        loss = tuner.train_epoch_simulated(epoch=1)
        assert isinstance(loss, float)
        assert loss > 0.0


class TestJointTraining:
    """Kiểm thử mô hình học đa nhiệm SupCon + CrossEntropy (Task #114)."""

    def test_joint_model_and_multi_task_loss(self):
        """Kiểm tra forward pass và loss của Joint Hybrid Model."""
        model = JointHybridSupConModel(num_classes=23, proj_dim=128)
        criterion = MultiTaskJointLoss(lambda_supcon=0.5, lambda_ce=1.0)

        # Batch 2 ảnh, mỗi ảnh có 2 views = 4 ảnh
        dummy_input = torch.randn(4, 3, 224, 224)
        dummy_labels = torch.tensor([0, 1])

        logits, z = model(dummy_input)
        assert logits.shape == (4, 23)
        assert z.shape == (4, 128)

        # z_views: [2, 2, 128]
        z_views = z.view(2, 2, 128)
        total_loss, l_sc, l_ce = criterion(logits, z_views, dummy_labels)

        assert total_loss.item() > 0.0
        assert l_sc >= 0.0
        assert l_ce > 0.0

        # Kiểm tra gradient lan truyền
        total_loss.backward()
        assert model.projection_head.projection_mlp[0].weight.grad is not None


class TestClusteringAndEmbeddingMetrics:
    """Kiểm thử các phép đo hình học khoảng cách nội và liên lớp (Tasks #115, #116)."""

    def test_synthetic_embeddings_and_distances(self):
        """Kiểm tra tính toán khoảng cách nội/liên lớp và tính tách biệt sau SupCon."""
        X_pre, X_post, y = generate_synthetic_embeddings(
            n_samples_per_class=10, n_classes=4, dim=64, seed=42
        )
        class_names = ["Lớp 0", "Lớp 1", "Lớp 2", "Lớp 3"]

        geo_pre = compute_geometric_distances(X_pre, y, class_names)
        geo_post = compute_geometric_distances(X_post, y, class_names)

        # Khoảng cách nội lớp sau SupCon phải nhỏ hơn hoặc bằng
        assert (
            geo_post["mean_intra_class_compactness"]
            <= geo_pre["mean_intra_class_compactness"]
        )
        # Tỷ số phân tách sau SupCon phải cao hơn
        assert (
            geo_post["separation_to_compactness_ratio"]
            >= geo_pre["separation_to_compactness_ratio"]
        )


class TestHandcraftedFeatureExtractionAndFusion:
    """Kiểm thử trích xuất đặc trưng truyền thống và mô hình hợp nhất (Task #117)."""

    def test_handcrafted_extractor_single_and_batch(self):
        """Kiểm tra trích xuất đặc trưng LBP, Haralick, Color, Gabor."""
        extractor = HandcraftedFeatureExtractor()

        # Ảnh đơn [224, 224, 3] uint8
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        feats = extractor.extract_single_image(img)
        assert feats.shape == (178,)
        assert not np.isnan(feats).any()

        # Batch ảnh dạng tensor [2, 3, 224, 224]
        batch_tensor = torch.rand(2, 3, 224, 224)
        batch_feats = extractor.extract_batch(batch_tensor)
        assert batch_feats.shape == (2, 178)

    def test_hybrid_handcrafted_deep_fusion_model_forward(self):
        """Kiểm tra forward pass qua mô hình hợp nhất đa nhánh Gated Fusion."""
        model = HybridHandcraftedDeepFusionModel(num_classes=23, handcrafted_in_dim=178)

        dummy_img = torch.randn(2, 3, 224, 224)
        dummy_hc = torch.randn(2, 178)

        # 1. Truyền cả ảnh và đặc trưng truyền thống
        logits = model(dummy_img, dummy_hc)
        assert logits.shape == (2, 23)

        # 2. Kiểm tra lan truyền gradient
        loss = logits.sum()
        loss.backward()
        assert model.fused_classifier[2].weight.grad is not None
