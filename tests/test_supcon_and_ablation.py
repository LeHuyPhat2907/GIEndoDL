"""Unit tests cho các Mô-đun Ablation, Visualization và Two-Stage SupCon (Tasks #106-#111)."""

# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.evaluation.ablation_component_incremental import (
    CNNOnlyBaseline,
    CNNCBAMVariant,
)
from src.models.hybrid_cnn_cbam_transformer import build_hybrid_model
from src.models.supcon_loss import SupConLoss
from src.models.supcon_projection_head import (
    HybridModelWithProjectionHead,
    SupConProjectionHead,
)
from src.training.two_stage_supcon_pipeline import TwoStageSupConTrainer


class TestSupConLoss:
    """Kiểm thử tính năng, tính ổn định số học và logic của SupConLoss (Task #109)."""

    def test_loss_decreases_as_similar_embeddings_align(self):
        """Kiểm tra: Loss giảm khi các vector biểu diễn cùng lớp tiến gần nhau."""
        criterion = SupConLoss(temperature=0.07)

        # Batch 4 mẫu, 2 views, dim 128. Giả định 2 lớp: lớp 0 (mẫu 0, 1) và lớp 1 (mẫu 2, 3)
        labels = torch.tensor([0, 0, 1, 1])

        # Trường hợp 1: Embeddings ngẫu nhiên chưa hội tụ
        torch.manual_seed(42)
        features_random = torch.randn(4, 2, 128)
        loss_random = criterion(features_random, labels)

        # Trường hợp 2: Embeddings của các mẫu cùng lớp rất gần nhau (similarity cao)
        # và khác lớp thì xa nhau
        c0_center = torch.randn(1, 1, 128)
        c1_center = torch.randn(1, 1, 128) * -1.0  # ngược hướng
        c0_center = F.normalize(c0_center, dim=-1)
        c1_center = F.normalize(c1_center, dim=-1)

        features_clustered = torch.zeros(4, 2, 128)
        # Thêm nhiễu rất nhỏ quanh trọng tâm lớp
        features_clustered[0:2] = c0_center + 0.01 * torch.randn(2, 2, 128)
        features_clustered[2:4] = c1_center + 0.01 * torch.randn(2, 2, 128)
        loss_clustered = criterion(features_clustered, labels)

        assert (
            loss_clustered.item() < loss_random.item()
        ), f"Loss của embeddings phân cụm ({loss_clustered.item():.4f}) phải nhỏ hơn ngẫu nhiên ({loss_random.item():.4f})"

    def test_numerical_stability_with_extreme_values(self):
        """Kiểm tra: Loss không bị NaN hoặc Inf khi giá trị đầu vào rất lớn hoặc rất nhỏ."""
        criterion = SupConLoss(
            temperature=0.01
        )  # Nhiệt độ rất nhỏ làm tăng độ nhạy scale
        features = torch.randn(8, 2, 64) * 100.0  # Giá trị scale cực lớn
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss = criterion(features, labels)
        assert not torch.isnan(
            loss
        ), "SupConLoss bị lỗi NaN với thang nhiệt độ nhỏ / giá trị lớn"
        assert not torch.isinf(
            loss
        ), "SupConLoss bị lỗi Inf với thang nhiệt độ nhỏ / giá trị lớn"
        assert loss.item() >= 0.0, "Loss phải không âm"

    def test_self_supervised_mode_without_labels(self):
        """Kiểm tra: Chế độ SimCLR tự giám sát khi labels = None."""
        criterion = SupConLoss(temperature=0.07)
        features = torch.randn(4, 2, 128)
        loss_simclr = criterion(features, labels=None)

        assert loss_simclr.item() > 0.0
        assert not torch.isnan(loss_simclr)


class TestSupConProjectionHead:
    """Kiểm thử cấu trúc Projection Head (Task #110)."""

    def test_projection_head_shapes_and_l2_normalization(self):
        """Kiểm tra: Vector đầu ra z có shape chuẩn và độ dài L2 = 1.0."""
        proj = SupConProjectionHead(
            in_dim=512, hidden_dim=512, out_dim=128, normalize=True
        )

        # 2D input: [B, 512]
        h_2d = torch.randn(4, 512)
        z_2d = proj(h_2d)
        assert z_2d.shape == (4, 128)
        l2_norms_2d = torch.norm(z_2d, p=2, dim=-1)
        assert torch.allclose(l2_norms_2d, torch.ones_like(l2_norms_2d), atol=1e-5)

        # 3D input: [B, 2, 512] (2 views)
        h_3d = torch.randn(4, 2, 512)
        z_3d = proj(h_3d)
        assert z_3d.shape == (4, 2, 128)
        l2_norms_3d = torch.norm(z_3d, p=2, dim=-1)
        assert torch.allclose(l2_norms_3d, torch.ones_like(l2_norms_3d), atol=1e-5)

    def test_hybrid_model_with_projection_and_discard(self):
        """Kiểm tra: Ghép Projection Head vào Hybrid Model và tháo bỏ chuyển sang Stage 2."""
        base_model = build_hybrid_model(
            num_classes=23,
            pretrained_backbone=False,
            cbam_placement="stage4",
            transformer_layers=1,
            nhead=4,
            d_model=256,
            token_pooling="gap",
        )
        hybrid_proj = HybridModelWithProjectionHead(
            base_hybrid_model=base_model,
            proj_hidden_dim=256,
            proj_out_dim=128,
        )

        # Forward với batch 2 views: [2, 2, 3, 224, 224]
        dummy_views = torch.randn(2, 2, 3, 224, 224)
        z, h = hybrid_proj(dummy_views, return_representation=True)

        assert z.shape == (2, 2, 128)
        assert h.shape == (2, 2, 256)

        # Tháo bỏ projection head
        recovered_model = hybrid_proj.discard_projection_head()
        assert not hasattr(hybrid_proj, "projection_head")

        # Forward kiểm tra mô hình nguyên bản cho Stage 2
        logits = recovered_model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 23)


class TestTwoStagePipeline:
    """Kiểm thử quy trình và luồng gradient trong Two-Stage Training Pipeline (Task #111)."""

    def test_dry_run_stage1_and_stage2_transitions(self):
        """Kiểm tra: Hoạt động của Trainer qua cả Stage 1 và Stage 2."""
        trainer = TwoStageSupConTrainer(num_classes=23)

        # Stage 1 dry run
        res_s1 = trainer.run_dry_run_step_stage1(batch_size=2)
        assert res_s1["stage"] == 1
        assert res_s1["loss"] > 0
        assert res_s1["gradient_flowing"] is True

        # Chuyển giao sang Stage 2
        trainer.transition_to_stage2()
        assert trainer.current_stage == 2

        # Stage 2 dry run
        res_s2 = trainer.run_dry_run_step_stage2(batch_size=2)
        assert res_s2["stage"] == 2
        assert res_s2["loss"] > 0
        assert res_s2["gradient_flowing"] is True


class TestAblationVariants:
    """Kiểm thử các biến thể kiến trúc trong Thí nghiệm Cắt bỏ (Tasks #106, #107)."""

    def test_baseline_and_cbam_variants_forward(self):
        """Kiểm tra forward pass của CNN-only và CNN-CBAM."""
        dummy_input = torch.randn(2, 3, 224, 224)

        # 1. CNN-only baseline
        cnn_only = CNNOnlyBaseline(num_classes=23, pretrained=False)
        logits_cnn = cnn_only(dummy_input)
        assert logits_cnn.shape == (2, 23)

        # 2. CNN-CBAM variant
        cnn_cbam = CNNCBAMVariant(num_classes=23, pretrained=False)
        logits_cbam = cnn_cbam(dummy_input)
        assert logits_cbam.shape == (2, 23)
