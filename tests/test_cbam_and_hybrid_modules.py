"""Kiểm thử đơn vị chuyên sâu cho CBAM Modules và các thành phần Mô hình Lai (Tasks #96-#101).

Kiểm tra:
1. Task #96: Channel Attention (shape, shared MLP, attention map, gradient).
2. Task #97: Spatial Attention (shape, 7x7 conv, spatial map, gradient).
3. Task #98: CBAM Backbone với các Selective Placement ('stage4', 'stage3_4', 'all_stages', 'residual_blocks').
4. Task #99: Feature-to-Token Bridge (projection, CLS token, position embedding 1D & 2D).
5. Task #100: Transformer Encoder (configurable depth L=2..4, MHSA, attention weights).
6. Task #101: Classification Head (GAP vs CLS vs Hybrid pooling modes).
7. End-to-end forward pass và lan truyền ngược gradient trên toàn bộ kiến trúc lai.
"""

# ruff: noqa: E402
from pathlib import Path
import sys
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bridge_feature_to_token import FeatureToTokenBridge
from src.models.cbam_integrated_backbone import (
    CBAMIntegratedResNet50,
    count_cbam_parameters,
)
from src.models.cbam_modules import CBAM, ChannelAttention, SpatialAttention
from src.models.classification_head import HybridClassificationHead
from src.models.hybrid_cnn_cbam_transformer import build_hybrid_model
from src.models.transformer_encoder import (
    ConfigurableTransformerEncoder,
)


# ==============================================================================
# 1. TASK #96: CHANNEL ATTENTION MODULE TESTS
# ==============================================================================


def test_channel_attention_shapes_and_weights():
    """Kiểm tra shape đầu ra và giá trị của Channel Attention Module (Task #96)."""
    b, c, h, w = 2, 256, 14, 14
    x = torch.randn(b, c, h, w, requires_grad=True)

    cam = ChannelAttention(in_planes=c, ratio=16)
    out, c_map = cam(x, return_map=True)

    # 1. Shape kiểm tra
    assert out.shape == (b, c, h, w)
    assert c_map.shape == (b, c, 1, 1)

    # 2. Giá trị attention map nằm trong khoảng (0, 1)
    assert (c_map >= 0.0).all() and (c_map <= 1.0).all()

    # 3. Lan truyền gradient
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_channel_attention_shared_mlp():
    """Kiểm tra MLP chia sẻ trọng số giữa 2 nhánh Pooling (Task #96)."""
    cam = ChannelAttention(in_planes=64, ratio=8)
    dummy = torch.randn(2, 64, 7, 7)

    # Đảm bảo shared_mlp được dùng cho cả avg_pool và max_pool
    avg_feat = cam.shared_mlp(cam.avg_pool(dummy))
    max_feat = cam.shared_mlp(cam.max_pool(dummy))
    assert avg_feat.shape == (2, 64, 1, 1)
    assert max_feat.shape == (2, 64, 1, 1)


# ==============================================================================
# 2. TASK #97: SPATIAL ATTENTION MODULE TESTS
# ==============================================================================


def test_spatial_attention_shapes_and_kernels():
    """Kiểm tra shape và cơ chế tích chập không gian của Spatial Attention Module (Task #97)."""
    b, c, h, w = 2, 512, 28, 28
    x = torch.randn(b, c, h, w, requires_grad=True)

    # Thử nghiệm với kernel 7x7 (chuẩn) và kernel 3x3
    for k in [7, 3]:
        sam = SpatialAttention(kernel_size=k)
        out, s_map = sam(x, return_map=True)

        assert out.shape == (b, c, h, w)
        assert s_map.shape == (b, 1, h, w)
        assert (s_map >= 0.0).all() and (s_map <= 1.0).all()

    # Lan truyền gradient
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_cbam_sequential_execution():
    """Kiểm tra khối CBAM hoàn chỉnh thực thi tuần tự CAM -> SAM (Tasks #96, #97)."""
    cbam = CBAM(in_planes=128, ratio=16, kernel_size=7)
    x = torch.randn(2, 128, 14, 14)
    out, c_map, s_map = cbam(x, return_maps=True)

    assert out.shape == (2, 128, 14, 14)
    assert c_map.shape == (2, 128, 1, 1)
    assert s_map.shape == (2, 1, 14, 14)


# ==============================================================================
# 3. TASK #98: CBAM BACKBONE SELECTIVE PLACEMENTS
# ==============================================================================


@pytest.mark.parametrize(
    "placement", ["stage4", "stage3_4", "all_stages", "residual_blocks"]
)
def test_cbam_backbone_placements(placement):
    """Kiểm tra các chiến lược định vị CBAM trong CNN Backbone (Task #98)."""
    model = CBAMIntegratedResNet50(pretrained=False, placement=placement)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out, aux = model(
            dummy_input, return_multiscale=True, return_attention_maps=True
        )

    assert out.shape == (2, 2048, 7, 7)
    assert "stage1" in aux["multiscale_features"]
    assert "stage4" in aux["multiscale_features"]

    # Kiểm tra tính toán tham số
    param_info = count_cbam_parameters(placement)
    assert param_info["total_backbone_params"] > 23_000_000
    assert param_info["cbam_added_params"] > 0


# ==============================================================================
# 4. TASK #99: FEATURE-TO-TOKEN BRIDGE
# ==============================================================================


def test_feature_to_token_bridge_with_and_without_cls():
    """Kiểm tra Feature-to-Token conversion với và không có CLS token (Task #99)."""
    b, c, h, w = 2, 2048, 7, 7
    feat = torch.randn(b, c, h, w)

    # 1. Có CLS token: 49 spatial + 1 CLS = 50 tokens
    bridge_cls = FeatureToTokenBridge(
        in_channels=2048, d_model=512, grid_size=(7, 7), use_cls_token=True
    )
    tokens_cls = bridge_cls(feat)
    assert tokens_cls.shape == (b, 50, 512)

    # 2. Không có CLS token (dùng cho GAP): 49 spatial tokens
    bridge_gap = FeatureToTokenBridge(
        in_channels=2048, d_model=512, grid_size=(7, 7), use_cls_token=False
    )
    tokens_gap = bridge_gap(feat)
    assert tokens_gap.shape == (b, 49, 512)

    # 3. Thử nghiệm với Positional Embedding 2D SinCos
    bridge_sincos = FeatureToTokenBridge(
        in_channels=2048, d_model=512, grid_size=(7, 7), pos_embed_type="sincos2d"
    )
    tokens_sincos = bridge_sincos(feat)
    assert tokens_sincos.shape == (b, 50, 512)


# ==============================================================================
# 5. TASK #100: TRANSFORMER ENCODER BLOCK
# ==============================================================================


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_configurable_transformer_encoder(depth):
    """Kiểm tra Transformer Encoder với độ sâu cấu hình L=2, 3, 4 (Task #100)."""
    encoder = ConfigurableTransformerEncoder(
        depth=depth, dim=512, num_heads=8, mlp_ratio=4.0
    )
    tokens = torch.randn(2, 50, 512)

    out, attns = encoder(tokens, return_all_attentions=True)
    assert out.shape == (2, 50, 512)
    assert len(attns) == depth
    # Attention map shape: [B, N, N]
    assert attns[0].shape == (2, 50, 50)


# ==============================================================================
# 6. TASK #101: CLASSIFICATION HEAD POOLING MODES
# ==============================================================================


@pytest.mark.parametrize("pooling", ["gap", "cls", "hybrid"])
def test_classification_head_pooling_modes(pooling):
    """Kiểm tra các chế độ pooling của Classification Head (Task #101)."""
    head = HybridClassificationHead(
        d_model=512, num_classes=23, pooling=pooling, has_cls_token=True
    )
    tokens = torch.randn(2, 50, 512)

    logits, emb = head(tokens, return_embeddings=True)
    assert logits.shape == (2, 23)
    assert emb.shape == (2, 512)


def test_classification_head_gap_without_cls():
    """Kiểm tra GAP khi chuỗi thuần túy chỉ có 49 visual tokens (Task #101)."""
    head = HybridClassificationHead(
        d_model=512, num_classes=23, pooling="gap", has_cls_token=False
    )
    tokens = torch.randn(2, 49, 512)
    logits = head(tokens)
    assert logits.shape == (2, 23)


# ==============================================================================
# 7. END-TO-END HYBRID MODEL INTEGRATION & GRADIENT FLOW (TASKS #94-#101)
# ==============================================================================


def test_full_hybrid_model_end_to_end_gradient_flow():
    """Kiểm tra toàn diện lan truyền ngược gradient trên mô hình lai hoàn chỉnh."""
    model = build_hybrid_model(
        num_classes=23,
        pretrained_backbone=False,
        cbam_placement="stage4",
        transformer_layers=2,
        token_pooling="gap",
    )
    model.train()

    dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits, aux = model(dummy_input, return_attention_weights=True)

    assert logits.shape == (2, 23)
    assert "encoded_tokens" in aux
    assert "transformer_attentions" in aux

    # Tính loss và backward
    target = torch.tensor([0, 5], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    loss.backward()

    # Đảm bảo gradient truyền đầy đủ về input và các thành phần chính
    assert dummy_input.grad is not None
    assert dummy_input.grad.abs().sum() > 0

    # Kiểm tra gradient tại Backbone Stem
    assert model.backbone.conv1.weight.grad is not None
    # Kiểm tra gradient tại CBAM Module
    assert model.backbone.cbam_modules["cbam4"].ca.shared_mlp[0].weight.grad is not None
    # Kiểm tra gradient tại Bridge Projection
    assert model.bridge.proj[0].weight.grad is not None
    # Kiểm tra gradient tại Transformer
    assert model.transformer_encoder.layers[0].attn.in_proj_weight.grad is not None
    # Kiểm tra gradient tại Classification Head
    assert model.head.fc.weight.grad is not None
