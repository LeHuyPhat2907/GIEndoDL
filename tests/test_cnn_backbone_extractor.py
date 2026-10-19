"""Unit tests cho Mô-đun Trích xuất Đặc trưng Đa Tỷ lệ CNN Backbone (Task #95).

Kiểm tra:
1. Độ phân giải (resolutions) và số kênh (channels) tại từng Stage 1 -> 4.
2. Cơ chế Hook-based extraction (CNNFeatureHooks): đăng ký, trích xuất và dọn dẹp (cleanup).
3. Lan truyền gradient (Backpropagation) qua các stage được trích xuất.
4. Đóng băng trọng số (Freeze stages).
5. Tích hợp trích xuất trên mô hình gốc torchvision ResNet-50 và HybridCNNCBAMTransformer.
"""

import pytest
import torch
from torchvision.models import resnet50

from src.models.cnn_backbone_extractor import (
    CNNFeatureHooks,
    build_backbone_extractor,
    extract_features_with_hooks,
)
from src.models.hybrid_cnn_cbam_transformer import build_hybrid_model


@pytest.fixture
def dummy_input():
    """Tạo tensor đầu vào giả lập [2, 3, 224, 224]."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def extractor():
    """Khởi tạo ResNet50StageExtractor không nạp pretrained weights cho test nhanh."""
    return build_backbone_extractor(pretrained=False, input_size=(224, 224))


def test_stage_output_shapes_and_channels(extractor, dummy_input):
    """Kiểm tra chính xác kích thước không gian và số channels tại từng stage 1 -> 4."""
    extractor.eval()
    with torch.no_grad():
        features = extractor(dummy_input, return_stem=True)

    b, _, h, w = dummy_input.shape

    # Kiểm tra sự hiện diện của các stage
    assert "stem" in features
    assert "stage1" in features
    assert "stage2" in features
    assert "stage3" in features
    assert "stage4" in features

    # 1. Stem: C=64, H/4, W/4
    assert features["stem"].shape == (b, 64, h // 4, w // 4)

    # 2. Stage 1: C=256, H/4, W/4 (56x56)
    assert features["stage1"].shape == (b, 256, h // 4, w // 4)

    # 3. Stage 2: C=512, H/8, W/8 (28x28)
    assert features["stage2"].shape == (b, 512, h // 8, w // 8)

    # 4. Stage 3: C=1024, H/16, W/16 (14x14)
    assert features["stage3"].shape == (b, 1024, h // 16, w // 16)

    # 5. Stage 4: C=2048, H/32, W/32 (7x7)
    assert features["stage4"].shape == (b, 2048, h // 32, w // 32)


def test_stage_metadata_consistency(extractor):
    """Kiểm tra tính nhất quán giữa siêu dữ liệu (Metadata) và cấu hình mô hình."""
    info = extractor.stage_info

    expected_resolutions = {
        "stem": (56, 56),
        "stage1": (56, 56),
        "stage2": (28, 28),
        "stage3": (14, 14),
        "stage4": (7, 7),
    }

    expected_channels = {
        "stem": 64,
        "stage1": 256,
        "stage2": 512,
        "stage3": 1024,
        "stage4": 2048,
    }

    assert extractor.get_stage_resolutions() == expected_resolutions
    assert extractor.get_stage_channels() == expected_channels

    for name in ["stage1", "stage2", "stage3", "stage4"]:
        meta = info[name]
        assert meta.channels == expected_channels[name]
        assert (meta.height, meta.width) == expected_resolutions[name]
        assert meta.receptive_field > 0


def test_hook_based_extraction(extractor, dummy_input):
    """Kiểm tra cơ chế Hook-based extraction: bắt đúng feature maps và tự động dọn dẹp hooks."""
    extractor.eval()

    # So sánh output giữa Direct Forward và Hook Extraction
    with torch.no_grad():
        direct_features = extractor(dummy_input, return_stem=True)

    hook_extractor = extractor.create_hook_extractor(detach=True)

    # Trước khi vào context, hooks đã được đăng ký
    assert hook_extractor._is_registered is True
    assert len(hook_extractor.hooks) == len(extractor.DEFAULT_TARGET_LAYERS)

    with hook_extractor:
        with torch.no_grad():
            _ = extractor(dummy_input)
        hooked_features = hook_extractor.features

        assert "stage1" in hooked_features
        assert "stage2" in hooked_features
        assert "stage3" in hooked_features
        assert "stage4" in hooked_features
        assert "stem" in hooked_features

        # Giá trị tensor từ hook phải trùng khớp hoàn hảo với direct forward
        for stage in ["stage1", "stage2", "stage3", "stage4"]:
            torch.testing.assert_close(
                hooked_features[stage], direct_features[stage], rtol=1e-5, atol=1e-5
            )

    # Sau khi thoát context manager, hooks phải được remove sạch sẽ
    assert hook_extractor._is_registered is False
    assert len(hook_extractor.hooks) == 0


def test_hook_gradient_flow(dummy_input):
    """Kiểm tra gradient flow qua hook khi detach=False."""
    extractor = build_backbone_extractor(pretrained=False)
    extractor.train()

    dummy_input.requires_grad_(True)

    target_layers = {"stage4": "layer4"}
    with CNNFeatureHooks(
        extractor, target_layers=target_layers, detach=False
    ) as hook_ext:
        _ = extractor(dummy_input)
        stage4_feat = hook_ext.features["stage4"]

        # Giả lập hàm loss trên stage4 feature map
        loss = stage4_feat.mean()
        loss.backward()

    # Đảm bảo gradient đã truyền ngược về input tensor
    assert dummy_input.grad is not None
    assert not torch.isnan(dummy_input.grad).any()
    assert dummy_input.grad.abs().sum() > 0


def test_extract_features_with_hooks_standalone_torchvision(dummy_input):
    """Kiểm tra hàm helper extract_features_with_hooks trên mô hình chuẩn torchvision ResNet-50."""
    standard_resnet = resnet50(weights=None)
    standard_resnet.eval()

    extracted = extract_features_with_hooks(
        model=standard_resnet,
        x=dummy_input,
        detach=True,
    )

    assert "stage1" in extracted
    assert "stage2" in extracted
    assert "stage3" in extracted
    assert "stage4" in extracted
    assert extracted["stage1"].shape == (2, 256, 56, 56)
    assert extracted["stage4"].shape == (2, 2048, 7, 7)


def test_freeze_stages():
    """Kiểm tra khả năng đóng băng các stage cụ thể."""
    # Đóng băng stage 0 (stem) và stage 1 (layer1)
    extractor = build_backbone_extractor(pretrained=False, freeze_stages=[0, 1])

    # Kiểm tra stem đã bị đóng băng
    assert not extractor.conv1.weight.requires_grad
    assert not extractor.bn1.weight.requires_grad

    # Kiểm tra stage 1 đã bị đóng băng
    for p in extractor.layer1.parameters():
        assert not p.requires_grad

    # Kiểm tra stage 2, 3, 4 vẫn mở gradient
    assert any(p.requires_grad for p in extractor.layer2.parameters())
    assert any(p.requires_grad for p in extractor.layer3.parameters())
    assert any(p.requires_grad for p in extractor.layer4.parameters())


def test_hook_invalid_layer_raises_error(dummy_input):
    """Kiểm tra xử lý ngoại lệ khi truyền tên submodule không tồn tại."""
    extractor = build_backbone_extractor(pretrained=False)
    invalid_targets = {"invalid_stage": "non_existent_layer_xyz"}

    with pytest.raises(AttributeError, match="không có module con"):
        CNNFeatureHooks(extractor, target_layers=invalid_targets)


def test_compatibility_with_hybrid_model(dummy_input):
    """Kiểm tra tích hợp trích xuất đặc trưng với mô hình Hybrid CNN-CBAM-Transformer (Task #94)."""
    hybrid = build_hybrid_model(num_classes=23, pretrained_backbone=False)
    hybrid.eval()

    # 1. Trích xuất qua phương thức built-in của HybridCNNCBAMTransformer
    with torch.no_grad():
        built_in_feats = hybrid.extract_features(dummy_input)

    assert built_in_feats["stage1"].shape == (2, 256, 56, 56)
    assert built_in_feats["stage4"].shape == (2, 2048, 7, 7)

    # 2. Trích xuất qua HookFeatureExtractor trên hybrid model
    hybrid_targets = {
        "stage1": "layer1",
        "stage4": "layer4",
        "cbam": "cbam4",
    }
    with CNNFeatureHooks(hybrid, target_layers=hybrid_targets, detach=True) as hooks:
        with torch.no_grad():
            _ = hybrid(dummy_input)
        hooked = hooks.features

    assert "stage1" in hooked
    assert "stage4" in hooked
    assert "cbam" in hooked
    assert hooked["cbam"].shape == (2, 2048, 7, 7)
