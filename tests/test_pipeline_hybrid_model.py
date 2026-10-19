"""Kiểm thử Đơn vị Toàn diện Pipeline Mô hình Lai Hybrid CNN-CBAM-Transformer (Task #102).

Kiểm tra:
1. Tính nhất quán của 10 điểm kết nối dữ liệu (Connection Points CP-1 -> CP-10).
2. Toàn vẹn lan truyền ngược Gradient (Gradient Flow không vanishing, không NaN).
3. Tính bất biến kích thước Batch (Batch Invariance: B = 1, 2, 4, 8).
4. Tính toàn vẹn khi chuyển đổi giữa chế độ Train và Eval.
"""

# ruff: noqa: E402
from pathlib import Path
import sys
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline_hybrid_model import HybridPipelineAuditor


@pytest.fixture
def auditor():
    """Khởi tạo auditor với mô hình lai đề xuất."""
    return HybridPipelineAuditor()


def test_pipeline_connection_points(auditor):
    """Kiểm tra 10 điểm kết nối dữ liệu (CP-1 -> CP-10) đều đạt chuẩn PASSED."""
    audits = auditor.audit_connection_points(batch_size=2)
    assert len(audits) == 10

    for cp in audits:
        assert cp.status == "PASSED"
        assert len(cp.input_shape) >= 2
        assert len(cp.output_shape) >= 2

    # Kiểm tra điểm xuất phát và kết thúc của pipeline
    assert audits[0].input_shape == [2, 3, 224, 224]
    assert audits[-1].output_shape == [2, 23]


def test_pipeline_gradient_flow(auditor):
    """Kiểm tra toàn bộ 13 thành phần kiến trúc đều nhận gradient hợp lệ."""
    grad_result = auditor.verify_gradient_flow()
    assert grad_result["all_components_passed"] is True
    assert grad_result["components_checked"] >= 12

    details = grad_result["details"]
    assert details["input_image"]["is_nonzero"] is True
    assert details["backbone_conv1"]["is_nonzero"] is True
    assert details["cbam_channel_attention"]["is_nonzero"] is True
    assert details["cbam_spatial_attention"]["is_nonzero"] is True
    assert details["bridge_projection"]["is_nonzero"] is True
    assert details["transformer_mhsa"]["is_nonzero"] is True
    assert details["classification_head"]["is_nonzero"] is True


@pytest.mark.parametrize("b_size", [1, 2, 4, 8])
def test_pipeline_batch_invariance(auditor, b_size):
    """Kiểm tra tính ổn định trên nhiều kích thước batch khác nhau."""
    res = auditor.verify_batch_invariance([b_size])
    assert res[b_size] is True


def test_pipeline_train_eval_modes(auditor):
    """Kiểm tra chuyển đổi trạng thái huấn luyện và đánh giá."""
    m = auditor.model

    # Chế độ Eval
    m.eval()
    with torch.no_grad():
        out_eval = m(torch.randn(2, 3, 224, 224))
    assert out_eval.shape == (2, 23)

    # Chế độ Train
    m.train()
    out_train = m(torch.randn(2, 3, 224, 224))
    assert out_train.shape == (2, 23)
