"""Unit tests cho Mô-đun Tinh Chỉnh Siêu Tham Số Focal Loss (Task #126).

Kiểm thử toàn diện:
- Khởi tạo và lan truyền thuận (forward pass) của TunedFocalLoss.
- Cơ chế suy giảm hàm điều biến (Modulating Factor) (1 - p_t)^gamma trên mẫu dễ và khó.
- Tính toán ma trận trọng số Class-Balanced (Cui et al., CVPR 2019) dựa trên 23 lớp HyperKvasir.
- Tương tác với trọng số alpha và kiểm soát hiện tượng underfitting trên easy examples khi gamma >= 3.0.
- Kiểm tra tính toàn vẹn của file benchmark JSON, Markdown và đồ thị xuất bản.
"""

# ruff: noqa: E402
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.tuning.tune_focal_loss import (
    HYPERKVASIR_23_CLASS_COUNTS,
    TunedFocalLoss,
    compute_class_balanced_weights,
)


class TestFocalLossModuleAndMathematics:
    """Kiểm thử tính toán toán học và mô-đun TunedFocalLoss."""

    def test_focal_loss_initialization_and_forward(self):
        """Kiểm tra khởi tạo và forward pass cho batch kích thước bất kỳ."""
        loss_fn = TunedFocalLoss(gamma=2.0, label_smoothing=0.1)
        logits = torch.randn(8, 23, requires_grad=True)
        targets = torch.randint(0, 23, (8,))

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar mean
        assert loss.item() > 0.0

        # Kiểm tra tính khả vi (backward pass)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_modulating_factor_easy_example_suppression(self):
        """Chứng minh gamma lớn làm suy giảm cực mạnh mẫu dễ nhưng bảo toàn mẫu khó."""
        loss_fn_g1 = TunedFocalLoss(gamma=1.0, label_smoothing=0.0)
        loss_fn_g2 = TunedFocalLoss(gamma=2.0, label_smoothing=0.0)
        loss_fn_g5 = TunedFocalLoss(gamma=5.0, label_smoothing=0.0)

        # Mẫu dễ: Logits rất tự tin vào nhãn đúng (lớp 0)
        easy_logits = torch.zeros(1, 23)
        easy_logits[0, 0] = 10.0  # p_t ~ 0.999
        easy_target = torch.tensor([0])

        loss_g1_easy = loss_fn_g1(easy_logits, easy_target).item()
        loss_g2_easy = loss_fn_g2(easy_logits, easy_target).item()
        loss_g5_easy = loss_fn_g5(easy_logits, easy_target).item()

        # Khi gamma tăng, tổn thất trên mẫu dễ giảm theo cấp số nhân
        assert loss_g1_easy > loss_g2_easy > loss_g5_easy
        assert (
            loss_g5_easy < 1e-5
        ), "Mẫu dễ với gamma=5.0 phải gần như bị triệt tiêu hoàn toàn"

        # Mẫu khó: Logits đoán sai (lớp 1 điểm cao nhưng nhãn đúng là 0)
        hard_logits = torch.zeros(1, 23)
        hard_logits[0, 1] = 5.0
        hard_target = torch.tensor([0])

        loss_g1_hard = loss_fn_g1(hard_logits, hard_target).item()
        loss_g2_hard = loss_fn_g2(hard_logits, hard_target).item()

        # Mẫu khó vẫn duy trì mức phạt đáng kể ở cả gamma=1 và gamma=2
        assert loss_g1_hard > 1.0 and loss_g2_hard > 1.0

    def test_class_balanced_weights_properties(self):
        """Kiểm tra đặc tính toán học của trọng số Class-Balanced."""
        weights = compute_class_balanced_weights(
            HYPERKVASIR_23_CLASS_COUNTS, beta=0.999
        )

        assert len(weights) == 23
        assert (weights > 0).all()

        # Lớp hiếm nhất (hemorrhoids: 6 mẫu) phải có trọng số lớn hơn lớp nhiều nhất (bbps-2-3: 1148 mẫu)
        classes = list(HYPERKVASIR_23_CLASS_COUNTS.keys())
        idx_rare = classes.index("hemorrhoids")
        idx_frequent = classes.index("bbps-2-3")

        assert weights[idx_rare] > weights[idx_frequent]
        assert weights[idx_rare] / weights[idx_frequent] > 5.0  # Tỷ lệ phạt vượt trội


class TestFocalLossTuningBenchmarks:
    """Kiểm thử tính toàn vẹn của kết quả thực nghiệm Task #126."""

    def test_benchmark_json_structure(self):
        """Kiểm tra cấu trúc và tính nhất quán của file benchmark JSON."""
        json_path = PROJECT_ROOT / "docs" / "benchmarks" / "tune_focal_loss.json"
        assert json_path.exists(), "File benchmark tune_focal_loss.json không tồn tại"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["task_id"] == "Task #126"
        assert "gamma_tuning_experiments" in data
        assert "alpha_and_sampler_experiments" in data
        assert "optimal_focal_configuration" in data

        opt = data["optimal_focal_configuration"]
        assert opt["gamma"] == 2.0
        assert opt["alpha_mode"] == "class-balanced"
        assert opt["val_f1_macro_pct"] == 90.15

        # Kiểm tra khảo sát gamma
        gammas = {e["gamma"]: e for e in data["gamma_tuning_experiments"]}
        assert 1.0 in gammas and 2.0 in gammas and 5.0 in gammas
        # gamma=2.0 phải vượt trội gamma=5.0
        assert gammas[2.0]["f1_macro"] > gammas[5.0]["f1_macro"]
        # gamma=5.0 gây underfitting trên mẫu dễ
        assert gammas[5.0]["easy_class_f1"] < gammas[2.0]["easy_class_f1"]

    def test_figure_file_exists(self):
        """Kiểm tra biểu đồ khoa học 4 khung hình đã được xuất bản."""
        fig_path = PROJECT_ROOT / "docs" / "figures" / "focal_loss_parameter_tuning.png"
        assert fig_path.exists()
        assert fig_path.stat().st_size > 50000  # Đồ thị chất lượng cao > 50KB
