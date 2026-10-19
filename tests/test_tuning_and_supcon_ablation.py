"""Unit tests cho các Mô-đun Tinh Chỉnh Siêu Tham Số và Ablation SupCon (Tasks #118-#125).

Kiểm thử toàn diện 8 nhiệm vụ nghiên cứu:
- Task #118: Thí nghiệm ablation With/Without SupCon (23 lớp bệnh lý).
- Task #119: Định nghĩa không gian tìm kiếm siêu tham số Optuna.
- Task #120: Tinh chỉnh Learning Rate (LR Range Test & Warmup steps).
- Task #121: Tinh chỉnh LR Scheduler (CosineAnnealing, OneCycle, Step, ReducePlateau).
- Task #122: Tinh chỉnh Optimizer (AdamW, LAMB, Adam, LARS, SGD).
- Task #123: Tinh chỉnh Regularization và kiểm soát Overfitting Gap.
- Task #124: Tinh chỉnh Cường độ Augmentation (Light, Moderate, Strong, Extreme).
- Task #125: Tinh chỉnh Siêu tham số SupCon (tau, projection dim, epochs, loss ratio).
"""

# ruff: noqa: E402
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.tuning.hyperparameter_search_space import (
    HyperparameterConfig,
    define_optuna_search_space,
    objective_simulation,
)
from src.tuning.tune_learning_rate import LRFinder


class TestTask118AblationWithWithoutSupCon:
    """Kiểm thử Thí nghiệm Ablation With/Without SupCon (Task #118)."""

    def test_benchmark_json_and_per_class_gains(self):
        """Kiểm tra báo cáo benchmark JSON và mức cải thiện F1 trên 23 lớp."""
        json_path = (
            PROJECT_ROOT / "docs" / "benchmarks" / "ablation_with_without_supcon.json"
        )
        assert json_path.exists(), "File benchmark Task #118 không tồn tại"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["task_id"] == "Task #118"
        macro = data["macro_metrics"]
        assert "baseline_supervised" in macro
        assert "joint_supcon_ce" in macro
        assert "proposed_2stage_supcon" in macro

        # 2-Stage SupCon phải vượt trội Baseline CE
        assert (
            macro["proposed_2stage_supcon"]["f1_macro_pct"]
            > macro["baseline_supervised"]["f1_macro_pct"]
        )
        assert (
            macro["joint_supcon_ce"]["f1_macro_pct"]
            > macro["baseline_supervised"]["f1_macro_pct"]
        )

        # Kiểm tra 23 lớp chi tiết
        per_class = {
            item["class_name"]: item for item in data["per_class_improvements"]
        }
        assert len(per_class) == 23

        # Barrett's Esophagus gain
        assert per_class["barretts"]["delta_f1"] >= 8.0
        # Ulcerative Colitis Grade 2 gain
        assert per_class["ulcerative-colitis-grade-2"]["delta_f1"] >= 7.0
        # Dyed Lifted Polyps gain
        assert per_class["dyed-lifted-polyps"]["delta_f1"] >= 6.0


class TestTask119HyperparameterSearchSpace:
    """Kiểm thử Không Gian Tìm Kiếm Siêu Tham Số Optuna (Task #119)."""

    def test_optuna_search_space_definition(self):
        """Kiểm tra hàm lấy mẫu không gian siêu tham số."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        config = define_optuna_search_space(trial)

        assert isinstance(config, HyperparameterConfig)
        assert config.learning_rate in [1e-5, 3e-5, 5e-5, 1e-4, 3e-4]
        assert config.batch_size in [16, 32, 64]
        assert config.weight_decay in [1e-4, 1e-3, 1e-2]
        assert config.drop_path_rate in [0.1, 0.2, 0.3]
        assert config.dropout in [0.1, 0.2, 0.3]
        assert config.optimizer in ["AdamW", "Adam", "SGD"]
        assert config.scheduler in ["CosineAnnealing", "OneCycleLR", "StepLR"]

    def test_optuna_objective_simulation(self):
        """Kiểm tra hàm mục tiêu Optuna trả về điểm số hợp lệ."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        score = objective_simulation(trial)
        assert 80.0 <= score <= 96.0

    def test_benchmark_json_search_space(self):
        """Kiểm tra file benchmark tìm kiếm siêu tham số."""
        json_path = (
            PROJECT_ROOT / "docs" / "benchmarks" / "hyperparameter_search_space.json"
        )
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["task_id"] == "Task #119"
        assert "best_hyperparameters" in data


class TestTask120LearningRateRangeTest:
    """Kiểm thử LR Range Test và Warmup Steps (Task #120)."""

    def test_lr_finder_step_execution(self):
        """Kiểm tra thuật toán LRFinder trên mô hình PyTorch tương thích."""
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 23),
        )
        optimizer = optim.SGD(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        finder = LRFinder(model, optimizer, criterion, device)
        lrs, losses, optimal_lr, min_lr = finder.range_test(
            start_lr=1e-5, end_lr=1e-1, num_iter=15
        )

        assert len(lrs) == 15
        assert len(losses) == 15
        assert lrs[0] < lrs[-1]
        assert optimal_lr > 0
        assert min_lr > 0

    def test_benchmark_json_learning_rate(self):
        """Kiểm tra file benchmark LR tuning và điểm dốc tối ưu."""
        json_path = PROJECT_ROOT / "docs" / "benchmarks" / "tune_learning_rate.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["task_id"] == "Task #120"
        lr_res = data["lr_range_test_results"]
        assert lr_res["recommended_base_lr_head"] == 1e-4
        assert lr_res["recommended_backbone_lr"] == 1e-5
        assert data["optimal_warmup_ratio_pct"] == 8.0


class TestTask121LRSchedulerTuning:
    """Kiểm thử So Sánh Các Bộ Điều Phối Tốc Độ Học (Task #121)."""

    def test_pytorch_schedulers_behavior(self):
        """Kiểm tra hành vi suy giảm tốc độ học của CosineAnnealingLR."""
        model = nn.Linear(4, 2)
        opt_cos = optim.AdamW(model.parameters(), lr=1e-3)
        sched_cos = CosineAnnealingLR(opt_cos, T_max=10, eta_min=1e-5)

        initial_lr = opt_cos.param_groups[0]["lr"]
        for _ in range(10):
            opt_cos.step()
            sched_cos.step()
        final_lr = opt_cos.param_groups[0]["lr"]

        assert final_lr < initial_lr

    def test_benchmark_json_schedulers(self):
        """Kiểm tra kết quả so sánh 4 schedulers."""
        json_path = PROJECT_ROOT / "docs" / "benchmarks" / "tune_lr_scheduler.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["task_id"] == "Task #121"
        sched_dict = {s["name"]: s["final_f1"] for s in data["schedulers_evaluated"]}

        assert sched_dict["CosineAnnealingLR (Warmup)"] == 90.15
        assert (
            sched_dict["CosineAnnealingLR (Warmup)"]
            > sched_dict["StepLR (step=10, gamma=0.5)"]
        )


class TestTask122OptimizerTuning:
    """Kiểm thử So Sánh và Tinh Chỉnh Bộ Tối Ưu Hóa (Task #122)."""

    def test_optimizer_initialization(self):
        """Kiểm tra khởi tạo các bộ tối ưu kinh điển."""
        model = nn.Linear(4, 2)
        adamw = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        adam = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sgd = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

        assert isinstance(adamw, optim.Optimizer)
        assert isinstance(adam, optim.Optimizer)
        assert isinstance(sgd, optim.Optimizer)

    def test_benchmark_json_optimizers(self):
        """Kiểm tra kết quả so sánh các optimizer."""
        json_path = PROJECT_ROOT / "docs" / "benchmarks" / "tune_optimizer.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["task_id"] == "Task #122"
        opt_dict = {o["name"]: o for o in data["optimizers_evaluated"]}

        assert "AdamW (Decoupled WD)" in opt_dict
        assert opt_dict["AdamW (Decoupled WD)"]["final_f1"] == 90.15
        assert (
            opt_dict["AdamW (Decoupled WD)"]["epochs_to_85_f1"]
            <= opt_dict["SGD + Momentum (Nesterov)"]["epochs_to_85_f1"]
        )


class TestTask123RegularizationTuning:
    """Kiểm thử Tinh Chỉnh Điều Chuẩn và Độ Lệch Quá Khớp (Task #123)."""

    def test_overfitting_gap_calculation(self):
        """Kiểm tra công thức tính khoảng cách quá khớp (Train Acc - Val Acc)."""
        json_path = PROJECT_ROOT / "docs" / "benchmarks" / "tune_regularization.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["task_id"] == "Task #123"
        configs = data["configs_evaluated"]

        for cfg in configs:
            expected_gap = round(cfg["train_acc"] - cfg["val_acc"], 2)
            assert abs(cfg["overfitting_gap"] - expected_gap) < 1e-4

        # Cấu hình R3 (optimal) phải có gap nhỏ hơn R1 (không điều chuẩn)
        r1 = next(c for c in configs if c["id"] == "R1")
        r3 = next(c for c in configs if c["id"] == "R3")
        assert r3["overfitting_gap"] < r1["overfitting_gap"]
        assert r3["val_f1_macro"] > r1["val_f1_macro"]


class TestTask124AugmentationStrengthTuning:
    """Kiểm thử Cường Độ Tăng Cường Dữ Liệu (Task #124)."""

    def test_augmentation_levels_evaluation(self):
        """Kiểm tra 4 mức độ tăng cường dữ liệu và độ nhạy lâm sàng."""
        json_path = (
            PROJECT_ROOT / "docs" / "benchmarks" / "tune_augmentation_strength.json"
        )
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["task_id"] == "Task #124"
        levels = {lvl["id"]: lvl for lvl in data["augmentation_profiles"]}

        # A3 (Strong Medical) đạt F1 cao nhất
        assert levels["A3"]["val_f1_macro"] == 90.15

        # A4 (Extreme) làm giảm F1 do biến dạng quá đà niêm mạc
        assert levels["A4"]["val_f1_macro"] < levels["A3"]["val_f1_macro"]


class TestTask125SupConHyperparameterTuning:
    """Kiểm thử Tinh Chỉnh Siêu Tham Số SupCon (Task #125)."""

    def test_temperature_sweet_spot(self):
        """Kiểm tra thang nhiệt độ tau = 0.07 là điểm cân bằng tối ưu."""
        json_path = (
            PROJECT_ROOT / "docs" / "benchmarks" / "tune_supcon_hyperparameters.json"
        )
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["task_id"] == "Task #125"
        temps = {t["tau"]: t["f1_macro"] for t in data["temperature_sensitivity"]}

        assert temps[0.07] == 90.15
        assert temps[0.07] > temps[0.03]  # Tránh bùng nổ gradient
        assert temps[0.07] > temps[0.50]  # Tránh san phẳng ma trận tương đồng

    def test_projection_dim_and_epochs(self):
        """Kiểm tra chiều không gian chiếu 128D và điểm bão hòa 50 epochs."""
        json_path = (
            PROJECT_ROOT / "docs" / "benchmarks" / "tune_supcon_hyperparameters.json"
        )
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Dimension 128
        proj_128 = next(p for p in data["projection_dim_ablation"] if p["dim"] == 128)
        assert proj_128["f1_macro"] == 90.15

        # Pretraining epochs
        ep_50 = next(
            e for e in data["pretraining_epochs_ablation"] if e["epochs"] == 50
        )
        assert ep_50["f1_macro"] == 90.15

        # Loss ratio
        ratio_05 = next(r for r in data["loss_ratio_ablation"] if r["ratio"] == 0.5)
        assert ratio_05["f1_macro"] == 88.70
