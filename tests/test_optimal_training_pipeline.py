"""Unit tests cho Pipeline Huấn Luyện Sản Xuất Mô Hình Tối Ưu (Task #127).

Kiểm thử toàn diện:
- Cấu hình chuẩn hóa Kaggle: configs/optimal_training_kaggle_config.json.
- Thuật toán Early Stopping với patience = 15 trên metric val_macro_f1.
- Khởi tạo và lan truyền thuận của OptimalModelTrainer trên thiết bị hiện hành.
- Quản lý mốc trọng số (Dual Checkpoints: best_model.pth & last_model.pth).
- Tính nhất quán của file benchmark JSON, Markdown và đồ thị hội tụ 150 epochs.
"""

# ruff: noqa: E402
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.training.train_optimal_model import EarlyStopping, OptimalModelTrainer


class TestTask127ProductionConfiguration:
    """Kiểm thử file cấu hình sản xuất của Task #127."""

    def test_kaggle_config_integrity(self):
        """Kiểm tra cấu hình tích hợp đầy đủ siêu tham số từ Tasks #120-#126."""
        config_path = PROJECT_ROOT / "configs" / "optimal_training_kaggle_config.json"
        assert config_path.exists(), "File cấu hình Kaggle không tồn tại"

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        assert cfg["task_id"] == "Task #127"
        assert cfg["training"]["full_epochs"] == 150
        assert cfg["training"]["early_stopping_patience"] == 15
        assert cfg["training"]["early_stopping_metric"] == "val_macro_f1"

        # Kiểm tra kiến trúc
        arch = cfg["architecture"]
        assert arch["model_name"] == "CNN-CBAM-Transformer"
        assert arch["transformer_layers"] == 4
        assert arch["transformer_heads"] == 8
        assert arch["transformer_dim"] == 512

        # Kiểm tra tối ưu hóa và suy giảm tốc độ học
        opt = cfg["optimizer"]
        assert opt["name"] == "AdamW"
        assert opt["discriminative_learning_rates"]["classifier_head_lr"] == 1e-4
        assert opt["discriminative_learning_rates"]["backbone_lr"] == 1e-5

        # Kiểm tra hàm mất mát
        loss_cfg = cfg["loss_function"]
        assert loss_cfg["gamma"] == 2.0
        assert loss_cfg["alpha_mode"] == "class-balanced"
        assert loss_cfg["class_balanced_beta"] == 0.999


class TestTask127EarlyStoppingAndTrainer:
    """Kiểm thử thuật toán Early Stopping và khung huấn luyện OptimalModelTrainer."""

    def test_early_stopping_trigger(self):
        """Kiểm tra cơ chế đếm patience = 15 và ngắt sớm."""
        stopper = EarlyStopping(patience=15, mode="max")

        # Điểm tăng dần
        assert stopper.step(80.0) is True
        assert stopper.step(82.5) is True
        assert stopper.counter == 0
        assert stopper.early_stop is False

        # 14 bước không cải thiện
        for _ in range(14):
            assert stopper.step(82.0) is False
            assert stopper.early_stop is False

        # Bước thứ 15 không cải thiện -> Kích hoạt early stop
        assert stopper.step(81.9) is False
        assert stopper.early_stop is True

    def test_optimal_model_trainer_forward_pass(self):
        """Kiểm tra forward pass và tính toán loss của OptimalModelTrainer."""
        trainer = OptimalModelTrainer()

        dummy_x = torch.randn(2, 3, 224, 224, device=trainer.device)
        dummy_y = torch.tensor([0, 5], device=trainer.device)

        logits = trainer.model(dummy_x)
        assert logits.shape == (2, 23)

        loss = trainer.criterion(logits, dummy_y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0.0

    def test_checkpoint_saving_and_loading(self, tmp_path):
        """Kiểm tra lưu trữ checkpoint best_model.pth và tải lại trạng thái."""
        trainer = OptimalModelTrainer()
        trainer.checkpoint_dir = tmp_path

        saved_path = trainer.save_checkpoint(
            epoch=10, val_f1=90.15, val_acc=93.85, is_best=True
        )
        assert saved_path.exists()

        checkpoint = torch.load(saved_path, map_location="cpu")
        assert checkpoint["epoch"] == 10
        assert checkpoint["val_macro_f1"] == 90.15
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint


class TestTask127BenchmarksAndArtifacts:
    """Kiểm thử tính toàn vẹn của kết quả benchmark và đồ thị xuất bản Task #127."""

    def test_benchmark_json_and_convergence_figure(self):
        """Kiểm tra file benchmark JSON và đồ thị hội tụ 150 epochs."""
        json_path = PROJECT_ROOT / "docs" / "benchmarks" / "train_optimal_model.json"
        assert (
            json_path.exists()
        ), "File benchmark train_optimal_model.json không tồn tại"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["task_id"] == "Task #127"
        assert data["best_metrics"]["val_macro_f1_pct"] >= 90.0
        assert data["best_metrics"]["val_accuracy_pct"] >= 93.0

        fig_path = (
            PROJECT_ROOT
            / "docs"
            / "figures"
            / "optimal_training_convergence_curves.png"
        )
        assert (
            fig_path.exists()
        ), "Biểu đồ hội tụ optimal_training_convergence_curves.png không tồn tại"
        assert fig_path.stat().st_size > 50000

        # Kiểm tra notebook Kaggle
        nb_path = PROJECT_ROOT / "notebooks" / "kaggle_train_optimal_model.ipynb"
        assert nb_path.exists(), "Kaggle notebook không tồn tại"
        assert nb_path.stat().st_size > 1000
