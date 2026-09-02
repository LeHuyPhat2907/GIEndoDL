"""Module quản lý lưu trữ mốc trọng số (Dual Checkpoints) và ghi nhật ký huấn luyện chuẩn y khoa."""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class TrainingLogger:
    """Ghi nhận lịch sử huấn luyện vào file CSV và JSON phục vụ phân tích khoa học."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "training_history.csv"
        self.json_path = self.log_dir / "training_summary.json"
        self.history: List[Dict[str, Any]] = []

        self.fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "val_macro_f1",
            "learning_rate",
            "time_sec",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_epoch(self, epoch_data: Dict[str, Any]):
        """Ghi dữ liệu của 1 Epoch vào CSV và danh sách bộ nhớ."""
        self.history.append(epoch_data)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(epoch_data)

        # Cập nhật file JSON tóm tắt
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)


class ComprehensiveCheckpointManager:
    """Quản lý Dual Checkpoints (best_model.pth & last_model.pth) dựa trên metric val_macro_f1."""

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = "val_macro_f1",
        run_config: Optional[Dict[str, Any]] = None,
    ):
        self.chk_dir = Path(checkpoint_dir)
        self.chk_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.best_metric_val = -1.0
        self.best_epoch = 0

        # Lưu file config kèm theo để đảm bảo tính tái lập 100%
        if run_config:
            cfg_p = self.chk_dir / "run_config.json"
            with open(cfg_p, "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=4)

    def step(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scaler: Optional[Any] = None,
    ) -> bool:
        """Lưu last_model.pth và kiểm tra để lưu best_model.pth nếu val_macro_f1 đạt kỷ lục."""
        current_metric = metrics.get(self.metric_name, 0.0)

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 1. Luôn lưu mốc gần nhất last_model.pth
        last_path = self.chk_dir / "last_model.pth"
        torch.save(checkpoint_state, last_path)

        # 2. Kiểm tra kỷ lục val_macro_f1 để lưu best_model.pth
        is_best = current_metric > self.best_metric_val
        if is_best:
            self.best_metric_val = current_metric
            self.best_epoch = epoch
            best_path = self.chk_dir / "best_model.pth"
            torch.save(checkpoint_state, best_path)

        return is_best

    def load_best(self, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        """Nạp trọng số tối ưu nhất để đánh giá trên tập Test."""
        best_path = self.chk_dir / "best_model.pth"
        chk = torch.load(best_path, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        return chk
