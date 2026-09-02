"""Module khung huấn luyện học sâu mô-đun hóa (Modular PyTorch Training Engine)."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """Theo dõi chỉ số kiểm thử và kích hoạt dừng sớm khi mô hình ngừng tiến bộ."""

    def __init__(self, patience: int = 7, mode: str = "max", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True

        improved = (
            (current_score > self.best_score + self.delta)
            if self.mode == "max"
            else (current_score < self.best_score - self.delta)
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ModelCheckpoint:
    """Quản lý lưu trữ và nạp các mốc trọng số tốt nhất (Best Checkpoints)."""

    def __init__(self, checkpoint_dir: str, filename: str = "best_model.pth"):
        self.save_dir = Path(checkpoint_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.save_dir / filename

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metric_val: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_val": metric_val,
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, self.filepath)

    def load(self, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        checkpoint = torch.load(self.filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint


class ModularTrainer:
    """Khung điều phối huấn luyện và kiểm định mô hình học sâu toàn diện."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Tự động kích hoạt AMP khi có GPU
        self.use_amp = self.device.type == "cuda" and self.config.get("use_amp", True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        patience = self.config.get("early_stopping_patience", 7)
        self.early_stopping = EarlyStopping(patience=patience, mode="max")

        chk_dir = self.config.get(
            "checkpoint_dir",
            os.path.join(os.getcwd(), "models", "checkpoints"),
        )
        self.checkpoint_manager = ModelCheckpoint(checkpoint_dir=chk_dir)

    def train_one_epoch(self) -> float:
        """Huấn luyện 1 epoch với Automatic Mixed Precision (AMP)."""
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            images = batch[0].to(self.device, non_blocking=True)
            targets = batch[1].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self) -> Tuple[float, float, float]:
        """Kiểm định mô hình trên tập validation và tính các chỉ số lâm sàng."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = accuracy_score(all_targets, all_preds) * 100.0
        val_macro_f1 = (
            f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
        )

        return val_loss, val_acc, val_macro_f1
