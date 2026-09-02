"""Module tích hợp Weights & Biases (W&B) với cơ chế phòng vệ Offline an toàn cho đề tài."""

import os
from typing import Any, Dict, Optional
import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """Wrapper quản lý ghi nhật ký thực nghiệm W&B với chế độ Offline/Disabled an toàn."""

    def __init__(
        self,
        project_name: str = "GIEndoDL-HyperKvasir-Benchmark",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "offline",  # 'online', 'offline', hoặc 'disabled'
    ):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.mode = mode
        self.run = None

        if WANDB_AVAILABLE:
            # Thiết lập biến môi trường
            os.environ["WANDB_SILENT"] = "true"
            if self.mode == "offline":
                os.environ["WANDB_MODE"] = "offline"

            self.run = wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                mode=self.mode,
                reinit=True,
            )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Ghi nhận các chỉ số huấn luyện (Loss, Acc, F1, LR, VRAM)."""
        # Tự động đo lường VRAM GPU nếu có
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024**2)
            metrics["system/gpu_memory_allocated_mb"] = round(vram_mb, 1)

        if WANDB_AVAILABLE and self.run is not None:
            wandb.log(metrics, step=step)

    def finish(self):
        """Đóng phiên làm việc W&B một cách an toàn."""
        if WANDB_AVAILABLE and self.run is not None:
            wandb.finish()
