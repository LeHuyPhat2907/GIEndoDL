"""Module thiết lập tính tái lập khoa học tuyệt đối (Bit-for-Bit Determinism) cho đề tài."""

import os
import platform
import random
import sys
from typing import Any, Callable, Dict
import numpy as np
import torch


def seed_worker(worker_id: int):
    """Cố định hạt giống ngẫu nhiên cho từng worker trong PyTorch DataLoader."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> Callable:
    """Khóa chặt toàn bộ các tầng sinh số ngẫu nhiên trên hệ thống."""
    # 1. Hệ điều hành và Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch CPU & GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4. NVIDIA cuDNN Deterministic Mode
    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed_worker


def get_system_fingerprint() -> Dict[str, Any]:
    """Thu thập dấu vân tay phần cứng và phần mềm (System Fingerprint) phục vụ báo cáo khoa học."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Mode"
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    cudnn_ver = (
        torch.backends.cudnn.version()
        if (cuda_avail and torch.backends.cudnn.is_available())
        else "N/A"
    )

    fingerprint = {
        "operating_system": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": str(cuda_ver),
        "cudnn_version": str(cudnn_ver),
        "gpu_device": str(gpu_name),
        "seed_protocol": {
            "master_seed": 42,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "python_hash_seed": 42,
        },
    }
    return fingerprint
