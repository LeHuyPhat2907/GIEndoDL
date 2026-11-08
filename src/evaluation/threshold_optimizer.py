# -*- coding: utf-8 -*-
"""Suy luận hậu kỳ: Dynamic Threshold Tuning & Temperature Scaling (Task D49 / Run 7).

Đặc tả kỹ thuật:
1. Temperature Scaling: Làm mềm phân phối xác suất p_c = softmax(z_c / T) với T in [1.2, 1.5] (mặc định T = 1.3).
2. Per-Class Threshold Optimization:
   - Tìm kiếm vector ngưỡng tau = [tau_1, ..., tau_23] tối đa hóa Macro F1 trên tập Validation.
   - Hàm quyết định: y_hat = argmax_c (P_c / tau_c).
   - Ngưỡng tham chiếu khởi tạo:
     + Lớp đa số (polyps, pylorus, cecum, bbps-2-3): tau_c ~ 0.55 - 0.65
     + Lớp trung bình (esophagitis, z-line, v.v.):   tau_c ~ 0.35 - 0.45
     + Lớp cực hiếm (hemorrhoids, barretts, uc-grade-1-2, ileum): tau_c ~ 0.15 - 0.22
3. Tối ưu hóa trực tiếp bằng thuật toán Nelder-Mead (Scipy Optimize).
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score


# 4 Lớp cực hiếm cần độ nhạy cao
EXTREME_RARE_CLASSES = {
    "hemorrhoids",
    "barretts",
    "ulcerative-colitis-grade-1-2",
    "ileum",
}

# Lớp đa số áp đảo gradient
MAJORITY_CLASSES = {
    "bbps-2-3",
    "cecum",
    "polyps",
    "pylorus",
}


def apply_temperature_scaling(
    logits: Union[np.ndarray, List[List[float]]],
    temperature: float = 1.3,
) -> np.ndarray:
    """Hiệu chuẩn logits với hệ số nhiệt độ T (Temperature Scaling)."""
    z = np.array(logits, dtype=np.float64)
    # Tránh tràn số: trừ max
    z_scaled = (z - np.max(z, axis=-1, keepdims=True)) / max(temperature, 1e-4)
    exp_z = np.exp(z_scaled)
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    return probs.astype(np.float32)


def get_default_prior_thresholds(class_names: List[str]) -> np.ndarray:
    """Khởi tạo vector ngưỡng tham chiếu lâm sàng ban đầu tau_0."""
    tau = np.zeros(len(class_names), dtype=np.float64)
    for i, cname in enumerate(class_names):
        if cname in EXTREME_RARE_CLASSES:
            tau[i] = 0.18  # Ngưỡng kích hoạt nhạy cảm cho lớp hiếm
        elif cname in MAJORITY_CLASSES:
            tau[i] = 0.58  # Ngưỡng chặt chẽ chống lấn át cho lớp đa số
        else:
            tau[i] = 0.38  # Ngưỡng cân bằng cho lớp trung bình
    return tau


def predict_with_thresholds(
    probs: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Dự đoán nhãn theo công thức: y_hat = argmax_c (P_c / tau_c)."""
    # Tránh chia cho 0 hoặc giá trị quá nhỏ
    safe_tau = np.maximum(tau, 1e-4)
    scaled_probs = probs / safe_tau
    return np.argmax(scaled_probs, axis=-1)


def optimize_per_class_thresholds(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    class_names: List[str],
    max_iter: int = 300,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Tối ưu hóa vector ngưỡng tau để tối đa hóa Macro F1 trên tập Validation."""
    init_tau = get_default_prior_thresholds(class_names)

    val_targets = np.array(val_targets, dtype=int)

    def objective(tau_candidate: np.ndarray) -> float:
        # Giới hạn tau trong [0.05, 1.0]
        clipped_tau = np.clip(tau_candidate, 0.05, 1.0)
        preds = predict_with_thresholds(val_probs, clipped_tau)
        macro_f1 = f1_score(val_targets, preds, average="macro", zero_division=0)
        return -float(macro_f1)

    # Tối ưu hóa Nelder-Mead
    res = minimize(
        objective,
        x0=init_tau,
        method="Nelder-Mead",
        options={
            "maxiter": max_iter,
            "adaptive": True,
            "xatol": 1e-3,
            "fatol": 1e-4,
        },
    )

    best_tau = np.clip(res.x, 0.05, 1.0)

    # Đánh giá trước và sau tối ưu
    raw_preds = np.argmax(val_probs, axis=-1)
    raw_macro_f1 = f1_score(val_targets, raw_preds, average="macro", zero_division=0)
    raw_bal_acc = balanced_accuracy_score(val_targets, raw_preds)
    raw_acc = accuracy_score(val_targets, raw_preds)

    opt_preds = predict_with_thresholds(val_probs, best_tau)
    opt_macro_f1 = f1_score(val_targets, opt_preds, average="macro", zero_division=0)
    opt_bal_acc = balanced_accuracy_score(val_targets, opt_preds)
    opt_acc = accuracy_score(val_targets, opt_preds)

    stats = {
        "raw_macro_f1": float(raw_macro_f1),
        "raw_balanced_acc": float(raw_bal_acc),
        "raw_acc": float(raw_acc),
        "opt_macro_f1": float(opt_macro_f1),
        "opt_balanced_acc": float(opt_bal_acc),
        "opt_acc": float(opt_acc),
        "f1_gain": float(opt_macro_f1 - raw_macro_f1),
    }

    return best_tau, stats
