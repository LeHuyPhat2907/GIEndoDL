"""Module cân bằng màu sắc Reinhard (Reinhard Color Normalization) trên không gian LAB."""

import cv2
import numpy as np


class ReinhardColorNormalizer:
    """Bộ chuẩn hóa phân phối màu sắc theo ảnh tham chiếu y khoa (Reinhard et al. 2001)."""

    def __init__(self, target_means=None, target_stds=None):
        """Khởi tạo với thông số thống kê LAB tham chiếu (Target Reference).

        Nếu None, sẽ tự động fit thông qua ảnh tham chiếu chuẩn.
        """
        self.target_means = target_means
        self.target_stds = target_stds

    def fit(self, ref_img_bgr: np.ndarray):
        """Học phân phối màu sắc (mean, std) từ một ảnh tham chiếu y khoa chuẩn."""
        lab = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.target_means = [np.mean(lab[:, :, i]) for i in range(3)]  # L, A, B
        self.target_stds = [np.std(lab[:, :, i]) + 1e-6 for i in range(3)]  # L, A, B

    def transform(self, src_img_bgr: np.ndarray) -> np.ndarray:
        """Căn chỉnh phân phối màu của ảnh nguồn về phân phối của ảnh tham chiếu."""
        if self.target_means is None or self.target_stds is None:
            raise ValueError("Cần gọi hàm fit(ref_img) trước khi thực hiện transform!")

        lab_src = cv2.cvtColor(src_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_means = [np.mean(lab_src[:, :, i]) for i in range(3)]
        src_stds = [np.std(lab_src[:, :, i]) + 1e-6 for i in range(3)]

        norm_lab = np.zeros_like(lab_src)

        # Áp dụng công thức chuyển dịch Reinhard cho từng kênh L, A, B
        for i in range(3):
            norm_lab[:, :, i] = (
                (lab_src[:, :, i] - src_means[i]) * (self.target_stds[i] / src_stds[i])
            ) + self.target_means[i]

        # Giới hạn giá trị trong khoảng hợp lệ [0, 255]
        norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)
        norm_bgr = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2BGR)

        return norm_bgr
