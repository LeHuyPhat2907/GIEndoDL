"""Module chuẩn hóa kích thước ảnh nội soi: Direct Resize vs Letterbox Padding (Bicubic)."""

from typing import Tuple
import cv2
import numpy as np


class MedicalImageResizer:
    """Bộ chuẩn hóa kích thước ảnh y tế hỗ trợ cả 224x224 và 384x384."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        interpolation: int = cv2.INTER_CUBIC,
    ):
        """Khởi tạo resizer.

        Args:
            target_size: Kích thước đích (W, H) chuẩn hóa.
            interpolation: Thuật toán nội suy điểm ảnh (Mặc định Bicubic chất lượng cao).
        """
        self.target_size = target_size
        self.interpolation = interpolation

    def resize_direct(
        self, img_bgr: np.ndarray, target_size: Tuple[int, int] = None
    ) -> np.ndarray:
        """Kéo giãn ảnh trực tiếp về kích thước đích (Không viền đệm)."""
        t_size = target_size or self.target_size
        return cv2.resize(img_bgr, t_size, interpolation=self.interpolation)

    def resize_letterbox(
        self,
        img_bgr: np.ndarray,
        target_size: Tuple[int, int] = None,
        pad_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """Resize bảo toàn 100% tỷ lệ khung hình (Aspect Ratio) kết hợp đệm đối xứng.

        Chống hiện tượng biến dạng méo mó hình học của tổn thương polyp.
        """
        t_w, t_h = target_size or self.target_size
        h, w = img_bgr.shape[:2]

        # Tính toán hệ số tỷ lệ co giãn
        scale = min(t_w / w, t_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize ảnh giữ nguyên tỷ lệ
        resized_img = cv2.resize(
            img_bgr, (new_w, new_h), interpolation=self.interpolation
        )

        # Tạo khung ảnh đích với màu đệm pad_color
        canvas = np.full((t_h, t_w, 3), pad_color, dtype=np.uint8)

        # Căn giữa ảnh vào khung
        x_offset = (t_w - new_w) // 2
        y_offset = (t_h - new_h) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img

        return canvas
