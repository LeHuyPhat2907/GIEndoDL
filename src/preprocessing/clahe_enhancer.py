"""Module cân bằng độ sáng thích ứng CLAHE trên không gian màu LAB chuẩn y tế."""

from typing import Tuple
import cv2
import numpy as np


class CLAHEIlluminationNormalizer:
    """Bộ cân bằng độ sáng và tăng cường tương phản vi mạch trên không gian LAB."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        """Khởi tạo CLAHE normalizer.

        Args:
            clip_limit: Ngưỡng cắt đỉnh histogram để chống phóng đại nhiễu (Thường từ 1.5 - 3.0).
            tile_grid_size: Kích thước lưới chia nhỏ ảnh để xử lý cục bộ (Mặc định 8x8).
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply_clahe_lab(self, img_bgr: np.ndarray) -> np.ndarray:
        """Cân bằng sáng thích ứng trên kênh L của không gian màu CIE LAB.

        Bảo toàn 100% màu sắc nguyên bản của kênh A (Xanh lá - Đỏ) và B (Xanh
        dương - Vàng).
        """
        # 1. Chuyển từ BGR sang LAB
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 2. Áp dụng CLAHE riêng cho kênh độ sáng L (Luminance)
        cl_channel = self.clahe.apply(l_channel)

        # 3. Gộp lại các kênh và chuyển về BGR
        enhanced_lab = cv2.merge((cl_channel, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    def apply_standard_rgb_equalization(self, img_bgr: np.ndarray) -> np.ndarray:
        """Cân bằng sáng toàn cục trên RGB (Dùng để so sánh đối chứng nhược điểm)."""
        b, g, r = cv2.split(img_bgr)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.merge((b_eq, g_eq, r_eq))
