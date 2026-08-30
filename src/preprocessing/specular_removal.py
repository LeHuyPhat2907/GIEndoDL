"""Module phát hiện và xóa điểm lóa sáng (Specular Reflections) bằng thuật toán Inpainting."""

import cv2
import numpy as np


class SpecularReflectionHandler:
    """Bộ phát hiện và tái tạo vùng bị lóa sáng trên ảnh nội soi."""

    def __init__(
        self,
        v_thresh: int = 240,
        s_thresh: int = 45,
        gray_thresh: int = 245,
        inpaint_radius: int = 4,
    ):
        """Khởi tạo handler.

        Args:
            v_thresh: Ngưỡng độ sáng Value trong HSV (0 - 255).
            s_thresh: Ngưỡng độ bão hòa Saturation tối đa của đốm trắng lóa.
            gray_thresh: Ngưỡng cường độ Grayscale cực đại.
            inpaint_radius: Bán kính láng giềng dùng để tái tạo điểm ảnh.
        """
        self.v_thresh = v_thresh
        self.s_thresh = s_thresh
        self.gray_thresh = gray_thresh
        self.inpaint_radius = inpaint_radius

    def detect_specular_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Tạo mặt nạ nhị phân định vị chính xác tất cả các đốm lóa sáng."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Điều kiện 1: Sắc tố trắng lóa (Độ sáng V cao, Độ bão hòa S thấp)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        hsv_mask = (v_channel >= self.v_thresh) & (s_channel <= self.s_thresh)

        # Điều kiện 2: Cường độ pixel tiệm cận 255
        gray_mask = gray >= self.gray_thresh

        # Kết hợp cả 2 điều kiện
        combined_mask = (hsv_mask | gray_mask).astype(np.uint8) * 255

        # Dilation nở biên 3x3 để bao phủ trọn vẹn rìa gradient của đốm lóa
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        return dilated_mask

    def inpaint_specular(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray = None,
        method: str = "telea",
    ) -> np.ndarray:
        """Tái tạo các điểm lóa sáng bằng thuật toán Inpainting."""
        if mask is None:
            mask = self.detect_specular_mask(img_bgr)

        # Nếu không có đốm lóa nào, trả về ảnh gốc
        if np.sum(mask) == 0:
            return img_bgr

        inpaint_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        cleaned_bgr = cv2.inpaint(img_bgr, mask, self.inpaint_radius, inpaint_flag)
        return cleaned_bgr
