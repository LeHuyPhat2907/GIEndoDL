"""Module tự động phát hiện viền đen, cắt ROI và làm sạch nhiễu thiết bị nội soi."""

from typing import Tuple
import cv2
import numpy as np


class EndoscopeROIExtractor:
    """Bộ trích xuất vùng nội soi hữu ích (ROI) và lọc viền thiết bị chuẩn y tế."""

    def __init__(
        self,
        threshold_val: int = 15,
        min_area_ratio: float = 0.35,
        padding: int = 2,
    ):
        self.threshold_val = threshold_val
        self.min_area_ratio = min_area_ratio
        self.padding = padding

    def detect_roi_bbox(self, img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """Tìm tọa độ bounding box (x, y, w, h) của vùng nội soi hữu ích."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Phân đoạn nhị phân để tách vùng sáng niêm mạc khỏi viền đen
        _, thresh = cv2.threshold(gray, self.threshold_val, 255, cv2.THRESH_BINARY)

        # 2. Phép toán hình thái học (Morphological Closing) để lấp đầy các khoảng tối bên trong
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 3. Tìm các đường viền (Contours)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0, 0, w, h

        # 4. Lấy đường viền có diện tích lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Kiểm tra nếu diện tích đủ lớn (> min_area_ratio)
        if area < (h * w * self.min_area_ratio):
            return 0, 0, w, h

        x, y, bw, bh = cv2.boundingRect(largest_contour)

        # Thêm padding an toàn và kiểm tra biên
        x = max(0, x + self.padding)
        y = max(0, y + self.padding)
        bw = min(w - x, bw - 2 * self.padding)
        bh = min(h - y, bh - 2 * self.padding)

        return int(x), int(y), int(bw), int(bh)

    def crop_roi(self, img_bgr: np.ndarray) -> np.ndarray:
        """Cắt ảnh theo vùng ROI hữu ích."""
        x, y, w, h = self.detect_roi_bbox(img_bgr)
        return img_bgr[y : y + h, x : x + w]

    def remove_text_and_markers(
        self, img_bgr: np.ndarray, inpaint_radius: int = 3
    ) -> np.ndarray:
        """Phát hiện chữ số ngày giờ / marker góc màn hình và inpaint làm sạch."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        text_mask = np.zeros((h, w), dtype=np.uint8)

        # Góc dưới bên trái (vùng thường in timestamp và sơ đồ ống soi)
        corner_left = gray[int(h * 0.70) :, : int(w * 0.35)]
        _, thresh_corner = cv2.threshold(corner_left, 240, 255, cv2.THRESH_BINARY)
        text_mask[int(h * 0.70) :, : int(w * 0.35)] = thresh_corner

        if np.sum(text_mask) > 0:
            cleaned = cv2.inpaint(img_bgr, text_mask, inpaint_radius, cv2.INPAINT_TELEA)
            return cleaned
        return img_bgr
