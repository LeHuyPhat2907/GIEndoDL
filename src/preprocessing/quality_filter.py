"""Module Bộ lọc Chất lượng Quang học Tự động (Image Quality Filter).

Sử dụng thuật toán Laplacian Variance (độ sắc nét) và Ngưỡng phơi sáng
(Mean Brightness) để phát hiện và cô lập các ảnh nội soi phế phẩm (mờ nhòe cực đoan, quá tối).
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd


class ImageQualityFilter:
    """Bộ lọc chất lượng ảnh y tế tự động cho nội soi tiêu hóa."""

    def __init__(
        self,
        min_laplacian_var: float = 25.0,
        min_brightness: float = 30.0,
        max_brightness: float = 235.0,
        protected_classes: Optional[List[str]] = None,
    ):
        """Khởi tạo bộ lọc chất lượng.

        Args:
            min_laplacian_var: Ngưỡng mờ cực đoan (mặc định 25.0). Dưới ngưỡng này ảnh mất hoàn toàn chi tiết niêm mạc.
            min_brightness: Ngưỡng tối cực đoan (mặc định 30.0). Dưới ngưỡng này ảnh thiếu sáng nghiêm trọng.
            max_brightness: Ngưỡng chói sáng cháy trắng (mặc định 235.0).
            protected_classes: Danh sách các phân lớp thiểu số được bảo vệ (không tự động xóa nếu chưa xác nhận).
        """
        self.min_laplacian = min_laplacian_var
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.protected_classes = set(
            protected_classes or ["hemorrhoids", "ileum", "barretts"]
        )

    def evaluate_image(self, image_input) -> Dict:
        """Đánh giá chất lượng của 1 ảnh (có thể nhận Path hoặc numpy array BGR)."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                return {
                    "is_valid": False,
                    "laplacian_var": 0.0,
                    "mean_brightness": 0.0,
                    "rejection_reason": "cannot_read_file",
                }
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise ValueError(
                "Input phải là đường dẫn file (str/Path) hoặc numpy array."
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_br = float(np.mean(gray))

        rejection_reasons = []
        if lap_var < self.min_laplacian:
            rejection_reasons.append("extreme_motion_blur")
        if mean_br < self.min_brightness:
            rejection_reasons.append("severe_underexposure")
        if mean_br > self.max_brightness:
            rejection_reasons.append("severe_overexposure")

        is_valid = len(rejection_reasons) == 0

        return {
            "is_valid": is_valid,
            "laplacian_var": round(lap_var, 2),
            "mean_brightness": round(mean_br, 2),
            "rejection_reason": ";".join(rejection_reasons)
            if rejection_reasons
            else "none",
        }

    def filter_dataset_directory(
        self, raw_dir: str, max_workers: int = 8
    ) -> pd.DataFrame:
        """Quét và phân loại chất lượng toàn bộ thư mục dữ liệu."""
        raw_path = Path(raw_dir)
        files = [
            f
            for f in raw_path.rglob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        def _process(p: Path):
            eval_res = self.evaluate_image(p)
            class_name = p.parent.name
            # Bảo vệ lớp cực hiếm
            if not eval_res["is_valid"] and class_name in self.protected_classes:
                eval_res["is_valid"] = True
                eval_res["rejection_reason"] += ";protected_minority_override"

            return {
                "file_name": p.name,
                "relative_path": str(p.relative_to(raw_path)),
                "class_name": class_name,
                "laplacian_var": eval_res["laplacian_var"],
                "mean_brightness": eval_res["mean_brightness"],
                "is_valid": eval_res["is_valid"],
                "rejection_reason": eval_res["rejection_reason"],
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            records = list(executor.map(_process, files))

        return pd.DataFrame(records)
