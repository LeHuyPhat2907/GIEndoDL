"""Bộ Trích Xuất Đặc Trưng Kết Cấu và Màu Sắc Truyền Thống (Handcrafted Features) (Task #117).

ĐÓNG GÓP KỸ THUẬT (TASK #117):
- Trích xuất 4 nhóm đặc trưng thị giác truyền thống bổ trợ cho mạng nơ-ron học sâu:
  1. Local Binary Patterns (LBP): Nắm bắt vi cấu trúc bề mặt niêm mạc và kiểu hố rãnh (Pit Pattern) (P=8, R=1, uniform -> 10 bins).
  2. Ma trận đồng xuất mức xám (Gray-Level Co-occurrence Matrix - GLCM / Haralick):
     Đo lường 6 tính chất (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM) trên 4 góc hướng (0, 45, 90, 135 độ) -> 24 đặc trưng.
  3. Lược đồ màu đa kênh (Color Histogram):
     Trích xuất trên không gian màu HSV nhạy cảm thị giác y học (32 bins mỗi kênh H, S, V) -> 96 đặc trưng.
  4. Bộ lọc Gabor đa tỷ lệ và hướng (Gabor Filter Bank):
     4 tỷ lệ tần số x 6 góc hướng = 24 bộ lọc; tính giá trị trung bình (mean) và độ lệch chuẩn (std) của ảnh phản hồi -> 48 đặc trưng.
  Tổng cộng vector đặc trưng truyền thống: 10 + 24 + 96 + 48 = 178 chiều (hoặc 227 chiều khi mở rộng).
"""

# ruff: noqa: E402
from pathlib import Path
import sys
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class HandcraftedFeatureExtractor:
    """Bộ trích xuất 4 nhóm đặc trưng kết cấu và màu sắc truyền thống cho ảnh nội soi."""

    def __init__(
        self,
        lbp_points: int = 8,
        lbp_radius: int = 1,
        color_bins: int = 32,
    ):
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.color_bins = color_bins

        # Tạo trước ngân hàng 24 bộ lọc Gabor (4 scales x 6 orientations)
        self.gabor_filters = []
        ksize = 21
        for theta in np.arange(0, np.pi, np.pi / 6):  # 6 hướng: 0, 30, 60, 90, 120, 150
            for sigma in (2.0, 4.0):  # 2 scales sigma
                for lamda in (np.pi / 4, np.pi / 2):  # 2 bước sóng
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F
                    )
                    self.gabor_filters.append(kernel)

    def extract_single_image(self, img_rgb: np.ndarray) -> np.ndarray:
        """Trích xuất vector đặc trưng truyền thống từ 1 ảnh RGB (uint8 hoặc float [0, 1]).

        Returns:
            Vector 1D numpy array kích thước 178 chiều đã chuẩn hóa.
        """
        if img_rgb.dtype != np.uint8:
            if img_rgb.max() <= 1.0:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            else:
                img_rgb = img_rgb.astype(np.uint8)

        # Đổi sang ảnh xám
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Local Binary Patterns (LBP)
        lbp = local_binary_pattern(
            gray, self.lbp_points, self.lbp_radius, method="uniform"
        )
        n_bins = self.lbp_points + 2  # 10 bins cho P=8 uniform
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )

        # 2. GLCM / Haralick Texture
        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )
        properties = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ]
        haralick_feats = []
        for prop in properties:
            val = graycoprops(glcm, prop).ravel()
            haralick_feats.extend(val)
        haralick_arr = np.array(haralick_feats, dtype=np.float32)

        # 3. Color Histogram (HSV)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180]).ravel()
        s_hist = cv2.calcHist([hsv], [1], None, [self.color_bins], [0, 256]).ravel()
        v_hist = cv2.calcHist([hsv], [2], None, [self.color_bins], [0, 256]).ravel()
        color_hist = np.concatenate([h_hist, s_hist, v_hist])
        color_hist = color_hist / (color_hist.sum() + 1e-8)

        # 4. Gabor Filter Responses
        gabor_feats = []
        for kernel in self.gabor_filters:
            fimg = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor_feats.append(float(np.mean(fimg)))
            gabor_feats.append(float(np.std(fimg)))
        gabor_arr = np.array(gabor_feats, dtype=np.float32)

        # Chuẩn hóa min-max z-score cho từng thành phần
        haralick_norm = (haralick_arr - np.mean(haralick_arr)) / (
            np.std(haralick_arr) + 1e-8
        )
        gabor_norm = (gabor_arr - np.mean(gabor_arr)) / (np.std(gabor_arr) + 1e-8)

        # Ghép nối vector: 10 + 24 + 96 + 48 = 178 chiều
        feature_vector = np.concatenate(
            [lbp_hist, haralick_norm, color_hist, gabor_norm]
        )
        return feature_vector.astype(np.float32)

    def extract_batch(self, images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Trích xuất vector đặc trưng cho một batch ảnh [B, 3, H, W] hoặc [B, H, W, 3]."""
        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
            if images_np.ndim == 4 and images_np.shape[1] == 3:
                # [B, 3, H, W] -> [B, H, W, 3]
                images_np = np.transpose(images_np, (0, 2, 3, 1))
        else:
            images_np = images

        batch_feats = []
        for i in range(len(images_np)):
            feat = self.extract_single_image(images_np[i])
            batch_feats.append(feat)

        return torch.from_numpy(np.array(batch_feats, dtype=np.float32))
