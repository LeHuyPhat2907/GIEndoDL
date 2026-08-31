"""Module phân đoạn tổn thương tự động và trích xuất Patch ROI (U-Net & Bounding Box Extraction)."""

from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Khối tích chập kép (Conv -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LightweightUNet(nn.Module):
    """Kiến trúc U-Net gọn nhẹ tối ưu hóa cho phân đoạn Polyp nội soi."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        d1 = self.up1(x4)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.conv_up3(d3)

        logits = self.outc(d3)
        return torch.sigmoid(logits)


class LesionROIExtractor:
    """Bộ trích xuất và cắt cô lập vùng tổn thương polyp (ROI Patch Extractor)."""

    def __init__(self, pad_ratio: float = 0.15, min_size: int = 32):
        self.pad_ratio = pad_ratio
        self.min_size = min_size

    def extract_bboxes_from_mask(
        self, mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Tìm bounding boxes (x, y, w, h) của các khối u từ mặt nạ phân đoạn."""
        h, w = mask.shape[:2]
        bin_mask = (mask > 127).astype(np.uint8)

        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (self.min_size * self.min_size):
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Thêm khoảng đệm ngữ cảnh an toàn (Contextual Padding)
            pad_w = int(bw * self.pad_ratio)
            pad_h = int(bh * self.pad_ratio)

            x1 = max(0, bx - pad_w)
            y1 = max(0, by - pad_h)
            x2 = min(w, bx + bw + pad_w)
            y2 = min(h, by + bh + pad_h)

            bboxes.append((x1, y1, x2 - x1, y2 - y1))

        # Nếu không có contour nào đạt chuẩn, lấy toàn bộ khung hình
        if not bboxes:
            bboxes.append((0, 0, w, h))

        return bboxes

    def crop_lesion_patch(
        self, img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Cắt lấy patch ảnh vùng tổn thương độ nét cao."""
        x, y, w, h = bbox
        return img_bgr[y : y + h, x : x + w]
