"""Các mô-đun Tái hiệu chuẩn Chú ý Tích chập (CBAM - Woo et al., ECCV 2018).

Phục vụ:
- Task #96: Implement CBAM Channel Attention module
  * Nén không gian qua Global Average Pooling và Global Max Pooling.
  * Shared MLP (MLP chia sẻ trọng số) với tỷ lệ giảm kênh r=16.
  * Kích hoạt Sigmoid tạo Channel Attention Map M_c ∈ R^{B x C x 1 x 1}.
  * Tái hiệu chuẩn trọng số kênh (tập trung vào câu hỏi "WHAT" - đặc trưng vi mạch, sắc tố).

- Task #97: Implement CBAM Spatial Attention module
  * Nén kênh qua Channel-wise AvgPool và MaxPool dọc theo trục C.
  * Ghép kênh thành tensor 2 x H x W.
  * Tích chập Conv2d 7x7 (padding 3) và Sigmoid tạo Spatial Attention Map M_s ∈ R^{B x 1 x H x W}.
  * Định vị vùng tổn thương (tập trung vào câu hỏi "WHERE" - ranh giới polyp, ổ loét).
"""

import sys
from typing import Tuple
import torch
import torch.nn as nn

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ChannelAttention(nn.Module):
    """Mô-đun Chú ý Kênh (Channel Attention Module - CAM) chuẩn mực (Task #96).

    Cơ chế toán học:
    M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
           = σ( W_1(W_0(F_avg^c)) + W_1(W_0(F_max^c)) )
    trong đó:
    - W_0 ∈ R^{(C/r) x C}, W_1 ∈ R^{C x (C/r)} là các trọng số được chia sẻ giữa hai nhánh.
    - r là hệ số giảm kênh (reduction ratio, mặc định r=16).
    """

    def __init__(self, in_planes: int, ratio: int = 16, min_planes: int = 8):
        """Khởi tạo Channel Attention Module.

        Args:
            in_planes: Số kênh đầu vào C.
            ratio: Tỷ lệ nén kênh r (mặc định 16).
            min_planes: Số kênh nén tối thiểu để tránh bottleneck quá hẹp khi C nhỏ.
        """
        super().__init__()
        self.in_planes = in_planes
        self.ratio = ratio

        # 1. Hai nhánh Pooling toàn cục nén không gian H x W -> 1 x 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 2. Shared MLP chia sẻ trọng số hoàn toàn giữa nhánh Avg và Max
        reduced_planes = max(in_planes // ratio, min_planes)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, bias=False),
        )

        # 3. Hàm kích hoạt cổng Sigmoid đưa trọng số về khoảng (0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, return_map: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass Channel Attention.

        Args:
            x: Tensor đầu vào [B, C, H, W].
            return_map: Nếu True, trả về cả bản đồ trọng số kênh M_c [B, C, 1, 1].

        Returns:
            Tensor đã được tái hiệu chuẩn [B, C, H, W] hoặc tuple (out, channel_map).
        """
        # Trích xuất vector thống kê kênh
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        # Cộng gộp và áp dụng Sigmoid
        channel_map = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]

        # Tái hiệu chuẩn bằng phép nhân từng phần tử dọc theo trục kênh
        out = x * channel_map

        if return_map:
            return out, channel_map
        return out


class SpatialAttention(nn.Module):
    """Mô-đun Chú ý Không gian (Spatial Attention Module - SAM) chuẩn mực (Task #97).

    Cơ chế toán học:
    M_s(F) = σ( f^{7x7}( [AvgPool(F); MaxPool(F)] ) )
    trong đó:
    - AvgPool(F) và MaxPool(F) là pooling dọc theo trục kênh (Channel-wise pooling).
    - [AvgPool(F); MaxPool(F)] là phép ghép nối kênh (concatenation) tạo tensor 2 x H x W.
    - f^{7x7} là tầng tích chập kích thước 7x7 với padding 3 để bảo toàn kích thước không gian.
    """

    def __init__(self, kernel_size: int = 7):
        """Khởi tạo Spatial Attention Module.

        Args:
            kernel_size: Kích thước bộ lọc không gian (mặc định 7, có thể chọn 3).
        """
        super().__init__()
        assert kernel_size in (
            3,
            7,
        ), "Kernel size của Spatial Attention phải là 3 hoặc 7."
        padding = kernel_size // 2

        # Tầng tích chập 2 kênh -> 1 kênh để hòa trộn thông tin không gian
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, return_map: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass Spatial Attention.

        Args:
            x: Tensor đầu vào [B, C, H, W].
            return_map: Nếu True, trả về cả bản đồ trọng số không gian M_s [B, 1, H, W].

        Returns:
            Tensor đã được định vị không gian [B, C, H, W] hoặc tuple (out, spatial_map).
        """
        # Channel-wise pooling dọc theo chiều C (dim=1)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Ghép nối hai bản đồ 1 kênh thành tensor 2 kênh
        cat_features = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        # Tích chập 7x7 và Sigmoid
        spatial_map = self.sigmoid(self.conv(cat_features))  # [B, 1, H, W]

        # Tái hiệu chuẩn không gian
        out = x * spatial_map

        if return_map:
            return out, spatial_map
        return out


class CBAM(nn.Module):
    """Khối Tái hiệu chuẩn Không gian và Kênh hoàn chỉnh (CBAM).

    Thực thi tuần tự:
    F' = M_c(F) ⊗ F
    F'' = M_s(F') ⊗ F'
    """

    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        """Khởi tạo CBAM Module.

        Args:
            in_planes: Số kênh đầu vào C.
            ratio: Tỷ lệ nén kênh r của Channel Attention.
            kernel_size: Kích thước kernel của Spatial Attention.
        """
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(
        self, x: torch.Tensor, return_maps: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass CBAM.

        Args:
            x: Tensor đầu vào [B, C, H, W].
            return_maps: Nếu True, trả về cả (out, channel_map, spatial_map).
        """
        if return_maps:
            x_ca, c_map = self.ca(x, return_map=True)
            x_out, s_map = self.sa(x_ca, return_map=True)
            return x_out, c_map, s_map
        else:
            x_out = self.ca(x)
            x_out = self.sa(x_out)
            return x_out
