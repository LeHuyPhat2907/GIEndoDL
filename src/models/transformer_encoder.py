"""Mô-đun Khối Transformer Encoder Tinh gọn có thể Cấu hình (Task #100).

ĐÓNG GÓP KỸ THUẬT (TASK #100):
- Thiết kế khối Transformer Encoder tuân thủ cấu trúc Pre-LayerNorm (Pre-LN) chuẩn mực:
  * z'_l = z_{l-1} + MHSA(LN_1(z_{l-1}))
  * z_l  = z'_l + FFN(LN_2(z'_l))
- Pre-LN đảm bảo độ dốc gradient lan truyền ổn định, triệt tiêu nguy cơ bùng nổ/triệt tiêu gradient,
  không yêu cầu warmup phức tạp.
- Khả năng cấu hình linh hoạt:
  * depth (số tầng L, khuyến nghị 2-4 tầng theo ghi chú Task #100).
  * dim (d_model, mặc định 512).
  * num_heads (số đầu tự chú ý h, mặc định 8, head_dim = 64).
  * mlp_ratio (tỷ lệ mở rộng FFN, mặc định 4.0 -> dim_feedforward = 2048).
  * dropout (tỷ lệ dropout, mặc định 0.1).
- Hỗ trợ xuất Attention Weights tại từng tầng phục vụ giải thích mô hình (Explainability / Attention Rollout).
"""

import sys
from typing import List, Tuple, Union
import torch
import torch.nn as nn

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class TransformerEncoderBlock(nn.Module):
    """Khối Transformer Encoder đơn lẻ với kiến trúc Pre-LayerNorm (Pre-LN)."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """Khởi tạo một khối Transformer Encoder.

        Args:
            d_model: Kích thước vector biểu diễn token D (mặc định 512).
            num_heads: Số lượng attention heads h (mặc định 8, head_dim = 64).
            mlp_ratio: Hệ số nhân kích thước tầng FFN (mặc định 4.0 -> FFN = 2048).
            dropout: Xác suất Dropout (mặc định 0.1).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        dim_feedforward = int(d_model * mlp_ratio)

        # 1. Nhánh Pre-LN Multi-Head Self-Attention (MHSA)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        # 2. Nhánh Pre-LN Feed-Forward Network (FFN)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass qua một khối Transformer Encoder.

        Args:
            x: Tensor tokens đầu vào [B, N, D].
            return_attn_weights: Có trả về ma trận trọng số attention [B, N, N] hay không.

        Returns:
            Tensor tokens đã được biểu diễn ngữ cảnh toàn cục [B, N, D].
        """
        # Pre-LN MHSA + Residual
        norm_x1 = self.norm1(x)
        attn_out, attn_weights = self.attn(
            norm_x1, norm_x1, norm_x1, need_weights=return_attn_weights
        )
        x = x + self.drop_attn(attn_out)

        # Pre-LN FFN + Residual
        norm_x2 = self.norm2(x)
        x = x + self.mlp(norm_x2)

        if return_attn_weights:
            return x, attn_weights
        return x


class ConfigurableTransformerEncoder(nn.Module):
    """Mô-đun Compact Transformer Encoder đa tầng có thể cấu hình linh hoạt (Task #100)."""

    def __init__(
        self,
        depth: int = 2,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """Khởi tạo Transformer Encoder đa tầng.

        Args:
            depth: Số lượng khối Transformer L (mặc định 2, khuyến nghị 2..4).
            dim: Số chiều embedding D (mặc định 512).
            num_heads: Số đầu tự chú ý h (mặc định 8).
            mlp_ratio: Hệ số nhân kích thước tầng FFN (mặc định 4.0).
            dropout: Tỷ lệ Dropout (mặc định 0.1).
        """
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        # Chuẩn hóa LayerNorm cuối cùng sau chuỗi các khối Pre-LN
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, return_all_attentions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass chuỗi tokens qua toàn bộ L tầng Transformer.

        Args:
            x: Chuỗi tokens [B, N, D].
            return_all_attentions: Có trả về danh sách trọng số attention của từng tầng hay không.

        Returns:
            Tensor tokens đầu ra [B, N, D] hoặc (out, list_of_attn_weights).
        """
        attn_maps = []
        for layer in self.layers:
            if return_all_attentions:
                x, attn = layer(x, return_attn_weights=True)
                attn_maps.append(attn)
            else:
                x = layer(x)

        x_out = self.final_norm(x)

        if return_all_attentions:
            return x_out, attn_maps
        return x_out
