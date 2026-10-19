"""Mô-đun Cầu nối Chuyển đổi Đặc trưng sang Visual Tokens (Task #99: Feature-to-Token Conversion).

ĐÓNG GÓP KỸ THUẬT (TASK #99):
- Đóng vai trò cầu nối bản lề (Bridge Module) kết nối giữa miền Không gian Tích chập (CNN-CBAM)
  và miền Chuỗi Tự chú ý (Transformer Sequence).
- Cơ chế hoạt động:
  1. (Tùy chọn) Tầng chiếu tuyến tính / 1x1 Conv giảm chiều: C_in (2048) -> D (512) + Norm + GELU.
  2. Duỗi phẳng không gian (Spatial Flattening): Reshape từ [B, D, H, W] -> [B, H*W, D].
  3. (Tùy chọn) Ghép nối Learnable [CLS] Token vào đầu chuỗi: [B, N, D] -> [B, N+1, D].
  4. Bổ sung Learnable Positional Embedding để bù đắp tính bất biến hoán vị của Transformer:
     z_0 = [x_cls; x_1; ...; x_N] + E_pos.
  5. Áp dụng Dropout điều hòa chuỗi token trước khi nạp vào Transformer Encoder.
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


class FeatureToTokenBridge(nn.Module):
    """Mô-đun Chuyển đổi Feature Map sang Token Sequence cho Transformer (Task #99)."""

    def __init__(
        self,
        in_channels: int = 2048,
        d_model: int = 512,
        grid_size: Tuple[int, int] = (7, 7),
        use_projection: bool = True,
        use_cls_token: bool = True,
        dropout: float = 0.1,
        pos_embed_type: str = "learnable",  # 'learnable' hoặc 'sincos2d'
    ):
        """Khởi tạo FeatureToTokenBridge.

        Args:
            in_channels: Số kênh đầu vào từ CNN-CBAM (mặc định 2048 tại Stage 4 ResNet-50).
            d_model: Số chiều vector token cho Transformer Encoder (mặc định 512).
            grid_size: Kích thước lưới không gian (H, W), mặc định (7, 7) ứng với N=49 tokens.
            use_projection: Có sử dụng tầng chiếu giảm số kênh 2048 -> d_model hay không.
            use_cls_token: Có ghép token đại diện phân loại [CLS] vào đầu chuỗi hay không.
            dropout: Tỷ lệ Dropout sau khi cộng Positional Embedding.
            pos_embed_type: Loại positional embedding ('learnable' hoặc 'sincos2d').
        """
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.grid_size = grid_size
        self.num_spatial_tokens = grid_size[0] * grid_size[1]
        self.use_projection = use_projection
        self.use_cls_token = use_cls_token
        self.pos_embed_type = pos_embed_type

        # 1. Tầng chiếu không gian / giảm số kênh (Projection Layer)
        if use_projection:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
            )
            token_dim = d_model
        else:
            self.proj = nn.Identity()
            token_dim = in_channels
            assert (
                token_dim == d_model
            ), f"Khi use_projection=False, in_channels ({in_channels}) phải bằng d_model ({d_model})."

        # 2. Learnable [CLS] Token
        self.total_tokens = self.num_spatial_tokens + (1 if use_cls_token else 0)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # 3. Positional Embedding
        if pos_embed_type == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.total_tokens, token_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sincos2d":
            self.register_buffer(
                "pos_embed",
                self._build_2d_sincos_position_embedding(
                    d_model=token_dim,
                    grid_size=grid_size,
                    has_cls_token=use_cls_token,
                ),
            )
        else:
            raise ValueError(f"Không hỗ trợ loại pos_embed_type: {pos_embed_type}")

        # 4. Dropout
        self.pos_drop = nn.Dropout(p=dropout)

    def _build_2d_sincos_position_embedding(
        self, d_model: int, grid_size: Tuple[int, int], has_cls_token: bool
    ) -> torch.Tensor:
        """Sinh ma trận mã hóa vị trí 2D Sine-Cosine cố định cho lưới H x W."""
        h, w = grid_size
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # [2, W, H]
        grid = torch.stack(grid, dim=0).reshape(2, 1, h, w)

        dim_per_coord = d_model // 2
        omega = torch.arange(dim_per_coord // 2, dtype=torch.float32)
        omega /= dim_per_coord / 2.0
        omega = 1.0 / (10000**omega)  # (D/4,)

        out_w = torch.einsum("m,d->md", grid[0].flatten(), omega)  # (HW, D/4)
        out_h = torch.einsum("m,d->md", grid[1].flatten(), omega)  # (HW, D/4)

        emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (HW, D/2)
        emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (HW, D/2)
        spatial_emb = torch.cat([emb_w, emb_h], dim=1)  # (HW, D)

        if has_cls_token:
            cls_emb = torch.zeros(1, d_model)
            full_emb = torch.cat([cls_emb, spatial_emb], dim=0)
        else:
            full_emb = spatial_emb

        return full_emb.unsqueeze(0)  # [1, N, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi feature map [B, C, H, W] thành token sequence [B, N_seq, D].

        Args:
            x: Feature map đầu vào từ CNN-CBAM [B, C, H, W].

        Returns:
            Tensor chuỗi tokens sẵn sàng nạp vào Transformer Encoder:
            - [B, 50, 512] nếu use_cls_token=True (49 visual + 1 CLS).
            - [B, 49, 512] nếu use_cls_token=False.
        """
        b = x.shape[0]

        # 1. Giảm chiều qua Conv 1x1 + BN + GELU: [B, C, H, W] -> [B, D, H, W]
        x_proj = self.proj(x)

        # 2. Duỗi phẳng chiều không gian: [B, D, H, W] -> [B, D, H*W] -> [B, H*W, D]
        tokens = x_proj.flatten(2).transpose(1, 2)  # [B, 49, D]

        # 3. Ghép [CLS] token vào đầu chuỗi nếu được kích hoạt
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(b, -1, -1)  # [B, 1, D]
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, 50, D]

        # 4. Cộng Positional Embedding và áp dụng Dropout
        tokens = self.pos_drop(tokens + self.pos_embed)

        return tokens
