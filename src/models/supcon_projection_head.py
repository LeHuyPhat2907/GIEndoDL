"""Mô-đun Đầu Chiếu Tương Phản (Projection Head) cho Huấn Luyện SupCon (Task #110).

Theo công trình:
- "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR) - Chen et al., ICML 2020.
- "Supervised Contrastive Learning" - Khosla et al., NeurIPS 2020.

ĐÓNG GÓP KỸ THUẬT (TASK #110):
- Thiết kế mạng MLP phi tuyến tính (Non-linear MLP Projection Head):
  Linear(in_dim, hidden_dim) -> BatchNorm1d -> ReLU -> Linear(hidden_dim, out_dim).
- Mặc định: in_dim = 512 (đầu ra của Transformer GAP), hidden_dim = 512, out_dim = 128 (hoặc 256).
- Tích hợp chuẩn hóa L2 (L2 Normalization) trực tiếp tại đầu ra để đưa biểu diễn lên mặt cầu siêu không gian S^{d-1}.
- Cơ chế linh hoạt:
  * Cho phép tháo rời (discard / detach) sau khi kết thúc Stage 1 (Pretraining).
  * Chuyển đổi trơn tru sang Classification Head (Linear Classifier) trong Stage 2 (Fine-tuning).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConProjectionHead(nn.Module):
    """Đầu chiếu MLP phi tuyến tính phục vụ học biểu diễn tương phản SupCon (Task #110)."""

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 128,
        normalize: bool = True,
        dropout: float = 0.0,
    ):
        """Khởi tạo Projection Head.

        Args:
            in_dim: Kích thước vector biểu diễn đầu vào h (mặc định 512 từ Transformer GAP).
            hidden_dim: Kích thước tầng ẩn MLP (mặc định 512).
            out_dim: Kích thước không gian tương phản z (mặc định 128).
            normalize: Tự động chuẩn hóa L2 vector đầu ra z (mặc định True).
            dropout: Xác suất Dropout nếu cần điều hòa regularization.
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.normalize = normalize

        layers = [
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
        self.projection_mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Thực hiện biến đổi hình chiếu h -> z.

        Args:
            h: Tensor đặc trưng [batch_size, in_dim] hoặc [batch_size, n_views, in_dim].

        Returns:
            Tensor không gian tương phản z có cùng số chiều ngoài và kích thước out_dim.
        """
        original_shape = h.shape

        if len(original_shape) == 3:
            # Dạng [B, n_views, in_dim] -> làm phẳng thành [B * n_views, in_dim] để đi qua BatchNorm1d
            b, v, d = original_shape
            flat_h = h.view(b * v, d)
            z = self.projection_mlp(flat_h)
            if self.normalize:
                z = F.normalize(z, p=2, dim=1)
            return z.view(b, v, self.out_dim)

        elif len(original_shape) == 2:
            z = self.projection_mlp(h)
            if self.normalize:
                z = F.normalize(z, p=2, dim=1)
            return z

        else:
            raise ValueError(
                f"Đầu vào Projection Head yêu cầu tensor 2D hoặc 3D, nhận tensor với shape {original_shape}"
            )


class HybridModelWithProjectionHead(nn.Module):
    """Bộ khung kết hợp mô hình Hybrid CNN-CBAM-Transformer với Projection Head cho Stage 1.

    Cung cấp hàm tháo rời projection head để chuyển sang Stage 2 Fine-tuning.
    """

    def __init__(
        self,
        base_hybrid_model: nn.Module,
        proj_hidden_dim: int = 512,
        proj_out_dim: int = 128,
    ):
        super().__init__()
        self.base_model = base_hybrid_model
        in_dim = getattr(base_hybrid_model, "d_model", 512)
        self.projection_head = SupConProjectionHead(
            in_dim=in_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
            normalize=True,
        )

    def forward(
        self, x: torch.Tensor, return_representation: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass cho giai đoạn học tương phản Stage 1.

        Args:
            x: Batch ảnh hoặc đa góc nhìn [B, 3, H, W] hoặc [B, n_views, 3, H, W].
            return_representation: Có trả về vector biểu diễn h không.

        Returns:
            z: Vector tương phản đã chuẩn hóa L2 [B, out_dim] hoặc [B, n_views, out_dim].
            h: Vector biểu diễn ẩn nếu return_representation=True.
        """
        # Nếu x có đa góc nhìn [B, n_views, 3, H, W]
        if len(x.shape) == 5:
            b, n_views, c, h_img, w_img = x.shape
            x_flat = x.view(b * n_views, c, h_img, w_img)
            # Trích xuất biểu diễn h qua backbone + transformer GAP
            # Trong base_hybrid_model: dùng forward với return_attention_weights=True để lấy pooled_embedding
            _, aux = self.base_model(x_flat, return_attention_weights=True)
            h = aux["pooled_embedding"]  # [B * n_views, d_model]
            z = self.projection_head(h)  # [B * n_views, out_dim]
            z = z.view(b, n_views, -1)
            if return_representation:
                h = h.view(b, n_views, -1)
                return z, h
            return z, None

        # Trường hợp đơn góc nhìn [B, 3, H, W]
        _, aux = self.base_model(x, return_attention_weights=True)
        h = aux["pooled_embedding"]
        z = self.projection_head(h)
        if return_representation:
            return z, h
        return z, None

    def discard_projection_head(self) -> nn.Module:
        """Tháo bỏ Projection Head và trả về base model nguyên bản sẵn sàng cho Stage 2."""
        del self.projection_head
        return self.base_model
