"""Mô-đun Đầu Phân Loại Đa Cơ Chế Pooling cho Ảnh Y Tế (Task #101: Classification Head).

ĐÓNG GÓP KỸ THUẬT (TASK #101):
- Hỗ trợ 3 chiến lược tổng hợp biểu diễn từ chuỗi Tokens:
  1. 'gap' (Global Average Pooling qua chuỗi visual tokens):
     - Ghi chú Task #101: "GAP thường ổn định hơn class token cho medical imaging".
     - Lý do y học: Tổn thương nội soi tiêu hóa (viêm đại tràng diện rộng, mạng lưới mao mạch Barrett)
       phân bổ trên nhiều vùng không gian; GAP tích hợp thông tin từ toàn bộ các tokens, tránh phụ thuộc
       đơn cực vào [CLS] token và ổn định gradient trên các lớp hiếm.
  2. 'cls' (Trích xuất Class Token z[:, 0]):
     - Cơ chế tiêu chuẩn của Vision Transformer (Dosovitskiy et al., 2020).
  3. 'hybrid' (Hợp nhất CLS + GAP):
     - Ghép nối cả biểu diễn toàn cục [CLS] và trung bình không gian GAP: [z_cls; z_gap].
- Cấu trúc chuẩn hóa:
  Pooled Vector -> LayerNorm -> Dropout(p=0.1) -> Linear(d_model -> num_classes).
- Hỗ trợ trích xuất feature embeddings phục vụ trực quan hóa t-SNE / UMAP và phân tích cụm.
"""

from typing import Literal, Tuple, Union
import torch
import torch.nn as nn

PoolingStrategy = Literal["gap", "cls", "hybrid"]


class HybridClassificationHead(nn.Module):
    """Đầu phân loại linh hoạt hỗ trợ GAP, CLS Token và Hybrid Fusion (Task #101)."""

    def __init__(
        self,
        d_model: int = 512,
        num_classes: int = 23,
        pooling: PoolingStrategy = "gap",
        dropout: float = 0.1,
        has_cls_token: bool = True,
    ):
        """Khởi tạo Hybrid Classification Head.

        Args:
            d_model: Số chiều biểu diễn của tokens (mặc định 512).
            num_classes: Số lượng phân lớp mục tiêu (mặc định 23 lớp HyperKvasir).
            pooling: Chiến lược pooling ('gap', 'cls', hoặc 'hybrid').
            dropout: Xác suất Dropout chống quá khớp (mặc định 0.1).
            has_cls_token: Chuỗi token đầu vào có chứa [CLS] ở vị trí 0 hay không.
        """
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.pooling = pooling
        self.has_cls_token = has_cls_token

        if pooling == "cls" and not has_cls_token:
            raise ValueError(
                "Không thể sử dụng pooling='cls' khi chuỗi token không có [CLS] token (has_cls_token=False)."
            )

        # Đối với pooling='hybrid', cần tầng chiếu kết hợp 2*d_model -> d_model
        if pooling == "hybrid":
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
            )

        # Khối chuẩn hóa và phân loại chính
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(d_model, num_classes)

        # Khởi tạo trọng số Linear
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def extract_pooled_feature(self, tokens: torch.Tensor) -> torch.Tensor:
        """Tổng hợp chuỗi tokens [B, N, D] thành một vector biểu diễn duy nhất [B, D]."""
        if self.pooling == "cls":
            # Lấy token [CLS] tại index 0
            pooled = tokens[:, 0]
        elif self.pooling == "gap":
            # Global Average Pooling qua các visual tokens
            # Nếu có [CLS] ở vị trí 0, chỉ tính trung bình trên các visual tokens (index 1..N-1)
            if self.has_cls_token:
                visual_tokens = tokens[:, 1:]
            else:
                visual_tokens = tokens
            pooled = torch.mean(visual_tokens, dim=1)
        elif self.pooling == "hybrid":
            cls_token = tokens[:, 0]
            visual_tokens = tokens[:, 1:] if self.has_cls_token else tokens
            gap_token = torch.mean(visual_tokens, dim=1)
            fused = torch.cat([cls_token, gap_token], dim=-1)
            pooled = self.fusion_proj(fused)
        else:
            raise ValueError(f"Chiến lược pooling không hợp lệ: {self.pooling}")

        return pooled

    def forward(
        self, tokens: torch.Tensor, return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass qua classification head.

        Args:
            tokens: Chuỗi tokens từ Transformer Encoder [B, N, D].
            return_embeddings: Có trả về vector embedding [B, D] trước tầng Linear hay không.

        Returns:
            Logits phân loại [B, num_classes] hoặc tuple (logits, embeddings).
        """
        # 1. Trích xuất vector đặc trưng sau pooling
        feat = self.extract_pooled_feature(tokens)  # [B, D]

        # 2. Chuẩn hóa LayerNorm và Dropout
        normed_feat = self.norm(feat)
        dropped_feat = self.drop(normed_feat)

        # 3. Chiếu tuyến tính sang 23 logits lớp bệnh lý
        logits = self.fc(dropped_feat)

        if return_embeddings:
            return logits, feat
        return logits
