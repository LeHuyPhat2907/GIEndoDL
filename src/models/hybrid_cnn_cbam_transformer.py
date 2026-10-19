"""Kiến trúc Tổng thể Mô hình Lai: Hybrid CNN-CBAM-Transformer (Tasks #94, #96-#101).

ĐÓNG GÓP KHOA HỌC CHÍNH (CORE CONTRIBUTION) CỦA LUẬN VĂN:
Tích hợp 4 trụ cột công nghệ thị giác máy tính trong một kiến trúc thống nhất:
1. CNN Backbone (ResNet-50) (Task #95): Trích xuất đặc trưng cục bộ phân cấp đa tầng (Stage 1 -> 4).
2. CBAM Modules (Task #96, #97, #98): Tái hiệu chuẩn kênh (Channel Attention - lọc màu vi mạch niêm mạc)
   và không gian (Spatial Attention - định vị tổn thương) với các chiến lược Selective Placement.
3. Feature-to-Token Bridge (Task #99): Chuyển đổi miền feature map -> chuỗi N=49 visual tokens + Learnable Positional Embedding.
4. Compact Transformer Encoder (Task #100): Mô hình hóa ngữ cảnh tầm xa toàn cảnh giữa 49 visual tokens.
5. Hybrid Classification Head (Task #101): Tổng hợp biểu diễn bằng GAP (Global Average Pooling) hoặc [CLS] token
   -> phân loại chính xác 23 lớp bệnh lý nội soi tiêu hóa HyperKvasir.
"""

# ruff: noqa: E402
from pathlib import Path
import sys
from typing import Any, Dict, Literal, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.models.bridge_feature_to_token import FeatureToTokenBridge
from src.models.cbam_integrated_backbone import CBAMIntegratedResNet50
from src.models.classification_head import HybridClassificationHead
from src.models.transformer_encoder import (
    ConfigurableTransformerEncoder,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Giữ lại alias để đảm bảo tương thích ngược
CompactTransformerEncoder = ConfigurableTransformerEncoder


class HybridCNNCBAMTransformer(nn.Module):
    """Kiến trúc Tổng thể Mô hình Lai Hybrid CNN-CBAM-Transformer hoàn chỉnh (Tasks #94-#101).

    Cấu trúc luồng dữ liệu (Dataflow):
    1. Ảnh đầu vào: [B, 3, 224, 224]
    2. ResNet-50 Backbone tích hợp CBAM (Stage 1 -> 4):
       - Stem:    [B, 64, 56, 56]
       - Stage 1: [B, 256, 56, 56]
       - Stage 2: [B, 512, 28, 28]
       - Stage 3: [B, 1024, 14, 14]
       - Stage 4: [B, 2048, 7, 7] (được tái hiệu chuẩn bằng CBAM)
    3. Feature-to-Token Bridge:
       - Chiếu 1x1 Conv: 2048 -> d_model (512)
       - Duỗi phẳng không gian: 7x7 -> 49 visual tokens
       - Ghép [CLS] token (nếu pooling='cls' hoặc 'hybrid'): chuỗi 50 tokens [B, 50, 512]
       - Cộng Learnable Positional Embedding
    4. Compact Transformer Encoder:
       - L tầng MHSA (L=2..4, d=512, h=8, FFN=2048): [B, N, 512]
    5. Hybrid Classification Head:
       - GAP (Global Average Pooling) hoặc [CLS] Token -> LayerNorm -> Dropout -> Linear(23)
    """

    def __init__(
        self,
        num_classes: int = 23,
        pretrained_backbone: bool = True,
        cbam_placement: Literal["stage4", "stage3_4", "all_stages"] = "stage4",
        transformer_layers: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        token_pooling: Literal["gap", "cls", "hybrid"] = "gap",
        dropout: float = 0.1,
        cbam_ratio: int = 16,
        cbam_kernel_size: int = 7,
        use_multistage_cbam: bool = False,
    ):
        """Khởi tạo Hybrid CNN-CBAM-Transformer.

        Args:
            num_classes: Số lượng phân lớp (mặc định 23 lớp).
            pretrained_backbone: Nạp trọng số ImageNet cho ResNet-50.
            cbam_placement: Vị trí đặt khối CBAM ('stage4', 'stage3_4', 'all_stages').
            transformer_layers: Số tầng Transformer Encoder (L=2..4).
            d_model: Số chiều biểu diễn của visual token (512).
            nhead: Số attention heads (8).
            dim_feedforward: Kích thước tầng ẩn FFN (2048).
            token_pooling: Cơ chế pooling đầu ra ('gap', 'cls', 'hybrid').
            dropout: Xác suất Dropout.
            cbam_ratio: Hệ số giảm kênh r trong Channel Attention (16).
            cbam_kernel_size: Kích thước kernel trong Spatial Attention (7).
            use_multistage_cbam: Tương thích ngược (True tương đương cbam_placement='all_stages').
        """
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.token_pooling = token_pooling

        # Xử lý cờ tương thích ngược
        if use_multistage_cbam:
            cbam_placement = "all_stages"
        self.cbam_placement = cbam_placement

        # 1. CNN Backbone tích hợp CBAM (Task #95, #96, #97, #98)
        self.backbone = CBAMIntegratedResNet50(
            pretrained=pretrained_backbone,
            placement=cbam_placement,
            cbam_ratio=cbam_ratio,
            cbam_kernel_size=cbam_kernel_size,
        )

        # 2. Feature-to-Token Bridge (Task #99)
        use_cls = token_pooling in ("cls", "hybrid")
        self.bridge = FeatureToTokenBridge(
            in_channels=2048,
            d_model=d_model,
            grid_size=(7, 7),
            use_projection=True,
            use_cls_token=use_cls,
            dropout=dropout,
            pos_embed_type="learnable",
        )

        # 3. Compact Transformer Encoder (Task #100)
        mlp_ratio = dim_feedforward / d_model
        self.transformer_encoder = ConfigurableTransformerEncoder(
            depth=transformer_layers,
            dim=d_model,
            num_heads=nhead,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # 4. Hybrid Classification Head (Task #101)
        self.head = HybridClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            pooling=token_pooling,
            dropout=dropout,
            has_cls_token=use_cls,
        )

    # Thuộc tính truy xuất trực tiếp các tầng con của Backbone mà không nhân đôi cây module
    @property
    def conv1(self):
        return self.backbone.conv1

    @property
    def bn1(self):
        return self.backbone.bn1

    @property
    def relu(self):
        return self.backbone.relu

    @property
    def maxpool(self):
        return self.backbone.maxpool

    @property
    def layer1(self):
        return self.backbone.layer1

    @property
    def layer2(self):
        return self.backbone.layer2

    @property
    def layer3(self):
        return self.backbone.layer3

    @property
    def layer4(self):
        return self.backbone.layer4

    @property
    def cbam4(self):
        return (
            self.backbone.cbam_modules["cbam4"]
            if "cbam4" in self.backbone.cbam_modules
            else None
        )

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Trích xuất feature maps tại từng Stage của Backbone (Task #95)."""
        _, aux = self.backbone(x, return_multiscale=True)
        return aux["multiscale_features"]

    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass hoàn chỉnh qua kiến trúc Hybrid CNN-CBAM-Transformer."""
        # 1. Trích xuất đặc trưng qua CNN-CBAM backbone
        f_stage4, aux_backbone = (
            self.backbone(
                x,
                return_multiscale=return_attention_weights,
                return_attention_maps=return_attention_weights,
            )
            if return_attention_weights
            else (self.backbone(x), {})
        )

        # 2. Chuyển đổi Feature map -> Visual Tokens + Position Embedding
        tokens = self.bridge(f_stage4)  # [B, 49 hoặc 50, 512]

        # 3. Biến đổi ngữ cảnh toàn cảnh qua Transformer Encoder
        if return_attention_weights:
            encoded_tokens, transformer_attns = self.transformer_encoder(
                tokens, return_all_attentions=True
            )
        else:
            encoded_tokens = self.transformer_encoder(tokens)

        # 4. Phân loại qua Classification Head
        logits, pooled_feat = (
            self.head(encoded_tokens, return_embeddings=True)
            if return_attention_weights
            else (self.head(encoded_tokens), None)
        )

        if return_attention_weights:
            aux_features = {
                "multiscale": aux_backbone.get("multiscale_features", {}),
                "cbam_attention_maps": aux_backbone.get("attention_maps", {}),
                "encoded_tokens": encoded_tokens,
                "transformer_attentions": transformer_attns,
                "pooled_embedding": pooled_feat,
            }
            return logits, aux_features

        return logits


def build_hybrid_model(
    num_classes: int = 23,
    pretrained_backbone: bool = True,
    cbam_placement: Literal["stage4", "stage3_4", "all_stages"] = "stage4",
    transformer_layers: int = 2,
    d_model: int = 512,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    token_pooling: Literal["gap", "cls", "hybrid"] = "gap",
    dropout: float = 0.1,
    use_multistage_cbam: bool = False,
) -> HybridCNNCBAMTransformer:
    """Hàm Factory khởi tạo mô hình lai tối ưu cho Luận văn."""
    return HybridCNNCBAMTransformer(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        cbam_placement=cbam_placement,
        transformer_layers=transformer_layers,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        token_pooling=token_pooling,
        dropout=dropout,
        use_multistage_cbam=use_multistage_cbam,
    )


if __name__ == "__main__":
    print("=" * 100)
    print(
        "🔬 KIỂM THỬ KHỞI TẠO MÔ HÌNH HYBRID CNN-CBAM-TRANSFORMER HOÀN CHỈNH (TASKS #94-#101)"
    )
    print("=" * 100)

    # Thử nghiệm với pooling='gap' (khuyến nghị cho ảnh y tế)
    model = build_hybrid_model(
        num_classes=23,
        pretrained_backbone=False,
        cbam_placement="stage4",
        transformer_layers=2,
        token_pooling="gap",
    )
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out, aux = model(dummy_input, return_attention_weights=True)

    print(f"✅ Input shape:                  {dummy_input.shape}")
    print(f"✅ Encoded tokens shape:         {aux['encoded_tokens'].shape}")
    print(f"✅ Logits output shape:          {out.shape}")
    print(f"✅ Pooling strategy:             {model.token_pooling.upper()}")
    print(f"✅ CBAM placement:               {model.cbam_placement.upper()}")

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"📊 Tổng số tham số (Total Params): {total_params / 1e6:.2f} M ({total_params:,} parameters)"
    )
    print("=" * 100)
