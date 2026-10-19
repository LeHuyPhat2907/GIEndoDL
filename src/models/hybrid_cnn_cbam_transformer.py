"""Module cài đặt kiến trúc lai đề xuất: Hybrid CNN-CBAM-Transformer (Task #94).

ĐÓNG GÓP KHOA HỌC CHÍNH (CORE CONTRIBUTION) CỦA LUẬN VĂN:
- Tích hợp 3 trụ cột công nghệ thị giác máy tính trong một kiến trúc thống nhất:
    1. CNN Backbone (ResNet-50): Trích xuất đặc trưng cục bộ đa tầng, giữ trọn vẹn inductive bias (tính bất biến quy nạp và cục bộ).
    2. CBAM Modules (Convolutional Block Attention Module): Tái hiệu chuẩn kênh (Channel Attention - lọc màu vi mạch niêm mạc) và không gian (Spatial Attention - định vị tổn thương).
    3. Compact Transformer Encoder: Mô hình hóa ngữ cảnh tầm xa toàn cục trên N=49 visual tokens (chi phí O(49^2) siêu nhẹ).
    4. Classification Head: Chuẩn hóa LayerNorm và phân loại chính xác 23 lớp bệnh lý nội soi tiêu hóa HyperKvasir.
"""

import sys
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ChannelAttention(nn.Module):
    """Mô-đun Chú ý Kênh (Channel Attention Module - CAM) của CBAM.

    Cơ chế:
    - Nén không gian bằng đồng thời Average Pooling và Max Pooling.
    - Đưa qua mạng MLP chia sẻ trọng số (Shared MLP) với hệ số giảm số kênh ratio r.
    - Cộng gộp hai vector biểu diễn và áp dụng hàm kích hoạt Sigmoid để sinh trọng số tái hiệu chuẩn kênh M_c.
    """

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights


class SpatialAttention(nn.Module):
    """Mô-đun Chú ý Không gian (Spatial Attention Module - SAM) của CBAM.

    Cơ chế:
    - Nén chiều kênh bằng Average Pooling và Max Pooling dọc theo trục kênh (Channel-wise).
    - Ghép nối 2 bản đồ đặc trưng 1xHxW thành tensor 2xHxW.
    - Áp dụng tầng tích chập kernel 7x7 và hàm Sigmoid để sinh trọng số tập trung không gian M_s.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (
            3,
            7,
        ), "Kernel size của Spatial Attention phải là 3 hoặc 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(spatial_features))
        return x * spatial_weights


class CBAM(nn.Module):
    """Khối Tái hiệu chuẩn Không gian và Kênh hoàn chỉnh (CBAM - Woo et al., ECCV 2018).

    Thứ tự xử lý tuần tự: Feature Map -> Channel Attention -> Spatial Attention -> Refined Feature Map.
    """

    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.ca(x)
        x_out = self.sa(x_out)
        return x_out


class TransformerEncoderBlock(nn.Module):
    """Khối Transformer Encoder chuẩn mực với kiến trúc Pre-LayerNorm (Pre-LN)."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN MHSA
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # Pre-LN FFN
        x = x + self.mlp(self.norm2(x))
        return x


class CompactTransformerEncoder(nn.Module):
    """Mô-đun Compact Transformer Encoder tinh gọn cho N=49 visual tokens."""

    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class HybridCNNCBAMTransformer(nn.Module):
    """Kiến trúc Tổng thể Mô hình Lai Hybrid CNN-CBAM-Transformer (Task #94 & Đóng góp chính Luận văn).

    Cấu trúc luồng dữ liệu (Dataflow):
    1. Ảnh đầu vào: [B, 3, 224, 224]
    2. ResNet-50 Stem + Layer 1-4:
       - Stem:       [B, 64, 56, 56]
       - Stage 1:    [B, 256, 56, 56]
       - Stage 2:    [B, 512, 28, 28]
       - Stage 3:    [B, 1024, 14, 14]
       - Stage 4:    [B, 2048, 7, 7]
    3. Khối CBAM Attention (tại Stage 4):
       - Tái hiệu chuẩn kênh màu tổn thương và không gian: [B, 2048, 7, 7]
    4. Bridge Projection (Tầng chiếu giảm số kênh):
       - Conv2d 1x1 + BN + GELU: [B, 512, 7, 7]
    5. Tokenization & Positional Embedding:
       - Trải phẳng không gian: 7x7 -> 49 visual tokens
       - Ghép thêm learnable [CLS] token: chuỗi dài 50 tokens [B, 50, 512]
       - Cộng 1D Learnable Positional Embedding [1, 50, 512]
    6. Compact Transformer Encoder:
       - L tầng MHSA (L=2..4, d=512, h=8, FFN=2048): [B, 50, 512]
    7. Classification Head:
       - Trích xuất [CLS] token: [B, 512]
       - LayerNorm -> Dropout -> Linear -> Logits 23 Lớp: [B, 23]
    """

    def __init__(
        self,
        num_classes: int = 23,
        pretrained_backbone: bool = True,
        transformer_layers: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cbam_ratio: int = 16,
        cbam_kernel_size: int = 7,
        use_multistage_cbam: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_multistage_cbam = use_multistage_cbam

        # 1. Khởi tạo CNN Backbone (ResNet-50)
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        base_resnet = models.resnet50(weights=weights)

        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        self.layer1 = base_resnet.layer1  # Stage 1: C=256,  H=56, W=56
        self.layer2 = base_resnet.layer2  # Stage 2: C=512,  H=28, W=28
        self.layer3 = base_resnet.layer3  # Stage 3: C=1024, H=14, W=14
        self.layer4 = base_resnet.layer4  # Stage 4: C=2048, H=7,  W=7

        # 2. Khối CBAM Modules
        if self.use_multistage_cbam:
            self.cbam1 = CBAM(256, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            self.cbam2 = CBAM(512, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            self.cbam3 = CBAM(1024, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            self.cbam4 = CBAM(2048, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
        else:
            self.cbam4 = CBAM(2048, ratio=cbam_ratio, kernel_size=cbam_kernel_size)

        # 3. Bridge Projection: Giảm kênh từ 2048 -> d_model (512)
        self.bridge_proj = nn.Sequential(
            nn.Conv2d(2048, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # 4. Tokenizer & Embedding parameters
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 50, d_model)
        )  # 49 visual tokens + 1 cls token
        self.pos_drop = nn.Dropout(p=dropout)

        # Khởi tạo trọng số token chuẩn xác
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 5. Compact Transformer Encoder
        self.transformer_encoder = CompactTransformerEncoder(
            num_layers=transformer_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # 6. Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Trích xuất feature maps tại từng Stage của Backbone (Phục vụ Task #95 & Phân tích Đa Tỷ lệ)."""
        x_stem = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        s1 = self.layer1(x_stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return {
            "stem": x_stem,
            "stage1": s1,
            "stage2": s2,
            "stage3": s3,
            "stage4": s4,
        }

    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass hoàn chỉnh qua kiến trúc Hybrid CNN-CBAM-Transformer."""
        b = x.shape[0]

        # 1. Trích xuất đặc trưng CNN Backbone
        x_stem = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        s1 = self.layer1(x_stem)
        if self.use_multistage_cbam:
            s1 = self.cbam1(s1)

        s2 = self.layer2(s1)
        if self.use_multistage_cbam:
            s2 = self.cbam2(s2)

        s3 = self.layer3(s2)
        if self.use_multistage_cbam:
            s3 = self.cbam3(s3)

        s4 = self.layer4(s3)  # [B, 2048, 7, 7]

        # 2. Tái hiệu chuẩn không gian và kênh qua CBAM
        f_cbam = self.cbam4(s4)  # [B, 2048, 7, 7]

        # 3. Chiếu đặc trưng xuống d_model (512)
        f_proj = self.bridge_proj(f_cbam)  # [B, 512, 7, 7]

        # 4. Trải phẳng không gian thành N=49 Visual Tokens
        tokens = f_proj.flatten(2).transpose(1, 2)  # [B, 49, 512]

        # 5. Ghép CLS Token và cộng Positional Embedding
        cls_tokens = self.cls_token.expand(b, -1, -1)  # [B, 1, 512]
        x_tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, 50, 512]
        x_tokens = self.pos_drop(x_tokens + self.pos_embed)

        # 6. Biến đổi toàn cục qua Compact Transformer Encoder
        encoded_tokens = self.transformer_encoder(x_tokens)  # [B, 50, 512]

        # 7. Trích xuất biểu diễn tại CLS Token để phân loại
        cls_repr = encoded_tokens[:, 0]  # [B, 512]
        logits = self.head(cls_repr)  # [B, num_classes]

        if return_attention_weights:
            aux_features = {
                "stage4_raw": s4,
                "stage4_cbam": f_cbam,
                "encoded_tokens": encoded_tokens,
                "cls_repr": cls_repr,
            }
            return logits, aux_features

        return logits


def build_hybrid_model(
    num_classes: int = 23,
    pretrained_backbone: bool = True,
    transformer_layers: int = 2,
    d_model: int = 512,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    use_multistage_cbam: bool = False,
) -> HybridCNNCBAMTransformer:
    """Hàm khởi tạo (Factory Function) mô hình lai đề xuất cho Luận văn."""
    model = HybridCNNCBAMTransformer(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        transformer_layers=transformer_layers,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_multistage_cbam=use_multistage_cbam,
    )
    return model


if __name__ == "__main__":
    print("=" * 90)
    print("🔬 KIỂM THỬ KHỞI TẠO MÔ HÌNH HYBRID CNN-CBAM-TRANSFORMER (TASK #94)")
    print("=" * 90)

    model = build_hybrid_model(
        num_classes=23, pretrained_backbone=False, transformer_layers=2
    )
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out, aux = model(dummy_input, return_attention_weights=True)

    print(f"✅ Input shape:         {dummy_input.shape}")
    print(f"✅ Stage 4 Raw shape:   {aux['stage4_raw'].shape}")
    print(f"✅ Stage 4 CBAM shape:  {aux['stage4_cbam'].shape}")
    print(f"✅ Encoded tokens shape:{aux['encoded_tokens'].shape}")
    print(f"✅ Logits output shape: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"📊 Tổng số tham số (Total Params):     {total_params / 1e6:.2f} M ({total_params:,} parameters)"
    )
    print(f"📊 Tham số có thể huấn luyện (Trainable): {trainable_params / 1e6:.2f} M")
    print("=" * 90)
