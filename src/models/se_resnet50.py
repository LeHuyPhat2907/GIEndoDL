"""Module xây dựng kiến trúc SE-ResNet-50 (Squeeze-and-Excitation ResNet-50) cho phân loại nội soi tiêu hóa (Task #83).

Đặc điểm kiến trúc:
- Tích hợp khối Squeeze-and-Excitation (SE Module - Hu et al., CVPR 2018) vào từng khối Bottleneck của ResNet-50.
- Cơ chế Channel Attention:
    1. Squeeze: Global Average Pooling nén không gian HxW thành vector kênh 1x1xC.
    2. Excitation: Hai tầng Fully Connected với hàm kích hoạt ReLU và Sigmoid để học tương quan phi tuyến giữa các kênh tổn thương.
    3. Scale: Nhân trọng số kênh học được với Feature Map gốc.
- Đóng vai trò nghiên cứu bóc tách (Ablation Study): So sánh hiệu quả của Channel Attention độc lập so với Vanilla ResNet-50 (không có Attention) và CBAM (Channel + Spatial Attention).
"""

from typing import Optional
import torch
import torch.nn as nn


class SEModule(nn.Module):
    """Khối Squeeze-and-Excitation chuẩn (Channel Attention)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class SEBottleneckWrapper(nn.Module):
    """Bọc khối Bottleneck của torchvision ResNet-50 với cơ chế SE Attention ở ngõ ra."""

    def __init__(self, original_block: nn.Module, channels: int, reduction: int = 16):
        super().__init__()
        self.original_block = original_block
        self.se = SEModule(channels=channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trong ResNet Bottleneck chuẩn: out = relu(conv3(x) + shortcut)
        # Khi tích hợp SE: out_conv = se(conv3(x)) + shortcut -> relu(out_conv)
        identity = x

        out = self.original_block.conv1(x)
        out = self.original_block.bn1(out)
        out = self.original_block.relu(out)

        out = self.original_block.conv2(out)
        out = self.original_block.bn2(out)
        out = self.original_block.relu(out)

        out = self.original_block.conv3(out)
        out = self.original_block.bn3(out)

        out = self.se(out)

        if self.original_block.downsample is not None:
            identity = self.original_block.downsample(x)

        out += identity
        out = self.original_block.relu(out)
        return out


def build_se_resnet50(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình SE-ResNet-50 chuẩn với đầu phân loại 23 lớp y sinh (Task #83).

    Ưu tiên tải từ timm ('seresnet50'). Nếu không có, tự động chuyển sang cơ chế chèn SEModule vào torchvision ResNet-50.
    """
    model = None
    use_timm = False

    # 1. Ưu tiên tải từ thư viện timm
    try:
        import timm

        for model_name in ["seresnet50", "seresnet50.a1_in1k", "seresnet50d"]:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                )
                use_timm = True
                print(f"✅ Đã tải mô hình SE-ResNet-50 ({model_name}) thành công từ thư viện timm!")
                break
            except Exception:
                continue
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang tự động chuyển sang cấu trúc SE-ResNet-50 tùy biến từ torchvision...")

    # 2. Cơ chế Fallback an toàn: tích hợp SEModule vào torchvision ResNet-50
    if model is None:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = resnet50(weights=weights)

        # Chèn SEModule vào từng khối Bottleneck của cả 4 tầng
        layers = [base_resnet.layer1, base_resnet.layer2, base_resnet.layer3, base_resnet.layer4]
        for layer in layers:
            for i, block in enumerate(layer):
                out_channels = block.conv3.out_channels
                layer[i] = SEBottleneckWrapper(block, channels=out_channels, reduction=16)

        # Thay thế tầng Fully Connected cuối cùng (2048 -> num_classes)
        in_features = base_resnet.fc.in_features
        base_resnet.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        model = base_resnet
        print(f"✅ Đã xây dựng thành công kiến trúc SE-ResNet-50 tùy biến từ torchvision (pretrained={pretrained})!")

    # 3. Đóng băng backbone nếu có yêu cầu
    if freeze_backbone:
        if use_timm:
            for name, param in model.named_parameters():
                if "classifier" not in name and "head" not in name and "fc" not in name:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if "fc" not in name and "se" not in name:
                    param.requires_grad = False

    return model
