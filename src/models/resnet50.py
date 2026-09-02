"""Module xây dựng kiến trúc mô hình chuẩn ResNet-50 Baseline cho phân loại nội soi tiêu hóa."""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_baseline(
    num_classes: int = 23,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình ResNet-50 với tầng phân loại 23 lớp chuẩn y khoa."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Thay thế tầng Fully Connected cuối cùng (2048 -> num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model
