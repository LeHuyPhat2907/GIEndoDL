"""Module xây dựng kiến trúc mô hình chuẩn ResNet-101 Baseline cho phân loại nội soi tiêu hóa (Task #77)."""

import torch.nn as nn
from torchvision.models import ResNet101_Weights, resnet101


def build_resnet101_baseline(
    num_classes: int = 23,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình ResNet-101 với tầng phân loại 23 lớp chuẩn y khoa.
    
    Được sử dụng trong Task #77 để kiểm chứng hiện tượng Diminishing Returns từ độ sâu mạng
    khi so sánh trực tiếp với ResNet-50 (Task #76).
    """
    weights = ResNet101_Weights.DEFAULT if pretrained else None
    model = resnet101(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Thay thế tầng Fully Connected cuối cùng (2048 -> num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.45),
        nn.Linear(in_features, num_classes),
    )

    return model
