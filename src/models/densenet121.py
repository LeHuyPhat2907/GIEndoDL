"""Module xây dựng kiến trúc mô hình chuẩn DenseNet-121 Baseline cho phân loại nội soi tiêu hóa (Task #78)."""

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_densenet121_baseline(
    num_classes: int = 23,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình DenseNet-121 với tầng phân loại 23 lớp chuẩn y khoa.
    
    Được sử dụng trong Task #78 để kiểm chứng và so sánh Dense Connections (Feature Reuse)
    với Residual Connections của họ ResNet (Task #76 và Task #77).
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Thay thế tầng Classifier cuối cùng (in_features = 1024 -> num_classes = 23)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.45),
        nn.Linear(in_features, num_classes),
    )

    return model
