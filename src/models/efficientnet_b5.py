"""Module xây dựng kiến trúc mô hình chuẩn EfficientNet-B5 Baseline cho phân loại nội soi tiêu hóa (Task #80).

Sử dụng thư viện timm (với cơ chế fallback an toàn sang torchvision nếu máy chưa cài đặt timm).
Hỗ trợ độ phân giải chuẩn y khoa 456x456 (Compound Scaling cho EfficientNet-B5).
"""

import torch.nn as nn


def build_efficientnet_b5_baseline(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.4,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình EfficientNet-B5 với đầu phân loại 23 lớp chuẩn Y sinh học.

    Task #80: Load pretrained EfficientNet-B5 (timm library); Fine-tune; Input size 456x456.
    """
    model = None
    use_timm = False

    # 1. Ưu tiên tải từ thư viện timm theo đúng yêu cầu Task #80
    try:
        import timm

        model = timm.create_model(
            "efficientnet_b5",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        use_timm = True
        print(f"✅ Đã tải mô hình EfficientNet-B5 thành công từ thư viện timm (pretrained={pretrained})!")
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang tự động chuyển sang torchvision.models.efficientnet_b5...")

    # 2. Cơ chế Fallback an toàn với torchvision nếu máy chưa có timm
    if model is None:
        from torchvision.models import EfficientNet_B5_Weights, efficientnet_b5

        weights = EfficientNet_B5_Weights.DEFAULT if pretrained else None
        model = efficientnet_b5(weights=weights)

        # Thay thế tầng Classifier cuối cùng (in_features = 2048 -> num_classes = 23)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        print(f"✅ Đã tải mô hình EfficientNet-B5 thành công từ torchvision (pretrained={pretrained})!")

    # 3. Đóng băng backbone nếu có yêu cầu
    if freeze_backbone:
        if use_timm:
            for name, param in model.named_parameters():
                if "classifier" not in name and "head" not in name:
                    param.requires_grad = False
        else:
            for param in model.features.parameters():
                param.requires_grad = False

    return model
