"""Module xây dựng kiến trúc mô hình chuẩn EfficientNet-B4 Baseline cho phân loại nội soi tiêu hóa (Task #79).

Sử dụng thư viện timm (với cơ chế fallback an toàn sang torchvision nếu chưa cài đặt timm).
Hỗ trợ độ phân giải chuẩn y khoa 380x380 (Sweet spot giữa performance và computation).
"""

import torch.nn as nn


def build_efficientnet_b4_baseline(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.4,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình EfficientNet-B4 với đầu phân loại 23 lớp chuẩn Y sinh học.

    Task #79: Load pretrained EfficientNet-B4 (timm library); Fine-tune; Input size 380x380.
    """
    model = None
    use_timm = False

    # 1. Ưu tiên tải từ thư viện timm theo đúng yêu cầu Task #79
    try:
        import timm

        model = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        use_timm = True
        print(f"✅ Đã tải mô hình EfficientNet-B4 thành công từ thư viện timm (pretrained={pretrained})!")
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang tự động chuyển sang torchvision.models.efficientnet_b4...")

    # 2. Cơ chế Fallback an toàn với torchvision nếu máy chưa có timm
    if model is None:
        from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = efficientnet_b4(weights=weights)

        # Thay thế tầng Classifier cuối cùng (in_features = 1792 -> num_classes = 23)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        print(f"✅ Đã tải mô hình EfficientNet-B4 thành công từ torchvision (pretrained={pretrained})!")

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
