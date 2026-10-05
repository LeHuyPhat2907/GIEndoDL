"""Module xây dựng kiến trúc mô hình chuẩn EfficientNetV2-S Baseline cho phân loại nội soi tiêu hóa (Task #81).

Đặc điểm kiến trúc:
- Sử dụng khối Fused-MBConv ở các tầng đầu (thay thế depthwise + pointwise conv) giúp tăng tốc độ xử lý GPU lên 2-3 lần.
- Kết hợp thư viện timm với cơ chế fallback tự động sang torchvision.models.efficientnet_v2_s.
- Phù hợp với chiến lược huấn luyện tăng dần độ phân giải (Progressive Resizing / Learning Strategy).
"""

import torch.nn as nn


def build_efficientnet_v2_s_baseline(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình EfficientNetV2-S với đầu phân loại 23 lớp chuẩn Y sinh học.

    Task #81: Load EfficientNetV2-S (improved training + architecture); Dùng Progressive Resizing strategy.
    """
    model = None
    use_timm = False

    # 1. Ưu tiên tải từ thư viện timm
    try:
        import timm

        # Thử các mã tên mô hình phổ biến trong timm
        for model_name in ["efficientnetv2_s", "tf_efficientnetv2_s.in21k_ft_in1k", "efficientnetv2_rw_s"]:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                )
                use_timm = True
                print(f"✅ Đã tải mô hình EfficientNetV2-S ({model_name}) thành công từ thư viện timm!")
                break
            except Exception:
                continue
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang tự động chuyển sang torchvision.models.efficientnet_v2_s...")

    # 2. Cơ chế Fallback an toàn với torchvision
    if model is None:
        from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)

        # Thay thế tầng Classifier cuối cùng (in_features = 1280 -> num_classes = 23)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        print(f"✅ Đã tải mô hình EfficientNetV2-S thành công từ torchvision (pretrained={pretrained})!")

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
