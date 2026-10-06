"""Module xây dựng kiến trúc Vision Transformer (ViT-Base/16) cho phân loại nội soi tiêu hóa (Task #86).

Đặc điểm kiến trúc:
- Mô hình Transformer thuần túy (Pure Self-Attention) đầu tiên ứng dụng cho phân loại ảnh nội soi tiêu hóa.
- Kiến trúc ViT-Base/16:
    * Patch size: 16x16 -> ảnh 224x224 được chia thành 14x14 = 196 patches không gian.
    * Embedding dimension: 768.
    * 12 Transformer Encoder layers, 12 Attention Heads (Multi-Head Self-Attention - MHSA).
    * Tổng số tham số: ~86.6 Triệu tham số (86.6M params).
- Cơ chế toàn cục (Global Context): Không dùng phép tích chập (Convolution), mô hình học mối quan hệ xa giữa các vùng niêm mạc qua cơ chế tương tác đa đầu (Self-Attention).
"""

from typing import Optional
import torch
import torch.nn as nn


def build_vit_base_16(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình Vision Transformer ViT-Base/16 với đầu phân loại 23 lớp chuẩn Y sinh (Task #86).

    Ưu tiên tải từ timm ('vit_base_patch16_224') nạp trọng số ImageNet-21K fine-tune.
    Fallback an toàn sang torchvision.models.vit_b_16.
    """
    model = None
    use_timm = False

    # 1. Ưu tiên tải từ thư viện timm
    try:
        import timm

        for model_name in [
            "vit_base_patch16_224",
            "vit_base_patch16_224.augreg_in21k_ft_in1k",
            "vit_base_patch16_224.orig_in21k_ft_in1k",
        ]:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                )
                use_timm = True
                print(f"✅ Đã tải mô hình ViT-Base/16 ({model_name}) thành công từ thư viện timm!")
                break
            except Exception:
                continue
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang tự động chuyển sang torchvision.models.vit_b_16...")

    # 2. Cơ chế Fallback an toàn với torchvision
    if model is None:
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)

        # Thay thế classifier head: model.heads.head (in_features = 768 -> num_classes = 23)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        print(f"✅ Đã tải mô hình ViT-Base/16 thành công từ torchvision (pretrained={pretrained})!")

    # 3. Đóng băng backbone nếu có yêu cầu
    if freeze_backbone:
        if use_timm:
            for name, param in model.named_parameters():
                if "head" not in name and "classifier" not in name:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if "heads" not in name:
                    param.requires_grad = False

    return model
