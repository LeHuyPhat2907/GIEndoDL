"""Module xây dựng kiến trúc Vision Transformer ViT-Base/32 cho phân loại nội soi tiêu hóa (Task #87).

Đặc điểm kiến trúc:
- Kiến trúc Vision Transformer với kích thước Patch lớn: 32x32 pixel.
- So sánh cấu trúc với ViT-Base/16:
    * Patch size: 32x32 -> ảnh 224x224 chỉ cần chia thành 7x7 = 49 patches không gian (thay vì 196 patches).
    * Token sequence length: 50 tokens (49 patch tokens + 1 [CLS] token) thay vì 197 tokens.
    * Độ phức tạp Self-Attention: O(N^2) giảm từ 197^2 = 38,809 xuống 50^2 = 2,500 (giảm ~15.5 lần!).
    * Tốc độ xử lý: Huấn luyện và suy luận nhanh hơn đáng kể, tiêu thụ VRAM thấp hơn.
- Đánh đổi (Trade-off): Phân tích ảnh hưởng của kích thước patch lớn đối với khả năng nhận diện tổn thương niêm mạc nhỏ (Ablation on Token Granularity).
"""

from typing import Optional
import torch
import torch.nn as nn


def build_vit_base_32(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình Vision Transformer ViT-Base/32 với đầu phân loại 23 lớp chuẩn Y sinh (Task #87).

    Ưu tiên tải từ timm ('vit_base_patch32_224').
    Fallback an toàn sang torchvision.models.vit_b_32.
    """
    model = None
    use_timm = False

    # 1. Ưu tiên tải từ thư viện timm
    try:
        import timm

        for model_name in [
            "vit_base_patch32_224",
            "vit_base_patch32_224.augreg_in21k_ft_in1k",
            "vit_base_patch32_224.orig_in21k_ft_in1k",
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
                print(f"✅ Đã tải mô hình ViT-Base/32 ({model_name}) thành công từ thư viện timm!")
                break
            except Exception:
                continue
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang tự động chuyển sang torchvision.models.vit_b_32...")

    # 2. Cơ chế Fallback an toàn với torchvision
    if model is None:
        from torchvision.models import ViT_B_32_Weights, vit_b_32

        weights = ViT_B_32_Weights.DEFAULT if pretrained else None
        model = vit_b_32(weights=weights)

        # Thay thế classifier head: model.heads.head (in_features = 768 -> num_classes = 23)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        print(f"✅ Đã tải mô hình ViT-Base/32 thành công từ torchvision (pretrained={pretrained})!")

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
