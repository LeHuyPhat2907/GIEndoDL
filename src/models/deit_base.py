"""Module xây dựng kiến trúc Data-efficient Image Transformer (DeiT-Base) cho phân loại nội soi tiêu hóa (Task #88).

Đặc điểm kiến trúc:
- Kiến trúc DeiT-Base (Touvron et al., ICML 2021 - Facebook AI Research).
- Cơ chế Distillation Token:
    * Ngoài Class Token ([CLS]) truyền thống, DeiT bổ sung một Distillation Token ([DIST]) chuyên biệt để học tri thức chuyển giao từ mô hình CNN giáo viên.
    * Giúp Vision Transformer học hiệu quả trên các tập dữ liệu có quy mô vừa và nhỏ (như dữ liệu ảnh y khoa HyperKvasir) mà không cần nạp trước từ kho dữ liệu khổng lồ JFT-300M.
    * Đầu ra phân loại là sự kết hợp (Ensemble) giữa Class Head và Distillation Head.
- Kích thước mô hình: Patch 16x16, 12 layers, 12 heads, dim 768 (~86.6M tham số).
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class DeiTBaseWrapper(nn.Module):
    """Wrapper bao bọc mô hình DeiT có Distillation Token nhằm đảm bảo đầu ra tương thích hoàn toàn chuẩn PyTorch."""

    def __init__(self, base_model: nn.Module, is_distilled: bool = True):
        super().__init__()
        self.base_model = base_model
        self.is_distilled = is_distilled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x)
        if isinstance(out, (tuple, list)):
            # Kết hợp dự đoán của cả Class Head và Distillation Head
            cls_logits, dist_logits = out[0], out[1]
            return (cls_logits + dist_logits) / 2.0
        return out


def build_deit_base(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    use_distillation: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình DeiT-Base với đầu phân loại 23 lớp chuẩn Y sinh (Task #88).

    Ưu tiên tải từ timm ('deit_base_distilled_patch16_224').
    Fallback sang torchvision ViT-Base nếu không có mạng.
    """
    model = None
    use_timm = False
    is_dist = False

    # 1. Ưu tiên tải từ timm
    try:
        import timm

        candidates = (
            ["deit_base_distilled_patch16_224", "deit_base_patch16_224"]
            if use_distillation
            else ["deit_base_patch16_224"]
        )

        for model_name in candidates:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                )
                use_timm = True
                is_dist = "distilled" in model_name
                print(f"✅ Đã tải mô hình DeiT-Base ({model_name}) thành công từ thư viện timm! (Distillation Token: {is_dist})")
                break
            except Exception:
                continue
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang tự động chuyển sang cấu trúc ViT-Base fallback...")

    # 2. Cơ chế Fallback an toàn với torchvision
    if model is None:
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        base_vit = vit_b_16(weights=weights)

        in_features = base_vit.heads.head.in_features
        base_vit.heads.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        model = base_vit
        is_dist = False
        print(f"✅ Đã tải mô hình Transformer fallback thành công từ torchvision (pretrained={pretrained})!")

    # 3. Đóng băng backbone nếu có yêu cầu
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name and "classifier" not in name:
                param.requires_grad = False

    # 4. Bao bọc với DeiTBaseWrapper để luôn xuất ra tensor chuẩn (batch_size, num_classes)
    return DeiTBaseWrapper(model, is_distilled=is_dist)
