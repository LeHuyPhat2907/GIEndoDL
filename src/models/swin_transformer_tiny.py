"""Module xây dựng kiến trúc Swin Transformer-Tiny (Swin-T) cho nhận diện và phân loại tổn thương nội soi tiêu hóa (Task #89).

Đặc điểm kiến trúc:
- Kiến trúc Swin Transformer (Liu et al., ICCV 2021 - Best Paper / Marr Prize).
- Shifted Window based Self-Attention (W-MSA & SW-MSA):
    * Tính self-attention cục bộ trong các cửa sổ không chồng lấn MxM (M=7).
    * Dịch chuyển cửa sổ ở các layer xen kẽ giúp trao đổi thông tin xuyên vùng (cross-window connections).
    * Giảm độ phức tạp tính toán từ bậc 2 O((HW)^2) của ViT chuẩn xuống tuyến tính O(HW).
- Biểu diễn phân cấp đa tỉ lệ (Hierarchical Feature Maps):
    * Sử dụng Patch Merging để giảm kích thước đặc trưng (H/4 -> H/8 -> H/16 -> H/32).
    * Cực kỳ thích hợp với ảnh nội soi tiêu hóa, nơi tổn thương có kích thước biến thiên rất lớn (từ polyp micro vài mm đến khối u loét lớn).
- Kích thước gọn nhẹ (Compact & Efficient):
    * ~28.3M tham số (chỉ bằng ~1/3 so với ViT-Base 86M hay DeiT-Base 86.6M).
- Tương thích 100% với timm và fallback an toàn với torchvision.
"""

import sys
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_swin_tiny(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.2,
    window_size: int = 7,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình Swin Transformer-Tiny với đầu phân loại 23 lớp chuẩn Y sinh (Task #89).

    Ưu tiên tải từ timm ('swin_tiny_patch4_window7_224').
    Fallback sang torchvision swin_t nếu môi trường chưa cài timm.

    Args:
        num_classes: Số lớp đầu ra (mặc định 23 lớp HyperKvasir).
        pretrained: Sử dụng trọng số ImageNet tiền huấn luyện.
        drop_rate: Tỉ lệ Dropout ở classifier head.
        drop_path_rate: Tỉ lệ Stochastic Depth (DropPath) cho các transformer blocks.
        window_size: Kích thước cửa sổ Self-Attention cục bộ (mặc định 7).
        freeze_backbone: Đóng băng các tầng trích xuất đặc trưng hay không.

    Returns:
        Mô hình PyTorch nn.Module sẵn sàng huấn luyện / suy luận.
    """
    model = None
    source = ""

    # 1. Ưu tiên tải từ thư viện timm
    try:
        import timm

        candidates = [
            "swin_tiny_patch4_window7_224",
            "swin_tiny_patch4_window7_224.ms_in1k",
            "swin_s3_tiny_224",
        ]

        for model_name in candidates:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                )
                source = f"timm ({model_name})"
                print(
                    f"✅ Đã tải thành công Swin Transformer-Tiny từ {source} (drop_path_rate={drop_path_rate}, pretrained={pretrained})"
                )
                break
            except Exception:
                continue
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang kích hoạt cơ chế Fallback sang torchvision...")

    # 2. Cơ chế Fallback sang torchvision
    if model is None:
        try:
            import torchvision.models as tvm
            from torchvision.models import Swin_T_Weights, swin_t

            weights = Swin_T_Weights.DEFAULT if pretrained else None
            base_model = swin_t(weights=weights)

            in_features = base_model.head.in_features
            base_model.head = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(in_features, num_classes),
            )
            model = base_model
            source = "torchvision.models.swin_t"
            print(
                f"✅ Đã tải thành công Swin Transformer-Tiny từ {source} (pretrained={pretrained})"
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ Không thể khởi tạo Swin Transformer-Tiny từ cả timm và torchvision: {e}"
            )

    # 3. Đóng băng backbone nếu có yêu cầu
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name and "classifier" not in name:
                param.requires_grad = False
        print("🔒 Đã đóng băng toàn bộ backbone Swin-Tiny, chỉ huấn luyện classifier head.")

    return model


def get_swin_tiny_model_info(model: nn.Module) -> Dict[str, Any]:
    """Trích xuất thông số kỹ thuật của mô hình Swin-T cho báo cáo luận văn."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_architecture": "Swin Transformer-Tiny (Swin-T)",
        "paper": "Liu et al., ICCV 2021 (Best Paper - Marr Prize)",
        "input_resolution": "224x224",
        "patch_size": "4x4",
        "window_size": 7,
        "stages": 4,
        "feature_dimensions": [96, 192, 384, 768],
        "attention_mechanism": "Shifted Window Self-Attention (W-MSA & SW-MSA)",
        "complexity": "Linear O(HW) with respect to image dimensions",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(total_params * 4 / (1024 * 1024), 2),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("🔬 KIỂM THỬ KHỞI TẠO SWIN TRANSFORMER-TINY (TASK #89)...")
    print("=" * 80)

    test_model = build_swin_tiny(num_classes=23, pretrained=False)
    info = get_swin_tiny_model_info(test_model)

    for k, v in info.items():
        print(f"  * {k}: {v}")

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = test_model(dummy_input)
    print(f"\n✅ Tensor đầu ra kiểm thử: shape = {out.shape} (Kỳ vọng: [2, 23])")
    print("=" * 80)
