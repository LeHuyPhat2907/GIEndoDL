"""Mô-đun Tích hợp CBAM vào CNN Backbone (Task #98).

Chức năng:
1. Tích hợp linh hoạt khối CBAM (Channel & Spatial Attention) vào ResNet-50.
2. Hỗ trợ các chiến lược định vị chọn lọc (Selective Placement Strategies):
   - 'stage4': Đặt CBAM sau Stage 4 (khuyến nghị cho kiến trúc lai, chi phí tham số tối ưu).
   - 'stage3_4': Đặt CBAM sau Stage 3 và Stage 4 (cân bằng đa tỷ lệ và hiệu quả).
   - 'all_stages': Đặt CBAM tại tất cả các Stages 1, 2, 3, 4.
   - 'residual_blocks': Tích hợp CBAM vào bên trong từng khối Bottleneck Residual.
3. Hỗ trợ trích xuất attention maps (Channel & Spatial) để trực quan hóa vùng chú ý lâm sàng.
"""

# ruff: noqa: E402
from pathlib import Path
import sys
from typing import Any, Dict, Literal, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.models.cbam_modules import CBAM

CBAMPlacement = Literal["stage4", "stage3_4", "all_stages", "residual_blocks"]


class CBAMBottleneck(nn.Module):
    """Khối Bottleneck Residual tích hợp sẵn CBAM ở cuối khối (Woo et al., 2018)."""

    def __init__(
        self,
        base_bottleneck: nn.Module,
        cbam_ratio: int = 16,
        cbam_kernel_size: int = 7,
    ):
        super().__init__()
        self.conv1 = base_bottleneck.conv1
        self.bn1 = base_bottleneck.bn1
        self.conv2 = base_bottleneck.conv2
        self.bn2 = base_bottleneck.bn2
        self.conv3 = base_bottleneck.conv3
        self.bn3 = base_bottleneck.bn3
        self.relu = base_bottleneck.relu
        self.downsample = base_bottleneck.downsample
        self.stride = base_bottleneck.stride

        out_channels = base_bottleneck.conv3.out_channels
        self.cbam = CBAM(
            in_planes=out_channels, ratio=cbam_ratio, kernel_size=cbam_kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Tái hiệu chuẩn kênh và không gian bằng CBAM trước khi cộng identity
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CBAMIntegratedResNet50(nn.Module):
    """CNN Backbone (ResNet-50) tích hợp CBAM với các chế độ định vị có thể cấu hình (Task #98)."""

    def __init__(
        self,
        pretrained: bool = True,
        placement: CBAMPlacement = "stage4",
        cbam_ratio: int = 16,
        cbam_kernel_size: int = 7,
    ):
        """Khởi tạo ResNet-50 tích hợp CBAM.

        Args:
            pretrained: Sử dụng trọng số ImageNet tiền huấn luyện.
            placement: Chiến lược định vị CBAM ('stage4', 'stage3_4', 'all_stages', 'residual_blocks').
            cbam_ratio: Hệ số giảm kênh r trong Channel Attention.
            cbam_kernel_size: Kích thước kernel trong Spatial Attention.
        """
        super().__init__()
        self.placement = placement
        self.cbam_ratio = cbam_ratio
        self.cbam_kernel_size = cbam_kernel_size

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = resnet50(weights=weights)

        # 1. Stem
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        # 2. Các tầng Backbone và CBAM tương ứng theo chiến lược placement
        if placement == "residual_blocks":
            # Tích hợp CBAM vào từng Bottleneck block riêng lẻ
            self.layer1 = self._wrap_layer_with_cbam(base_resnet.layer1)
            self.layer2 = self._wrap_layer_with_cbam(base_resnet.layer2)
            self.layer3 = self._wrap_layer_with_cbam(base_resnet.layer3)
            self.layer4 = self._wrap_layer_with_cbam(base_resnet.layer4)
            self.cbam_modules = nn.ModuleDict()
        else:
            self.layer1 = base_resnet.layer1  # C=256
            self.layer2 = base_resnet.layer2  # C=512
            self.layer3 = base_resnet.layer3  # C=1024
            self.layer4 = base_resnet.layer4  # C=2048

            self.cbam_modules = nn.ModuleDict()
            if placement in ("all_stages",):
                self.cbam_modules["cbam1"] = CBAM(
                    256, ratio=cbam_ratio, kernel_size=cbam_kernel_size
                )
                self.cbam_modules["cbam2"] = CBAM(
                    512, ratio=cbam_ratio, kernel_size=cbam_kernel_size
                )
            if placement in ("stage3_4", "all_stages"):
                self.cbam_modules["cbam3"] = CBAM(
                    1024, ratio=cbam_ratio, kernel_size=cbam_kernel_size
                )
            if placement in ("stage4", "stage3_4", "all_stages"):
                self.cbam_modules["cbam4"] = CBAM(
                    2048, ratio=cbam_ratio, kernel_size=cbam_kernel_size
                )

    def _wrap_layer_with_cbam(self, layer: nn.Sequential) -> nn.Sequential:
        """Gói từng khối Bottleneck bên trong layer bằng CBAMBottleneck."""
        wrapped_blocks = [
            CBAMBottleneck(
                block,
                cbam_ratio=self.cbam_ratio,
                cbam_kernel_size=self.cbam_kernel_size,
            )
            for block in layer
        ]
        return nn.Sequential(*wrapped_blocks)

    def forward(
        self,
        x: torch.Tensor,
        return_multiscale: bool = False,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass qua CNN-CBAM backbone.

        Args:
            x: Tensor đầu vào [B, 3, 224, 224].
            return_multiscale: Nếu True, trả về feature maps tại tất cả các stages.
            return_attention_maps: Nếu True, trả về các attention maps của CBAM.

        Returns:
            Tensor đặc trưng cuối cùng [B, 2048, 7, 7] hoặc (out, aux_dict).
        """
        aux_dict: Dict[str, Any] = {"attention_maps": {}}

        # Stem
        x_stem = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Stage 1
        s1 = self.layer1(x_stem)
        if "cbam1" in self.cbam_modules:
            if return_attention_maps:
                s1, c_map1, s_map1 = self.cbam_modules["cbam1"](s1, return_maps=True)
                aux_dict["attention_maps"]["stage1"] = {
                    "channel": c_map1,
                    "spatial": s_map1,
                }
            else:
                s1 = self.cbam_modules["cbam1"](s1)

        # Stage 2
        s2 = self.layer2(s1)
        if "cbam2" in self.cbam_modules:
            if return_attention_maps:
                s2, c_map2, s_map2 = self.cbam_modules["cbam2"](s2, return_maps=True)
                aux_dict["attention_maps"]["stage2"] = {
                    "channel": c_map2,
                    "spatial": s_map2,
                }
            else:
                s2 = self.cbam_modules["cbam2"](s2)

        # Stage 3
        s3 = self.layer3(s2)
        if "cbam3" in self.cbam_modules:
            if return_attention_maps:
                s3, c_map3, s_map3 = self.cbam_modules["cbam3"](s3, return_maps=True)
                aux_dict["attention_maps"]["stage3"] = {
                    "channel": c_map3,
                    "spatial": s_map3,
                }
            else:
                s3 = self.cbam_modules["cbam3"](s3)

        # Stage 4
        s4 = self.layer4(s3)
        if "cbam4" in self.cbam_modules:
            if return_attention_maps:
                s4, c_map4, s_map4 = self.cbam_modules["cbam4"](s4, return_maps=True)
                aux_dict["attention_maps"]["stage4"] = {
                    "channel": c_map4,
                    "spatial": s_map4,
                }
            else:
                s4 = self.cbam_modules["cbam4"](s4)

        if return_multiscale:
            aux_dict["multiscale_features"] = {
                "stem": x_stem,
                "stage1": s1,
                "stage2": s2,
                "stage3": s3,
                "stage4": s4,
            }

        if return_multiscale or return_attention_maps:
            return s4, aux_dict

        return s4


def count_cbam_parameters(placement: CBAMPlacement) -> Dict[str, Any]:
    """Tính toán chi tiết số tham số bổ sung của từng chiến lược tích hợp CBAM."""
    model = CBAMIntegratedResNet50(pretrained=False, placement=placement)
    base_model = resnet50(weights=None)

    total_params = sum(p.numel() for p in model.parameters())
    base_params = sum(p.numel() for p in base_model.parameters()) - sum(
        p.numel() for p in base_model.fc.parameters()
    )  # Trừ fc head 1000 classes

    cbam_added_params = total_params - base_params

    return {
        "placement": placement,
        "total_backbone_params": total_params,
        "total_backbone_params_m": round(total_params / 1e6, 3),
        "cbam_added_params": cbam_added_params,
        "cbam_added_params_k": round(cbam_added_params / 1e3, 2),
        "percentage_added": round((cbam_added_params / base_params) * 100, 2),
    }
