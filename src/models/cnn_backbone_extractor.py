"""Mô-đun Trích xuất Đặc trưng Đa Tỷ lệ từ CNN Backbone (Task #95).

ĐÓNG GÓP & CHỨC NĂNG (TASK #95):
1. Lấy feature maps từ các stages của CNN backbone (Stage 1 -> 4).
2. Xác định chi tiết feature map resolution, số channels, effective stride và receptive field tại mỗi stage.
3. Cơ chế Hook-based extraction (CNNFeatureHooks): Can thiệp phi xâm lấn (non-intrusive) vào mạng nơ-ron
   bằng PyTorch forward hooks, tự động dọn dẹp bộ nhớ và hỗ trợ context manager an toàn.
4. Tương thích hoàn toàn với ResNet-50 đã được chọn làm optimal backbone (Task #85 & #94)
   và sẵn sàng mở rộng cho các kiến trúc CNN khác (EfficientNet, DenseNet).
5. Cung cấp nền tảng đa tỷ lệ (multi-scale features) cho khối CBAM (Task #96, #97) và Transformer encoder.
"""

from dataclasses import asdict, dataclass
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class StageMetadata:
    """Thông tin đặc tính kỹ thuật của từng Stage trong CNN Backbone."""

    stage_name: str
    layer_name: str
    channels: int
    height: int
    width: int
    downsampling_factor: int
    receptive_field: int
    num_blocks: int
    clinical_role: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CNNFeatureHooks:
    """Bộ trích xuất đặc trưng dựa trên PyTorch Forward Hooks (Hook-based extraction).

    Ưu điểm kỹ thuật:
    - Phi xâm lấn (Non-intrusive): Không cần viết lại cấu trúc forward của backbone có sẵn.
    - An toàn bộ nhớ: Hỗ trợ context manager `with CNNFeatureHooks(...) as hooks:`,
      tự động gỡ bỏ hook sau khi hoàn tất forward pass.
    - Hỗ trợ cả inference (detach tensor) và training (giữ nguyên computation graph cho backpropagation).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Dict[str, str],
        detach: bool = False,
    ):
        """Khởi tạo Hook Feature Extractor.

        Args:
            model: Mạng nơ-ron backbone (hoặc mô hình tổng thể).
            target_layers: Từ điển ánh xạ {tên_stage: tên_submodule_trong_model}.
                           Ví dụ: {'stage1': 'layer1', 'stage2': 'layer2', ...}
            detach: Nếu True, tensor lưu trữ sẽ được detach() khỏi computation graph
                    để tiết kiệm VRAM khi chỉ cần trích xuất đặc trưng / suy luận.
        """
        self.model = model
        self.target_layers = target_layers
        self.detach = detach
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._is_registered = False

        self._register_hooks()

    def _get_submodule(self, module_path: str) -> nn.Module:
        """Truy xuất submodule theo đường dẫn phân cấp (vd: 'layer1.2.conv3' hoặc 'layer1')."""
        curr = self.model
        for part in module_path.split("."):
            if hasattr(curr, part):
                curr = getattr(curr, part)
            else:
                raise AttributeError(
                    f"Mô hình {self.model.__class__.__name__} không có module con '{part}' "
                    f"trong đường dẫn '{module_path}'."
                )
        return curr

    def _make_hook(self, stage_name: str) -> Callable:
        """Tạo hàm callback cho hook của từng stage cụ thể."""

        def hook(
            module: nn.Module, input_tensor: Any, output_tensor: torch.Tensor
        ) -> None:
            if self.detach:
                self.features[stage_name] = output_tensor.detach()
            else:
                self.features[stage_name] = output_tensor

        return hook

    def _register_hooks(self) -> None:
        """Đăng ký hook vào từng tầng mục tiêu."""
        if self._is_registered:
            return

        for stage_name, layer_path in self.target_layers.items():
            submodule = self._get_submodule(layer_path)
            handle = submodule.register_forward_hook(self._make_hook(stage_name))
            self.hooks.append(handle)

        self._is_registered = True

    def remove(self) -> None:
        """Gỡ bỏ toàn bộ hooks đã đăng ký để giải phóng tài nguyên và tránh memory leak."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self._is_registered = False

    def clear(self) -> None:
        """Xóa bộ đệm features đã lưu."""
        self.features.clear()

    def __enter__(self) -> "CNNFeatureHooks":
        self.clear()
        if not self._is_registered:
            self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Thực thi forward pass qua mô hình và trả về features đã trích xuất."""
        self.clear()
        _ = self.model(x)
        return dict(self.features)


class ResNet50StageExtractor(nn.Module):
    """Mô-đun Trích xuất Đặc trưng Đa Tầng ResNet-50 Backbone chuyên dụng (Task #95).

    Kiến trúc chuẩn của ResNet-50 cho ảnh đầu vào kích thước [B, 3, 224, 224]:
    -----------------------------------------------------------------------------------------------------
    Stage      Tầng PyTorch      Cấu trúc Bottleneck            Resolution (HxW)  Channels  Stride   RF
    -----------------------------------------------------------------------------------------------------
    Stem       conv1+bn+maxpool  7x7, s=2 -> 3x3 maxpool, s=2   56 x 56           64        4x       35
    Stage 1    layer1            3x Bottleneck [64, 64, 256]    56 x 56           256       4x       67
    Stage 2    layer2            4x Bottleneck [128, 128, 512]  28 x 28           512       8x       163
    Stage 3    layer3            6x Bottleneck [256, 256, 1024] 14 x 14           1024      16x      355
    Stage 4    layer4            3x Bottleneck [512, 512, 2048] 7 x 7             2048      32x      499
    -----------------------------------------------------------------------------------------------------
    """

    DEFAULT_TARGET_LAYERS = {
        "stem": "maxpool",
        "stage1": "layer1",
        "stage2": "layer2",
        "stage3": "layer3",
        "stage4": "layer4",
    }

    def __init__(
        self,
        pretrained: bool = True,
        input_size: Tuple[int, int] = (224, 224),
        freeze_stages: Optional[List[int]] = None,
    ):
        """Khởi tạo ResNet50StageExtractor.

        Args:
            pretrained: Có nạp trọng số ImageNet tiền huấn luyện hay không.
            input_size: Kích thước ảnh đầu vào chuẩn (H, W), mặc định (224, 224).
            freeze_stages: Danh sách các stage cần đóng băng trọng số (vd: [0, 1] để đóng băng stem & stage1).
        """
        super().__init__()
        self.input_size = input_size
        self.pretrained = pretrained

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = resnet50(weights=weights)

        # 1. Stem: giảm độ phân giải ban đầu 224x224 -> 56x56
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        # 2. Bốn giai đoạn xử lý phân cấp (Hierarchical Stages)
        self.layer1 = base_resnet.layer1  # Stage 1: 56x56, 256ch
        self.layer2 = base_resnet.layer2  # Stage 2: 28x28, 512ch
        self.layer3 = base_resnet.layer3  # Stage 3: 14x14, 1024ch
        self.layer4 = base_resnet.layer4  # Stage 4: 7x7,   2048ch

        # Đóng băng các stage nếu được chỉ định
        if freeze_stages:
            self._apply_freeze(freeze_stages)

        # Lưu thông số siêu dữ liệu (Metadata) cho từng stage
        self._metadata = self._build_metadata()

    def _apply_freeze(self, stages: List[int]) -> None:
        """Đóng băng gradient của các stage được chọn nhằm fine-tune ổn định."""
        stage_map = {
            0: [self.conv1, self.bn1],
            1: [self.layer1],
            2: [self.layer2],
            3: [self.layer3],
            4: [self.layer4],
        }
        for stg in stages:
            if stg in stage_map:
                for module in stage_map[stg]:
                    for p in module.parameters():
                        p.requires_grad = False

    def _build_metadata(self) -> Dict[str, StageMetadata]:
        """Xây dựng bảng thông số chi tiết của từng stage phục vụ nghiên cứu & tài liệu hoá."""
        h, w = self.input_size
        return {
            "stem": StageMetadata(
                stage_name="stem",
                layer_name="conv1 + bn1 + relu + maxpool",
                channels=64,
                height=h // 4,
                width=w // 4,
                downsampling_factor=4,
                receptive_field=35,
                num_blocks=1,
                clinical_role="Trích xuất cạnh thô, gradient ánh sáng nội soi, vùng lóa sáng (specular reflection)",
            ),
            "stage1": StageMetadata(
                stage_name="stage1",
                layer_name="layer1 (3 Bottlenecks)",
                channels=256,
                height=h // 4,
                width=w // 4,
                downsampling_factor=4,
                receptive_field=67,
                num_blocks=3,
                clinical_role="Mạng lưới vi mạch niêm mạc (micro-vascular network), vi cấu trúc bề mặt (crypt architecture)",
            ),
            "stage2": StageMetadata(
                stage_name="stage2",
                layer_name="layer2 (4 Bottlenecks)",
                channels=512,
                height=h // 8,
                width=w // 8,
                downsampling_factor=8,
                receptive_field=163,
                num_blocks=4,
                clinical_role="Mô thức rãnh niêm mạc (pit patterns), viền xung huyết viêm loét, cấu trúc mạch máu lớn",
            ),
            "stage3": StageMetadata(
                stage_name="stage3",
                layer_name="layer3 (6 Bottlenecks)",
                channels=1024,
                height=h // 16,
                width=w // 16,
                downsampling_factor=16,
                receptive_field=355,
                num_blocks=6,
                clinical_role="Biên giới khối polyp, bờ ổ loét, thành nếp gấp dạ dày/đại tràng, vùng chuyển tiếp niêm mạc",
            ),
            "stage4": StageMetadata(
                stage_name="stage4",
                layer_name="layer4 (3 Bottlenecks)",
                channels=2048,
                height=h // 32,
                width=w // 32,
                downsampling_factor=32,
                receptive_field=499,
                num_blocks=3,
                clinical_role="Ngữ nghĩa bệnh lý cấp cao (23 phân lớp HyperKvasir), phân biệt polyp vs trĩ vs viêm loét",
            ),
        }

    @property
    def stage_info(self) -> Dict[str, StageMetadata]:
        """Truy xuất bảng thuộc tính của các stage."""
        return self._metadata

    def get_stage_channels(self) -> Dict[str, int]:
        """Lấy số lượng channels tại mỗi stage."""
        return {name: meta.channels for name, meta in self._metadata.items()}

    def get_stage_resolutions(self) -> Dict[str, Tuple[int, int]]:
        """Lấy độ phân giải (H, W) tại mỗi stage đối với input_size hiện tại."""
        return {
            name: (meta.height, meta.width) for name, meta in self._metadata.items()
        }

    def forward(
        self,
        x: torch.Tensor,
        return_stem: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass trích xuất trực tiếp toàn bộ feature maps từ Stage 1 -> Stage 4.

        Args:
            x: Tensor ảnh đầu vào [B, 3, H, W].
            return_stem: Nếu True, trả về cả đặc trưng của stem [B, 64, H/4, W/4].

        Returns:
            Dictionary chứa các feature maps:
            - 'stage1': [B, 256, H/4, W/4]
            - 'stage2': [B, 512, H/8, W/8]
            - 'stage3': [B, 1024, H/16, W/16]
            - 'stage4': [B, 2048, H/32, W/32]
            - (tùy chọn) 'stem': [B, 64, H/4, W/4]
        """
        # Stem
        x_stem = self.conv1(x)
        x_stem = self.bn1(x_stem)
        x_stem = self.relu(x_stem)
        x_stem = self.maxpool(x_stem)

        # Stage 1-4
        s1 = self.layer1(x_stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        features = {
            "stage1": s1,
            "stage2": s2,
            "stage3": s3,
            "stage4": s4,
        }
        if return_stem:
            features["stem"] = x_stem

        return features

    def create_hook_extractor(self, detach: bool = False) -> CNNFeatureHooks:
        """Tạo đối tượng Hook-based extractor kết nối trực tiếp vào các module con của instance này."""
        return CNNFeatureHooks(
            model=self,
            target_layers=self.DEFAULT_TARGET_LAYERS,
            detach=detach,
        )


def build_backbone_extractor(
    pretrained: bool = True,
    input_size: Tuple[int, int] = (224, 224),
    freeze_stages: Optional[List[int]] = None,
) -> ResNet50StageExtractor:
    """Factory function tạo bộ trích xuất đặc trưng ResNet-50 cho Luận văn."""
    return ResNet50StageExtractor(
        pretrained=pretrained,
        input_size=input_size,
        freeze_stages=freeze_stages,
    )


def extract_features_with_hooks(
    model: nn.Module,
    x: torch.Tensor,
    target_layers: Optional[Dict[str, str]] = None,
    detach: bool = False,
) -> Dict[str, torch.Tensor]:
    """Hàm tiện ích trích xuất đặc trưng từ bất kỳ CNN model nào thông qua PyTorch forward hooks.

    Args:
        model: PyTorch model (vd: torchvision resnet50, timm model, custom CNN).
        x: Tensor ảnh đầu vào [B, 3, H, W].
        target_layers: Mapping tên stage -> tên tầng con (mặc định lấy theo ResNet-50).
        detach: Có detach tensor khỏi computational graph hay không.

    Returns:
        Dict chứa feature maps tại các layers được chỉ định.
    """
    if target_layers is None:
        target_layers = ResNet50StageExtractor.DEFAULT_TARGET_LAYERS

    with CNNFeatureHooks(
        model=model, target_layers=target_layers, detach=detach
    ) as hook_extractor:
        with torch.set_grad_enabled(not detach):
            _ = model(x)
        extracted = dict(hook_extractor.features)

    return extracted


if __name__ == "__main__":
    print("=" * 100)
    print("🔬 KIỂM THỬ MÔ-ĐUN TRÍCH XUẤT ĐẶC TRƯNG ĐA TẦNG CNN BACKBONE (TASK #95)")
    print("=" * 100)

    extractor = build_backbone_extractor(pretrained=False, input_size=(224, 224))
    extractor.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"📥 Input Tensor Shape: {dummy_input.shape}\n")

    # 1. Trích xuất trực tiếp
    print("--- 1. DIRECT MULTI-STAGE FORWARD EXTRACTION ---")
    with torch.no_grad():
        stage_feats = extractor(dummy_input, return_stem=True)

    for name, feat in stage_feats.items():
        meta = extractor.stage_info[name]
        print(
            f"  • {name.upper():<7} | Shape: {str(list(feat.shape)):<18} | "
            f"Stride: {meta.downsampling_factor:>2}x | RF: {meta.receptive_field:>3}px | "
            f"Channels: {meta.channels:>4} | {meta.clinical_role}"
        )

    # 2. Trích xuất bằng Hook
    print("\n--- 2. HOOK-BASED EXTRACTION (CNNFeatureHooks) ---")
    hook_extractor = extractor.create_hook_extractor(detach=True)
    with hook_extractor:
        _ = extractor(dummy_input)
        hooked_feats = hook_extractor.features

    for name, feat in hooked_feats.items():
        print(
            f"  • [HOOK] {name.upper():<7} | Intercepted Shape: {str(list(feat.shape)):<18}"
        )

    print("\n" + "=" * 100)
    print("✅ HOÀN TẤT KIỂM TRA MÔ-ĐUN TRÍCH XUẤT ĐẶC TRƯNG BACKBONE (TASK #95)")
    print("=" * 100)
