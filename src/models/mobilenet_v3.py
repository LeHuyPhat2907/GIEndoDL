"""Module xây dựng kiến trúc MobileNetV3-Large cho phân loại nội soi tiêu hóa (Task #82).

Đặc điểm kiến trúc:
- Kiến trúc mạng siêu nhẹ (Lightweight CNN) được thiết kế qua Neural Architecture Search (NAS).
- Sử dụng khối inverted residual với Depthwise Separable Convolution, Squeeze-and-Excitation (SE), và hàm kích hoạt Hard-Swish.
- Phục vụ làm Baseline Efficiency cho triển khai hệ thống biên (Edge Devices) và hỗ trợ chẩn đoán thời gian thực (Real-time Video Endoscopy).
- Hỗ trợ đo lường độ trễ suy luận (Inference Latency, Throughput FPS) trên cả GPU và CPU.
"""

from typing import Dict, Tuple
import time
import torch
import torch.nn as nn


def build_mobilenet_v3_large(
    num_classes: int = 23,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Khởi tạo mô hình MobileNetV3-Large với đầu phân loại 23 lớp chuẩn Y sinh học.

    Task #82: Load pretrained MobileNetV3; Fine-tune; Đo inference time.
    """
    model = None
    use_timm = False

    # 1. Ưu tiên tải từ thư viện timm
    try:
        import timm

        for model_name in [
            "mobilenetv3_large_100",
            "mobilenetv3_large_100.ra_in1k",
            "tf_mobilenetv3_large_100",
        ]:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                )
                use_timm = True
                print(f"✅ Đã tải mô hình MobileNetV3-Large ({model_name}) thành công từ thư viện timm!")
                break
            except Exception:
                continue
    except ImportError:
        print("⚠️ Chưa phát hiện thư viện timm. Đang tự động chuyển sang torchvision.models.mobilenet_v3_large...")

    # 2. Cơ chế Fallback an toàn với torchvision
    if model is None:
        from torchvision.models import (
            MobileNet_V3_Large_Weights,
            mobilenet_v3_large,
        )

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)

        # Cấu trúc classifier của torchvision mobilenet_v3_large:
        # classifier[0]: Linear(960, 1280), classifier[1]: Hardswish(), classifier[2]: Dropout(p=0.2), classifier[3]: Linear(1280, 1000)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=drop_rate)
        model.classifier[3] = nn.Linear(in_features, num_classes)
        print(f"✅ Đã tải mô hình MobileNetV3-Large thành công từ torchvision (pretrained={pretrained})!")

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


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cuda",
    num_runs: int = 200,
    warmup: int = 30,
) -> Dict[str, float]:
    """Đo lường chính xác độ trễ suy luận (Latency) và số khung hình/giây (FPS) chuẩn Y khoa.

    Hỗ trợ đánh giá khả năng triển khai thời gian thực (Real-time Video Endoscopy >= 30 FPS).
    """
    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    model = model.to(dev)
    model.eval()

    dummy_input = torch.randn(input_size, device=dev)

    # 1. Warmup GPU/CPU để ổn định cache và xung nhịp phần cứng
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    # 2. Thực hiện đo lường chính xác qua nhiều lần lặp
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
            timings.append((end - start) * 1000.0)  # milliseconds

    timings = timings[10:]  # loại bỏ ngoại lai ban đầu
    mean_latency_ms = float(sum(timings) / len(timings))
    min_latency_ms = float(min(timings))
    max_latency_ms = float(max(timings))
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    return {
        "device": str(dev),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "min_latency_ms": round(min_latency_ms, 2),
        "max_latency_ms": round(max_latency_ms, 2),
        "fps": round(fps, 1),
        "realtime_ready_30fps": fps >= 30.0,
        "realtime_ready_60fps": fps >= 60.0,
    }
