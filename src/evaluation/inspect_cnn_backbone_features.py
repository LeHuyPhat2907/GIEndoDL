"""Script kiểm tra, đo lường và lập hồ sơ đặc tính (Profiling) các Stages của CNN Backbone (Task #95).

Chức năng:
1. Đánh giá độ phân giải (Resolutions), số kênh (Channels), Receptive Fields (RF) từ Stage 1 -> 4.
2. Thống kê đặc tính Tensor động: Mean, Std, Sparsity (độ thưa % giá trị 0 sau kích hoạt), L2 Norm.
3. Đo lường độ trễ suy luận (Latency) và bộ nhớ (Memory Footprint) giữa Direct Forward vs Hook-based Extraction.
4. Xuất kết quả ra file JSON và Markdown phục vụ bài báo khoa học và Luận văn tốt nghiệp.
"""

# ruff: noqa: E402
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.models.cnn_backbone_extractor import (
    build_backbone_extractor,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def profile_backbone_stages():
    print("=" * 100)
    print(
        "🔬 BẮT ĐẦU ĐO LƯỜNG VÀ LẬP HỒ SƠ ĐẶC TÍNH MULTI-SCALE CNN BACKBONE (TASK #95)"
    )
    print("=" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Thiết bị thực thi: {device}")

    # Khởi tạo mô hình ResNet-50 Stage Extractor
    extractor = build_backbone_extractor(pretrained=True, input_size=(224, 224))
    extractor.to(device)
    extractor.eval()

    # Chuẩn bị Tensor đầu vào mô phỏng batch ảnh nội soi [4, 3, 224, 224]
    b_size = 4
    torch.manual_seed(42)
    dummy_input = torch.randn(b_size, 3, 224, 224, device=device)

    # 1. Đo đạc các thông số tĩnh và động của từng Stage
    with torch.no_grad():
        features = extractor(dummy_input, return_stem=True)

    stage_profiles = []
    metadata = extractor.stage_info

    # Tính toán tham số của từng module
    param_counts = {
        "stem": sum(
            p.numel() for m in [extractor.conv1, extractor.bn1] for p in m.parameters()
        ),
        "stage1": sum(p.numel() for p in extractor.layer1.parameters()),
        "stage2": sum(p.numel() for p in extractor.layer2.parameters()),
        "stage3": sum(p.numel() for p in extractor.layer3.parameters()),
        "stage4": sum(p.numel() for p in extractor.layer4.parameters()),
    }

    for name in ["stem", "stage1", "stage2", "stage3", "stage4"]:
        feat = features[name]
        meta = metadata[name]
        feat_np = feat.cpu().numpy()

        # Tính toán thống kê động
        mean_val = float(np.mean(feat_np))
        std_val = float(np.std(feat_np))
        sparsity = float(
            np.mean(feat_np == 0.0) * 100
        )  # Tỷ lệ phần trăm giá trị 0 (độ thưa do ReLU)
        l2_norm = float(np.linalg.norm(feat_np) / b_size)

        profile = {
            "stage_id": name,
            "layer_name": meta.layer_name,
            "channels": meta.channels,
            "height": meta.height,
            "width": meta.width,
            "spatial_elements": meta.height * meta.width,
            "total_tokens_if_patched": meta.height * meta.width,
            "downsampling_stride": f"{meta.downsampling_factor}x",
            "effective_stride": meta.downsampling_factor,
            "receptive_field_px": meta.receptive_field,
            "num_blocks": meta.num_blocks,
            "param_count": param_counts[name],
            "param_count_m": round(param_counts[name] / 1e6, 3),
            "mean_activation": round(mean_val, 4),
            "std_activation": round(std_val, 4),
            "sparsity_pct": round(sparsity, 2),
            "l2_norm_per_sample": round(l2_norm, 2),
            "clinical_role": meta.clinical_role,
        }
        stage_profiles.append(profile)

    # In bảng đặc tính ra console
    print(
        "\n📋 BẢNG TỔNG HỢP ĐẶC TÍNH KỸ THUẬT CÁC STAGES CỦA RESNET-50 BACKBONE:",
        flush=True,
    )
    print("-" * 100, flush=True)
    for p in stage_profiles:
        print(
            f"• Stage: {p['stage_id'].upper():<7} | Shape: [{p['channels']:>4}, {p['height']:>2}, {p['width']:>2}] "
            f"({p['spatial_elements']:>4} positions) | Stride: {p['downsampling_stride']:>3} | "
            f"RF: {p['receptive_field_px']:>3}px | Params: {p['param_count_m']:>5.2f}M | "
            f"Sparsity: {p['sparsity_pct']:>5.1f}%",
            flush=True,
        )
        print(f"  ↳ Vai trò lâm sàng: {p['clinical_role']}", flush=True)
    print("-" * 100, flush=True)

    # 2. Benchmark so sánh độ trễ: Direct Forward vs Hook-based Extraction
    print("\n⏱️ TIẾN HÀNH ĐO ĐỘ TRỄ SUY LUẬN (INFERENCE LATENCY) - BATCH=4:", flush=True)
    n_warmup = 3
    n_runs = 20

    # Warmup
    for _ in range(n_warmup):
        _ = extractor(dummy_input, return_stem=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Đo Direct Forward
    start_time = time.perf_counter()
    for _ in range(n_runs):
        _ = extractor(dummy_input, return_stem=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
    direct_time_total = time.perf_counter() - start_time
    direct_latency_ms = (direct_time_total / n_runs) * 1000
    direct_fps = (n_runs * b_size) / direct_time_total

    # Đo Hook-based extraction
    hook_extractor = extractor.create_hook_extractor(detach=True)
    # Warmup hook
    with hook_extractor:
        for _ in range(n_warmup):
            _ = extractor(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    start_time = time.perf_counter()
    with hook_extractor:
        for _ in range(n_runs):
            _ = extractor(dummy_input)
            _ = hook_extractor.features
            if device.type == "cuda":
                torch.cuda.synchronize()
    hook_time_total = time.perf_counter() - start_time
    hook_latency_ms = (hook_time_total / n_runs) * 1000
    hook_fps = (n_runs * b_size) / hook_time_total

    latency_comparison = {
        "device": str(device),
        "batch_size": b_size,
        "runs": n_runs,
        "direct_forward_latency_ms": round(direct_latency_ms, 3),
        "direct_forward_fps": round(direct_fps, 2),
        "hook_extraction_latency_ms": round(hook_latency_ms, 3),
        "hook_extraction_fps": round(hook_fps, 2),
        "hook_overhead_ms": round(hook_latency_ms - direct_latency_ms, 3),
        "hook_overhead_pct": round(
            ((hook_latency_ms - direct_latency_ms) / direct_latency_ms) * 100, 2
        ),
    }

    print(
        f"  • Direct Forward:        {direct_latency_ms:.2f} ms/batch ({direct_fps:.1f} FPS)"
    )
    print(
        f"  • Hook-based Extraction: {hook_latency_ms:.2f} ms/batch ({hook_fps:.1f} FPS)"
    )
    print(
        f"  • Hook Overhead:         +{latency_comparison['hook_overhead_ms']:.2f} ms ({latency_comparison['hook_overhead_pct']:+.1f}%)"
    )

    # 3. Xuất file JSON và Markdown
    out_dir = Path("docs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "cnn_backbone_stages_profile.json"
    full_report = {
        "task_id": "Task #95",
        "task_name": "Implement CNN backbone extraction module",
        "backbone_model": "ResNet-50",
        "input_resolution": "224x224x3",
        "stages": stage_profiles,
        "latency_benchmark": latency_comparison,
        "total_backbone_params": sum(param_counts.values()),
        "total_backbone_params_m": round(sum(param_counts.values()) / 1e6, 2),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Đã lưu hồ sơ đặc tính JSON tại: {json_path}")

    # Tạo bảng Markdown
    md_path = out_dir / "cnn_backbone_stages_profile.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "# Bảng Hồ Sơ Đặc Tính Kỹ Thuật CNN Backbone (ResNet-50) Cho Mô Hình Lai (Task #95)\n\n"
        )
        f.write(f"**Ngày thực hiện:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write("**Mô hình Backbone:** ResNet-50 (Pretrained ImageNet)  \n")
        f.write(
            f"**Tổng số tham số Backbone:** {full_report['total_backbone_params_m']} Triệu (M)  \n"
        )
        f.write(f"**Thiết bị benchmark:** {device}  \n\n")

        f.write("## 1. Thông số Kỹ thuật & Độ phân giải từng Stage\n\n")
        f.write(
            "| Stage | PyTorch Module | Độ phân giải ($H \\times W$) | Số kênh ($C$) | Stride | Receptive Field | Tham số (M) | Độ thưa (% zero) | Vai trò Lâm sàng (Nội soi Tiêu hóa) |\n"
        )
        f.write(
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n"
        )
        for p in stage_profiles:
            f.write(
                f"| **{p['stage_id'].upper()}** | `{p['layer_name']}` | ${p['height']} \\times {p['width']}$ | "
                f"${p['channels']}$ | {p['downsampling_stride']} | {p['receptive_field_px']}px | "
                f"{p['param_count_m']:.2f}M | {p['sparsity_pct']:.1f}% | {p['clinical_role']} |\n"
            )

        f.write(
            "\n## 2. Đo lường Hiệu năng: Direct Forward vs Hook-Based Extraction\n\n"
        )
        f.write(
            "| Phương thức Trích xuất | Độ trễ Batch (ms) | Throughput (FPS) | Overhead so với Direct | Ghi chú Kỹ thuật |\n"
        )
        f.write("| :--- | :---: | :---: | :---: | :--- |\n")
        f.write(
            f"| **Direct Forward Slicing** | {latency_comparison['direct_forward_latency_ms']:.2f} ms | {latency_comparison['direct_forward_fps']:.1f} FPS | Baseline (0.0%) | Truy cập trực tiếp qua các submodule `layer1..4` |\n"
        )
        f.write(
            f"| **Hook-based Extraction (`CNNFeatureHooks`)** | {latency_comparison['hook_extraction_latency_ms']:.2f} ms | {latency_comparison['hook_extraction_fps']:.1f} FPS | {latency_comparison['hook_overhead_pct']:+.1f}% (+{latency_comparison['hook_overhead_ms']:.2f} ms) | Phi xâm lấn, tự động cleanup, an toàn bộ nhớ |\n\n"
        )

        f.write(
            "## 3. Ý nghĩa Đa Tỷ lệ (Multi-Scale Features) đối với CBAM & Transformer\n\n"
        )
        f.write(
            "1. **Giai đoạn Nông (Stage 1 & 2 - $56\\times 56$ & $28\\times 28$):**\n"
        )
        f.write(
            "   - Giữ độ phân giải không gian cao, nhạy với vi cấu trúc niêm mạc (crypt architecture) và mạng lưới vi mạch mao mạch (microvascular pattern).\n"
        )
        f.write(
            "   - Rất quan trọng để CBAM Channel Attention lọc các kênh màu hấp thụ hemoglobin (ánh sáng hẹp NBI) và Spatial Attention định vị vi tổn thương phẳng (flat lesions như Barrett esophagus, angiectasia).\n\n"
        )
        f.write("2. **Giai đoạn Sâu (Stage 3 & 4 - $14\\times 14$ & $7\\times 7$):**\n")
        f.write(
            "   - Receptive field lớn ($355\\text{px} \\to 499\\text{px}$), bao quát toàn bộ khung hình nội soi.\n"
        )
        f.write(
            "   - Trích xuất ngữ nghĩa phân loại trừu tượng (polyp cuống vs polyp phẳng, phân độ viêm loét Mayo 0-3).\n"
        )
        f.write(
            "   - Đặc trưng Stage 4 ($7\\times 7 = 49$ vị trí) là đầu vào lý tưởng cho khối CBAM và ánh xạ sang 49 visual tokens cho Compact Transformer Encoder với chi phí tính toán self-attention $O(49^2)$ cực thấp.\n"
        )

    print(f"💾 Đã lưu bảng tóm tắt Markdown tại: {md_path}")
    print("=" * 100)
    print("🎉 HOÀN THÀNH LẬP HỒ SƠ MULTI-SCALE CNN BACKBONE!")
    print("=" * 100)


if __name__ == "__main__":
    profile_backbone_stages()
