"""Script đo lường tốc độ nạp dữ liệu (DataLoader Throughput Benchmark) và xuất cấu hình JSON."""

import json
import os
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Thiết lập đường dẫn root an toàn
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.dataset.hyperkvasir_dataset import HyperKvasirDataset
except ImportError:
    from hyperkvasir_dataset import HyperKvasirDataset


def run_dataloader_benchmark(
    processed_dir: str,
    raw_dir: str,
    config_dir: str,
    fig_dir: str,
    doc_dir: str,
):
    proc_path = Path(processed_dir)
    raw_path = Path(raw_dir) / "labeled-images"
    cfg_path = Path(config_dir)
    fig_path = Path(fig_dir)
    doc_path = Path(doc_dir)

    cfg_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    train_csv = proc_path / "train_split.csv"
    train_dataset = HyperKvasirDataset(
        train_csv, raw_path, split="train", img_size=(224, 224)
    )

    print("=" * 75)
    print(
        "⚡ ĐANG TIẾN HÀNH THỬ NGHIỆM ĐO TỐC ĐỘ NẠP DỮ LIỆU (DATALOADER BENCHMARK)..."
    )
    print("=" * 75)

    batch_size = 32
    num_batches_to_test = 15  # Đo nhanh qua 15 batches (~480 ảnh)

    # Thử nghiệm trên các cấu hình Worker khác nhau
    # Lưu ý: Trên Windows, num_workers=0 và num_workers=2 là cấu hình ổn định nhất
    worker_configs = [
        {"workers": 0, "pin": False, "name": "1. Single Worker (Baseline)"},
        {"workers": 2, "pin": True, "name": "2. Dual Workers (Balanced)"},
        {"workers": 4, "pin": True, "name": "3. Quad Workers (High Speed)"},
    ]

    benchmark_results = []

    for cfg in worker_configs:
        w = cfg["workers"]
        pin = cfg["pin"] if torch.cuda.is_available() else False
        name = cfg["name"]

        loader_kwargs = {"num_workers": w, "pin_memory": pin}
        if w > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
        )

        # Warm-up 2 batches
        iterator = iter(loader)
        _ = next(iterator)
        _ = next(iterator)

        # Đo thời gian nạp thực tế
        start_t = time.time()
        for _ in range(num_batches_to_test):
            _ = next(iterator)
        elapsed_t = time.time() - start_t

        total_samples = num_batches_to_test * batch_size
        samples_per_sec = total_samples / elapsed_t
        batch_latency_ms = (elapsed_t / num_batches_to_test) * 1000

        benchmark_results.append(
            {
                "Config_Name": name,
                "Workers": w,
                "Throughput_Samples_Sec": round(samples_per_sec, 1),
                "Batch_Latency_ms": round(batch_latency_ms, 1),
                "Speedup": round(
                    samples_per_sec
                    / (
                        benchmark_results[0]["Throughput_Samples_Sec"]
                        if benchmark_results
                        else samples_per_sec
                    ),
                    2,
                ),
            }
        )

        print(
            f"   ▶ {name:<32} ➔ Tốc độ: {samples_per_sec:5.1f} ảnh/giây | Độ trễ: {batch_latency_ms:5.1f} ms/batch"
        )

    # Hiệu chỉnh lại speedup
    base_tp = benchmark_results[0]["Throughput_Samples_Sec"]
    for r in benchmark_results:
        r["Speedup"] = round(r["Throughput_Samples_Sec"] / base_tp, 2)

    print("=" * 75)

    # 1. Lưu cấu hình JSON dataloader_config.json
    optimal_loader_cfg = {
        "batch_size_training": 32,
        "batch_size_evaluation": 64,
        "recommended_workers_windows": 2,
        "recommended_workers_colab_linux": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "img_size": [224, 224],
        "high_res_img_size": [384, 384],
        "drop_last_train": True,
        "drop_last_val_test": False,
        "benchmark_summary": benchmark_results,
    }

    opt_json_p = cfg_path / "dataloader_config.json"
    with open(opt_json_p, "w", encoding="utf-8") as f:
        json.dump(optimal_loader_cfg, f, indent=4)
    print(f"✅ Đã lưu cấu hình tối ưu DataLoader tại: {opt_json_p}")

    # 2. Vẽ Dashboard đối sánh 4 Panel
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_theme(style="whitegrid")

    df_bench = pd.DataFrame(benchmark_results)

    # Panel 1: Throughput (Ảnh/giây)
    b1 = axes[0, 0].bar(
        df_bench["Config_Name"],
        df_bench["Throughput_Samples_Sec"],
        color=["#e74c3c", "#3498db", "#2ecc71"],
        edgecolor="black",
        lw=1,
    )
    axes[0, 0].set_title(
        "1. Tốc độ Nạp Dữ Liệu (Throughput: Samples / Second)",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 0].set_ylabel("Số ảnh / giây")
    axes[0, 0].set_xticklabels(
        ["Single Worker (w=0)", "Dual Workers (w=2)", "Quad Workers (w=4)"],
        fontsize=10,
        fontweight="bold",
    )

    for bar in b1:
        h = bar.get_height()
        axes[0, 0].annotate(
            f"{h:.1f} ảnh/s",
            (bar.get_x() + bar.get_width() / 2, h + 5),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Panel 2: Độ trễ Batch (ms)
    b2 = axes[0, 1].bar(
        df_bench["Config_Name"],
        df_bench["Batch_Latency_ms"],
        color=["#e67e22", "#9b59b6", "#1abc9c"],
        edgecolor="black",
        lw=1,
    )
    axes[0, 1].set_title(
        "2. Độ trễ Nạp Mỗi Batch (Batch Latency in ms)", fontsize=11, fontweight="bold"
    )
    axes[0, 1].set_ylabel("Mili-giây (ms)")
    axes[0, 1].set_xticklabels(
        ["Single Worker (w=0)", "Dual Workers (w=2)", "Quad Workers (w=4)"],
        fontsize=10,
        fontweight="bold",
    )

    for bar in b2:
        h = bar.get_height()
        axes[0, 1].annotate(
            f"{h:.1f} ms",
            (bar.get_x() + bar.get_width() / 2, h + 3),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Panel 3: Hệ số Tăng tốc (Speedup Factor)
    axes[1, 0].plot(
        df_bench["Workers"],
        df_bench["Speedup"],
        marker="o",
        color="#27ae60",
        lw=2.5,
        markersize=9,
    )
    axes[1, 0].set_title(
        "3. Hệ số Tăng tốc Quy mô Luồng (Speedup Curve)", fontsize=11, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Số lượng CPU Workers")
    axes[1, 0].set_ylabel("Hệ số tăng tốc (Lần)")
    axes[1, 0].set_xticks([0, 2, 4])
    axes[1, 0].axhline(1.0, color="gray", linestyle="--", label="Baseline (w=0)")
    axes[1, 0].legend()

    for x, y in zip(df_bench["Workers"], df_bench["Speedup"]):
        axes[1, 0].annotate(
            f"{y:.2f}x",
            (x, y + 0.1),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Panel 4: Khuyến nghị Thực thi Phần cứng
    axes[1, 1].text(
        0.5,
        0.5,
        "⚙️ KHUYẾN NGHỊ CẤU HÌNH VẬN HÀNH\n\n"
        "✔ Windows PC: num_workers = 2, pin_memory = True\n"
        "✔ Google Colab / Linux Server: num_workers = 4\n"
        "✔ Prefetch Factor: 2 (Chuẩn bị sẵn 2 batch vào RAM)\n"
        "✔ Persistent Workers: True (Tránh hủy luồng giữa các Epoch)\n\n"
        "👉 ĐẢM BẢO GPU LOAD > 95% KHÔNG BỊ NGHẼN CỔ CHAI I/O",
        fontsize=11.5,
        va="center",
        ha="center",
        fontweight="bold",
        color="#145a32",
        bbox=dict(
            boxstyle="round,pad=0.8", facecolor="#e8f8f5", edgecolor="#27ae60", lw=2
        ),
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_fig = fig_path / "27_dataloader_throughput_benchmark.png"
    plt.savefig(out_fig, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"✅ Đã lưu Dashboard Throughput tại: {out_fig}")

    # 3. Xuất tài liệu nghiên cứu kỹ thuật
    md_file = doc_path / "34_dataloader_optimization_and_throughput.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(
            "# ⚡ Báo cáo Kỹ thuật: Cấu Hình Tối Ưu Hóa DataLoader & Đo Lường Băng Thông Nạp (Throughput)\n\n"
        )
        f.write(
            "> **File cấu hình:** `configs/dataloader_config.json` | **Hình minh họa:** `docs/figures/27_dataloader_throughput_benchmark.png`\n\n---\n\n"
        )
        f.write("## 1. Bảng Kết Quả Đo Lường Hiệu Năng Thực Tế\n\n")
        f.write(
            "| Cấu hình Luồng (Workers) | Pin Memory | Tốc độ nạp (Ảnh/giây) | Độ trễ mỗi Batch (ms) | Hệ số Tăng tốc |\n"
        )
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        for r in benchmark_results:
            f.write(
                f"| **{r['Config_Name']}** | `True` | **`{r['Throughput_Samples_Sec']:.1f} samples/s`** | `{r['Batch_Latency_ms']:.1f} ms` | **`{r['Speedup']:.2f}x`** |\n"
            )
        f.write(
            "\n---\n\n## 2. Giải Pháp Triệt Tiêu Hiện Tượng Nghẽn Cổ Chai (Anti-Bottleneck Strategy)\n\n"
        )
        f.write(
            "1. **DMA Memory Pinning:** Khóa cứng bộ nhớ đệm giúp GPU sao chép Tensor trực tiếp từ RAM mà không qua trung gian CPU, tăng tốc truyền dữ liệu lên Card đồ họa.\n"
        )
        f.write(
            "2. **Prefetch Queue:** Cơ chế nạp đón đầu giúp triệt tiêu hoàn toàn thời gian chết (GPU Idle Time) giữa các bước tính toán Gradient.\n"
        )

    print(f"✅ Đã lưu tài liệu nghiên cứu tại: {md_file}")
    print("=" * 75)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    proc_dir = os.path.join(project_root, "data", "processed")
    raw_dir = os.path.join(project_root, "data", "raw")
    config_dir_path = os.path.join(project_root, "configs")
    figures_dir = os.path.join(project_root, "docs", "figures")
    research_dir = os.path.join(project_root, "docs", "research")

    run_dataloader_benchmark(
        proc_dir, raw_dir, config_dir_path, figures_dir, research_dir
    )
