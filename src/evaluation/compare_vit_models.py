"""Module tổng hợp, đối chiếu và phân tích toàn diện tất cả các mô hình Vision Transformer (Task #92).

Nhiệm vụ:
- Task #92: Tạo bảng so sánh SOTA chuẩn y khoa: Model, Params, FLOPs, Accuracy, F1-macro, AUC, Training Time, Inference Time.
- Đối chiếu chuyên sâu 6 mô hình Vision Transformer (#86 - #91) với 8 mô hình CNN (#76 - #83).
- Xuất bản bảng Master Table dạng CSV, JSON và Markdown cho bài báo khoa học.
"""

import json
from pathlib import Path
import sys
from typing import Optional
import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# Cấu hình siêu dữ liệu kiến trúc & chỉ số thực nghiệm chuẩn của 6 mô hình Vision Transformer (#86 - #91)
VIT_MASTER_DATA = [
    {
        "model_id": "#86",
        "model_name": "ViT-Base/16",
        "architecture": "Standard Vision Transformer",
        "attention": "Global Multi-Head Self-Attention",
        "input_resolution": "224x224",
        "params_m": 85.80,
        "flops_g": 17.58,
        "accuracy": 87.34,
        "f1_macro": 60.41,
        "weighted_f1": 87.19,
        "auc_roc": 98.89,
        "training_time": "1.23 h (~88.8s/ep)",
        "gpu_latency_ms": 9.85,
        "gpu_fps": 101.5,
        "cpu_latency_ms": 586.65,
        "cpu_fps": 1.7,
        "key_characteristics": "Mốc cơ sở Transformer kinh điển, 196 tokens phẳng (16x16), thiếu inductive bias nên Macro F1 thấp (60.41%), 40 ca báo động giả polyp.",
    },
    {
        "model_id": "#87",
        "model_name": "ViT-Base/32",
        "architecture": "Coarse-Patch Vision Transformer",
        "attention": "Global Multi-Head Self-Attention (Coarse)",
        "input_resolution": "224x224",
        "params_m": 87.45,
        "flops_g": 4.41,
        "accuracy": 85.51,
        "f1_macro": 60.02,
        "weighted_f1": 85.16,
        "auc_roc": 99.14,
        "training_time": "0.42 h (~30.6s/ep)",
        "gpu_latency_ms": 3.42,
        "gpu_fps": 292.4,
        "cpu_latency_ms": 206.29,
        "cpu_fps": 4.8,
        "key_characteristics": "Token thô (32x32) chỉ 49 tokens, tính toán nhẹ (4.4 GFLOPs) nhưng mất mát đặc trưng niêm mạc vi mô, gây 83 ca báo động giả polyp.",
    },
    {
        "model_id": "#88",
        "model_name": "DeiT-Base Distilled",
        "architecture": "Distilled Vision Transformer",
        "attention": "Distillation Token + Self-Attention",
        "input_resolution": "224x224",
        "params_m": 85.84,
        "flops_g": 17.65,
        "accuracy": 87.06,
        "f1_macro": 63.05,
        "weighted_f1": 86.88,
        "auc_roc": 98.43,
        "training_time": "1.12 h (~80.8s/ep)",
        "gpu_latency_ms": 10.45,
        "gpu_fps": 95.7,
        "cpu_latency_ms": 703.86,
        "cpu_fps": 1.4,
        "key_characteristics": "Chưng cất tri thức từ CNN teacher giúp bù đắp inductive bias, tăng vọt F1 ở các lớp viêm ruột (+20% F1), nhưng vẫn dính 79 ca báo động giả polyp.",
    },
    {
        "model_id": "#89",
        "model_name": "Swin-Base",
        "architecture": "Hierarchical Window Transformer",
        "attention": "Shifted Window Attention (W-MSA & SW-MSA)",
        "input_resolution": "224x224",
        "params_m": 86.77,
        "flops_g": 15.44,
        "accuracy": 89.73,
        "f1_macro": 69.78,
        "weighted_f1": 90.18,
        "auc_roc": 99.48,
        "training_time": "1.92 h (~138.3s/ep)",
        "gpu_latency_ms": 14.85,
        "gpu_fps": 67.3,
        "cpu_latency_ms": 1176.32,
        "cpu_fps": 0.9,
        "key_characteristics": "QUÁN QUÂN MACRO F1 TOÀN BỘ BENCHMARK (69.78%), cấu trúc phân cấp đa tầng triệt tiêu 98.7% báo động giả polyp (chỉ 1 ca FP, Precision 99.50%).",
    },
    {
        "model_id": "#90",
        "model_name": "Swin-Tiny",
        "architecture": "Lightweight Hierarchical Transformer",
        "attention": "Shifted Window Attention (Lightweight)",
        "input_resolution": "224x224",
        "params_m": 27.54,
        "flops_g": 4.49,
        "accuracy": 89.97,
        "f1_macro": 65.19,
        "weighted_f1": 89.88,
        "auc_roc": 99.05,
        "training_time": "0.78 h (~56.2s/ep)",
        "gpu_latency_ms": 5.62,
        "gpu_fps": 177.9,
        "cpu_latency_ms": 440.39,
        "cpu_fps": 2.3,
        "key_characteristics": "QUÁN QUÂN ACCURACY & GỌN NHẸ HỌ TRANSFORMER (89.97% Acc, 27.5M params), Cohen Kappa 0.8912, huấn luyện nhanh gấp 2.5 lần Swin-Base.",
    },
    {
        "model_id": "#91",
        "model_name": "CaiT-S24",
        "architecture": "Class-Attention Two-Stage Transformer",
        "attention": "Patch Self-Attention + Class-Attention",
        "input_resolution": "224x224",
        "params_m": 46.54,
        "flops_g": 9.35,
        "accuracy": 82.98,
        "f1_macro": 59.74,
        "weighted_f1": 82.86,
        "auc_roc": 99.40,
        "training_time": "12.94 h (~820s/ep)",
        "gpu_latency_ms": 18.20,
        "gpu_fps": 54.9,
        "cpu_latency_ms": 685.29,
        "cpu_fps": 1.5,
        "key_characteristics": "Quá khớp nghiêm trọng sau epoch 8 do 24 tầng sâu phẳng, Class-Attention không hợp ảnh niêm mạc liên tục, nhầm 93 ca môn vị sang polyp, 159 ca báo động giả.",
    },
]


def generate_vit_comparison_report(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Tổng hợp và đối chiếu toàn bộ 6 mô hình Vision Transformer trong đề tài (Task #92)."""
    if output_dir is None:
        output_dir = ROOT_DIR / "docs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for item in VIT_MASTER_DATA:
        records.append(
            {
                "Task ID": item["model_id"],
                "Model": item["model_name"],
                "Architecture": item["architecture"],
                "Attention Mechanism": item["attention"],
                "Resolution": item["input_resolution"],
                "Params (M)": item["params_m"],
                "FLOPs (G)": item["flops_g"],
                "Accuracy (%)": item["accuracy"],
                "F1-macro (%)": item["f1_macro"],
                "Weighted F1 (%)": item["weighted_f1"],
                "AUC-ROC (%)": item["auc_roc"],
                "Training Time (50ep)": item["training_time"],
                "GPU Latency (ms)": item["gpu_latency_ms"],
                "GPU FPS": item["gpu_fps"],
                "CPU Latency (ms)": item["cpu_latency_ms"],
                "CPU FPS": item["cpu_fps"],
            }
        )

    df = pd.DataFrame(records)

    print("\n" + "=" * 135)
    print(
        "📊 BẢNG MASTER TABLE ĐỐI CHIẾU 6 MÔ HÌNH VISION TRANSFORMER (TASK #92 - MAIN TABLE BÀI BÁO KHOA HỌC)"
    )
    print("=" * 135)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))
    print("=" * 135 + "\n")

    # Lưu bảng CSV, JSON và Markdown
    csv_path = output_dir / "vit_master_benchmark.csv"
    json_path = output_dir / "vit_master_benchmark.json"
    md_table_path = output_dir / "vit_master_benchmark.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(VIT_MASTER_DATA, f, indent=4, ensure_ascii=False)

    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write(
            "# 📊 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 6 MÔ HÌNH VISION TRANSFORMER (TASK #92)\n\n"
        )
        f.write(
            "*Bảng đối chuẩn chính (Main Paper Table) đánh giá toàn diện các biến thể Vision Transformer trên tập HyperKvasir (23 lớp, Fold 0).*\n\n"
        )
        try:
            f.write(df.to_markdown(index=False) + "\n\n")
        except Exception:
            f.write(df.to_string(index=False) + "\n\n")

    print(f"💾 Đã lưu bảng CSV: {csv_path}")
    print(f"💾 Đã lưu bảng JSON: {json_path}")
    print(f"💾 Đã lưu bảng Markdown: {md_table_path}")

    # Tạo bảng đối chiếu liên họ kiến trúc: CNNs (#76 - #83) vs Vision Transformers (#86 - #91)
    cnn_csv_path = output_dir / "cnn_master_benchmark.csv"
    if cnn_csv_path.exists():
        cnn_df = pd.read_csv(cnn_csv_path)
        combined_df = pd.concat([cnn_df, df], ignore_index=True)
        combined_csv = output_dir / "cnn_vs_vit_cross_benchmark.csv"
        combined_md = output_dir / "cnn_vs_vit_cross_benchmark.md"

        combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
        with open(combined_md, "w", encoding="utf-8") as f:
            f.write(
                "# 🏆 BẢNG ĐỐI CHIẾU TOÀN DIỆN LIÊN KIẾN TRÚC: 8 CNNs vs 6 VISION TRANSFORMERS (PHASE 6 & PHASE 7)\n\n"
            )
            f.write(
                "*Bảng tổng hợp SOTA toàn diện 14 mô hình học sâu trên tập dữ liệu nội soi HyperKvasir (23 lớp, Fold 0).*\n\n"
            )
            try:
                f.write(combined_df.to_markdown(index=False) + "\n\n")
            except Exception:
                f.write(combined_df.to_string(index=False) + "\n\n")
        print(f"💾 Đã lưu bảng so sánh liên kiến trúc: {combined_csv}")
        print(f"💾 Đã lưu bảng Markdown so sánh liên kiến trúc: {combined_md}")

    return df


if __name__ == "__main__":
    generate_vit_comparison_report()
