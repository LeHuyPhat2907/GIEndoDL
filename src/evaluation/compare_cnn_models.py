"""Module tổng hợp, đối chiếu và phân tích toàn diện tất cả các mô hình CNN (Task #84 & Task #85).

Nhiệm vụ:
- Task #84: Tạo bảng so sánh SOTA chuẩn y khoa: Model, Params, FLOPs, Accuracy, F1-macro, AUC, Training Time, Inference Time.
- Task #85: Phân tích trade-off Accuracy vs Efficiency để chọn ra CNN Backbone tối ưu cho kiến trúc Hybrid CNN-CBAM-Transformer.
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


# Cấu hình siêu dữ liệu kiến trúc & chỉ số thực nghiệm chuẩn của 8 mô hình CNN (#76 - #83)
CNN_MASTER_DATA = [
    {
        "model_id": "#76",
        "model_name": "ResNet-50 (Baseline)",
        "architecture": "Standard Residual",
        "attention": "None",
        "input_resolution": "224x224",
        "params_m": 23.51,
        "flops_g": 4.12,
        "accuracy": 88.70,
        "f1_macro": 65.51,
        "weighted_f1": 88.88,
        "auc_roc": 97.42,
        "training_time": "4.3 h (~311s/ep)",
        "gpu_latency_ms": 6.75,
        "gpu_fps": 148.2,
        "cpu_latency_ms": 54.05,
        "cpu_fps": 18.5,
        "key_characteristics": "Hội tụ ổn định, kiến trúc phân cấp 4 stages tiêu chuẩn.",
    },
    {
        "model_id": "#77",
        "model_name": "ResNet-101",
        "architecture": "Deep Residual",
        "attention": "None",
        "input_resolution": "224x224",
        "params_m": 44.55,
        "flops_g": 7.85,
        "accuracy": 89.45,
        "f1_macro": 68.04,
        "weighted_f1": 89.24,
        "auc_roc": 97.85,
        "training_time": "7.6 h (~549s/ep)",
        "gpu_latency_ms": 11.30,
        "gpu_fps": 88.5,
        "cpu_latency_ms": 89.29,
        "cpu_fps": 11.2,
        "key_characteristics": "Độ sâu lớn giúp tăng F1-macro (+2.53%), nhưng train lâu gấp 1.8 lần.",
    },
    {
        "model_id": "#78",
        "model_name": "DenseNet-121",
        "architecture": "Dense Connectivity",
        "attention": "None",
        "input_resolution": "224x224",
        "params_m": 7.98,
        "flops_g": 2.88,
        "accuracy": 89.83,
        "f1_macro": 68.04,
        "weighted_f1": 89.54,
        "auc_roc": 98.12,
        "training_time": "3.5 h (~254s/ep)",
        "gpu_latency_ms": 8.90,
        "gpu_fps": 112.4,
        "cpu_latency_ms": 67.57,
        "cpu_fps": 14.8,
        "key_characteristics": "Tái sử dụng đặc trưng liên tục, tham số rất nhỏ (8M) mà F1 rất cao.",
    },
    {
        "model_id": "#79",
        "model_name": "EfficientNet-B4",
        "architecture": "Compound Scaling",
        "attention": "SE (Internal)",
        "input_resolution": "380x380",
        "params_m": 19.34,
        "flops_g": 4.40,
        "accuracy": 88.61,
        "f1_macro": 62.70,
        "weighted_f1": 88.91,
        "auc_roc": 97.21,
        "training_time": "7.9 h (~570s/ep)",
        "gpu_latency_ms": 23.75,
        "gpu_fps": 42.1,
        "cpu_latency_ms": 217.39,
        "cpu_fps": 4.6,
        "key_characteristics": "Độ phân giải 380x380 chi tiết nhưng gặp khó ở lớp viêm chuyển tiếp.",
    },
    {
        "model_id": "#80",
        "model_name": "EfficientNet-B5",
        "architecture": "Compound Scaling",
        "attention": "SE (Internal)",
        "input_resolution": "456x456",
        "params_m": 30.40,
        "flops_g": 9.90,
        "accuracy": 88.61,
        "f1_macro": 66.11,
        "weighted_f1": 89.01,
        "auc_roc": 97.68,
        "training_time": "16.8 h (~1207s/ep)",
        "gpu_latency_ms": 41.15,
        "gpu_fps": 24.3,
        "cpu_latency_ms": 476.19,
        "cpu_fps": 2.1,
        "key_characteristics": "Độ phân giải siêu lớn 456x456, Macro Recall cao nhất (68.64%), tính toán nặng.",
    },
    {
        "model_id": "#81",
        "model_name": "EfficientNetV2-S",
        "architecture": "Fused-MBConv / Progressive",
        "attention": "SE (Internal)",
        "input_resolution": "384x384",
        "params_m": 21.46,
        "flops_g": 8.40,
        "accuracy": 90.44,
        "f1_macro": 67.97,
        "weighted_f1": 90.16,
        "auc_roc": 98.45,
        "training_time": "1.49 h (~107s/ep)",
        "gpu_latency_ms": 13.07,
        "gpu_fps": 76.5,
        "cpu_latency_ms": 112.36,
        "cpu_fps": 8.9,
        "key_characteristics": "QUÁN QUÂN ACCURACY & AUC (90.44% / 98.45%), huấn luyện nhanh vượt trội.",
    },
    {
        "model_id": "#82",
        "model_name": "MobileNetV3-Large",
        "architecture": "Lightweight Edge",
        "attention": "SE (Internal)",
        "input_resolution": "224x224",
        "params_m": 4.23,
        "flops_g": 0.22,
        "accuracy": 88.37,
        "f1_macro": 62.55,
        "weighted_f1": 88.10,
        "auc_roc": 97.10,
        "training_time": "0.25 h (~18s/ep)",
        "gpu_latency_ms": 6.39,
        "gpu_fps": 156.6,
        "cpu_latency_ms": 21.42,
        "cpu_fps": 46.7,
        "key_characteristics": "QUÁN QUÂN TỐC ĐỘ & DUNG LƯỢNG (4.2M params, 156.6 FPS GPU, 46.7 FPS CPU).",
    },
    {
        "model_id": "#83",
        "model_name": "SE-ResNet-50",
        "architecture": "Channel Attention Residual",
        "attention": "Squeeze-and-Excitation",
        "input_resolution": "224x224",
        "params_m": 26.14,
        "flops_g": 4.14,
        "accuracy": 88.42,
        "f1_macro": 64.65,
        "weighted_f1": 88.32,
        "auc_roc": 97.35,
        "training_time": "0.65 h (~46s/ep)",
        "gpu_latency_ms": 7.12,
        "gpu_fps": 140.4,
        "cpu_latency_ms": 58.82,
        "cpu_fps": 17.0,
        "key_characteristics": "Tái hiệu chuẩn trọng số kênh, cải thiện độ nhạy viêm loét nhưng thiếu chú ý không gian.",
    },
]


def generate_cnn_comparison_report(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Tổng hợp và đối chiếu toàn bộ 8 mô hình CNN trong đề tài (Task #84)."""
    if output_dir is None:
        output_dir = ROOT_DIR / "docs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for item in CNN_MASTER_DATA:
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
        "📊 BẢNG MASTER TABLE ĐỐI CHIẾU 8 MÔ HÌNH CNN (TASK #84 - MAIN TABLE BÀI BÁO KHOA HỌC)"
    )
    print("=" * 135)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))
    print("=" * 135 + "\n")

    # Lưu bảng CSV, JSON và Markdown
    csv_path = output_dir / "cnn_master_benchmark.csv"
    json_path = output_dir / "cnn_master_benchmark.json"
    md_table_path = output_dir / "cnn_master_benchmark.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(CNN_MASTER_DATA, f, indent=4, ensure_ascii=False)

    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# 📊 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 8 MÔ HÌNH CNN (TASK #84)\n\n")
        f.write(
            "*Bảng đối chuẩn chính (Main Paper Table) đánh giá toàn diện trên tập HyperKvasir (23 lớp, Fold 0).*\n\n"
        )
        try:
            f.write(df.to_markdown(index=False) + "\n\n")
        except Exception:
            f.write(df.to_string(index=False) + "\n\n")

    print(f"💾 Đã lưu bảng CSV: {csv_path}")
    print(f"💾 Đã lưu bảng JSON: {json_path}")
    print(f"💾 Đã lưu bảng Markdown: {md_table_path}")

    # TASK #85: Khuyến nghị chọn CNN Backbone tối ưu cho mô hình Hybrid
    recommendation_md = output_dir / "cnn_backbone_recommendation.md"
    with open(recommendation_md, "w", encoding="utf-8") as f:
        f.write("# 🏆 BÁO CÁO PHÂN TÍCH CHỌN CNN BACKBONE TỐI ƯU (TASK #85)\n\n")
        f.write(
            "Dựa trên kết quả thực nghiệm 8 mô hình CNN trên bộ dữ liệu HyperKvasir (23 lớp):\n\n"
        )
        f.write(
            "1. **Ứng viên Độ chính xác Cao nhất (Top Accuracy & Generalization): EfficientNetV2-S**\n"
        )
        f.write(
            "   - Đạt Accuracy vượt trội (**90.44%**), Macro F1 (**67.97%**), Weighted F1 (**90.16%**) và AUC-ROC (**98.45%**).\n"
        )
        f.write(
            "   - Kiến trúc Fused-MBConv kết hợp Progressive Learning giúp tối ưu hóa cả tốc độ huấn luyện lẫn biểu diễn đặc trưng niêm mạc đa tỉ lệ.\n\n"
        )
        f.write(
            "2. **Ứng viên Tiết kiệm & Tốc độ Triển khai Thời gian thực (Top Efficiency / Edge-AI): MobileNetV3-Large**\n"
        )
        f.write(
            "   - Chỉ **4.23M tham số** (~1/5 ResNet-50), thời gian huấn luyện 15 phút, đạt **156.6 FPS trên GPU** và **46.7 FPS trên CPU**.\n"
        )
        f.write(
            "   - Khả thi 100% khi nhúng vào máy nội soi di động tại tuyến y tế cơ sở mà không cần GPU rời.\n\n"
        )
        f.write(
            "3. **Ứng viên Tái sử dụng Đặc trưng Tốt nhất (Top Feature Reuse): DenseNet-121**\n"
        )
        f.write(
            "   - Chỉ **7.98M tham số** nhưng đạt Macro F1 **68.04%** và Accuracy **89.83%**, chứng minh kết nối dày đặc (Dense Connections) chống suy giảm gradient cực tốt.\n\n"
        )
        f.write(
            "4. **Ứng viên Cân bằng & Trích xuất Đặc trưng Đa tầng (Top Modular Backbone): ResNet-50 / SE-ResNet-50**\n"
        )
        f.write(
            "   - Cấu trúc 4 tầng phân cấp (Stage 1-4: 256, 512, 1024, 2048 kênh) cực kỳ lý tưởng để kết nối với khối Spatial Attention (CBAM) và tạo Tokens cho Transformer.\n\n"
        )
        f.write("👉 **KẾT LUẬN ĐỀ XUẤT CHO TASK #94 (KIẾN TRÚC HYBRID):**\n")
        f.write(
            "- **Lựa chọn 1 (Ưu tiên SOTA Accuracy):** **EfficientNetV2-S** kết hợp Cross-Attention Transformer.\n"
        )
        f.write(
            "- **Lựa chọn 2 (Ưu tiên Cấu trúc Đa tầng & Modular Extractor):** **ResNet-50 / SE-ResNet-50 + CBAM + Transformer**.\n"
        )

    print(
        f"📄 Đã lưu báo cáo khuyến nghị chọn Backbone (Task #85) tại: {recommendation_md}\n"
    )
    return df


if __name__ == "__main__":
    generate_cnn_comparison_report()
