# -*- coding: utf-8 -*-
"""Tạo notebook kaggle_train_production_run7.ipynb chuẩn mực phục vụ Task D49 (23 Phân lớp SOTA)."""

import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏥 GIEndoDL: TASK D49 - HUẤN LUYỆN SẢN XUẤT ĐỈNH CAO RUN 7 (23 PHÂN LỚP NGUYÊN BẢN HYPERKVASIR SOTA)\n",
                "## Đề tài: Ứng dụng Học Sâu trong Nhận diện và Phân loại Tổn thương Đường Tiêu hóa từ Ảnh Nội soi\n",
                "**Học viên thực hiện:** Lê Huy Phát | **Đơn vị:** Đại học Công Thương TP.HCM (HUIT)  \n",
                "**Mô hình Đề xuất:** Hybrid CNN-CBAM-Transformer (ResNet-50 + CBAM Stage 4 + 2-layer Transformer + GAP Head)  \n",
                "**Bộ Dữ liệu:** 23 phân lớp nguyên bản HyperKvasir (Độ mất cân bằng cực đại 191:1)  \n",
                "**Mục tiêu Task D49:** Khắc phục triệt để hiện tượng sụp đổ Macro F1 ở Run 1–3 bằng 4 điều kiện cốt lõi, đưa toàn bộ 23 lớp đạt trạng thái nhận diện hội tụ bền vững, không lớp nào bị $F_1 = 0\\%$, xuất trọn bộ **24 tệp kết quả lâm sàng** chuẩn y khoa.\n",
                "\n",
                "---\n",
                "### 📋 4 Trụ Cột Kỹ Thuật Đột Phá Trong Run 7:\n",
                "1. **Phân hoạch Dữ liệu Nâng cao (StratifiedGroupKFold + Fallback Cực tiểu số):** Khóa chặt 100% `patient_video_cluster_id` (Zero Video Leakage = 0.00000000%), đảm bảo các lớp siêu hiếm (`hemorrhoids`, `ileum`, `ulcerative-colitis-grade-1-2`) có $\\ge 2$ mẫu độc lập trong validation.\n",
                "2. **Targeted Minority Augmentation & Sampler:** `WeightedRandomSampler` với số mũ $\\gamma = 0.60$ ($w_i = (1 / N_c)^{0.60}$) kết hợp biến dạng quang học chuyên biệt cho 4 lớp tử huyệt: ElasticTransform ($\\alpha=120, \\sigma=9.6$), GridDistortion và ColorJitter hemoglobin.\n",
                "3. **Hàm mất mát Phân tầng (Class-Balanced Focal Loss + Ordinal Cost Matrix):** $\\beta=0.9999, \\gamma=2.0$, tích hợp ma trận phạt khoảng cách thứ bậc Mayo Grade ($0.1 \\times |\\Delta \\text{rank}|$) và phạt nặng ($2.0$) khi bỏ sót tổn thương biến thành mốc giải phẫu lành tính.\n",
                "4. **Suy luận Hậu kỳ (Temperature Scaling & Per-Class Threshold Optimization):** Hiệu chuẩn xác suất $T=1.3$ và tối ưu hóa vector ngưỡng $\\boldsymbol{\\tau} = [\\tau_1, \\dots, \\tau_{23}]$ bằng thuật toán Nelder-Mead tối đa hóa trực tiếp Val Macro F1.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. KIỂM TRA PHẦN CỨNG GPU & MÔI TRƯỜNG KAGGLE\n",
                "!nvidia-smi\n",
                "\n",
                "import os\n",
                "import sys\n",
                "import time\n",
                "from pathlib import Path\n",
                "import torch\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print('=' * 80)\n",
                "print(f'🚀 PyTorch Version: {torch.__version__}')\n",
                "print(f'⚡ Device: {device} | CUDA Available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'🎮 GPU Model: {torch.cuda.get_device_name(0)}')\n",
                "    print(f'💾 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')\n",
                "print('=' * 80)",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. CÀI ĐẶT CÁC THƯ VIỆN BỔ TRỢ\n",
                "!pip install -q albumentations timm scikit-learn matplotlib seaborn tqdm scipy",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. CẬP NHẬT MÃ NGUỒN MỚI NHẤT TỪ GITHUB\n",
                "import os\n",
                "import subprocess\n",
                "from pathlib import Path\n",
                "\n",
                'REPO_DIR = Path("/kaggle/working/GIEndoDL")\n',
                "if not REPO_DIR.exists():\n",
                '    print("⚡ Đang clone mã nguồn GIEndoDL từ GitHub...")\n',
                '    subprocess.run(["git", "clone", "https://github.com/LeHuyPhat2907/GIEndoDL.git", str(REPO_DIR)], check=True)\n',
                "\n",
                "%cd /kaggle/working/GIEndoDL\n",
                'print("⚡ Đang kéo các bản cập nhật mới nhất từ GitHub...")\n',
                "!git pull origin main\n",
                'print(f"📂 Thư mục làm việc hiện tại: {Path.cwd()}")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. KIỂM TRA BỘ DỮ LIỆU HYPERKVASIR (NẾU CHƯA CÓ, TỰ ĐỘNG TẢI TỪ SIMULA)\n",
                'raw_dir = Path("/kaggle/working/GIEndoDL/data/raw/labeled-images")\n',
                "if not raw_dir.exists() or len(list(raw_dir.glob('*/*/*.jpg'))) < 1000:\n",
                "    has_input = False\n",
                '    if Path("/kaggle/input").exists():\n',
                '        for root, _, files in os.walk("/kaggle/input"):\n',
                "            if any(f.lower().endswith('.jpg') for f in files):\n",
                "                has_input = True\n",
                '                print(f"✅ Đã tìm thấy dữ liệu ảnh trong /kaggle/input!")\n',
                "                break\n",
                "    if not has_input:\n",
                '        print("🌐 Dữ liệu chưa tồn tại, bắt đầu tải trực tiếp từ máy chủ Simula (3.9 GB)...")\n',
                "        !python src/dataset/download_hyperkvasir.py\n",
                "else:\n",
                '    print(f"✅ Dữ liệu HyperKvasir đã có sẵn sàng tại: {raw_dir}")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. THỰC THI HUẤN LUYỆN SẢN XUẤT PRODUCTION RUN 7 (23 PHÂN LỚP SOTA - 50 EPOCHS)\n",
                "!python src/training/train_production_run7.py \\\n",
                "    --device cuda \\\n",
                "    --epochs 50 \\\n",
                "    --batch_size 32 \\\n",
                "    --img_size 224 \\\n",
                "    --fast \\\n",
                "    --lr_backbone 1.5e-5 \\\n",
                "    --lr_transformer 8e-5 \\\n",
                "    --lr_head 3.5e-4 \\\n",
                "    --weight_decay 2e-2 \\\n",
                "    --dropout 0.35 \\\n",
                "    --t_0 15 \\\n",
                "    --t_mult 1 \\\n",
                "    --swa_start 20 \\\n",
                "    --beta_sampler 0.60 \\\n",
                "    --beta_cb 0.9999 \\\n",
                "    --gamma 2.0 \\\n",
                "    --label_smoothing 0.08 \\\n",
                "    --lambda_ord 0.35 \\\n",
                "    --temperature 1.3 \\\n",
                "    --num_workers 4 \\\n",
                "    --output_dir outputs/optimal_hybrid_model_run7",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. HIỂN THỊ TRỰC QUAN HÓA KẾT QUẢ ĐỐI SÁNH TỐI ƯU & MA TRẬN NHẦM LẪN 23 LỚP\n",
                "from IPython.display import Image, display, Markdown\n",
                "\n",
                'out_dir = Path("/kaggle/working/GIEndoDL/outputs/optimal_hybrid_model_run7")\n',
                "\n",
                'print("📊 ĐỒ THỊ ĐỐI SÁNH TỐI ƯU 4 GÓC ĐỘ (LOSS, MACRO F1, ACCURACY, FPS):")\n',
                'if (out_dir / "optimal_training_convergence_curves.png").exists():\n',
                '    display(Image(filename=str(out_dir / "optimal_training_convergence_curves.png")))\n',
                "\n",
                'print("📊 MA TRẬN NHẦM LẪN CHUẨN HÓA (23x23 CONFUSION MATRIX):")\n',
                'if (out_dir / "confusion_matrix_normalized.png").exists():\n',
                '    display(Image(filename=str(out_dir / "confusion_matrix_normalized.png")))\n',
                "\n",
                'print("📊 BẢNG XẾP HẠNG F1-SCORE TRÊN TOÀN BỘ 23 PHÂN LỚP:")\n',
                'if (out_dir / "per_class_f1_ranking.png").exists():\n',
                '    display(Image(filename=str(out_dir / "per_class_f1_ranking.png")))\n',
                "\n",
                'print("📊 BIỂU ĐỒ ĐỐI CHIẾU PRECISION VS RECALL:")\n',
                'if (out_dir / "per_class_precision_recall.png").exists():\n',
                '    display(Image(filename=str(out_dir / "per_class_precision_recall.png")))',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. HIỂN THỊ BÁO CÁO LÂM SÀNG TOÀN DIỆN (CLINICAL REPORT)\n",
                'report_file = out_dir / "clinical_report.md"\n',
                "if report_file.exists():\n",
                '    with open(report_file, "r", encoding="utf-8") as f:\n',
                "        display(Markdown(f.read()))\n",
                "else:\n",
                '    print("Chưa tìm thấy clinical_report.md")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. TẠO LIÊN KẾT TẢI VỀ 1-CLICK CHO FILE ZIP TOÀN BỘ KẾT QUẢ RUN 7\n",
                "from IPython.display import FileLink, display\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                'zip_file = Path("/kaggle/working/GIEndoDL/outputs/optimal_hybrid_model_run7_results.zip")\n',
                "if not zip_file.exists():\n",
                '    zip_file = Path("/kaggle/working/optimal_hybrid_model_run7_results.zip")\n',
                "\n",
                "print('=' * 85)\n",
                "if zip_file.exists():\n",
                '    print(f"📦 TẬP TIN NÉN KẾT QUẢ RUN 7 ĐÃ SẴN SÀNG: {zip_file.name} ({zip_file.stat().st_size / (1024**2):.2f} MB)")\n',
                '    print("📥 Bấm vào đường dẫn bên dưới để tải trực tiếp toàn bộ 24 file về máy:")\n',
                '    if Path("outputs/optimal_hybrid_model_run7_results.zip").exists():\n',
                '        display(FileLink("outputs/optimal_hybrid_model_run7_results.zip"))\n',
                '    elif Path("optimal_hybrid_model_run7_results.zip").exists():\n',
                '        display(FileLink("optimal_hybrid_model_run7_results.zip"))\n',
                "    else:\n",
                "        display(FileLink(str(zip_file)))\n",
                "else:\n",
                '    print("Không tìm thấy file zip kết quả!")\n',
                "print('=' * 85)",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 4,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent.parent
notebooks_dir = repo_dir / "notebooks"
notebooks_dir.mkdir(parents=True, exist_ok=True)
out_path = notebooks_dir / "kaggle_train_production_run7.ipynb"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(
    f"✅ Đã tạo thành công file Jupyter Notebook: {out_path} ({out_path.stat().st_size:,} bytes)"
)
