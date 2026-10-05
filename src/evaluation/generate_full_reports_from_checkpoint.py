"""Script trích xuất đầy đủ 100% tất cả các kết quả lâm sàng từ Checkpoint đã huấn luyện.

Công dụng:
- Sử dụng cho bất kỳ mô hình nào đã train xong (ResNet-50, ResNet-101, DenseNet-121, EfficientNet-B4).
- Tự động nạp `best_checkpoint.pth`, đánh giá lại tập validation, xuất toàn bộ 9 bộ kết quả:
  1. fold_metrics.json (Accuracy, F1, Kappa, MCC, Top-k)
  2. confusion_matrix_raw.csv & confusion_matrix_normalized.csv
  3. confusion_matrix_raw.png & confusion_matrix_normalized.png (Heatmap 300 DPI)
  4. per_class_metrics.csv & .json (Đầy đủ 23 lớp)
  5. per_class_f1_ranking.png & per_class_precision_recall.png
  6. training_curves.png (Dashboard 4 đồ thị)
  7. val_predictions.csv (Dự đoán chi tiết từng ảnh kèm xác suất 23 lớp)
  8. error_cases.csv (Danh sách ca chẩn đoán nhầm xếp theo độ tự tin)
  9. clinical_report.md & classification_report.txt (Báo cáo y sinh học)
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset.dataloader_factory import get_dataloaders
from src.evaluation.clinical_result_exporter import export_all_clinical_results
from src.models.densenet121 import build_densenet121_baseline
from src.models.efficientnet_b4 import build_efficientnet_b4_baseline
from src.models.efficientnet_b5 import build_efficientnet_b5_baseline
from src.models.resnet50 import build_resnet50_baseline
from src.models.resnet101 import build_resnet101_baseline


def main():
    parser = argparse.ArgumentParser(description="Xuất toàn bộ kết quả lâm sàng từ Checkpoint có sẵn")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Đường dẫn thư mục chứa checkpoint (ví dụ: models/checkpoints/resnet50_5folds/fold_0)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["resnet50", "resnet101", "densenet121", "efficientnet_b4", "efficientnet_b5"],
        help="Loại kiến trúc mô hình",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Số thứ tự Fold (Mặc định: 0)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Kích thước ảnh vuông đầu vào (Mặc định: 224, B4 là 380, B5 là 456)",
    )
    parser.add_argument(
        "--raw_images_dir",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "labeled-images"),
        help="Đường dẫn tới thư mục labeled-images",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "5folds"),
        help="Đường dẫn tới thư mục 5folds",
    )
    args = parser.parse_args()

    chk_dir = Path(args.checkpoint_dir)
    ckpt_file = chk_dir / "best_checkpoint.pth"
    if not ckpt_file.exists():
        ckpt_file = chk_dir / "checkpoint_best.pth"
    if not ckpt_file.exists():
        ckpt_file = chk_dir / "last_checkpoint.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Thiết bị: {device}")
    print(f"🔍 Thư mục Checkpoint: {chk_dir}")

    # Khởi tạo mô hình
    if args.model_type == "resnet50":
        model = build_resnet50_baseline(num_classes=23, pretrained=False)
    elif args.model_type == "resnet101":
        model = build_resnet101_baseline(num_classes=23, pretrained=False)
    elif args.model_type == "densenet121":
        model = build_densenet121_baseline(num_classes=23, pretrained=False)
    elif args.model_type == "efficientnet_b4":
        model = build_efficientnet_b4_baseline(num_classes=23, pretrained=False)
    elif args.model_type == "efficientnet_b5":
        model = build_efficientnet_b5_baseline(num_classes=23, pretrained=False)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {args.model_type}")

    best_epoch = -1
    if ckpt_file.exists():
        print(f"📥 Đang nạp trọng số từ: {ckpt_file}")
        state = torch.load(ckpt_file, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            best_epoch = state.get("epoch", -1)
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
            best_epoch = state.get("epoch", -1)
        else:
            model.load_state_dict(state)
    else:
        print(f"⚠️ Không tìm thấy file trọng số .pth tại {chk_dir}. Đánh giá trên cấu trúc hiện tại.")

    model = model.to(device)
    model.eval()

    # Nạp DataLoader Validation
    fold_dir = Path(args.processed_dir) / f"fold_{args.fold}"
    loaders = get_dataloaders(
        processed_dir=str(fold_dir),
        raw_images_dir=args.raw_images_dir,
        batch_size=32,
        num_workers=4,
        img_size=(args.img_size, args.img_size),
        preload_ram=True,
    )
    val_loader = loaders["val"]

    print("🚀 Đang tiến hành suy luận và thu thập dự đoán trên tập kiểm định...")
    all_preds, all_targets, all_probs, all_filenames = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            imgs = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)
            filenames = batch[2]
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outs = model(imgs)
                probs = torch.softmax(outs, dim=-1)
            all_preds.extend(outs.argmax(dim=-1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_filenames.extend(filenames)

    class_names = [val_loader.dataset.idx_to_class[i] for i in range(len(val_loader.dataset.idx_to_class))]

    history_df = None
    if (chk_dir / "history.csv").exists():
        history_df = pd.read_csv(chk_dir / "history.csv")
    elif (chk_dir / "training_history.csv").exists():
        history_df = pd.read_csv(chk_dir / "training_history.csv")

    res50_base_dir = ROOT_DIR / "models" / "checkpoints" / "resnet50_5folds"

    export_all_clinical_results(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        filenames=all_filenames,
        class_names=class_names,
        history_df=history_df,
        output_dir=chk_dir,
        model_name=args.model_type.upper(),
        fold_idx=args.fold,
        best_epoch=best_epoch,
        res50_comparison_dir=res50_base_dir if res50_base_dir.exists() and args.model_type != "resnet50" else None,
    )


if __name__ == "__main__":
    main()
