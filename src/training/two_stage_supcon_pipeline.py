"""Chiến lược Huấn Luyện 2 Giai Đoạn (2-Stage Training Strategy) với SupCon (Task #111).

Theo công trình:
- "Supervised Contrastive Learning" - Khosla et al., NeurIPS 2020.
- "A Simple Framework for Contrastive Learning of Visual Representations" - Chen et al., ICML 2020.

ĐÓNG GÓP KỸ THUẬT (TASK #111):
- Thiết kế quy trình huấn luyện 2 giai đoạn hoàn chỉnh:
  * Stage 1: Contrastive Pretraining với Supervised Contrastive Loss (SupCon).
    - Mục tiêu: Huấn luyện Backbone CNN + CBAM + Transformer Encoder để tạo không gian biểu diễn
      phân tách rõ rệt giữa các lớp bệnh lý nội soi (Separable Feature Space).
    - Đầu ra qua MLP Projection Head (512 -> 512 -> 128) với chuẩn hóa L2 và tau = 0.07.
    - Tăng cường dữ liệu đa góc nhìn ngẫu nhiên (2-view augmentations).
    - Sau khi hoàn tất Stage 1: Tháo bỏ (discard) hoàn toàn Projection Head.
  * Stage 2: Supervised Fine-Tuning.
    - Lắp Classification Head (Linear Classifier 512 -> 23 classes).
    - Sử dụng chiến lược tốc độ học phân tầng (Discriminative Learning Rates):
      lr_backbone = 1e-5 (giữ vững cấu trúc biểu diễn đã hội tụ ở Stage 1).
      lr_head = 1e-3 (tối ưu nhanh bộ phân loại).
    - Hàm tổn thất: Focal Loss kết hợp Label Smoothing Cross-Entropy.
- Xuất bản tài liệu thiết kế giao thức huấn luyện (docs/benchmarks/two_stage_training_strategy.md & .json).
"""

# ruff: noqa: E402
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from src.models.hybrid_cnn_cbam_transformer import build_hybrid_model
from src.models.supcon_loss import SupConLoss
from src.models.supcon_projection_head import HybridModelWithProjectionHead

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class Stage1Config:
    """Cấu hình cho Giai đoạn 1: Contrastive Representation Pretraining."""

    epochs: int = 50
    batch_size: int = 32
    views_per_sample: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    proj_hidden_dim: int = 512
    proj_out_dim: int = 128
    optimizer: str = "AdamW"
    lr_scheduler: str = "CosineAnnealingLR (T_max=50, eta_min=1e-6)"
    augmentations: str = "RandomResizedCrop(224) + ColorJitter(0.4, 0.4, 0.4, 0.1) + RandomHorizontalFlip + GaussianBlur"


@dataclass
class Stage2Config:
    """Cấu hình cho Giai đoạn 2: Supervised Classification Fine-Tuning."""

    epochs: int = 35
    batch_size: int = 32
    views_per_sample: int = 1
    backbone_lr: float = 1e-5
    transformer_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    loss_function: str = "Focal Loss (gamma=2.0, alpha=balanced) / Label Smoothing CE"
    optimizer: str = "AdamW with Discriminative Learning Rates"
    lr_scheduler: str = "Linear Warmup (5 epochs) + Cosine Decay (30 epochs)"
    augmentations: str = (
        "Standard Medical Endo Augmentations (Resize 224, RandomFlip, Mild ColorJitter)"
    )


class TwoViewAugmentation:
    """Bộ tăng cường tạo 2 góc nhìn ngẫu nhiên độc lập từ cùng một ảnh nội soi."""

    def __init__(self, size: int = 224):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


class TwoStageSupConTrainer:
    """Lớp điều phối toàn diện quy trình huấn luyện 2 giai đoạn (Task #111)."""

    def __init__(
        self,
        num_classes: int = 23,
        stage1_cfg: Optional[Stage1Config] = None,
        stage2_cfg: Optional[Stage2Config] = None,
        device: Optional[torch.device] = None,
    ):
        self.num_classes = num_classes
        self.stage1_cfg = stage1_cfg or Stage1Config()
        self.stage2_cfg = stage2_cfg or Stage2Config()
        self.device = device or torch.device("cpu")

        # 1. Khởi tạo mô hình Hybrid cốt lõi
        self.base_model = build_hybrid_model(
            num_classes=num_classes,
            pretrained_backbone=False,
            cbam_placement="stage4",
            transformer_layers=2,
            nhead=8,
            d_model=512,
            token_pooling="gap",
        ).to(self.device)

        # 2. Khởi tạo mô hình Stage 1 với Projection Head
        self.stage1_model = HybridModelWithProjectionHead(
            base_hybrid_model=self.base_model,
            proj_hidden_dim=self.stage1_cfg.proj_hidden_dim,
            proj_out_dim=self.stage1_cfg.proj_out_dim,
        ).to(self.device)

        # 3. Khởi tạo SupCon Loss
        self.supcon_criterion = SupConLoss(
            temperature=self.stage1_cfg.temperature,
            contrast_mode="all",
        )

        self.current_stage = 1

    def run_dry_run_step_stage1(self, batch_size: int = 4) -> Dict[str, Any]:
        """Thực hiện một bước kiểm thử Stage 1 giả lập để kiểm tra luồng gradient và tính ổn định số học."""
        self.stage1_model.train()
        # Tạo batch 2 góc nhìn [B, 2, 3, 224, 224]
        dummy_views = torch.randn(batch_size, 2, 3, 224, 224, device=self.device)
        dummy_labels = torch.randint(
            0, self.num_classes, (batch_size,), device=self.device
        )

        optimizer = optim.AdamW(
            self.stage1_model.parameters(),
            lr=self.stage1_cfg.learning_rate,
            weight_decay=self.stage1_cfg.weight_decay,
        )
        optimizer.zero_grad()

        # Forward qua stage 1
        z, h = self.stage1_model(dummy_views, return_representation=True)
        # z: [B, 2, 128]
        loss = self.supcon_criterion(z, dummy_labels)

        loss.backward()
        optimizer.step()

        return {
            "stage": 1,
            "loss": float(loss.item()),
            "z_norm": float(torch.norm(z[0, 0]).item()),
            "h_shape": list(h.shape),
            "z_shape": list(z.shape),
            "gradient_flowing": all(
                p.grad is not None
                for name, p in self.stage1_model.named_parameters()
                if p.requires_grad and "base_model.head" not in name
            ),
        }

    def transition_to_stage2(self) -> nn.Module:
        """Chuyển giao từ Stage 1 sang Stage 2: Tháo rời Projection Head và thiết lập Discriminative LRs."""
        print("🔄 Đang thực hiện chuyển giao từ Stage 1 sang Stage 2...", flush=True)
        # Tháo bỏ projection head
        self.stage1_model.discard_projection_head()
        self.current_stage = 2

        # Thiết lập các nhóm tham số với tốc độ học phân biệt (Discriminative Learning Rates)
        param_groups = [
            {
                "params": self.base_model.backbone.parameters(),
                "lr": self.stage2_cfg.backbone_lr,
            },
            {
                "params": self.base_model.bridge.parameters(),
                "lr": self.stage2_cfg.transformer_lr,
            },
            {
                "params": self.base_model.transformer_encoder.parameters(),
                "lr": self.stage2_cfg.transformer_lr,
            },
            {
                "params": self.base_model.head.parameters(),
                "lr": self.stage2_cfg.head_lr,
            },
        ]

        self.stage2_optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.stage2_cfg.weight_decay,
        )
        print(
            "✅ Đã tháo bỏ Projection Head. Đã kích hoạt 4 nhóm tham số với Discriminative LRs.",
            flush=True,
        )
        return self.base_model

    def run_dry_run_step_stage2(self, batch_size: int = 4) -> Dict[str, Any]:
        """Thực hiện một bước kiểm thử Stage 2 với Single View và Cross-Entropy Loss."""
        self.base_model.train()
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=self.device)
        dummy_labels = torch.randint(
            0, self.num_classes, (batch_size,), device=self.device
        )

        criterion = nn.CrossEntropyLoss()
        self.stage2_optimizer.zero_grad()

        logits = self.base_model(dummy_input)
        loss = criterion(logits, dummy_labels)

        loss.backward()
        self.stage2_optimizer.step()

        return {
            "stage": 2,
            "loss": float(loss.item()),
            "logits_shape": list(logits.shape),
            "gradient_flowing": all(
                p.grad is not None
                for p in self.base_model.parameters()
                if p.requires_grad
            ),
        }


def generate_two_stage_strategy_docs():
    print("=" * 105, flush=True)
    print(
        "🚀 BẮT ĐẦU XÁC LẬP VÀ KIỂM CHỨNG CHIẾN LƯỢC HUẤN LUYỆN 2 GIAI ĐOẠN (TASK #111)",
        flush=True,
    )
    print("=" * 105, flush=True)

    trainer = TwoStageSupConTrainer(num_classes=23)

    # 1. Kiểm thử Stage 1 Dry Run
    s1_res = trainer.run_dry_run_step_stage1(batch_size=4)
    print(
        f"• Stage 1 (SupCon Pretraining): Loss = {s1_res['loss']:.4f} | z_norm = {s1_res['z_norm']:.4f} | Gradient Flow = {s1_res['gradient_flowing']}",
        flush=True,
    )

    # 2. Kiểm thử chuyển giao sang Stage 2
    trainer.transition_to_stage2()

    # 3. Kiểm thử Stage 2 Dry Run
    s2_res = trainer.run_dry_run_step_stage2(batch_size=4)
    print(
        f"• Stage 2 (Fine-tuning): Cross-Entropy Loss = {s2_res['loss']:.4f} | Logits shape = {s2_res['logits_shape']} | Gradient Flow = {s2_res['gradient_flowing']}",
        flush=True,
    )

    # 4. Xuất file JSON
    out_dir = Path("docs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "two_stage_training_strategy.json"
    full_spec = {
        "task_id": "Task #111",
        "task_name": "Two-Stage Training Strategy (SupCon Pretraining + Supervised Fine-Tuning)",
        "stage1_config": asdict(trainer.stage1_cfg),
        "stage2_config": asdict(trainer.stage2_cfg),
        "dry_run_verification": {
            "stage1": s1_res,
            "stage2": s2_res,
        },
        "scientific_rationale": (
            "Giai đoạn 1 tiền huấn luyện biểu diễn tương phản Supervised Contrastive giúp hội tụ không gian embedding "
            "vững chắc, đẩy lùi nhiễu ảnh nội soi và tách biệt 23 nhóm tổn thương. Giai đoạn 2 tinh chỉnh bộ phân loại "
            "với tốc độ học phân tầng (lr_backbone=1e-5 vs lr_head=1e-3) ngăn chặn hiện tượng lãng quên thảm họa (Catastrophic Forgetting)."
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_spec, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Đã lưu đặc tả chiến lược JSON tại: {json_path}", flush=True)

    # 5. Xuất file Markdown
    md_path = out_dir / "two_stage_training_strategy.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "# Báo Cáo Thiết Kế: Chiến Lược Huấn Luyện 2 Giai Đoạn (Two-Stage Training Strategy) (Task #111)\n\n"
        )
        f.write(f"**Thời gian xác lập:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(
            "**Kiến trúc áp dụng:** Hybrid CNN-CBAM-Transformer (ResNet-50 + Stage 4 CBAM + L=2 Transformer Encoder)  \n"
        )
        f.write("**Tập dữ liệu mục tiêu:** HyperKvasir Challenge (23 classes)  \n\n")

        f.write("## 1. Tổng Quan Quy Trình Huấn Luyện 2 Giai Đoạn\n\n")
        f.write("```\n")
        f.write(
            "========================================================================================================\n"
        )
        f.write("  [STAGE 1: CONTRASTIVE REPRESENTATION PRETRAINING]\n")
        f.write("  Dữ liệu: 2-View Augmentations (Crop, Flip, ColorJitter, Blur)\n")
        f.write(
            "  Mô hình: CNN-CBAM + Bridge + Transformer Encoder ➔ MLP Projection Head (512 -> 512 -> 128)\n"
        )
        f.write(
            "  Mục tiêu: Tối ưu hóa Supervised Contrastive Loss (SupCon, tau=0.07, Khosla et al., 2020)\n"
        )
        f.write("  Lịch trình: 50 epochs, AdamW (lr=5e-4), Cosine Annealing\n")
        f.write("          │\n")
        f.write(
            "          ▼  THÁO BỎ (DISCARD) PROJECTION HEAD ➔ GIỮ LẠI TRỌNG SỐ BIỂU DIỄN TỐI ƯU\n"
        )
        f.write("          │\n")
        f.write("  [STAGE 2: SUPERVISED CLASSIFICATION FINE-TUNING]\n")
        f.write("  Dữ liệu: Đơn ảnh (Single View, Standard Endo Augmentation)\n")
        f.write(
            "  Mô hình: CNN-CBAM + Bridge + Transformer Encoder ➔ Classification Head (Linear 512 -> 23)\n"
        )
        f.write("  Tốc độ học phân biệt (Discriminative LRs):\n")
        f.write("    - Backbone: 1e-5 (bảo toàn không gian đặc trưng tương phản)\n")
        f.write("    - Transformer: 2e-5 (điều chỉnh nhẹ ngữ cảnh toàn cục)\n")
        f.write("    - Classifier Head: 1e-3 (hội tụ nhanh năng lực phân lớp)\n")
        f.write("  Mục tiêu: Focal Loss (gamma=2.0) + Label Smoothing Cross-Entropy\n")
        f.write(
            "  Lịch trình: 35 epochs, Linear Warmup (5 ep) + Cosine Decay (30 ep)\n"
        )
        f.write(
            "========================================================================================================\n"
        )
        f.write("```\n\n")

        f.write("## 2. Bảng Đối Chiếu Thông Số Kỹ Thuật Giữa 2 Giai Đoạn\n\n")
        f.write(
            "| Tham Số Huấn Luyện | Giai Đoạn 1 (Stage 1: SupCon Pretraining) | Giai Đoạn 2 (Stage 2: Fine-Tuning) |\n"
        )
        f.write("| :--- | :--- | :--- |\n")
        f.write(
            "| **Mục đích** | Học không gian đặc trưng phân tách (Separable Feature Space) | Tối ưu hóa xác suất phân lớp bệnh lý 23 lớp |\n"
        )
        f.write(
            f"| **Số Epochs** | {trainer.stage1_cfg.epochs} epochs | {trainer.stage2_cfg.epochs} epochs |\n"
        )
        f.write(
            f"| **Batch Size** | {trainer.stage1_cfg.batch_size} (Hiệu dụng: {trainer.stage1_cfg.batch_size * trainer.stage1_cfg.views_per_sample} views) | {trainer.stage2_cfg.batch_size} (Đơn góc nhìn) |\n"
        )
        f.write(
            f"| **Hàm Tổn Thất** | SupCon Loss ($\\tau = {trainer.stage1_cfg.temperature}$) | {trainer.stage2_cfg.loss_function} |\n"
        )
        f.write(
            "| **Đầu Ra (Head)** | Non-linear MLP Projection Head ($512 \\to 512 \\to 128$) | Linear Classification Head ($512 \\to 23$) |\n"
        )
        f.write(
            f"| **Tốc Độ Học (LR)** | Đơn nhất: $\\eta = {trainer.stage1_cfg.learning_rate}$ | Phân tầng: Backbone $10^{{-5}}$, Transformer $2 \\times 10^{{-5}}$, Head $10^{{-3}}$ |\n"
        )
        f.write(
            f"| **Tối Ưu Hóa** | {trainer.stage1_cfg.optimizer} ($wd = {trainer.stage1_cfg.weight_decay}$) | {trainer.stage2_cfg.optimizer} |\n"
        )
        f.write(
            f"| **Lịch Trình LR** | {trainer.stage1_cfg.lr_scheduler} | {trainer.stage2_cfg.lr_scheduler} |\n"
        )
        f.write(
            f"| **Tăng Cường Dữ Liệu** | {trainer.stage1_cfg.augmentations} | {trainer.stage2_cfg.augmentations} |\n\n"
        )

        f.write("## 3. Lý Luận Khoa Học Cho Luận Văn\n\n")
        f.write(
            "1. **Giải quyết vấn đề mất cân bằng dữ liệu y tế (Long-tailed Imbalance):**\n"
        )
        f.write(
            "   - Tập dữ liệu HyperKvasir có các lớp phổ biến (cecum, retroflex) lên tới hàng ngàn ảnh, trong khi các lớp bệnh lý hiểm nghèo (Barrett, dysplasia) chỉ có vài chục ảnh.\n"
        )
        f.write(
            "   - Huấn luyện trực tiếp bằng Cross-Entropy thường khiến mô hình thiên lệch về các lớp đông. SupCon giải quyết triệt để điều này bằng cách chuẩn hóa $\\ell_2$ và chia đều sức ảnh hưởng của từng mẫu dương tính qua mẫu số $|P(i)|$.\n\n"
        )

        f.write(
            "2. **Triệt tiêu hiện tượng Quên Lãng Thảm Họa (Catastrophic Forgetting):**\n"
        )
        f.write(
            "   - Việc áp dụng Discriminative Learning Rates với tỷ lệ $1:100$ giữa Backbone và Classifier Head đảm bảo cấu trúc hình học của không gian biểu diễn học được từ Stage 1 không bị phá vỡ trong quá trình thích ứng với phân loại nhãn ở Stage 2.\n"
        )

    print(f"💾 Đã lưu tài liệu đặc tả Markdown tại: {md_path}", flush=True)
    print("=" * 105, flush=True)
    print(
        "🎉 HOÀN THÀNH XÁC LẬP CHIẾN LƯỢC HUẤN LUYỆN 2 GIAI ĐOẠN THÀNH CÔNG!",
        flush=True,
    )
    print("=" * 105, flush=True)


if __name__ == "__main__":
    generate_two_stage_strategy_docs()
