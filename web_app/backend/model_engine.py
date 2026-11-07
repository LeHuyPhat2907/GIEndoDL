"""Engine Trí tuệ Nhân tạo & Giải thích Mô hình XAI (Grad-CAM++) cho Web App.

Nạp mô hình tối ưu Run 4: Hybrid CNN-CBAM-Transformer (31.4M tham số),
thực hiện tiền xử lý chuẩn y khoa, đo lường độ trễ suy diễn (Latency ms / FPS),
và kết xuất bản đồ chú ý Grad-CAM++ chỉ rõ vùng tổn thương bệnh học.
"""

from pathlib import Path
import sys
import time
import base64
import io
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

from src.models.hybrid_cnn_cbam_transformer import build_hybrid_model
from web_app.backend.clinical_metadata import CLINICAL_METADATA

# Bộ tham số chuẩn hóa đo đạc thực nghiệm riêng cho HyperKvasir
HYPERKVASIR_MEAN = [0.5729, 0.3557, 0.2515]
HYPERKVASIR_STD = [0.3105, 0.2116, 0.1834]


class InferenceEngine:
    """Inference Engine tải mô hình tối ưu Run 4 và sinh kết quả chẩn đoán + Grad-CAM++."""

    def __init__(self, checkpoint_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            # Ưu tiên tìm trong outputs/optimal_hybrid_model_run4 hoặc D:\KhoaLuanCuNhan
            candidates = [
                PROJECT_ROOT
                / "outputs"
                / "optimal_hybrid_model_run4"
                / "best_model.pth",
                Path(
                    r"D:\KhoaLuanCuNhan\optimal_hybrid_model_run4_results\best_model.pth"
                ),
                PROJECT_ROOT / "checkpoints" / "optimal_production" / "best_model.pth",
            ]
            for cand in candidates:
                if cand.exists():
                    checkpoint_path = str(cand)
                    break

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số best_model.pth: {checkpoint_path}"
            )

        print(f"[*] Đang nạp mô hình từ: {checkpoint_path} lên thiết bị: {self.device}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.num_classes = ckpt.get("num_classes", 19)
        self.classes: List[str] = ckpt.get("classes", list(CLINICAL_METADATA.keys()))

        cfg = ckpt.get("config", {})
        arch_cfg = cfg.get("architecture", {})

        # Khởi tạo kiến trúc CNN-CBAM-Transformer đúng cấu hình đã huấn luyện
        self.model = build_hybrid_model(
            num_classes=self.num_classes,
            pretrained_backbone=False,
            transformer_layers=arch_cfg.get("transformer_layers", 4),
            d_model=arch_cfg.get("transformer_dim", 512),
            nhead=arch_cfg.get("transformer_heads", 8),
            dim_feedforward=arch_cfg.get("transformer_ffn_dim", 2048),
            token_pooling=arch_cfg.get("pooling_mode", "hybrid"),
            dropout=cfg.get("regularization", {}).get("dropout", 0.1),
        )

        state_dict = ckpt["model_state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Đăng ký hook cho Grad-CAM++ trên tầng ResNet-50 Layer 4
        self.target_layer = self.model.backbone.layer4[-1]
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

        # Pipeline biến đổi ảnh chuẩn y khoa (224x224, HyperKvasir Normalization)
        self.transform = T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=HYPERKVASIR_MEAN, std=HYPERKVASIR_STD),
            ]
        )

        print("[+] Khởi tạo Inference Engine thành công! Sẵn sàng phục vụ 19 phân lớp.")

    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Thực hiện suy diễn và tính toán Grad-CAM++ trên ảnh đầu vào."""
        # Chuyển ảnh RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        orig_w, orig_h = pil_image.size
        orig_np = np.array(pil_image)

        # Tiền xử lý Tensor
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        # Bấm giờ suy diễn
        t0 = time.perf_counter()
        self.model.zero_grad()
        logits = self.model(input_tensor)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(inference_time_ms, 0.001)

        probs = F.softmax(logits, dim=1).squeeze(0)
        top5_probs, top5_indices = torch.topk(probs, k=min(5, self.num_classes))

        top5_probs = top5_probs.detach().cpu().numpy()
        top5_indices = top5_indices.detach().cpu().numpy()

        pred_idx = int(top5_indices[0])
        pred_label = self.classes[pred_idx]
        confidence = float(top5_probs[0]) * 100.0

        # Tính toán Grad-CAM++
        self.model.zero_grad()
        score = logits[0, pred_idx]
        score.backward(retain_graph=True)

        activations = self.activations.detach().cpu()  # [1, 2048, 7, 7]
        gradients = self.gradients.detach().cpu()  # [1, 2048, 7, 7]

        # Thuật toán Grad-CAM++
        # Trọng số bậc 2 và bậc 3 của đạo hàm
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        eps = 1e-7

        sum_act = activations.sum(dim=(2, 3), keepdim=True)
        alpha = grad_2 / (2.0 * grad_2 + sum_act * grad_3 + eps)
        alpha = F.relu(alpha)

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        cam_resized = cv2.resize(cam, (orig_w, orig_h))

        # Tạo Heatmap màu JET
        heatmap_u8 = np.uint8(255 * cam_resized)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Phủ nhiệt với ảnh gốc (tỷ lệ 50% - 50%)
        blended = np.uint8(orig_np * 0.55 + heatmap_color * 0.45)

        # Chuyển đổi Base64 để gửi về giao diện Web
        orig_b64 = self._np_to_base64(orig_np)
        heatmap_b64 = self._np_to_base64(blended)
        pure_cam_b64 = self._np_to_base64(heatmap_color)

        # Lấy thông tin lâm sàng
        meta = CLINICAL_METADATA.get(
            pred_label,
            {
                "name_vn": pred_label,
                "name_en": pred_label,
                "tract": "Hệ tiêu hóa",
                "category": "Phân loại bệnh lý",
                "severity": "warning",
                "severity_vn": "Cần theo dõi",
                "color": "#3b82f6",
                "recommendation": "Đề xuất bác sĩ thăm khám và đối chiếu lâm sàng.",
                "follow_up": "Theo dõi định kỳ.",
            },
        )

        # Danh sách Top-5 dự đoán
        top5_list = []
        for i in range(len(top5_indices)):
            c_idx = int(top5_indices[i])
            c_label = self.classes[c_idx]
            c_prob = float(top5_probs[i]) * 100.0
            c_meta = CLINICAL_METADATA.get(c_label, {})
            top5_list.append(
                {
                    "class_id": c_idx,
                    "label": c_label,
                    "name_vn": c_meta.get("name_vn", c_label),
                    "name_en": c_meta.get("name_en", c_label),
                    "probability": round(c_prob, 2),
                    "severity": c_meta.get("severity", "normal"),
                    "color": c_meta.get("color", "#3b82f6"),
                }
            )

        return {
            "status": "success",
            "pred_label": pred_label,
            "confidence": round(confidence, 2),
            "clinical_meta": meta,
            "top5": top5_list,
            "inference_metrics": {
                "latency_ms": round(inference_time_ms, 2),
                "fps": round(fps, 1),
                "device": str(self.device),
                "model_name": "Hybrid CNN-CBAM-Transformer (Run 4)",
                "total_parameters": "31.43M",
                "backbone": "ResNet-50",
                "attention": "CBAM Stage 4 + Compact Transformer (4 layers, 8 heads)",
            },
            "images": {
                "original_base64": orig_b64,
                "heatmap_base64": heatmap_b64,
                "pure_cam_base64": pure_cam_b64,
                "width": orig_w,
                "height": orig_h,
            },
        }

    def _np_to_base64(self, arr: np.ndarray) -> str:
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(
            "utf-8"
        )


# Singleton instance toàn cục
_ENGINE_INSTANCE = None


def get_engine() -> InferenceEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = InferenceEngine()
    return _ENGINE_INSTANCE
