"""Engine Trí tuệ Nhân tạo & Giải thích Mô hình XAI (Grad-CAM++) cho Web App.

Hỗ trợ đa mô hình thực nghiệm:
- Run 7 (Mặc định): Toàn vẹn 23 Phân lớp Bệnh học HyperKvasir (SOTA Production - 29,30M tham số).
- Run 4: 19 Phân lớp Lâm sàng gộp chuẩn y tế.
- Run 6: 17 Phân lớp Dữ liệu Sạch Data-Centric AI (DCAI SOTA).
"""

from pathlib import Path
import sys
import time
import base64
import io
import json
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

# Danh sách 23 lớp chuẩn từ classes_23.json
CLASSES_23_LIST = [
    "barretts",
    "barretts-short-segment",
    "bbps-0-1",
    "bbps-2-3",
    "cecum",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis-a",
    "esophagitis-b-d",
    "hemorrhoids",
    "ileum",
    "impacted-stool",
    "polyps",
    "pylorus",
    "retroflex-rectum",
    "retroflex-stomach",
    "ulcerative-colitis-grade-0-1",
    "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-1-2",
    "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-2-3",
    "ulcerative-colitis-grade-3",
    "z-line"
]

AVAILABLE_CHECKPOINTS = {
    "run7": {
        "name": "Run 7 (Toàn vẹn 23 Lớp - SOTA)",
        "num_classes": 23,
        "layers": 2,
        "pooling": "gap",
        "path": Path(r"D:\KhoaLuanCuNhan\optimal_hybrid_model_run7_results\best_model.pth"),
        "classes": CLASSES_23_LIST,
        "description": "Mô hình SOTA đề xuất trên toàn vẹn 23 lớp bệnh học HyperKvasir (Top-3 Acc 90.46%)"
    },
    "run4": {
        "name": "Run 4 (19 Phân lớp Lâm sàng)",
        "num_classes": 19,
        "layers": 4,
        "pooling": "hybrid",
        "path": Path(r"D:\KhoaLuanCuNhan\optimal_hybrid_model_run4_results\best_model.pth"),
        "classes": list(CLINICAL_METADATA.keys())[:19],
        "description": "Mô hình tiền trạm trên 19 lớp lâm sàng (F1 69.28%)"
    }
}


class InferenceEngine:
    """Inference Engine tải mô hình tối ưu Run 7 (hoặc Run 4) và sinh kết quả chẩn đoán + Grad-CAM++."""

    def __init__(self, model_key: str = "run7", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_key = model_key
        self.load_model(model_key)

        # Pipeline biến đổi ảnh chuẩn y khoa (224x224, HyperKvasir Normalization)
        self.transform = T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=HYPERKVASIR_MEAN, std=HYPERKVASIR_STD),
            ]
        )

    def load_model(self, model_key: str = "run7"):
        """Nạp trọng số mô hình theo khóa (run7 hoặc run4)."""
        cfg = AVAILABLE_CHECKPOINTS.get(model_key, AVAILABLE_CHECKPOINTS["run7"])
        ckpt_path = cfg["path"]

        # Fallback nếu đường dẫn chưa thấy
        if not ckpt_path.exists():
            cand = PROJECT_ROOT / "checkpoints" / "optimal_production" / "best_model.pth"
            if cand.exists():
                ckpt_path = cand

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file checkpoint: {ckpt_path}")

        print(f"[*] Đang nạp mô hình [{model_key.upper()}] từ: {ckpt_path} lên thiết bị: {self.device}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Tự động nhận diện số lớp qua head.fc.weight
        if "head.fc.weight" in state_dict:
            fc_classes = state_dict["head.fc.weight"].shape[0]
            self.num_classes = fc_classes
        else:
            self.num_classes = cfg["num_classes"]

        if self.num_classes == 23:
            self.classes = CLASSES_23_LIST
            self.transformer_layers = 2
            self.pooling = "gap"
            self.model_name = "Hybrid CNN-CBAM-Transformer (Run 7 SOTA - 23 Lớp Bệnh Học)"
            self.active_run = "Run 7"
        else:
            self.classes = cfg["classes"]
            self.transformer_layers = cfg.get("layers", 4)
            self.pooling = cfg.get("pooling", "hybrid")
            self.model_name = "Hybrid CNN-CBAM-Transformer (Run 4 - 19 Phân Lớp Lâm Sàng)"
            self.active_run = "Run 4"

        # Khởi tạo kiến trúc CNN-CBAM-Transformer đúng cấu hình đã huấn luyện
        self.model = build_hybrid_model(
            num_classes=self.num_classes,
            pretrained_backbone=False,
            cbam_placement="stage4",
            transformer_layers=self.transformer_layers,
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            token_pooling=self.pooling,
            dropout=0.1,
        )

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

        self.model_key = model_key
        print(f"[+] Khởi tạo thành công mô hình {self.active_run} với {self.num_classes} phân lớp bệnh học!")

    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Thực hiện suy diễn và tính toán Grad-CAM++ trên ảnh đầu vào."""
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
        pred_label = self.classes[pred_idx] if pred_idx < len(self.classes) else f"class_{pred_idx}"
        confidence = float(top5_probs[0]) * 100.0

        # Tính toán Grad-CAM++
        self.model.zero_grad()
        score = logits[0, pred_idx]
        score.backward(retain_graph=True)

        activations = self.activations.detach().cpu()  # [1, 2048, 7, 7]
        gradients = self.gradients.detach().cpu()  # [1, 2048, 7, 7]

        # Thuật toán Grad-CAM++
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        spatial_sum = activations.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * grad_2 + spatial_sum * grad_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))

        alphas = grad_2 / denom
        relu_grads = F.relu(score.exp() * gradients)
        weights = (alphas * relu_grads).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Nội suy song tuyến lên kích thước gốc của ảnh
        cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Phủ lên ảnh gốc (Alpha Blending 50-50)
        overlay = np.uint8(0.55 * orig_np + 0.45 * heatmap_colored)

        # Mã hóa Base64
        def np_to_b64(arr):
            im = Image.fromarray(arr)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        orig_b64 = np_to_b64(orig_np)
        heatmap_b64 = np_to_b64(overlay)
        pure_cam_b64 = np_to_b64(heatmap_colored)

        # Lấy metadata lâm sàng
        meta = CLINICAL_METADATA.get(
            pred_label,
            {
                "class_id": pred_idx,
                "name_vn": pred_label,
                "name_en": pred_label,
                "tract": "Đường tiêu hóa",
                "category": "Chẩn đoán hình ảnh học",
                "severity": "normal",
                "severity_vn": "Khảo sát lâm sàng",
                "color": "#0284c7",
                "recommendation": "Quan sát kỹ hình thái niêm mạc xung quanh tổn thương.",
                "follow_up": "Theo dõi định kỳ.",
            },
        )

        # Danh sách Top-5 dự đoán
        top5_list = []
        for i in range(len(top5_indices)):
            c_idx = int(top5_indices[i])
            c_label = self.classes[c_idx] if c_idx < len(self.classes) else f"class_{c_idx}"
            c_prob = float(top5_probs[i]) * 100.0
            c_meta = CLINICAL_METADATA.get(c_label, {})
            top5_list.append(
                {
                    "rank": i + 1,
                    "class_id": c_idx,
                    "label": c_label,
                    "name_vn": c_meta.get("name_vn", c_label),
                    "name_en": c_meta.get("name_en", c_label),
                    "probability": round(c_prob, 2),
                    "severity": c_meta.get("severity", "normal"),
                    "color": c_meta.get("color", "#0284c7"),
                }
            )

        return {
            "status": "success",
            "active_run": self.active_run,
            "pred_label": pred_label,
            "confidence": round(confidence, 2),
            "clinical_meta": meta,
            "top5": top5_list,
            "inference_metrics": {
                "latency_ms": round(inference_time_ms, 2),
                "fps": round(fps, 1),
                "device": str(self.device),
                "model_name": self.model_name,
                "total_parameters": "29.30M",
                "num_classes": self.num_classes,
                "backbone": "ResNet-50",
                "attention": f"CBAM Stage 4 + Compact Transformer ({self.transformer_layers} layers, 8 heads)",
            },
            "images": {
                "original_base64": orig_b64,
                "heatmap_base64": heatmap_b64,
                "pure_cam_base64": pure_cam_b64,
                "width": orig_w,
                "height": orig_h,
            },
        }


_ENGINE_INSTANCE = None


def get_engine(model_key: str = "run7") -> InferenceEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = InferenceEngine(model_key=model_key)
    elif _ENGINE_INSTANCE.model_key != model_key:
        _ENGINE_INSTANCE.load_model(model_key)
    return _ENGINE_INSTANCE
