"""Supervised Contrastive Loss (SupCon) (Task #109).

Theo công trình:
"Supervised Contrastive Learning" - Prannay Khosla et al., NeurIPS 2020.
https://arxiv.org/abs/2004.11362

ĐÓNG GÓP KỸ THUẬT (TASK #109):
- Kéo các mẫu cùng lớp bệnh lý (polyp, viêm loét, bình thường) lại gần nhau trong không gian đặc trưng;
  đẩy các mẫu khác lớp bệnh lý ra xa.
- Hỗ trợ đa góc nhìn (multi-view batching, ví dụ 2 views từ data augmentations khác nhau).
- Tự động chuẩn hóa L2 embeddings (L2 normalization).
- Kiểm soát thang nhiệt độ (temperature scaling tau, mặc định tau=0.07).
- Ổn định số học tuyệt đối (numerical stability via log-sum-exp trick với max subtraction).
- Tương thích cả chế độ có giám sát (Supervised Contrastive) lẫn tự giám sát (SimCLR / Self-Supervised).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Hỗ trợ cả Supervised Contrastive Loss (khi có nhãn lớp labels)
    lẫn Self-Supervised SimCLR Loss (khi không truyền nhãn labels).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ):
        """Khởi tạo SupConLoss.

        Args:
            temperature: Thang nhiệt độ tau (mặc định 0.07).
            contrast_mode: 'all' hoặc 'one' (mặc định 'all' cho Khosla 2020).
            base_temperature: Nhiệt độ cơ sở để chuẩn hóa loss scale (0.07).
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature phải lớn hơn 0, nhận giá trị: {temperature}")
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính toán Supervised Contrastive Loss.

        Args:
            features: Tensor vector đặc trưng [batch_size, n_views, dim]
                      hoặc [batch_size * n_views, dim].
            labels: Nhãn ground truth [batch_size]. Nếu None, trở thành SimCLR loss.
            mask: Ma trận mặt nạ tương phản tùy chọn [batch_size, batch_size].

        Returns:
            Scalar tensor giá trị tổn thất SupCon.
        """
        device = features.device

        # 1. Định hình lại features về dạng chuẩn: [batch_size, n_views, dim]
        if len(features.shape) < 3:
            # Dạng [B * views, dim] -> giả định 2 views nếu không chỉ định
            raise ValueError(
                f"`features` cần có ít nhất 3 chiều [batch_size, n_views, dim], nhận: {features.shape}"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        # 2. Xây dựng ma trận mặt nạ nhãn (label mask)
        if labels is not None and mask is not None:
            raise ValueError("Không thể truyền đồng thời cả `labels` và `mask`.")
        elif labels is None and mask is None:
            # Tự giám sát (SimCLR): Chỉ các góc nhìn khác nhau của cùng 1 ảnh là positive
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    f"Số lượng nhãn ({labels.shape[0]}) không khớp với batch size ({batch_size})"
                )
            # mask[i, j] = 1 nếu labels[i] == labels[j]
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 3. Chuẩn hóa L2 embeddings dọc theo chiều dim
        features = F.normalize(features, p=2, dim=2)

        # 4. Gom các views thành chuỗi tương phản
        contrast_count = n_views
        contrast_feature = torch.cat(
            torch.unbind(features, dim=1), dim=0
        )  # [B * views, dim]

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Chế độ contrast_mode không hợp lệ: {self.contrast_mode}")

        # 5. Tính ma trận tích vô hướng tương đồng Cosine chia cho nhiệt độ tau
        # anchor_dot_contrast: [B * anchor_count, B * contrast_count]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )

        # 6. Đảm bảo ổn định số học (Numerical Stability Trick): Trừ giá trị cực đại từng dòng
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 7. Nhân bản ma trận mask tương ứng với số views
        # mask: [batch_size, batch_size] -> tile thành [batch_size * anchor_count, batch_size * contrast_count]
        mask = mask.repeat(anchor_count, contrast_count)

        # 8. Mặt nạ loại bỏ sự tự tương phản với chính mình (self-contrast mask)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # 9. Tính toán xác suất Log-Sum-Exp ổn định
        # Mẫu số: Tổng exp của tất cả các mẫu trong batch ngoại trừ chính nó
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 10. Tính trung bình log-probability trên tập các cặp dương tính (positives)
        # mean_log_prob_pos = sum(mask * log_prob) / sum(mask)
        mask_pos_pairs = mask.sum(1)
        # Trường hợp mẫu không có cặp dương tính nào khác trong batch (chỉ có trong supervised với lớp đơn lẻ)
        mask_pos_pairs = torch.where(
            mask_pos_pairs == 0, torch.ones_like(mask_pos_pairs), mask_pos_pairs
        )
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # 11. Tổn thất SupCon tổng thể
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
