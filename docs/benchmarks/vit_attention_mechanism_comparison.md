# 🔬 SO SÁNH 4 CƠ CHẾ ATTENTION CỦA CÁC DÒNG VISION TRANSFORMER (TASK #92)

| Cơ chế Chú ý | Mô hình Đại diện | Công thức Toán học | Độ phức tạp | Trường tiếp nhận | Inductive Bias | Đánh giá trên Nội soi tiêu hóa |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Global Multi-Head Self-Attention | ViT-Base/16, ViT-Base/32 | Softmax(QK^T / sqrt(d)) * V | O(N^2 * C) quadratic | Toàn cục ngay từ Layer 1 | Rất yếu (chỉ qua patchify) | Thiếu phân cấp cục bộ, dễ nhầm môn vị/nếp gấp niêm mạc thành polyp (40-83 ca FP). |
| Distillation Token + Self-Attention | DeiT-Base Distilled | Attention trên [CLS, DIST, x_1..x_N] | O((N+2)^2 * C) quadratic | Toàn cục + Tiêm tri thức giáo viên CNN | Trung bình (truyền qua loss chưng cất) | Tăng vọt F1 ở các lớp viêm ruột (+20% F1), nhưng bản đồ phẳng 14x14 vẫn dính 79 ca FP polyp. |
| Shifted Window Attention (W-MSA / SW-MSA) | Swin-Base, Swin-Tiny | Softmax(QK^T / sqrt(d) + B) * V trong cửa sổ 7x7 | O(M^2 * H * W * C) linear | Cục bộ tầng nông -> Mở rộng toàn cục tầng sâu | Mạnh (nhờ phân cấp 4 stages & relative position bias) | SOTA TOÀN DIỆN (Macro F1 69.78%), triệt tiêu 98.7% báo động giả polyp (chỉ 1 ca FP duy nhất). |
| Two-Stage Class-Attention (CaiT) | CaiT-S24 | Patch-Patch Attn (24L) -> Class-Attn (2L) | O(N^2 * C) + O(N * C) | Tách rời biểu diễn patch và gom nhãn CLS | Rất yếu | THẤT BẠI LÂM SÀNG: Quá khớp sau epoch 8, CLS token gom nhầm 93 ca môn vị sang polyp, 159 ca FP. |
