# 📈 Báo cáo Kỹ thuật: Hệ Thống Giám Sát Thực Nghiệm & Tối Ưu Siêu Tham Số W&B

> **File cấu hình Sweeps:** `configs/wandb_sweep_config.json` | **Hình minh họa:** `docs/figures/39_wandb_experiment_tracking_dashboard.png`

---

## 1. Cơ Chế Giám Sát Thực Nghiệm Trực Quan (Experiment Tracking)

Hệ thống tích hợp lớp `WandbLogger` cho phép tự động ghi lại toàn bộ tiến trình huấn luyện:
- **Hàm mất mát:** `train/loss` và `val/loss` theo dõi sự hội tụ của mạng.
- **Chỉ số lâm sàng:** `val/accuracy`, `val/macro_f1`, và độ nhạy trên 23 lớp bệnh lý.
- **Phần cứng & Siêu tham số:** Tốc độ suy giảm Learning Rate (Cosine Annealing) và dung lượng VRAM GPU tiêu thụ.

---

## 2. Chiến Lược Tự Động Quét Siêu Tham Số (Bayesian Sweeps)

File cấu hình `wandb_sweep_config.json` thiết lập không gian tìm kiếm thông minh thông qua thuật toán Bayes, giúp tự động dò tìm bộ siêu tham số tốt nhất (Learning Rate từ $10^{-4}$ đến $10^{-3}$, Batch Size 16 hoặc 32) nhằm tối đa hóa Macro F1-Score trên tập kiểm thử.
