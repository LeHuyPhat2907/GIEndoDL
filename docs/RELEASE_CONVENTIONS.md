#  Quy ước Tagging & GitHub Release cho Dự án GIEndoDL

## 1. Cấu trúc Đặt tên Tag (Semantic Versioning)
Mỗi mốc quan trọng (Milestone) của đề tài sẽ được đánh tag theo định dạng: `v<Major>.<Minor>.<Patch>-<mô_tả>`

## 2. Bảng Danh sách các Mốc Release Chính

| Tag | Giai đoạn | Mô tả Mốc Phát hành (Milestone) | File đính kèm khi Release |
|:---|:---:|:---|:---|
| `v0.0.1-init` | GĐ 0 | Hoàn thành khởi tạo Repo, GitHub CI/CD & Quy trình làm việc | Mã nguồn khởi tạo |
| `v0.1.0-data` | GĐ 2–3 | Hoàn thành pipeline tiền xử lý HyperKvasir & ROI Segmentation | Metadata CSV, Script tiền xử lý |
| `v0.2.0-baseline-cnn` | GĐ 6 | Hoàn thành huấn luyện Baseline CNN (ResNet, EfficientNet) | Weights `resnet50_baseline.pth` |
| `v0.3.0-transformer` | GĐ 7 | Hoàn thành huấn luyện Vision Transformer & Swin-T | Weights `swin_transformer.pth` |
| `v0.5.0-proposed-model` | GĐ 8–10 | Hoàn thành mô hình đề xuất **CNN-CBAM-Transformer + SupCon** | Weights `giendodl_best_model.pth` |
| `v0.8.0-webapp-xai` | GĐ 12–14 | Hoàn thành Web App FastAPI tích hợp Grad-CAM++ & CBMIR | Model ONNX, Docker Image |
| `v1.0.0-final-thesis` | GĐ 16–17 | **Hoàn thành Khóa luận, Bài báo khoa học & Bảo vệ Hội đồng** | Toàn bộ Source Code + Báo cáo PDF |

## 3. Lệnh Git để tạo và đẩy Tag
```bash
# 1. Tạo Tag có ghi chú (Annotated Tag)
git tag -a v0.0.1-init -m "Milestone: Complete Phase 0 - Repository & Workflow Setup"

# 2. Đẩy Tag lên GitHub
git push origin v0.0.1-init
