# 📐 Báo cáo Kỹ thuật: Chiến lược Chuẩn hóa Kích thước & Tỷ lệ Khung hình Ảnh Nội soi

> **Module chính:** `src/preprocessing/image_resizer.py` | **Hình minh họa:** `docs/figures/14_image_resizing_comparison.png`

---

## 1. So sánh Ưu / Nhược điểm giữa 2 Chiến lược

| Chiến lược Resize | Độ méo mó hình học | Tốc độ xử lý (FPS) | Bảo toàn Pit-pattern | Mục tiêu ứng dụng |
|:---|:---:|:---:|:---:|:---|
| **Direct Resize (224x224)** | 🟡 Co giãn ~18% | 🟢 Cực nhanh ($>120$) | 🟡 Tương đối | Huấn luyện Baseline CNN & Web App thời gian thực |
| **Letterbox Padding (224x224)** | 🟢 **0% (Nguyên bản)** | 🟢 Nhanh ($>90$) | 🟢 Rất tốt | Huấn luyện mô hình chuẩn hóa hình thái học |
| **High-Res Letterbox (384x384)** | 🟢 **0% (Nguyên bản)** | 🟡 Trung bình ($>45$) | 🟢 **Hoàn hảo** | Huấn luyện **Mô hình đề xuất (CNN-CBAM-Transformer + SupCon)** |

---

## 2. Quyết định Kỹ thuật cho Đề tài

1. **Bảo tồn Cấu trúc Sinh học:** Việc sử dụng nội suy **Bicubic** kết hợp **Letterbox Padding** giúp giữ nguyên độ tròn của polyp và cấu trúc nếp gấp niêm mạc.
2. **Đa dạng Kích thước Thực nghiệm:** Đề tài duy trì cả 2 phiên bản kích thước (224x224 cho thực nghiệm so sánh tốc độ và 384x384 cho thực nghiệm tối ưu độ chính xác Macro F1-score).
