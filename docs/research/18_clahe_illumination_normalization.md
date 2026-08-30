# 💡 Báo cáo Kỹ thuật: Chuẩn hóa Độ sáng Thích ứng CLAHE trên Không gian Màu LAB

> **Module chính:** `src/preprocessing/clahe_enhancer.py` | **Hình minh họa:** `docs/figures/11_clahe_illumination_comparison.png`

---

## 1. Cơ sở Khoa học & So sánh Giải pháp

| Phương pháp tiền xử lý | Xử lý vùng lóa sáng | Bảo toàn màu sắc mô | Tăng cường vi mạch (Pit-pattern) | Đánh giá y tế |
|:---|:---:|:---:|:---:|:---:|
| **Ảnh gốc chưa xử lý** | ❌ Kém (chói lóa) | 🟢 Gốc | ❌ Bị chìm trong vùng tối | 🔴 Không tối ưu |
| **Cân bằng RGB toàn cục** | 🟡 Khá | 🔴 **Thất bại** (biến đổi màu) | 🟡 Nhiễu hạt | ❌ Nguy hiểm lâm sàng |
| **LAB-CLAHE (Đề xuất)** | 🟢 **Xuất sắc** | 🟢 **Bảo toàn 100%** | 🟢 **Rõ nét từng mao mạch** | 🟢 **Chuẩn Y khoa** |

---

## 2. Kết luận Kỹ thuật cho Khóa luận

Thuật toán CLAHE trên kênh L (Luminance) với tham số `clipLimit=2.0` và `tileGridSize=(8,8)` giúp làm nổi bật các hoa văn niêm mạc ẩn sâu trong bóng tối mà không làm méo mó đặc trưng màu sắc sinh học, tạo đầu vào lý tưởng cho khối Attention CBAM ở Giai đoạn 8.
