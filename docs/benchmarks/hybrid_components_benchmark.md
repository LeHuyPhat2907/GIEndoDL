# Khảo Sát Thực Nghiệm & Đánh Giá Các Thành Phần Kiến Trúc Lai (Tasks #96-#101)

**Thời gian thực hiện:** 2026-10-19 16:10:27
**Thiết bị thực thi:** cpu

## 1. Thí Nghiệm Định Vị Chọn Lọc CBAM Trong Backbone (Task #98)

| Vị trí Đặt CBAM | Tham số Backbone (M) | Tham số CBAM Bổ sung | Tỷ lệ Tăng (%) | Độ trễ (ms) | Nhận định Kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`stage4`** | 24.03M | +524.39K | +2.23% | 150.23 ms | Khuyến nghị tối ưu (Core Hybrid) |
| **`stage3_4`** | 24.16M | +655.56K | +2.79% | 143.50 ms | Đa tỷ lệ cân bằng |
| **`all_stages`** | 24.20M | +696.71K | +2.96% | 148.34 ms | Chi phí cao |
| **`residual_blocks`** | 26.02M | +2516.51K | +10.70% | 199.26 ms | Chuẩn Woo et al. 2018 |

## 2. Thí Nghiệm Cơ Chế Pooling Của Classification Head (Task #101)

| Chiến lược Pooling | Tổng Tham số Mô hình (M) | Số lượng Tokens | Độ trễ (ms) | Cơ sở Lâm sàng (Nội soi Y tế) |
| :--- | :---: | :---: | :---: | :--- |
| **`GAP`** | 31.43M | 49 tokens | 156.98 ms | Phù hợp nhất cho y tế: Tích lũy bằng chứng từ toàn bộ các vi tổn thương niêm mạc. |
| **`CLS`** | 31.43M | 50 tokens | 160.78 ms | Chuẩn ViT: Tập trung ngữ nghĩa đại diện toàn cục thông qua self-attention. |
| **`HYBRID`** | 31.95M | 50 tokens | 162.82 ms | Hợp nhất đa diện: Kết hợp cả biểu diễn CLS và trung bình không gian GAP. |

## 3. Thí Nghiệm Độ Sâu Tầng Transformer Encoder (Task #100)

| Độ sâu (Layers) | Tổng Tham số (M) | Tham số Transformer (M) | Độ trễ (ms) | Throughput (FPS) | Nhận xét |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **L = 2** | 31.43M | 6.31M | 163.43 ms | 12.2 FPS | Cấu hình tiêu chuẩn (Khuyến nghị) |
| **L = 3** | 34.58M | 9.46M | 272.76 ms | 7.3 FPS | Biểu diễn ngữ cảnh sâu hơn |
| **L = 4** | 37.73M | 12.61M | 274.82 ms | 7.3 FPS | Biểu diễn ngữ cảnh sâu hơn |
