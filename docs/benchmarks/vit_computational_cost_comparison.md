# ⚡ SO SÁNH CHI PHÍ TÍNH TOÁN & HIỆU QUẢ PHẦN CỨNG (COMPUTATIONAL COST - TASK #92)

| Mô hình | Params (M) | FLOPs (G) | Train Time | GPU FPS | CPU FPS | Chi phí phần cứng |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-Base/32 | 87.45 | 4.41 | 0.42 h (25 min) | 292.4 | 4.8 | Thấp nhất nhánh Base, siêu nhanh nhưng token 32x32 quá thô gây suy thoái độ chính xác. |
| Swin-Tiny | 27.54 | 4.49 | 0.78 h (46.9 min) | 177.9 | 2.3 | HIỆU QUẢ NHẤT (Top Balanced SOTA): 27.5M params, 89.97% Accuracy, huấn luyện nhanh gấp 2.5 lần Swin-Base. |
| DeiT-Base | 85.84 | 17.65 | 1.12 h (67 min) | 95.7 | 1.4 | Nặng nề, yêu cầu tiền tính toán forward pass từ Teacher CNN trong suốt quá trình train. |
| ViT-Base/16 | 85.8 | 17.58 | 1.23 h (74 min) | 101.5 | 1.7 | Nặng (17.58 GFLOPs), tốc độ CPU 1.7 FPS không thể triển khai thời gian thực. |
| Swin-Base | 86.77 | 15.44 | 1.92 h (115 min) | 67.3 | 0.9 | SOTA Lâm sàng cao nhất (69.78% Macro F1), nhưng CPU 0.9 FPS và 86.8M params cản trở nhúng Edge. |
| CaiT-S24 | 46.54 | 9.35 | 12.94 h (776 min) | 54.9 | 1.5 | ĐẮT ĐỎ PHI LÝ: Train mất gần 13 tiếng mà kết quả suy thoái nghiêm trọng (82.98% Acc). |
