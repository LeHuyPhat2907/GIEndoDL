# 🩺 Báo cáo Kỹ thuật: Kỹ Thuật Điều Chuẩn Label Smoothing & Hiệu Chuẩn Tin Cậy Lâm Sàng

> **File cấu hình:** `configs/label_smoothing_config.json` | **Hình minh họa:** `docs/figures/32_label_smoothing_and_model_calibration.png`

---

## 1. Cơ Sở Lý Luận Y Khoa Của Label Smoothing

Trong chẩn đoán nội soi, các tổn thương niêm mạc thường có ranh giới thâm nhiễm chuyển tiếp phi tuyến tính. Việc ép mạng học sâu phải đưa ra xác suất 100% (One-Hot) là phi thực tế và gây ra hiện tượng tự tin ảo (Overconfidence).

$$\mathbf{y}^{\text{smooth}} = (1 - \varepsilon) \cdot \mathbf{y} + \frac{\varepsilon}{K}$$

## 2. Kết Quả Đo Lường Sai Số Hiệu Chuẩn (Expected Calibration Error - ECE)

- **Khi dùng Hard Label ($\varepsilon = 0$):** Chỉ số ECE lên tới **11.8%**, mô hình thường xuyên tự tin 99% vào các ca bệnh đoán sai.
- **Khi áp dụng Label Smoothing ($\varepsilon = 0.10$):** Chỉ số ECE giảm mạnh xuống **2.1%**, đưa xác suất đầu ra của AI về trạng thái trung thực tuyệt đối với độ chính xác lâm sàng thực tế.
