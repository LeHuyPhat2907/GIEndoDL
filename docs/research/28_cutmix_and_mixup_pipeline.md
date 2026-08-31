# 🧬 Báo cáo Kỹ thuật: Kỹ Thuật Tăng Cường Điều Chuẩn CutMix & MixUp (Regularization)

> **Module chính:** `src/preprocessing/mixup_cutmix.py` | **Hình minh họa:** `docs/figures/21_cutmix_mixup_augmentations.png`

---

## 1. Cơ chế Toán học & Ý nghĩa Lâm sàng

1. **MixUp ($\lambda \in [0, 1]$):** Trộn tuyến tính không gian ảnh và vector nhãn Soft-Label. Kỹ thuật này ép mạng nơ-ron cư xử tuyến tính giữa các vùng chuyển tiếp, ngăn ngừa mô hình đưa ra dự đoán quá tự tin (Overconfidence).
2. **CutMix ($\mathbf{M} \in \{0, 1\}$):** Thay thế một vùng chữ nhật của ảnh $A$ bằng mô từ ảnh $B$. Buộc các tầng tích chập phải kích hoạt trên toàn bộ vùng niêm mạc thay vì chỉ phụ thuộc vào một đốm tổn thương đơn lẻ.

---

## 2. Kết quả Đạt được cho Khóa luận

- **Cải thiện độ chuẩn xác xác suất (Model Calibration):** Giảm trực tiếp sai số Expected Calibration Error (ECE), giúp xác suất xuất ra cho bác sĩ đáng tin cậy hơn.
- **Tăng cường năng lực chống nhiễu (Robustness):** Mô hình không bị 'sốc' khi gặp các ca bệnh đồng mắc (vừa có polyp vừa có viêm loét đại tràng).
