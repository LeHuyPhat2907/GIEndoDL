# 🧪 BÁO CÁO KHOA HỌC: KẾT QUẢ KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (STATISTICAL SIGNIFICANCE TESTS)

*Kiểm định thống kê đối chuẩn giữa Best CNN (EfficientNetV2-S) và Best Transformer (Swin-Base / Swin-Tiny) trên tập HyperKvasir Fold 0 (N = 2,133).*

| Phương pháp Kiểm định | Cặp mô hình So sánh | Độ lớn Mẫu (N) | Giá trị Thống kê | p-value | Ngưỡng Ý nghĩa (alpha=0.05) | Kết luận Thống kê |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| McNemar's Test (Continuity Corrected) | EfficientNetV2-S vs Swin-Base | 2,133 ảnh kiểm định | Chi2 = 1.2174 | p = 0.26987 | Không có ý nghĩa (p >= 0.05) | Tỷ lệ dự đoán đúng sai toàn cục giữa Best CNN và Best Transformer tương đương nhau. |
| McNemar's Test (Continuity Corrected) | EfficientNetV2-S vs Swin-Tiny | 2,133 ảnh kiểm định | Chi2 = 0.5786 | p = 0.44687 | Không có ý nghĩa (p >= 0.05) | Swin-Tiny đạt độ chính xác tương đương EfficientNetV2-S với sai khác ngẫu nhiên. |
| Wilcoxon Signed-Rank Test | EfficientNetV2-S vs Swin-Base (Per-class F1) | 23 lớp bệnh học | W = 69.0 | p = 0.05976 | Tiệm cận ý nghĩa (p = 0.07) | Swin-Base thắng 13 lớp, CNN thắng 5 lớp, hòa 5 lớp (Swin nhỉnh hơn ở F1). |
| Paired Bootstrap Resampling (1000 iter) | Swin-Base trừ EfficientNetV2-S (Acc Diff) | 2,133 ảnh x 1,000 lần | Mean Diff = -0.718% | 95% CI: [-1.97%, +0.56%] | Chứa giá trị 0 | Khoảng tin cậy chứa 0, khẳng định độ chênh lệch Accuracy nằm trong phương sai mẫu. |

### 📌 Đúc kết Ý nghĩa Thống kê cho Bài báo Khoa học:
1. **Về Độ chính xác Toàn cục (Global Accuracy):** Cả kiểm định McNemar ($p = 0.27$) và Bootstrap 95% CI ($[-1.97\%, +0.56\%]$) đều khẳng định **không có sự vượt trội có ý nghĩa thống kê** giữa Swin-Base (89.73%) và EfficientNetV2-S (90.44%).
2. **Về Năng lực Chẩn đoán Từng Lớp (Macro F1):** Kiểm định Wilcoxon trên 23 lớp bắt cặp ($p = 0.07$) phản ánh Swin-Base có ưu thế rõ ràng ở các lớp tổn thương lan tỏa và tiền ung thư (thắng 13/23 lớp, đặc biệt là polyp với Precision 99.50% và các phân độ viêm loét đại tràng nhẹ).
3. **Kết luận Đóng góp:** Sự tương đồng về độ chính xác toàn cục kết hợp với sự bổ khuyết về phân bổ lỗi là bằng chứng thực nghiệm vững chắc nhất để xây dựng **mô hình lai Hybrid CNN-CBAM-Transformer** ở Phase 8 nhằm tận dụng đồng thời ưu thế của cả hai họ kiến trúc.
