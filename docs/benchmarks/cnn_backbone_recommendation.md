# 🏆 BÁO CÁO PHÂN TÍCH CHỌN CNN BACKBONE TỐI ƯU (TASK #85)

Dựa trên kết quả thực nghiệm 8 mô hình CNN trên bộ dữ liệu HyperKvasir (23 lớp):

1. **Ứng viên Độ chính xác Cao nhất (Top Accuracy & Generalization): EfficientNetV2-S**
   - Đạt Accuracy vượt trội (**90.44%**), Macro F1 (**67.97%**), Weighted F1 (**90.16%**) và AUC-ROC (**98.45%**).
   - Kiến trúc Fused-MBConv kết hợp Progressive Learning giúp tối ưu hóa cả tốc độ huấn luyện lẫn biểu diễn đặc trưng niêm mạc đa tỉ lệ.

2. **Ứng viên Tiết kiệm & Tốc độ Triển khai Thời gian thực (Top Efficiency / Edge-AI): MobileNetV3-Large**
   - Chỉ **4.23M tham số** (~1/5 ResNet-50), thời gian huấn luyện 15 phút, đạt **156.6 FPS trên GPU** và **46.7 FPS trên CPU**.
   - Khả thi 100% khi nhúng vào máy nội soi di động tại tuyến y tế cơ sở mà không cần GPU rời.

3. **Ứng viên Tái sử dụng Đặc trưng Tốt nhất (Top Feature Reuse): DenseNet-121**
   - Chỉ **7.98M tham số** nhưng đạt Macro F1 **68.04%** và Accuracy **89.83%**, chứng minh kết nối dày đặc (Dense Connections) chống suy giảm gradient cực tốt.

4. **Ứng viên Cân bằng & Trích xuất Đặc trưng Đa tầng (Top Modular Backbone): ResNet-50 / SE-ResNet-50**
   - Cấu trúc 4 tầng phân cấp (Stage 1-4: 256, 512, 1024, 2048 kênh) cực kỳ lý tưởng để kết nối với khối Spatial Attention (CBAM) và tạo Tokens cho Transformer.

👉 **KẾT LUẬN ĐỀ XUẤT CHO TASK #94 (KIẾN TRÚC HYBRID):**
- **Lựa chọn 1 (Ưu tiên SOTA Accuracy):** **EfficientNetV2-S** kết hợp Cross-Attention Transformer.
- **Lựa chọn 2 (Ưu tiên Cấu trúc Đa tầng & Modular Extractor):** **ResNet-50 / SE-ResNet-50 + CBAM + Transformer**.
