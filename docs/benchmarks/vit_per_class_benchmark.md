# 📊 BẢNG ĐỐI CHIẾU HIỆU NĂNG TỪNG LỚP (PER-CLASS PERFORMANCE BENCHMARK - TASK #92)

*Bảng so sánh chi tiết điểm F1-Score (%) trên toàn bộ 23 lớp bệnh học nội soi giữa 6 mô hình Vision Transformer trên tập HyperKvasir Fold 0.*

| Class_ID | Class_Name | Organ_Group | Support (Số mẫu) | Clinical_System | ViT-16_F1 (%) | ViT-32_F1 (%) | DeiT-Base_F1 (%) | Swin-Base_F1 (%) | Swin-Tiny_F1 (%) | CaiT-S24_F1 (%) | Best_Model | Best_F1 (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 0 | barretts | Thực quản (Esophagus) | 8 | 1. Tiêu hóa Trên (Upper GI) | 40.0 | 40.0 | 61.54 | 28.57 | 15.38 | 16.67 | DeiT-Base | 61.54 |
| 1 | barretts-short-segment | Thực quản (Esophagus) | 11 | 1. Tiêu hóa Trên (Upper GI) | 33.33 | 16.67 | 29.63 | 27.27 | 33.33 | 36.36 | CaiT-S24 | 36.36 |
| 2 | bbps-0-1 | Chất lượng làm sạch ruột kém (BBPS 0-1) | 129 | 2. Đại trực tràng (Lower GI) | 95.75 | 95.69 | 96.9 | 97.28 | 96.88 | 96.47 | Swin-Base | 97.28 |
| 3 | bbps-2-3 | Chất lượng làm sạch ruột tốt (BBPS 2-3) | 229 | 2. Đại trực tràng (Lower GI) | 95.28 | 95.69 | 95.95 | 96.18 | 96.23 | 95.81 | Swin-Tiny | 96.23 |
| 4 | cecum | Manh tràng (Cecum Landmark) | 202 | 2. Đại trực tràng (Lower GI) | 96.48 | 94.95 | 95.41 | 98.78 | 98.01 | 94.3 | Swin-Base | 98.78 |
| 5 | dyed-lifted-polyps | Can thiệp điều trị (Therapeutic / EMR) | 201 | 3. Polyp & Diện cắt can thiệp | 93.09 | 89.92 | 93.73 | 94.68 | 95.24 | 92.8 | Swin-Tiny | 95.24 |
| 6 | dyed-resection-margins | Bờ diện cắt sau can thiệp (Resection Margin) | 198 | 3. Polyp & Diện cắt can thiệp | 93.37 | 91.63 | 95.17 | 94.79 | 95.48 | 93.09 | Swin-Tiny | 95.48 |
| 7 | esophagitis-a | Thực quản (Esophagus) | 81 | 1. Tiêu hóa Trên (Upper GI) | 52.12 | 52.51 | 43.61 | 64.92 | 61.82 | 45.8 | Swin-Base | 64.92 |
| 8 | esophagitis-b-d | Thực quản (Esophagus) | 52 | 1. Tiêu hóa Trên (Upper GI) | 69.72 | 68.42 | 74.55 | 75.0 | 70.8 | 71.19 | Swin-Base | 75.0 |
| 9 | hemorrhoids | Trĩ hậu môn trực tràng (Hemorrhoids) | 1 | 2. Đại trực tràng (Lower GI) | 0.0 | 0.0 | 0.0 | 100.0 | 66.67 | 0.0 | Swin-Base | 100.0 |
| 10 | ileum | Hồi tràng (Terminal Ileum) | 2 | 2. Đại trực tràng (Lower GI) | 0.0 | 66.67 | 0.0 | 66.67 | 0.0 | 0.0 | ViT-32 | 66.67 |
| 11 | impacted-stool | Phân tồn đọng (Impacted Stool) | 26 | 2. Đại trực tràng (Lower GI) | 77.42 | 78.79 | 76.92 | 78.79 | 80.0 | 80.0 | Swin-Tiny | 80.0 |
| 12 | polyps | Polyp tiền ung thư (Colorectal Polyps) | 206 | 3. Polyp & Diện cắt can thiệp | 89.44 | 81.31 | 82.47 | 98.28 | 94.37 | 70.8 | Swin-Base | 98.28 |
| 13 | pylorus | Dạ dày (Stomach - Pylorus) | 199 | 1. Tiêu hóa Trên (Upper GI) | 98.98 | 97.96 | 91.55 | 99.75 | 100.0 | 69.51 | Swin-Tiny | 100.0 |
| 14 | retroflex-rectum | Trực tràng (Retroflex Rectum) | 78 | 2. Đại trực tràng (Lower GI) | 96.25 | 92.99 | 96.3 | 96.89 | 97.44 | 94.34 | Swin-Tiny | 97.44 |
| 15 | retroflex-stomach | Dạ dày (Stomach - Retroflex) | 153 | 1. Tiêu hóa Trên (Upper GI) | 99.67 | 99.67 | 99.35 | 99.35 | 99.67 | 98.68 | ViT-16 | 99.67 |
| 16 | ulcerative-colitis-grade-0-1 | Viêm loét đại tràng (IBD - Grade 0-1) | 7 | 4. Viêm loét đại tràng (IBD) | 36.36 | 0.0 | 20.0 | 20.0 | 30.77 | 35.29 | ViT-16 | 36.36 |
| 17 | ulcerative-colitis-grade-1 | Viêm loét đại tràng (IBD - Grade 1) | 40 | 4. Viêm loét đại tràng (IBD) | 22.5 | 28.21 | 42.86 | 47.89 | 41.98 | 35.48 | Swin-Base | 47.89 |
| 18 | ulcerative-colitis-grade-1-2 | Viêm loét đại tràng (IBD - Grade 1-2) | 2 | 4. Viêm loét đại tràng (IBD) | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 22.22 | DeiT-Base | 40.0 |
| 19 | ulcerative-colitis-grade-2 | Viêm loét đại tràng (IBD - Grade 2) | 89 | 4. Viêm loét đại tràng (IBD) | 59.74 | 55.17 | 70.06 | 62.25 | 64.63 | 64.97 | DeiT-Base | 70.06 |
| 20 | ulcerative-colitis-grade-2-3 | Viêm loét đại tràng (IBD - Grade 2-3) | 6 | 4. Viêm loét đại tràng (IBD) | 0.0 | 0.0 | 0.0 | 5.41 | 14.29 | 12.5 | Swin-Tiny | 14.29 |
| 21 | ulcerative-colitis-grade-3 | Viêm loét đại tràng (IBD - Grade 3) | 26 | 4. Viêm loét đại tràng (IBD) | 55.56 | 52.17 | 58.46 | 67.65 | 57.69 | 61.22 | Swin-Base | 67.65 |
| 22 | z-line | Thực quản - Dạ dày (GE Junction) | 187 | 1. Tiêu hóa Trên (Upper GI) | 84.3 | 82.12 | 85.79 | 84.62 | 88.65 | 86.47 | Swin-Tiny | 88.65 |

### 🩺 Phân tích F1-Score Trung bình theo 4 Nhóm Bệnh học & Giải phẫu Lâm sàng:

| Hệ Cơ quan Lâm sàng | Số Lớp | ViT-16 F1 TB (%) | ViT-32 F1 TB (%) | DeiT-Base F1 TB (%) | Swin-Base F1 TB (%) | Swin-Tiny F1 TB (%) | CaiT-S24 F1 TB (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Tiêu hóa Trên (Upper GI) | 7 | 68.3 | 65.34 | 69.43 | 68.5 | 67.09 | 60.67 |
| 2. Đại trực tràng (Lower GI) | 7 | 65.88 | 74.97 | 65.93 | 90.66 | 76.46 | 65.85 |
| 3. Polyp & Diện cắt can thiệp | 3 | 91.97 | 87.62 | 90.46 | 95.92 | 95.03 | 85.56 |
| 4. Viêm loét đại tràng (IBD) | 6 | 29.03 | 22.59 | 38.56 | 33.87 | 34.89 | 38.61 |

### 📌 Đột phá lâm sàng nổi bật:
1. **Swin-Base (#89):** Áp đảo tại các tổn thương tiền ung thư then chốt: `polyps` (F1 98.28%, chỉ 1 ca FP duy nhất), `esophagitis-a` (64.92%), `esophagitis-b-d` (75.00%), `hemorrhoids` (100.00%) và `cecum` (98.78%).
2. **DeiT-Base (#88):** Nhờ chưng cất tri thức từ CNN teacher, mô hình vươn lên dẫn đầu ở nhóm viêm loét đại tràng mạn tính và niêm mạc Barrett: `barretts` (61.54%), `ulcerative-colitis-grade-1-2` (40.00%), `ulcerative-colitis-grade-2` (70.06%).
3. **Swin-Tiny (#90):** Đạt sự hoàn hảo tuyệt đối ở mốc giải phẫu môn vị `pylorus` (F1 100.00%, Precision 100%, Recall 100%) và dẫn đầu ở `z-line` (88.65%), `bbps-2-3` (96.23%), `dyed-resection-margins` (95.48%).
4. **CaiT-S24 (#91):** Gặp sự cố nghiêm trọng khi 93 ca môn vị bị gom nhầm thành polyp, khiến Precision của polyp tụt xuống chỉ còn 55.71%.
