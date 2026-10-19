# Bảng Hồ Sơ Đặc Tính Kỹ Thuật CNN Backbone (ResNet-50) Cho Mô Hình Lai (Task #95)

**Ngày thực hiện:** 2026-10-19 15:57:23
**Mô hình Backbone:** ResNet-50 (Pretrained ImageNet)
**Tổng số tham số Backbone:** 23.51 Triệu (M)
**Thiết bị benchmark:** cpu

## 1. Thông số Kỹ thuật & Độ phân giải từng Stage

| Stage | PyTorch Module | Độ phân giải ($H \times W$) | Số kênh ($C$) | Stride | Receptive Field | Tham số (M) | Độ thưa (% zero) | Vai trò Lâm sàng (Nội soi Tiêu hóa) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **STEM** | `conv1 + bn1 + relu + maxpool` | $56 \times 56$ | $64$ | 4x | 35px | 0.01M | 7.3% | Trích xuất cạnh thô, gradient ánh sáng nội soi, vùng lóa sáng (specular reflection) |
| **STAGE1** | `layer1 (3 Bottlenecks)` | $56 \times 56$ | $256$ | 4x | 67px | 0.22M | 42.6% | Mạng lưới vi mạch niêm mạc (micro-vascular network), vi cấu trúc bề mặt (crypt architecture) |
| **STAGE2** | `layer2 (4 Bottlenecks)` | $28 \times 28$ | $512$ | 8x | 163px | 1.22M | 54.6% | Mô thức rãnh niêm mạc (pit patterns), viền xung huyết viêm loét, cấu trúc mạch máu lớn |
| **STAGE3** | `layer3 (6 Bottlenecks)` | $14 \times 14$ | $1024$ | 16x | 355px | 7.10M | 61.3% | Biên giới khối polyp, bờ ổ loét, thành nếp gấp dạ dày/đại tràng, vùng chuyển tiếp niêm mạc |
| **STAGE4** | `layer4 (3 Bottlenecks)` | $7 \times 7$ | $2048$ | 32x | 499px | 14.96M | 98.0% | Ngữ nghĩa bệnh lý cấp cao (23 phân lớp HyperKvasir), phân biệt polyp vs trĩ vs viêm loét |

## 2. Đo lường Hiệu năng: Direct Forward vs Hook-Based Extraction

| Phương thức Trích xuất | Độ trễ Batch (ms) | Throughput (FPS) | Overhead so với Direct | Ghi chú Kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| **Direct Forward Slicing** | 830.37 ms | 4.8 FPS | Baseline (0.0%) | Truy cập trực tiếp qua các submodule `layer1..4` |
| **Hook-based Extraction (`CNNFeatureHooks`)** | 885.22 ms | 4.5 FPS | +6.6% (+54.85 ms) | Phi xâm lấn, tự động cleanup, an toàn bộ nhớ |

## 3. Ý nghĩa Đa Tỷ lệ (Multi-Scale Features) đối với CBAM & Transformer

1. **Giai đoạn Nông (Stage 1 & 2 - $56\times 56$ & $28\times 28$):**
   - Giữ độ phân giải không gian cao, nhạy với vi cấu trúc niêm mạc (crypt architecture) và mạng lưới vi mạch mao mạch (microvascular pattern).
   - Rất quan trọng để CBAM Channel Attention lọc các kênh màu hấp thụ hemoglobin (ánh sáng hẹp NBI) và Spatial Attention định vị vi tổn thương phẳng (flat lesions như Barrett esophagus, angiectasia).

2. **Giai đoạn Sâu (Stage 3 & 4 - $14\times 14$ & $7\times 7$):**
   - Receptive field lớn ($355\text{px} \to 499\text{px}$), bao quát toàn bộ khung hình nội soi.
   - Trích xuất ngữ nghĩa phân loại trừu tượng (polyp cuống vs polyp phẳng, phân độ viêm loét Mayo 0-3).
   - Đặc trưng Stage 4 ($7\times 7 = 49$ vị trí) là đầu vào lý tưởng cho khối CBAM và ánh xạ sang 49 visual tokens cho Compact Transformer Encoder với chi phí tính toán self-attention $O(49^2)$ cực thấp.
