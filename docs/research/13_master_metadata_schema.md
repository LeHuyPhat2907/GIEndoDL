# Đặc Tả Kỹ Thuật Master Metadata Schema & Cấu Trúc Lưu Trữ Dữ Liệu Hoàn Chỉnh (Task D24)

> **Mã tài liệu:** `DOC-RES-13-METADATA-SPEC`
> **Tệp lưu trữ chính thức:** `data/processed/hyperkvasir_final_master_metadata.csv` (10,662 dòng)
> **Tệp quy chiếu ánh xạ:** `data/processed/schema_mapping_v2.json`

---

## 1. Cấu Trúc Các Trường Dữ Liệu Trong Master Metadata CSV

| Tên Cột | Kiểu Dữ Liệu | Mục Đích Sử Dụng | Ví Dụ Giá Trị |
|:---|:---:|:---|:---|
| `image_id` | `string` | Khóa chính duy nhất định dạng số thứ tự chuẩn | `IMG_00001` |
| `file_name` | `string` | Tên tệp tin gốc của ảnh | `000e3fee-7f5c-4819-9f9c-4c983b68888a.jpg` |
| `relative_path` | `string` | Đường dẫn tương đối từ thư mục `labeled-images/` | `lower-gi-tract/anatomical-landmarks/cecum/...` |
| `original_class_name` | `string` | Tên lớp gốc ban đầu của HyperKvasir (23 lớp) | `cecum` |
| `cleaned_class_name` | `string` | Tên lớp sau khi hiệu đính Confident Learning | `cecum` |
| `final_class` | `string` | Tên phân lớp chính thức trong Schema 17 lớp mục tiêu | `cecum-and-ileum` |
| `organ_group` | `string` | Phân nhóm cơ quan giải phẫu lớn | `Lower GI Tract - Colorectum` |
| `patient_video_cluster_id`| `string` | Mã định danh cụm ca bệnh/video khử rò rỉ | `PATIENT_CLUSTER_0042` |
| `cluster_size` | `integer`| Số lượng khung hình cùng ca bệnh/thủ thuật | `3` |
| `cluster_type` | `string` | Loại cụm (`MULTI_FRAME` hoặc `SINGLETON`) | `MULTI_FRAME` |
| `bounding_box_flag` | `boolean`| Chỉ báo ảnh có tọa độ Bounding Box hay không | `True` / `False` |
| `bounding_box_coords` | `json_str`| Tọa độ hộp bao xmin, ymin, xmax, ymax (Kvasir-SEG) | `{"bbox": [{"xmin": 38, ...}]}` |
| `is_active_sample` | `boolean`| Cờ cho phép tham gia huấn luyện/thẩm định 17 lớp | `True` / `False` |
| `quarantine_reason` | `string` | Lý do cách ly nếu không tham gia huấn luyện chính | `ACTIVE`, `EXTREME_MICRO_CLASS_HOLDOUT` |
| `split_eligibility` | `string` | Phân loại phân chia (`TRAIN_VAL_TEST` / `HOLDOUT`) | `TRAIN_VAL_TEST` |
| `composite_256bit` | `string` | Chữ ký số đa băm tổ hợp 256-bit | `f0e1d2...` |
