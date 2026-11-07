# Báo Cáo Kỹ Thuật: Biến Dạng Cơ Sinh Học Thấu Kính Vòm & Nhu Động Ruột (Biomechanical Lens & Peristalsis)

> **Mã công việc:** Task D33 & Task D35
> **Kỹ thuật:** OpticalDistortion, GridDistortion, Peristaltic MotionBlur & GaussianBlur.

---

## 1. Mô Phỏng Độ Cong Thấu Kính Vòm Góc Rộng (Fisheye Lens Distortion)
- Đầu ống soi có góc nhìn cực rộng ($140^\circ - 170^\circ$) gây biến dạng cong vòm (Barrel Distortion) về phía rìa khung hình. `OpticalDistortion` (biên độ 0.08) mô phỏng trung thực sự co dãn này.
## 2. Mô Phỏng Nhu Động Ruột & Dịch Bọt Thấu Kính
- `MotionBlur` (kernel 3–7px) tái hiện hiện tượng dây soi bị giật nhẹ do sóng nhu động co bóp cơ trơn thành ruột hoặc thao tác rút ống soi nhanh của bác sĩ.
