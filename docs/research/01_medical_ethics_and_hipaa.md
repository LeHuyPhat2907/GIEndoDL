# 🛡️ Ghi chú Nghiên cứu: Đạo đức Y sinh, Tiêu chuẩn HIPAA & Quản trị Dữ liệu (Medical AI Ethics)

> **Mã tài liệu:** `DOC-RES-01-ETHICS`
> **Chủ đề:** Đạo đức Y tế, Quyền riêng tư Bệnh nhân (HIPAA/GDPR), Hướng dẫn TRIPOD-AI & Cơ sở lựa chọn Bộ dữ liệu HyperKvasir.
> **Tài liệu tham chiếu:** *Gastroenterology 2003;125:1298–1300* (Tremaine et al., Mayo Clinic) & *TRIPOD-AI Reporting Guidelines*.

---

## 1. Bối cảnh Pháp lý & Đạo đức Y sinh trong Nghiên cứu AI Y tế (HIPAA & IRB)

Theo Đạo luật về Trách nhiệm giải trình và Cung cấp bảo hiểm y tế (**HIPAA**) và các quy định về Bảo vệ Nghiên cứu Con người của Hội đồng Xét duyệt Thể chế (**IRB**):
1. **Sự đồng thuận của bệnh nhân (Informed Consent):** Mọi dữ liệu thu thập từ bệnh nhân đều bắt buộc phải có văn bản đồng thuận tham gia nghiên cứu (kể cả các nghiên cứu có nguy cơ tối thiểu).
2. **Khai thác dữ liệu (Data Mining):** Nghiên cứu viên không được tự ý rà soát hồ sơ bệnh án tự do nếu không có đề cương nghiên cứu gắn liền với câu hỏi cụ thể đã được IRB phê duyệt.
3. **Cơ sở dữ liệu máy tính hóa (Computerized Databases):** Việc duy trì dữ liệu bệnh nhân số lượng lớn phục vụ AI phải có ranh giới rõ ràng giữa dữ liệu lưu trữ lâm sàng và dữ liệu nghiên cứu mở.

---

## 2. Ba Điểm Luận điểm Cốt lõi ứng dụng vào Khóa luận & Bài báo

### 🎯 2.1. Đưa vào Chương 1 (Đặt vấn đề & Tính cấp thiết - Motivation)
* **Luận điểm từ Hiệp hội Tiêu hóa Hoa Kỳ (AGA):** Các bệnh lý đường tiêu hóa (ung thư đại trực tràng, viêm loét đại tràng, tổn thương tiền ung thư) tạo ra gánh nặng y tế và kinh tế khổng lồ cho cộng đồng.
* **Đóng góp của Đề tài:** Ứng dụng Học sâu (CADe/CADx) trong nội soi thời gian thực là giải pháp công nghệ mũi nhọn giúp hỗ trợ bác sĩ giảm tỷ lệ bỏ sót tổn thương (miss-rate), phát hiện sớm ung thư và tối ưu hóa chi phí điều trị.

### 🎯 2.2. Đưa vào Chương 2/Chương 3 (Cơ sở khoa học chọn Bộ dữ liệu HyperKvasir)
* **Thực trạng rào cản:** Việc tự thu thập và gán nhãn dữ liệu thô tại chỗ thường gặp rào cản pháp lý rất lớn về bảo mật y tế và thủ tục phê duyệt đạo đức kéo dài.
* **Cơ sở lựa chọn HyperKvasir:**
  - Bộ dữ liệu do **Phòng thí nghiệm Nghiên cứu Simula** và **Bệnh viện Bærum (Na Uy)** công bố.
  - Toàn bộ hình ảnh đã được giải danh tính (**De-identified**) theo chuẩn quốc tế, loại bỏ mọi siêu dữ liệu nhận dạng cá nhân.
  - Đã được cấp phép và phê duyệt đạo đức bởi Hội đồng Nghiên cứu Y tế Khu vực Na Uy (*Regional Committees for Medical and Health Research Ethics*).
  - Đảm bảo tính khoa học, hợp pháp và khả năng tái lập thực nghiệm (**Reproducibility**) cao nhất cho đề tài.

### 🎯 2.3. Đưa vào Chương 5 (Thiết kế Web App & Tuân thủ TRIPOD-AI)
* **Nguyên tắc Bảo mật Dữ liệu (Data Privacy in Inference):**
  - Hệ thống Web App của dự án tuân thủ nghiêm ngặt nguyên tắc: **Chỉ xử lý pixel ảnh nội soi thuần túy**.
  - Tự động lọc bỏ các siêu dữ liệu nhạy cảm (Tên bệnh nhân, Mã hồ sơ, Ngày sinh, Viện phí, Viện nội soi) trước khi gửi qua API tới Model Inference Engine.
  - Hỗ trợ lưu trữ lịch sử chẩn đoán dạng ẩn danh phục vụ cơ chế Active Learning.

---

## 3. Đoạn văn Trích dẫn Mẫu cho Quyển Khóa luận

> *"Trong nghiên cứu ứng dụng Trí tuệ nhân tạo vào Y tế, việc thu thập và khai thác dữ liệu hình ảnh lâm sàng luôn phải đối mặt với các rào cản nghiêm ngặt về đạo đức y sinh và quyền riêng tư bệnh nhân theo tiêu chuẩn quốc tế như **Đạo luật HIPAA** và hướng dẫn báo cáo y tế **TRIPOD-AI** (Tremaine et al., Gastroenterology). Do đó, đề tài lựa chọn tiếp cận bộ dữ liệu chuẩn hóa **HyperKvasir**—nơi toàn bộ hình ảnh đã được giải danh tính (De-identified) và phê duyệt bởi Hội đồng Đạo đức Y sinh Na Uy (Regional Committees for Medical and Health Research Ethics), đồng thời thiết kế hệ thống Web App tuân thủ tuyệt đối nguyên tắc không lưu trữ thông tin nhận dạng cá nhân của người bệnh."*

---

## 4. Trích dẫn Tài liệu Tham khảo (BibTeX Citation)

```bibtex
@article{tremaine2003hipaa,
  title={Challenges of HIPAA to Clinical Research},
  author={Tremaine, William J},
  journal={Gastroenterology},
  volume={125},
  number={4},
  pages={1298--1300},
  year={2003},
  publisher={Elsevier}
}
