"""Metadata Lâm sàng & Khuyến nghị Y khoa cho 19 Lớp Bệnh lý Nội soi Tiêu hóa.

Ánh xạ chi tiết 19 phân lớp tối ưu từ Run 4 sang:
- Tên tiếng Việt y khoa chuẩn mực
- Phân vùng giải phẫu (Upper GI vs Lower GI)
- Nhóm lâm sàng & Mức độ cảnh báo (Bình thường / Cảnh báo / Khẩn cấp)
- Khuyến nghị can thiệp lâm sàng theo hướng dẫn của ESGE (Hiệp hội Nội soi Tiêu hóa Châu Âu)
"""

from typing import Dict, Any

CLINICAL_METADATA: Dict[str, Dict[str, Any]] = {
    "barretts-esophagus": {
        "class_id": 0,
        "name_vn": "Barrett thực quản (Tổn thương tiền ung thư)",
        "name_en": "Barrett's Esophagus",
        "tract": "Đường tiêu hóa trên (Upper GI - Thực quản)",
        "category": "Tổn thương bệnh lý tiền ác tính",
        "severity": "critical",
        "severity_vn": "Khẩn cấp / Tiền ác tính",
        "color": "#ef4444",
        "recommendation": (
            "Phát hiện niêm mạc chuyển sản tuyến dạng Barrett tại đoạn xa thực quản. "
            "Khuyến nghị thực hiện nội soi phóng đại kết hợp dải tần hẹp (NBI) "
            "và sinh thiết 4 góc theo quy chuẩn Seattle Protocol để sàng lọc loạn sản (Dysplasia). "
            "Nếu có loạn sản mức độ cao, chỉ định cắt niêm mạc nội soi (EMR) hoặc triệt đốt vô tuyến (RFA)."
        ),
        "follow_up": "Nội soi kiểm tra định kỳ 1–3 năm tùy mức độ loạn sản.",
    },
    "bbps-0-1": {
        "class_id": 1,
        "name_vn": "Chuẩn bị ruột kém (Thang điểm BBPS 0–1)",
        "name_en": "Poor Bowel Preparation (BBPS 0-1)",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại tràng)",
        "category": "Chất lượng làm sạch niêm mạc",
        "severity": "warning",
        "severity_vn": "Hạn chế quan sát lâm sàng",
        "color": "#f59e0b",
        "recommendation": (
            "Phân tồn đọng nhiều, dịch đục che phủ phần lớn niêm mạc đại tràng không thể rửa trôi hết. "
            "Tỷ lệ bỏ sót polyp và tổn thương ung thư sớm rất cao (>30%). "
            "Khuyến nghị ngừng thủ thuật hoặc hướng dẫn bệnh nhân làm sạch ruột lại bằng thuốc nhuận tràng "
            "và đặt lịch nội soi lại trong vòng 1 năm."
        ),
        "follow_up": "Nội soi lại sớm sau khi làm sạch ruột chuẩn.",
    },
    "bbps-2-3": {
        "class_id": 2,
        "name_vn": "Chuẩn bị ruột tốt / xuất sắc (Thang điểm BBPS 2–3)",
        "name_en": "Good Bowel Preparation (BBPS 2-3)",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại tràng)",
        "category": "Chất lượng làm sạch niêm mạc",
        "severity": "normal",
        "severity_vn": "Chuẩn bị đạt yêu cầu chất lượng",
        "color": "#10b981",
        "recommendation": (
            "Niêm mạc đại tràng sạch hoàn toàn hoặc chỉ còn một ít dịch trong dễ dàng hút rửa. "
            "Trường nhìn nội soi tối ưu, đủ điều kiện khảo sát chi tiết mọi nếp gấp và góc khuất niêm mạc. "
            "Tiếp tục quy trình quan sát rút ống nội soi tối thiểu 6 phút."
        ),
        "follow_up": "Theo dõi định kỳ theo chỉ định sàng lọc chung.",
    },
    "cecum": {
        "class_id": 3,
        "name_vn": "Manh tràng (Mốc giải phẫu đáy đại tràng)",
        "name_en": "Cecum Landmark",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đáy manh tràng)",
        "category": "Mốc giải phẫu bình thường",
        "severity": "normal",
        "severity_vn": "Mốc giải phẫu bình thường",
        "color": "#10b981",
        "recommendation": (
            "Nhận diện rõ lỗ van hồi - manh tràng (Ileocecal valve) và lỗ ruột thừa (Appendiceal orifice). "
            "Xác nhận ống soi đã đến đáy manh tràng, hoàn thành 100% chiều dài đại tràng. "
            "Bắt đầu quy trình quan sát cẩn trọng khi rút ống soi ra ngoài."
        ),
        "follow_up": "Chụp lưu ảnh mốc giải phẫu làm bằng chứng hoàn thành thủ thuật.",
    },
    "dyed-lifted-polyps": {
        "class_id": 4,
        "name_vn": "Polyp đã tiêm nâng & nhuộm màu can thiệp",
        "name_en": "Dyed & Lifted Polyps",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại tràng)",
        "category": "Can thiệp điều trị nội soi",
        "severity": "warning",
        "severity_vn": "Đang thực hiện can thiệp",
        "color": "#3b82f6",
        "recommendation": (
            "Polyp đã được tiêm dung dịch xanh Indigo Carmine / Hyaluronic acid vào lớp dưới niêm mạc "
            "và nâng lên an toàn (Positive lifting sign - Dấu hiệu nâng dương tính). "
            "Đủ điều kiện thực hiện cắt polyp bằng thòng lọng điện (Snare Polypectomy) hoặc cắt tách dưới niêm mạc (ESD)."
        ),
        "follow_up": "Tiến hành cắt bỏ trọn vẹn và thu hồi bệnh phẩm gửi sinh thiết.",
    },
    "dyed-resection-margins": {
        "class_id": 5,
        "name_vn": "Ranh giới diện cắt sau thủ thuật nội soi",
        "name_en": "Dyed Resection Margins",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại tràng)",
        "category": "Can thiệp điều trị nội soi",
        "severity": "warning",
        "severity_vn": "Kiểm tra sau can thiệp",
        "color": "#3b82f6",
        "recommendation": (
            "Khảo sát diện cắt sau khi đã loại bỏ khối polyp. "
            "Cần kiểm tra kỹ mép cắt xem còn mô u sót (Residual tissue) hay không. "
            "Kiểm tra đáy diện cắt để phát hiện dấu hiệu thủng cơ hoặc chảy máu muộn, "
            "chủ động kẹp clip dự phòng nếu cần thiết."
        ),
        "follow_up": "Nội soi kiểm tra lại sau 3–6 tháng để loại trừ tái phát tại diện cắt.",
    },
    "esophagitis-a": {
        "class_id": 6,
        "name_vn": "Viêm thực quản trào ngược Los Angeles Độ A",
        "name_en": "Reflux Esophagitis (LA Grade A)",
        "tract": "Đường tiêu hóa trên (Upper GI - Thực quản)",
        "category": "Tổn thương viêm trào ngược",
        "severity": "warning",
        "severity_vn": "Tổn thương viêm nhẹ",
        "color": "#f59e0b",
        "recommendation": (
            "Xuất hiện một hoặc nhiều vết trợt rách niêm mạc thực quản kích thước <= 5 mm, "
            "chưa lan rộng giữa các nếp niêm mạc. "
            "Khuyến nghị kê đơn thuốc ức chế bơm proton (PPI) liều chuẩn trong 4–8 tuần "
            "kết hợp thay đổi lối sống, kiêng thức ăn kích thích và tránh nằm ngay sau ăn."
        ),
        "follow_up": "Tái khám và đánh giá lại triệu chứng lâm sàng sau 8 tuần.",
    },
    "esophagitis-b-d": {
        "class_id": 7,
        "name_vn": "Viêm thực quản trào ngược Los Angeles Độ B–D (Nặng)",
        "name_en": "Reflux Esophagitis (LA Grade B-D)",
        "tract": "Đường tiêu hóa trên (Upper GI - Thực quản)",
        "category": "Tổn thương viêm trợt loét tiến triển",
        "severity": "critical",
        "severity_vn": "Tổn thương viêm loét nặng",
        "color": "#ef4444",
        "recommendation": (
            "Vết trợt niêm mạc dài > 5 mm lan rộng nối liền giữa các nếp gấp hoặc chiếm > 75% chu vi lòng thực quản. "
            "Nguy cơ cao tiến triển hẹp thực quản hoặc loét chảy máu. "
            "Chỉ định điều trị PPI liều kép tích cực tối thiểu 8–12 tuần và sinh thiết loại trừ biến chứng ung thư biểu mô."
        ),
        "follow_up": "Bắt buộc nội soi kiểm tra lại sau 8–12 tuần sau đợt điều trị PPI.",
    },
    "hemorrhoids": {
        "class_id": 8,
        "name_vn": "Búi trĩ trực tràng / hậu môn",
        "name_en": "Hemorrhoids",
        "tract": "Đường tiêu hóa dưới (Lower GI - Trực tràng / Ống hậu môn)",
        "category": "Bệnh lý mạch máu hậu môn - trực tràng",
        "severity": "warning",
        "severity_vn": "Bệnh lý tĩnh mạch trực tràng",
        "color": "#f59e0b",
        "recommendation": (
            "Giãn búi tĩnh mạch trĩ nội hoặc trĩ ngoại tại vùng chuyển tiếp hậu môn - trực tràng. "
            "Đánh giá độ trĩ (Độ I–IV). Khuyến nghị điều trị nội khoa tăng cường chất xơ, "
            "thuốc trợ tĩnh mạch hoặc chỉ định thắt dây thun / phẫu thuật nếu trĩ độ III–IV có xuất huyết."
        ),
        "follow_up": "Khám chuyên khoa hậu môn trực tràng khi có triệu chứng sa trĩ hay chảy máu.",
    },
    "ileum": {
        "class_id": 9,
        "name_vn": "Hồi tràng (Mốc giải phẫu ruột non tận cùng)",
        "name_en": "Terminal Ileum",
        "tract": "Đường tiêu hóa dưới (Lower GI - Hồi tràng)",
        "category": "Mốc giải phẫu bình thường",
        "severity": "normal",
        "severity_vn": "Mốc giải phẫu bình thường",
        "color": "#10b981",
        "recommendation": (
            "Quan sát rõ hình ảnh nhung mao hồi tràng (Villi) bình thường và các nang lympho dưới niêm mạc. "
            "Không phát hiện dấu hiệu loét áp-tơ hay sỏi phân. "
            "Hoàn tất quy trình kiểm tra hồi tràng tận (thường chỉ định trong sàng lọc bệnh Crohn)."
        ),
        "follow_up": "Ghi nhận hình ảnh bình thường vào biên bản nội soi.",
    },
    "impacted-stool": {
        "class_id": 10,
        "name_vn": "Sỏi phân / Phân đóng khối tồn đọng",
        "name_en": "Impacted Stool",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại tràng)",
        "category": "Chất lượng làm sạch niêm mạc",
        "severity": "warning",
        "severity_vn": "Cản trở trường nhìn nội soi",
        "color": "#f59e0b",
        "recommendation": (
            "Khối phân khô cứng bám chặt vào thành đại tràng gây tắc nghẽn hoặc che khuất tổn thương bên dưới. "
            "Cần cẩn thận khi đẩy ống soi để tránh nguy cơ thủng đại tràng do chấn thương áp lực. "
            "Xem xét thụt tháo hoặc ngưng soi để chuẩn bị lại."
        ),
        "follow_up": "Xử lý táo bón và lên kế hoạch soi lại.",
    },
    "polyps": {
        "class_id": 11,
        "name_vn": "Polyp đại trực tràng (Tổn thương tiền ung thư)",
        "name_en": "Colorectal Polyps",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại trực tràng)",
        "category": "Tổn thương tiền ung thư hàng đầu",
        "severity": "critical",
        "severity_vn": "Khẩn cấp / Cần can thiệp cắt bỏ",
        "color": "#ef4444",
        "recommendation": (
            "Phát hiện khối polyp lồi vào lòng ruột (nguy cơ u tuyến Adenoma tiến triển thành ung thư đại trực tràng). "
            "Bác sĩ cần đánh giá kích thước, hình thái Paris (cuống/không cuống) và hoa văn bề mặt (Kudo Pit Pattern). "
            "Chỉ định cắt polyp qua nội soi (Polypectomy) và gửi giải phẫu bệnh trọn vẹn mẫu polyp."
        ),
        "follow_up": "Nội soi kiểm tra lại sau 1–3 năm tùy số lượng và giải phẫu bệnh polyp.",
    },
    "pylorus": {
        "class_id": 12,
        "name_vn": "Môn vị dạ dày (Mốc giải phẫu)",
        "name_en": "Pylorus Landmark",
        "tract": "Đường tiêu hóa trên (Upper GI - Lỗ môn vị dạ dày)",
        "category": "Mốc giải phẫu bình thường",
        "severity": "normal",
        "severity_vn": "Mốc giải phẫu bình thường",
        "color": "#10b981",
        "recommendation": (
            "Lỗ môn vị co bóp nhịp nhàng, niêm mạc quanh môn vị nhẵn bóng, không phù nề hay hẹp biến dạng. "
            "Không phát hiện ổ loét tiền môn vị. "
            "Chuẩn bị điều khiển đầu soi vượt qua môn vị vào hành tá tràng."
        ),
        "follow_up": "Lưu hình ảnh mốc giải phẫu dạ dày bình thường.",
    },
    "retroflex-rectum": {
        "class_id": 13,
        "name_vn": "Quan sát đảo ngược bóng trực tràng",
        "name_en": "Retroflexed Rectum",
        "tract": "Đường tiêu hóa dưới (Lower GI - Trực tràng)",
        "category": "Thao tác nội soi chuẩn",
        "severity": "normal",
        "severity_vn": "Thao tác kiểm tra đạt chuẩn",
        "color": "#10b981",
        "recommendation": (
            "Thao tác bẻ ngược đầu soi (Retroflexion) trong bóng trực tràng nhằm quan sát toàn diện "
            "vùng cổ trực tràng và bờ trên cơ thắt hậu môn. "
            "Kỹ thuật bắt buộc giúp không bỏ sót polyp nhỏ và khối u nằm tại góc khuất trực tràng."
        ),
        "follow_up": "Ghi nhận hoàn thành bước kiểm tra đảo ngược trực tràng.",
    },
    "retroflex-stomach": {
        "class_id": 14,
        "name_vn": "Quan sát đảo ngược phình vị & tâm vị dạ dày",
        "name_en": "Retroflexed Stomach",
        "tract": "Đường tiêu hóa trên (Upper GI - Phình vị & Tâm vị)",
        "category": "Thao tác nội soi chuẩn",
        "severity": "normal",
        "severity_vn": "Thao tác kiểm tra đạt chuẩn",
        "color": "#10b981",
        "recommendation": (
            "Thao tác soi quặt ngược hình chữ U trong dạ dày giúp kiểm tra toàn cảnh đáy phình vị, "
            "vùng tâm vị ôm sát dây soi và góc bờ cong nhỏ. "
            "Không phát hiện giãn tĩnh mạch phình vị (Fundal varices) hay u dưới niêm mạc (GIST)."
        ),
        "follow_up": "Lưu ảnh mốc quặt ngược dạ dày chuẩn.",
    },
    "ulcerative-colitis-grade-1": {
        "class_id": 15,
        "name_vn": "Viêm loét đại tràng Thang điểm Mayo Độ 1 (Nhẹ)",
        "name_en": "Ulcerative Colitis (Mayo Grade 1)",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại tràng)",
        "category": "Bệnh lý viêm ruột mạn tính (IBD)",
        "severity": "warning",
        "severity_vn": "Viêm mạn tính mức độ nhẹ",
        "color": "#f59e0b",
        "recommendation": (
            "Niêm mạc đại tràng sung huyết nhẹ, giảm cấu trúc lưới mạch máu, không có hiện tượng mủn trợt. "
            "Chỉ định sinh thiết đa vị trí khẳng định chẩn đoán IBD và điều trị duy trì "
            "bằng dẫn xuất 5-ASA (Mesalamine) đường uống hoặc đặt hậu môn."
        ),
        "follow_up": "Đánh giá đáp ứng lâm sàng sau 6–12 tuần.",
    },
    "ulcerative-colitis-grade-2": {
        "class_id": 16,
        "name_vn": "Viêm loét đại tràng Thang điểm Mayo Độ 2 (Trung bình)",
        "name_en": "Ulcerative Colitis (Mayo Grade 2)",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại tràng)",
        "category": "Bệnh lý viêm ruột mạn tính (IBD)",
        "severity": "critical",
        "severity_vn": "Viêm hoạt động mức độ trung bình",
        "color": "#ef4444",
        "recommendation": (
            "Sung huyết đỏ rõ rệt, mất hoàn toàn mạng mạch máu, niêm mạc mủn dễ chảy máu khi chạm nhẹ "
            "và xuất hiện các vết trợt loét nông. "
            "Khuyến nghị tăng cường phác đồ Corticosteroid toàn thân đợt cấp kết hợp 5-ASA liều cao "
            "và xét nghiệm Calprotectin trong phân."
        ),
        "follow_up": "Nội soi kiểm tra lại sau khi kiểm soát đợt viêm cấp.",
    },
    "ulcerative-colitis-grade-3": {
        "class_id": 17,
        "name_vn": "Viêm loét đại tràng Thang điểm Mayo Độ 3 (Rất nặng)",
        "name_en": "Ulcerative Colitis (Mayo Grade 3)",
        "tract": "Đường tiêu hóa dưới (Lower GI - Đại tràng)",
        "category": "Bệnh lý viêm ruột mạn tính (IBD)",
        "severity": "critical",
        "severity_vn": "Khẩn cấp / Nguy cơ biến chứng thủng",
        "color": "#b91c1c",
        "recommendation": (
            "Tổn thương loét sâu diện rộng, tự chảy máu tự phát, phủ nhiều giả mạc đục mủ. "
            "Nguy cơ cao biến chứng phình giãn đại tràng nhiễm độc (Toxic Megacolon) hoặc thủng ruột. "
            "Chỉ định nhập viện cấp cứu khẩn, hồi sức tích cực, Corticosteroid truyền tĩnh mạch "
            "hoặc xem xét thuốc sinh học (Infliximab/Vedolizumab)."
        ),
        "follow_up": "Theo dõi sát sinh hiệu nội trú và chỉ định hội chẩn ngoại khoa.",
    },
    "z-line": {
        "class_id": 18,
        "name_vn": "Đường Z-line (Ranh giới dạ dày – thực quản)",
        "name_en": "Z-Line (Gastroesophageal Junction)",
        "tract": "Đường tiêu hóa trên (Upper GI - Chuyển tiếp dạ dày - thực quản)",
        "category": "Mốc giải phẫu bình thường",
        "severity": "normal",
        "severity_vn": "Mốc giải phẫu bình thường",
        "color": "#10b981",
        "recommendation": (
            "Ranh giới chuyển tiếp rõ ràng giữa biểu mô lát tầng thực quản (màu nhạt) "
            "và biểu mô trụ dạ dày (màu hồng đậm). "
            "Đường Z-line đều đặn, không có tổn thương trợt vượt quá 5 mm và không có đảo niêm mạc nghi ngờ Barrett."
        ),
        "follow_up": "Ghi nhận mốc giải phẫu bình thường.",
    },
}
