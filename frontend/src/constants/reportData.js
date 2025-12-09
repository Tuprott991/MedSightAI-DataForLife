// Mock data báo cáo X-quang cho từng loại bệnh
export const reportTemplates = {
  "01_Tuberculosis": {
    bao_cao_x_quang: {
      MeSH: "Đông đặc, Tổn thương phổi, Thâm nhiễm, Hoại tử khô",
      loai_anh: "X-quang ngực tư thế PA",
      chi_dinh: "Ho kéo dài, sốt nhẹ về chiều, đổ mồ hôi đêm",
      so_sanh: "Không có ảnh trước để so sánh.",
      mo_ta:
        "Tim: Kích thước tim bình thường, không có dấu hiệu phì đại. Trung thất nằm giữa, không có dấu hiệu chèn ép. Phổi: Ghi nhận vùng đông đặc không đồng nhất tại thùy trên phổi phải, có xu hướng xơ hóa và co kéo. Xuất hiện các tổn thương dạng nốt mờ đục phân bố lan tỏa ở cả hai trường phổi. Có hình ảnh hoại tử khô với vùng trống khí ở trung tâm tổn thương. Các vùng phổi còn lại có tăng vân phế quản. Màng phổi: Không thấy tràn dịch màng phổi. Cơ hoành: Di động bình thường, góc sườn hoành nhọn. Hệ xương: Cấu trúc xương bình thường, không có tổn thương tiêu xương. Chất lượng hình ảnh: Đạt tiêu chuẩn chẩn đoán.",
      ket_luan:
        "Hình ảnh X-quang phù hợp với lao phổi hoạt động. Có đông đặc và tổn thương phổi thùy trên phải với hình ảnh hoại tử khô. Khuyến nghị xét nghiệm đờm tìm vi khuẩn lao và cân nhắc điều trị kháng lao.",
    },
  },

  "02_pneumonia": {
    bao_cao_x_quang: {
      MeSH: "Thâm nhiễm phổi, Đông đặc, Giãn động mạch chủ, Tim to",
      loai_anh: "X-quang ngực tư thế PA",
      chi_dinh: "Truyền dịch và kháng sinh hỗ trợ hô hấp",
      so_sanh: "So với phim chụp 3 tháng trước: tổn thương mới xuất hiện.",
      mo_ta:
        "Tim: Kích thước tim tăng nhẹ, chỉ số tim-ngực > 0.5. Bờ tim phải lồi ra, gợi ý giãn tâm thất phải. Trung thất giãn rộng, ghi nhận giãn động mạch chủ lên với đường kính khoảng 4.2 cm. Phổi: Có vùng đông đặc không khí phế nang rộng tại thùy dưới phổi phải và một phần thùy giữa. Ghi nhận hình ảnh viền khí phế quản (air bronchogram) trong vùng đông đặc. Có thâm nhiễm lan tỏa dạng vá ở cả hai trường phổi. Màng phổi: Không thấy tràn dịch màng phổi rõ rệt. Cơ hoành: Cơ hoành phải hơi cao, di động hạn chế. Hệ xương: Có vôi hóa nhẹ dọc theo cung động mạch chủ. Chất lượng hình ảnh: Tốt, phơi sáng đạt yêu cầu.",
      ket_luan:
        "Viêm phổi thùy dưới phải và thùy giữa với hình ảnh đông đặc rõ rệt. Đồng thời có tim to và giãn động mạch chủ lên. Khuyến nghị điều trị kháng sinh mạnh và theo dõi chức năng tim mạch.",
    },
  },

  "03_Otherdisease": {
    bao_cao_x_quang: {
      MeSH: "Xơ phổi, Dày màng phổi, Tổn thương kẽ",
      loai_anh: "X-quang ngực tư thế PA",
      chi_dinh: "Khó thở kéo dài, ho khan",
      so_sanh: "So với phim 6 tháng trước: tổn thương tiến triển chậm.",
      mo_ta:
        "Tim: Kích thước tim bình thường. Trung thất nằm giữa, không dấu hiệu bất thường. Phổi: Ghi nhận tăng vân kẽ lan tỏa ở cả hai trường phổi, đặc biệt rõ ở vùng đáy phổi. Có hình ảnh tổn thương dạng tổ ong (honeycomb pattern) tại các vùng ngoại vi. Xuất hiện các đường kẻ Kerley B tại đáy phổi. Có dày màng phổi hai bên, không kèm tràn dịch. Vùng phổi giữa có tăng mật độ không đồng nhất. Màng phổi: Dày màng phổi thành bên và đáy hai bên phổi. Cơ hoành: Vị trí và di động bình thường. Hệ xương: Không có tổn thương xương rõ rệt. Chất lượng hình ảnh: Đạt chuẩn chẩn đoán.",
      ket_luan:
        "Hình ảnh phù hợp với bệnh phổi kẽ mạn tính có xơ hóa và dày màng phổi. Có dấu hiệu tổn thương tiến triển so với phim trước. Khuyến nghị chụp CT phân giải cao và xét nghiệm chức năng hô hấp.",
    },
  },

  "04_Turbeculosis": {
    bao_cao_x_quang: {
      MeSH: "Vôi hóa, Nốt/Khối, Dày màng phổi, Xơ phổi",
      loai_anh: "X-quang ngực tư thế PA",
      chi_dinh: "Tiền sử điều trị lao, khám sức khỏe định kỳ",
      so_sanh: "So với phim 1 năm trước: tổn thương ổn định.",
      mo_ta:
        "Tim: Kích thước và hình dạng tim bình thường. Trung thất không có dịch chuyển. Phổi: Ghi nhận nhiều nốt vôi hóa đậm độ kích thước từ 3-8mm phân bố rải rác ở cả hai trường phổi, tập trung nhiều ở thùy trên. Có một khối vôi hóa kích thước 2.3 cm tại vùng rốn phổi phải. Xuất hiện xơ hóa và co kéo phế quản tại thùy trên phổi phải. Dày màng phổi vùng đỉnh hai bên. Các vùng phổi còn lại thông thoáng, không có tổn thương cấp tính. Màng phổi: Dày màng phổi vùng đỉnh, không có tràn dịch. Cơ hoành: Vị trí bình thường, di động tốt. Hệ xương: Không có tổn thương xương. Chất lượng hình ảnh: Tốt.",
      ket_luan:
        "Lao phổi cũ đã vôi hóa với nhiều nốt vôi hóa và một khối vôi hóa tại rốn phổi phải. Có xơ hóa và dày màng phổi thứ phát. Tổn thương ổn định, không có dấu hiệu tái hoạt động.",
    },
  },

  "05_pneumonia": {
    bao_cao_x_quang: {
      MeSH: "Tim to, Phù phổi, Tắc nghẽn tĩnh mạch",
      loai_anh: "X-quang ngực tư thế PA",
      chi_dinh: "Khó thở tăng dần, phù chi dưới",
      so_sanh: "So với phim 2 tuần trước: tim to tăng hơn.",
      mo_ta:
        "Tim: Kích thước tim tăng rõ rệt, chỉ số tim-ngực khoảng 0.62. Bờ tim trái lồi ra, gợi ý giãn tâm thất trái. Bờ tim phải cũng lồi ra. Hình ảnh phù hợp với tim to toàn bộ. Trung thất giãn rộng. Phổi: Có tăng mật độ lan tỏa hai trường phổi dạng đám mây, đặc biệt ở vùng rốn và đáy phổi. Viền mờ quanh rốn phổi (hilar haze). Có dấu hiệu phù phổi gian kẽ với đường Kerley B rõ. Mạch máu phổi nổi rõ và tắc nghẽn. Màng phổi: Không thấy tràn dịch màng phổi rõ rệt. Cơ hoành: Vị trí bình thường. Hệ xương: Không có bất thường. Chất lượng hình ảnh: Đạt yêu cầu.",
      ket_luan:
        "Tim to toàn bộ kèm theo dấu hiệu phù phổi gian kẽ và tắc nghẽn tĩnh mạch phổi. Hình ảnh phù hợp với suy tim sung huyết. Khuyến nghị siêu âm tim và điều trị suy tim tích cực.",
    },
  },

  "06_pneumonia": {
    bao_cao_x_quang: {
      MeSH: "Giãn động mạch chủ, Tim to, Vôi hóa động mạch chủ",
      loai_anh: "X-quang ngực tư thế PA",
      chi_dinh: "Đau ngực, khó thở khi gắng sức",
      so_sanh: "So với phim 1 năm trước: giãn động mạch chủ tăng thêm.",
      mo_ta:
        "Tim: Kích thước tim tăng, chỉ số tim-ngực khoảng 0.58. Hình ảnh tim hình giày, gợi ý giãn tâm thất trái. Đỉnh tim hạ thấp. Trung thất giãn rộng đáng kể. Động mạch chủ lên giãn rõ rệt với đường kính ước tính 5.1 cm. Phổi: Trường phổi hai bên thông thoáng, không có tổn thương thực chất rõ rệt. Có tăng vân phế quản nhẹ. Hình ảnh mạch máu phổi bình thường. Màng phổi: Không có dày màng phổi hay tràn dịch. Cơ hoành: Vị trí và di động bình thường. Hệ xương: Có vôi hóa dày dọc theo cung động mạch chủ và cả động mạch chủ xuống. Chất lượng hình ảnh: Rất tốt.",
      ket_luan:
        "Giãn động mạch chủ lên rõ rệt (5.1 cm) kèm tim to. Có vôi hóa động mạch chủ lan tỏa. Nguy cơ cao vỡ động mạch chủ. Khuyến nghị chụp CT động mạch chủ có cản quang và tư vấn phẫu thuật tim mạch khẩn cấp.",
    },
  },

  "07_Nofinding": {
    bao_cao_x_quang: {
      MeSH: "Không có",
      loai_anh: "X-quang ngực tư thế PA",
      chi_dinh: "Khám sức khỏe định kỳ",
      so_sanh: "So với phim năm trước: không có thay đổi đáng kể.",
      mo_ta:
        "Tim: Kích thước và hình dạng tim bình thường, chỉ số tim-ngực < 0.5. Bờ tim rõ nét, không có dấu hiệu phì đại buồng tim. Trung thất nằm giữa, không có dịch chuyển hay khối. Phổi: Trường phổi hai bên sáng, thông thoáng hoàn toàn. Hình ảnh phế quản và mạch máu phổi bình thường. Không có tổn thương thực chất, nốt, khối hay đông đặc. Vân phổi trong giới hạn sinh lý. Rốn phổi hai bên không có hạch to. Màng phổi: Không có dày màng phổi hay tràn dịch màng phổi. Góc sườn hoành hai bên nhọn. Cơ hoành: Vị trí bình thường, hình vòm tự nhiên, di động tốt. Hệ xương: Cấu trúc xương lồng ngực, xương sườn, đốt sống bình thường. Không có gãy xương, tiêu xương hay thoái hóa. Chất lượng hình ảnh: Xuất sắc, phơi sáng cân đối.",
      ket_luan:
        "X-quang ngực hoàn toàn bình thường. Không phát hiện bất thường về tim, phổi, màng phổi và hệ xương. Khuyến nghị khám sức khỏe định kỳ hàng năm.",
    },
  },
};

// Function lấy báo cáo theo đường dẫn ảnh bệnh nhân
export const getReportByImagePath = (imagePath) => {
  if (!imagePath) return null;

  for (const [key, report] of Object.entries(reportTemplates)) {
    if (imagePath.includes(key)) {
      return report;
    }
  }

  return null;
};

// Function tạo báo cáo với thông tin bệnh nhân
export const generatePatientReport = (patient) => {
  if (!patient) return null;

  const baseReport = getReportByImagePath(patient.image);
  if (!baseReport) return null;

  return {
    ...baseReport,
    thong_tin_benh_nhan: {
      ho_ten: patient.name,
      tuoi: patient.age,
      gioi_tinh: patient.gender,
      nhom_mau: patient.blood_type,
      ngay_chup: new Date().toLocaleDateString("vi-VN"),
      chan_doan_lam_sang: patient.diagnosis,
      bac_si_doc_phim: "BS. Nguyễn Văn A",
      ngay_doc_phim: new Date().toLocaleDateString("vi-VN"),
    },
    bao_cao_x_quang: baseReport.bao_cao_x_quang,
  };
};
