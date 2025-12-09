// Bilingual Medical Report Data
export const reportTemplatesBilingual = {
  "01_Tuberculosis": {
    bao_cao_x_quang: {
      MeSH: {
        vi: "Đông đặc, Tổn thương phổi, Thâm nhiễm, Hoại tử khô",
        en: "Consolidation, Lung Lesion, Infiltration, Caseous Necrosis",
      },
      loai_anh: {
        vi: "X-quang ngực tư thế PA",
        en: "Chest X-ray PA View",
      },
      chi_dinh: {
        vi: "Ho kéo dài, sốt nhẹ về chiều, đổ mồ hôi đêm",
        en: "Persistent cough, evening fever, night sweats",
      },
      so_sanh: {
        vi: "Không có ảnh trước để so sánh.",
        en: "No prior images for comparison.",
      },
      mo_ta: {
        vi: "Tim: Kích thước tim bình thường, không có dấu hiệu phì đại. Trung thất nằm giữa, không có dấu hiệu chèn ép. Phổi: Ghi nhận vùng đông đặc không đồng nhất tại thùy trên phổi phải, có xu hướng xơ hóa và co kéo. Xuất hiện các tổn thương dạng nốt mờ đục phân bố lan tỏa ở cả hai trường phổi. Có hình ảnh hoại tử khô với vùng trống khí ở trung tâm tổn thương. Các vùng phổi còn lại có tăng vân phế quản. Màng phổi: Không thấy tràn dịch màng phổi. Cơ hoành: Di động bình thường, góc sườn hoành nhọn. Hệ xương: Cấu trúc xương bình thường, không có tổn thương tiêu xương. Chất lượng hình ảnh: Đạt tiêu chuẩn chẩn đoán.",
        en: "Heart: Normal cardiac size, no signs of hypertrophy. Mediastinum is central with no compression. Lungs: Heterogeneous consolidation noted in the right upper lobe with tendency towards fibrosis and retraction. Multiple nodular opacities diffusely distributed in both lung fields. Caseous necrosis with central cavitation present. Remaining lung fields show increased bronchovascular markings. Pleura: No pleural effusion. Diaphragm: Normal mobility, sharp costophrenic angles. Skeletal system: Normal bone structure, no bone destruction. Image quality: Diagnostic standard.",
      },
      ket_luan: {
        vi: "Hình ảnh X-quang phù hợp với lao phổi hoạt động. Có đông đặc và tổn thương phổi thùy trên phải với hình ảnh hoại tử khô. Khuyến nghị xét nghiệm đờm tìm vi khuẩn lao và cân nhắc điều trị kháng lao.",
        en: "X-ray findings consistent with active pulmonary tuberculosis. Consolidation and right upper lobe lesions with caseous necrosis. Recommend sputum culture for TB bacteria and consider anti-tuberculosis treatment.",
      },
    },
  },

  "02_pneumonia": {
    bao_cao_x_quang: {
      MeSH: {
        vi: "Thâm nhiễm phổi, Đông đặc, Giãn động mạch chủ, Tim to",
        en: "Pulmonary Infiltration, Consolidation, Aortic Dilatation, Cardiomegaly",
      },
      loai_anh: {
        vi: "X-quang ngực tư thế PA",
        en: "Chest X-ray PA View",
      },
      chi_dinh: {
        vi: "Truyền dịch và kháng sinh hỗ trợ hô hấp",
        en: "IV fluids and antibiotics for respiratory support",
      },
      so_sanh: {
        vi: "So với phim chụp 3 tháng trước: tổn thương mới xuất hiện.",
        en: "Compared to 3 months prior: new lesions appeared.",
      },
      mo_ta: {
        vi: "Tim: Kích thước tim tăng nhẹ, chỉ số tim-ngực > 0.5. Bờ tim phải lồi ra, gợi ý giãn tâm thất phải. Trung thất giãn rộng, ghi nhận giãn động mạch chủ lên với đường kính khoảng 4.2 cm. Phổi: Có vùng đông đặc không khí phế nang rộng tại thùy dưới phổi phải và một phần thùy giữa. Ghi nhận hình ảnh viền khí phế quản (air bronchogram) trong vùng đông đặc. Có thâm nhiễm lan tỏa dạng vá ở cả hai trường phổi. Màng phổi: Không thấy tràn dịch màng phổi rõ rệt. Cơ hoành: Cơ hoành phải hơi cao, di động hạn chế. Hệ xương: Có vôi hóa nhẹ dọc theo cung động mạch chủ. Chất lượng hình ảnh: Tốt, phơi sáng đạt yêu cầu.",
        en: "Heart: Mildly enlarged, cardiothoracic ratio > 0.5. Right heart border bulges, suggesting right ventricular dilatation. Widened mediastinum with ascending aortic dilatation measuring approximately 4.2 cm. Lungs: Extensive airspace consolidation in the right lower lobe and part of the middle lobe. Air bronchogram visible within the consolidation. Patchy infiltrates scattered throughout both lung fields. Pleura: No significant pleural effusion. Diaphragm: Right hemidiaphragm slightly elevated, limited mobility. Skeletal system: Mild calcification along the aortic arch. Image quality: Good, adequate exposure.",
      },
      ket_luan: {
        vi: "Viêm phổi thùy dưới phải và thùy giữa với hình ảnh đông đặc rõ rệt. Đồng thời có tim to và giãn động mạch chủ lên. Khuyến nghị điều trị kháng sinh mạnh và theo dõi chức năng tim mạch.",
        en: "Right lower and middle lobe pneumonia with prominent consolidation. Concurrent cardiomegaly and ascending aortic dilatation. Recommend aggressive antibiotic therapy and cardiovascular monitoring.",
      },
    },
  },

  "03_Otherdisease": {
    bao_cao_x_quang: {
      MeSH: {
        vi: "Xơ phổi, Dày màng phổi, Tổn thương kẽ",
        en: "Pulmonary Fibrosis, Pleural Thickening, Interstitial Disease",
      },
      loai_anh: {
        vi: "X-quang ngực tư thế PA",
        en: "Chest X-ray PA View",
      },
      chi_dinh: {
        vi: "Khó thở kéo dài, ho khan",
        en: "Persistent dyspnea, dry cough",
      },
      so_sanh: {
        vi: "So với phim 6 tháng trước: tổn thương tiến triển chậm.",
        en: "Compared to 6 months prior: slow disease progression.",
      },
      mo_ta: {
        vi: "Tim: Kích thước tim bình thường. Trung thất nằm giữa, không dấu hiệu bất thường. Phổi: Ghi nhận tăng vân kẽ lan tỏa ở cả hai trường phổi, đặc biệt rõ ở vùng đáy phổi. Có hình ảnh tổn thương dạng tổ ong (honeycomb pattern) tại các vùng ngoại vi. Xuất hiện các đường kẻ Kerley B tại đáy phổi. Có dày màng phổi hai bên, không kèm tràn dịch. Vùng phổi giữa có tăng mật độ không đồng nhất. Màng phổi: Dày màng phổi thành bên và đáy hai bên phổi. Cơ hoành: Vị trí và di động bình thường. Hệ xương: Không có tổn thương xương rõ rệt. Chất lượng hình ảnh: Đạt chuẩn chẩn đoán.",
        en: "Heart: Normal cardiac size. Mediastinum is central, no abnormality. Lungs: Diffuse interstitial markings throughout both lung fields, particularly prominent at lung bases. Honeycomb pattern present in peripheral regions. Kerley B lines visible at lung bases. Bilateral pleural thickening without effusion. Mid-lung zones show heterogeneous increased density. Pleura: Thickening of lateral and basal pleura bilaterally. Diaphragm: Normal position and mobility. Skeletal system: No significant bone lesions. Image quality: Diagnostic standard.",
      },
      ket_luan: {
        vi: "Hình ảnh phù hợp với bệnh phổi kẽ mạn tính có xơ hóa và dày màng phổi. Có dấu hiệu tổn thương tiến triển so với phim trước. Khuyến nghị chụp CT phân giải cao và xét nghiệm chức năng hô hấp.",
        en: "Findings consistent with chronic interstitial lung disease with fibrosis and pleural thickening. Signs of disease progression compared to previous imaging. Recommend high-resolution CT and pulmonary function testing.",
      },
    },
  },

  "04_Turbeculosis": {
    bao_cao_x_quang: {
      MeSH: {
        vi: "Vôi hóa, Nốt/Khối, Dày màng phổi, Xơ phổi",
        en: "Calcification, Nodule/Mass, Pleural Thickening, Pulmonary Fibrosis",
      },
      loai_anh: {
        vi: "X-quang ngực tư thế PA",
        en: "Chest X-ray PA View",
      },
      chi_dinh: {
        vi: "Tiền sử điều trị lao, khám sức khỏe định kỳ",
        en: "History of TB treatment, routine health check",
      },
      so_sanh: {
        vi: "So với phim 1 năm trước: tổn thương ổn định.",
        en: "Compared to 1 year prior: stable lesions.",
      },
      mo_ta: {
        vi: "Tim: Kích thước và hình dạng tim bình thường. Trung thất không có dịch chuyển. Phổi: Ghi nhận nhiều nốt vôi hóa đậm độ kích thước từ 3-8mm phân bố rải rác ở cả hai trường phổi, tập trung nhiều ở thùy trên. Có một khối vôi hóa kích thước 2.3 cm tại vùng rốn phổi phải. Xuất hiện xơ hóa và co kéo phế quản tại thùy trên phổi phải. Dày màng phổi vùng đỉnh hai bên. Các vùng phổi còn lại thông thoáng, không có tổn thương cấp tính. Màng phổi: Dày màng phổi vùng đỉnh, không có tràn dịch. Cơ hoành: Vị trí bình thường, di động tốt. Hệ xương: Không có tổn thương xương. Chất lượng hình ảnh: Tốt.",
        en: "Heart: Normal cardiac size and shape. No mediastinal shift. Lungs: Multiple dense calcified nodules 3-8mm scattered throughout both lung fields, predominantly in upper lobes. A 2.3 cm calcified mass at right hilum. Fibrosis and bronchial distortion in right upper lobe. Apical pleural thickening bilaterally. Remaining lung fields clear, no acute lesions. Pleura: Apical pleural thickening, no effusion. Diaphragm: Normal position, good mobility. Skeletal system: No bone lesions. Image quality: Good.",
      },
      ket_luan: {
        vi: "Lao phổi cũ đã vôi hóa với nhiều nốt vôi hóa và một khối vôi hóa tại rốn phổi phải. Có xơ hóa và dày màng phổi thứ phát. Tổn thương ổn định, không có dấu hiệu tái hoạt động.",
        en: "Old calcified pulmonary tuberculosis with multiple calcified nodules and a calcified mass at right hilum. Secondary fibrosis and pleural thickening. Stable lesions, no signs of reactivation.",
      },
    },
  },

  "05_pneumonia": {
    bao_cao_x_quang: {
      MeSH: {
        vi: "Tim to, Phù phổi, Tắc nghẽn tĩnh mạch",
        en: "Cardiomegaly, Pulmonary Edema, Venous Congestion",
      },
      loai_anh: {
        vi: "X-quang ngực tư thế PA",
        en: "Chest X-ray PA View",
      },
      chi_dinh: {
        vi: "Khó thở tăng dần, phù chi dưới",
        en: "Progressive dyspnea, lower extremity edema",
      },
      so_sanh: {
        vi: "So với phim 2 tuần trước: tim to tăng hơn.",
        en: "Compared to 2 weeks prior: increased cardiomegaly.",
      },
      mo_ta: {
        vi: "Tim: Kích thước tim tăng rõ rệt, chỉ số tim-ngực khoảng 0.62. Bờ tim trái lồi ra, gợi ý giãn tâm thất trái. Bờ tim phải cũng lồi ra. Hình ảnh phù hợp với tim to toàn bộ. Trung thất giãn rộng. Phổi: Có tăng mật độ lan tỏa hai trường phổi dạng đám mây, đặc biệt ở vùng rốn và đáy phổi. Viền mờ quanh rốn phổi (hilar haze). Có dấu hiệu phù phổi gian kẽ với đường Kerley B rõ. Mạch máu phổi nổi rõ và tắc nghẽn. Màng phổi: Không thấy tràn dịch màng phổi rõ rệt. Cơ hoành: Vị trí bình thường. Hệ xương: Không có bất thường. Chất lượng hình ảnh: Đạt yêu cầu.",
        en: "Heart: Markedly enlarged, cardiothoracic ratio approximately 0.62. Left heart border bulges, suggesting left ventricular dilatation. Right heart border also bulges. Findings consistent with global cardiomegaly. Widened mediastinum. Lungs: Diffuse cloud-like increased density throughout both lung fields, particularly at hila and lung bases. Hilar haze present. Signs of interstitial pulmonary edema with prominent Kerley B lines. Pulmonary vessels are prominent and congested. Pleura: No significant pleural effusion. Diaphragm: Normal position. Skeletal system: No abnormality. Image quality: Adequate.",
      },
      ket_luan: {
        vi: "Tim to toàn bộ kèm theo dấu hiệu phù phổi gian kẽ và tắc nghẽn tĩnh mạch phổi. Hình ảnh phù hợp với suy tim sung huyết. Khuyến nghị siêu âm tim và điều trị suy tim tích cực.",
        en: "Global cardiomegaly with signs of interstitial pulmonary edema and pulmonary venous congestion. Findings consistent with congestive heart failure. Recommend echocardiography and aggressive heart failure management.",
      },
    },
  },

  "06_pneumonia": {
    bao_cao_x_quang: {
      MeSH: {
        vi: "Giãn động mạch chủ, Tim to, Vôi hóa động mạch chủ",
        en: "Aortic Dilatation, Cardiomegaly, Aortic Calcification",
      },
      loai_anh: {
        vi: "X-quang ngực tư thế PA",
        en: "Chest X-ray PA View",
      },
      chi_dinh: {
        vi: "Đau ngực, khó thở khi gắng sức",
        en: "Chest pain, exertional dyspnea",
      },
      so_sanh: {
        vi: "So với phim 1 năm trước: giãn động mạch chủ tăng thêm.",
        en: "Compared to 1 year prior: increased aortic dilatation.",
      },
      mo_ta: {
        vi: "Tim: Kích thước tim tăng, chỉ số tim-ngực khoảng 0.58. Hình ảnh tim hình giày, gợi ý giãn tâm thất trái. Đỉnh tim hạ thấp. Trung thất giãn rộng đáng kể. Động mạch chủ lên giãn rõ rệt với đường kính ước tính 5.1 cm. Phổi: Trường phổi hai bên thông thoáng, không có tổn thương thực chất rõ rệt. Có tăng vân phế quản nhẹ. Hình ảnh mạch máu phổi bình thường. Màng phổi: Không có dày màng phổi hay tràn dịch. Cơ hoành: Vị trí và di động bình thường. Hệ xương: Có vôi hóa dày dọc theo cung động mạch chủ và cả động mạch chủ xuống. Chất lượng hình ảnh: Rất tốt.",
        en: "Heart: Enlarged, cardiothoracic ratio approximately 0.58. Boot-shaped cardiac silhouette suggesting left ventricular dilatation. Apex displaced downward. Significantly widened mediastinum. Ascending aorta markedly dilated with estimated diameter of 5.1 cm. Lungs: Both lung fields clear, no significant parenchymal lesions. Mild increased bronchovascular markings. Normal pulmonary vasculature. Pleura: No pleural thickening or effusion. Diaphragm: Normal position and mobility. Skeletal system: Dense calcification along aortic arch and descending aorta. Image quality: Excellent.",
      },
      ket_luan: {
        vi: "Giãn động mạch chủ lên rõ rệt (5.1 cm) kèm tim to. Có vôi hóa động mạch chủ lan tỏa. Nguy cơ cao vỡ động mạch chủ. Khuyến nghị chụp CT động mạch chủ có cản quang và tư vấn phẫu thuật tim mạch khẩn cấp.",
        en: "Marked ascending aortic dilatation (5.1 cm) with cardiomegaly. Diffuse aortic calcification. High risk of aortic rupture. Recommend urgent CT angiography and cardiovascular surgery consultation.",
      },
    },
  },

  "07_Nofinding": {
    bao_cao_x_quang: {
      MeSH: {
        vi: "Không có",
        en: "None",
      },
      loai_anh: {
        vi: "X-quang ngực tư thế PA",
        en: "Chest X-ray PA View",
      },
      chi_dinh: {
        vi: "Khám sức khỏe định kỳ",
        en: "Routine health examination",
      },
      so_sanh: {
        vi: "So với phim năm trước: không có thay đổi đáng kể.",
        en: "Compared to previous year: no significant changes.",
      },
      mo_ta: {
        vi: "Tim: Kích thước và hình dạng tim bình thường, chỉ số tim-ngực < 0.5. Bờ tim rõ nét, không có dấu hiệu phì đại buồng tim. Trung thất nằm giữa, không có dịch chuyển hay khối. Phổi: Trường phổi hai bên sáng, thông thoáng hoàn toàn. Hình ảnh phế quản và mạch máu phổi bình thường. Không có tổn thương thực chất, nốt, khối hay đông đặc. Vân phổi trong giới hạn sinh lý. Rốn phổi hai bên không có hạch to. Màng phổi: Không có dày màng phổi hay tràn dịch màng phổi. Góc sườn hoành hai bên nhọn. Cơ hoành: Vị trí bình thường, hình vòm tự nhiên, di động tốt. Hệ xương: Cấu trúc xương lồng ngực, xương sườn, đốt sống bình thường. Không có gãy xương, tiêu xương hay thoái hóa. Chất lượng hình ảnh: Xuất sắc, phơi sáng cân đối.",
        en: "Heart: Normal cardiac size and shape, cardiothoracic ratio < 0.5. Sharp cardiac borders, no chamber hypertrophy. Mediastinum is central, no shift or mass. Lungs: Both lung fields clear and well-aerated. Normal bronchial and pulmonary vascular markings. No parenchymal lesions, nodules, masses, or consolidation. Lung markings within physiological limits. No hilar lymphadenopathy. Pleura: No pleural thickening or effusion. Sharp costophrenic angles bilaterally. Diaphragm: Normal position, smooth dome configuration, good mobility. Skeletal system: Normal chest wall, ribs, and vertebral structures. No fractures, bone destruction, or degenerative changes. Image quality: Excellent, balanced exposure.",
      },
      ket_luan: {
        vi: "X-quang ngực hoàn toàn bình thường. Không phát hiện bất thường về tim, phổi, màng phổi và hệ xương. Khuyến nghị khám sức khỏe định kỳ hàng năm.",
        en: "Completely normal chest X-ray. No abnormalities detected in heart, lungs, pleura, or skeletal system. Recommend annual routine health examination.",
      },
    },
  },
};

// Helper function to get text by language
export const getText = (obj, lang = "vi") => {
  if (!obj) return "";
  if (typeof obj === "string") return obj;
  return obj[lang] || obj.vi || "";
};

// Function to get report by image path with language support
export const getReportByImagePath = (imagePath, lang = "vi") => {
  if (!imagePath) return null;

  for (const [key, report] of Object.entries(reportTemplatesBilingual)) {
    if (imagePath.includes(key)) {
      const baoCao = report.bao_cao_x_quang;
      return {
        bao_cao_x_quang: {
          MeSH: getText(baoCao.MeSH, lang),
          loai_anh: getText(baoCao.loai_anh, lang),
          chi_dinh: getText(baoCao.chi_dinh, lang),
          so_sanh: getText(baoCao.so_sanh, lang),
          mo_ta: getText(baoCao.mo_ta, lang),
          ket_luan: getText(baoCao.ket_luan, lang),
        },
      };
    }
  }

  return null;
};

// Function to generate patient report with bilingual support
export const generatePatientReport = (patient, lang = "vi") => {
  if (!patient) return null;

  const baseReport = getReportByImagePath(patient.image, lang);
  if (!baseReport) return null;

  const labels =
    lang === "en"
      ? {
          ho_ten: patient.name,
          tuoi: patient.age,
          gioi_tinh:
            patient.gender === "Nam"
              ? "Male"
              : patient.gender === "Nữ"
                ? "Female"
                : patient.gender,
          nhom_mau: patient.blood_type,
          ngay_chup: new Date().toLocaleDateString(
            lang === "vi" ? "vi-VN" : "en-US"
          ),
          chan_doan_lam_sang: patient.diagnosis,
          bac_si_doc_phim: "Dr. Nguyen Van A",
          ngay_doc_phim: new Date().toLocaleDateString(
            lang === "vi" ? "vi-VN" : "en-US"
          ),
        }
      : {
          ho_ten: patient.name,
          tuoi: patient.age,
          gioi_tinh: patient.gender,
          nhom_mau: patient.blood_type,
          ngay_chup: new Date().toLocaleDateString("vi-VN"),
          chan_doan_lam_sang: patient.diagnosis,
          bac_si_doc_phim: "BS. Nguyễn Văn A",
          ngay_doc_phim: new Date().toLocaleDateString("vi-VN"),
        };

  return {
    ...baseReport,
    thong_tin_benh_nhan: labels,
    bao_cao_x_quang: baseReport.bao_cao_x_quang,
  };
};
