// Dữ liệu hình ảnh y tế được nhóm theo lần khám
export const medicalImagesGroups = [
  {
    id: 1,
    examDate: "2025-11-10",
    examType: "Khám Ngực",
    images: [
      {
        id: "IMG-001",
        type: "X-Quang Ngực PA",
        url: "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400",
        modality: "X-Quang",
        imageCode: "XR-CH-001",
      },
      {
        id: "IMG-002",
        type: "X-Quang Ngực Nghiêng",
        url: "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400",
        modality: "X-Quang",
        imageCode: "XR-CH-002",
      },
    ],
  },
  {
    id: 2,
    examDate: "2025-11-08",
    examType: "Chụp MRI Não",
    images: [
      {
        id: "IMG-003",
        type: "MRI Não T1",
        url: "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=400",
        modality: "MRI",
        imageCode: "MR-BR-001",
      },
      {
        id: "IMG-004",
        type: "MRI Não T2",
        url: "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=400",
        modality: "MRI",
        imageCode: "MR-BR-002",
      },
      {
        id: "IMG-005",
        type: "MRI Não FLAIR",
        url: "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=400",
        modality: "MRI",
        imageCode: "MR-BR-003",
      },
    ],
  },
  {
    id: 3,
    examDate: "2025-11-05",
    examType: "Chụp CT Bụng",
    images: [
      {
        id: "IMG-006",
        type: "CT Bụng Trước Cản Quang",
        url: "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?w=400",
        modality: "CT",
        imageCode: "CT-AB-001",
      },
      {
        id: "IMG-007",
        type: "CT Bụng Sau Cản Quang",
        url: "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?w=400",
        modality: "CT",
        imageCode: "CT-AB-002",
      },
    ],
  },
  {
    id: 4,
    examDate: "2025-11-03",
    examType: "Siêu Âm",
    images: [
      {
        id: "IMG-008",
        type: "Siêu Âm Gan",
        url: "https://images.unsplash.com/photo-1579154204601-01588f351e67?w=400",
        modality: "Siêu Âm",
        imageCode: "US-LV-001",
      },
      {
        id: "IMG-009",
        type: "Siêu Âm Thận",
        url: "https://images.unsplash.com/photo-1579154204601-01588f351e67?w=400",
        modality: "Siêu Âm",
        imageCode: "US-KD-001",
      },
    ],
  },
  {
    id: 5,
    examDate: "2025-11-02",
    examType: "X-Quang Cột Sống",
    images: [
      {
        id: "IMG-010",
        type: "X-Quang Cột Sống AP",
        url: "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400",
        modality: "X-Quang",
        imageCode: "XR-SP-001",
      },
      {
        id: "IMG-011",
        type: "X-Quang Cột Sống Nghiêng",
        url: "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400",
        modality: "X-Quang",
        imageCode: "XR-SP-002",
      },
    ],
  },
];

// Lấy bệnh nghi ngờ từ thư mục của bệnh nhân
const getSuspectedDiseaseFromPath = (imagePath) => {
  if (!imagePath) return [];

  if (
    imagePath.includes("01_Tuberculosis") ||
    imagePath.includes("04_Turbeculosis")
  ) {
    return [{ name: "Lao phổi", confidence: 94 }];
  } else if (
    imagePath.includes("02_pneumonia") ||
    imagePath.includes("05_pneumonia") ||
    imagePath.includes("06_pneumonia")
  ) {
    return [{ name: "Viêm phổi", confidence: 92 }];
  } else if (imagePath.includes("03_Otherdisease")) {
    return [{ name: "Bệnh phổi khác", confidence: 88 }];
  } else if (imagePath.includes("07_Nofinding")) {
    return [{ name: "Không có phát hiện bất thường", confidence: 98 }];
  }

  return [{ name: "Đang phân tích", confidence: 85 }];
};

// Lấy đường dẫn hình ảnh cho finding
export const getFindingImagePath = (findingText, patientImagePath) => {
  if (!patientImagePath) return null;

  // Lấy thư mục gốc từ đường dẫn bệnh nhân
  const basePathMatch = patientImagePath.match(/(.+\/)origin\.png/);
  if (!basePathMatch) return null;

  const basePath = basePathMatch[1];

  // Ánh xạ tên finding sang tên thư mục con và tên file
  const findingPathMap = {
    "Đông đặc phổi": "Consolidation/Untitled.jpeg",
    "Tổn thương phổi": "Lung Opacity/Untitled.jpeg",
    "Vôi hóa": "Calcification/Untitled.png",
    "Khối u phổi": "Nodule Mass/Untitled.jpeg",
    "Dày màng phổi": "Pleural Thickening/Untitled.png",
    "Xơ phổi": "Pulmonary fibrosis/Untitled.jpeg",
    "Phình động mạch chủ": "Aortic enlargement/Untitled.png",
    "Tim to": "Cardiomegaly/Pasted image.png",
    "Thâm nhiễm phổi": "Infiltration/Untitled.jpeg",
  };

  // Xử lý đặc biệt cho các thư mục khác nhau
  if (basePath.includes("01_Tuberculosis")) {
    if (findingText === "Tổn thương phổi") {
      return `${basePath}Lung Opacity/Untitled.png`;
    }
  } else if (basePath.includes("02_pneumonia")) {
    if (findingText === "Phình động mạch chủ") {
      return `${basePath}Aortic enlargement/Untitled.png`;
    } else if (findingText === "Tổn thương phổi") {
      return `${basePath}Lung Opacity/Untitled.jpeg`;
    }
  } else if (basePath.includes("03_Otherdisease")) {
    if (findingText === "Tổn thương phổi") {
      return `${basePath}Lung Opacity/Untitled.jpeg`;
    } else if (findingText === "Dày màng phổi") {
      return `${basePath}Pleural Thickening/Untitled.jpeg`;
    } else if (findingText === "Xơ phổi") {
      return `${basePath}Pulmory Fibrosis/Untitled.jpeg`;
    }
  } else if (basePath.includes("04_Turbeculosis")) {
    // Tất cả đã đúng trong map
  } else if (basePath.includes("05_pneumonia")) {
    if (findingText === "Tim to") {
      return `${basePath}Cardiomegaly/011244ab511b20130d846f5f8f0c3866.jpeg`;
    }
  } else if (basePath.includes("06_pneumonia")) {
    if (findingText === "Phình động mạch chủ") {
      return `${basePath}Aortic enlargement/Untitled.jpeg`;
    } else if (findingText === "Tim to") {
      return `${basePath}Cardiomegaly/Untitled.jpeg`;
    }
  }

  const folderPath = findingPathMap[findingText];
  if (!folderPath) return null;

  return `${basePath}${folderPath}`;
};

// Lấy danh sách findings từ thư mục của bệnh nhân
const getFindingsFromPath = (imagePath) => {
  if (!imagePath) return [];

  if (imagePath.includes("01_Tuberculosis")) {
    return [
      { id: 1, text: "Đông đặc phổi", severity: "high", confidence: 92 },
      { id: 2, text: "Tổn thương phổi", severity: "high", confidence: 88 },
    ];
  } else if (imagePath.includes("02_pneumonia")) {
    return [
      { id: 1, text: "Phình động mạch chủ", severity: "high", confidence: 91 },
      { id: 2, text: "Tim to", severity: "high", confidence: 89 },
      { id: 3, text: "Thâm nhiễm phổi", severity: "high", confidence: 94 },
      { id: 4, text: "Tổn thương phổi", severity: "medium", confidence: 86 },
    ];
  } else if (imagePath.includes("03_Otherdisease")) {
    return [
      { id: 1, text: "Tổn thương phổi", severity: "medium", confidence: 88 },
      { id: 2, text: "Dày màng phổi", severity: "medium", confidence: 85 },
      { id: 3, text: "Xơ phổi", severity: "medium", confidence: 82 },
    ];
  } else if (imagePath.includes("04_Turbeculosis")) {
    return [
      { id: 1, text: "Vôi hóa", severity: "medium", confidence: 85 },
      { id: 2, text: "Khối u phổi", severity: "high", confidence: 90 },
      { id: 3, text: "Dày màng phổi", severity: "medium", confidence: 87 },
      { id: 4, text: "Xơ phổi", severity: "medium", confidence: 83 },
    ];
  } else if (imagePath.includes("05_pneumonia")) {
    return [{ id: 1, text: "Tim to", severity: "high", confidence: 89 }];
  } else if (imagePath.includes("06_pneumonia")) {
    return [
      { id: 1, text: "Phình động mạch chủ", severity: "high", confidence: 91 },
      { id: 2, text: "Tim to", severity: "high", confidence: 89 },
    ];
  } else if (imagePath.includes("07_Nofinding")) {
    return [
      {
        id: 1,
        text: "Phổi bình thường, không có bất thường",
        severity: "low",
        confidence: 98,
      },
    ];
  }

  return [
    {
      id: 1,
      text: "Phát hiện mật độ bất thường ở góc phần tư trên bên phải",
      severity: "high",
      confidence: 92,
    },
    {
      id: 2,
      text: "Quan sát thấy vôi hóa nhẹ",
      severity: "medium",
      confidence: 87,
    },
    {
      id: 3,
      text: "Cấu trúc và sắp xếp xương bình thường",
      severity: "low",
      confidence: 96,
    },
  ];
};

// Tạo dữ liệu phân tích AI dựa trên chẩn đoán bệnh nhân
export const generateAnalysisData = (patientDiagnosis, patientImage) => {
  // Lấy findings dựa trên đường dẫn ảnh của bệnh nhân
  const findings = getFindingsFromPath(patientImage);
  // Lấy bệnh nghi ngờ dựa trên đường dẫn ảnh
  const suspectedDiseases = getSuspectedDiseaseFromPath(patientImage);

  return {
    aiConfidence: 94.5,
    diagnosis: patientDiagnosis || "Đang Phân Tích",
    suspectedDiseases: suspectedDiseases,
    findings: findings,
    metrics: [
      { label: "Kích Thước Tổn Thương", value: "2.3 cm", status: "warning" },
      { label: "Mật Độ", value: "145 HU", status: "normal" },
      { label: "Thể Tích", value: "12.5 cm³", status: "warning" },
      {
        label: "Tốc Độ Tăng Trưởng",
        value: "+5% (30 ngày)",
        status: "critical",
      },
      { label: "Tăng Cường Cản Quang", value: "42 HU", status: "normal" },
      { label: "Suy Giảm", value: "38 HU", status: "normal" },
    ],
    recommendations: [
      "Chụp hình ảnh theo dõi sau 3 tháng",
      "Tư vấn với khoa ung bướu",
      "Cân nhắc sinh thiết để phân tích chi tiết",
      "Theo dõi chặt chẽ các triệu chứng của bệnh nhân",
    ],
  };
};
