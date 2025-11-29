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

// Tạo dữ liệu phân tích AI
export const generateAnalysisData = (patientDiagnosis) => ({
  aiConfidence: 94.5,
  diagnosis: patientDiagnosis || "Đang Phân Tích",
  suspectedDiseases: [
    {
      name: "Bệnh Động Mạch Vành",
      confidence: 92,
    },
    {
      name: "Nhồi Máu Cơ Tim Cấp",
      confidence: 85,
    },
    {
      name: "Bệnh Cơ Tim",
      confidence: 78,
    },
  ],
  findings: [
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
  ],
  metrics: [
    { label: "Kích Thước Tổn Thương", value: "2.3 cm", status: "warning" },
    { label: "Mật Độ", value: "145 HU", status: "normal" },
    { label: "Thể Tích", value: "12.5 cm³", status: "warning" },
    { label: "Tốc Độ Tăng Trưởng", value: "+5% (30 ngày)", status: "critical" },
    { label: "Tăng Cường Cản Quang", value: "42 HU", status: "normal" },
    { label: "Suy Giảm", value: "38 HU", status: "normal" },
  ],
  recommendations: [
    "Chụp hình ảnh theo dõi sau 3 tháng",
    "Tư vấn với khoa ung bướu",
    "Cân nhắc sinh thiết để phân tích chi tiết",
    "Theo dõi chặt chẽ các triệu chứng của bệnh nhân",
  ],
});
