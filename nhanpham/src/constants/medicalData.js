// Mock medical images data grouped by examination
export const medicalImagesGroups = [
  {
    id: 1,
    examDate: "2025-11-10",
    examType: "Chest Examination",
    images: [
      {
        id: "IMG-001",
        type: "X-Ray Chest PA",
        url: "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400",
        modality: "X-Ray",
        imageCode: "XR-CH-001",
      },
      {
        id: "IMG-002",
        type: "X-Ray Chest Lateral",
        url: "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400",
        modality: "X-Ray",
        imageCode: "XR-CH-002",
      },
    ],
  },
  {
    id: 2,
    examDate: "2025-11-08",
    examType: "Brain MRI Study",
    images: [
      {
        id: "IMG-003",
        type: "MRI Brain T1",
        url: "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=400",
        modality: "MRI",
        imageCode: "MR-BR-001",
      },
      {
        id: "IMG-004",
        type: "MRI Brain T2",
        url: "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=400",
        modality: "MRI",
        imageCode: "MR-BR-002",
      },
      {
        id: "IMG-005",
        type: "MRI Brain FLAIR",
        url: "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=400",
        modality: "MRI",
        imageCode: "MR-BR-003",
      },
    ],
  },
  {
    id: 3,
    examDate: "2025-11-05",
    examType: "Abdominal CT Scan",
    images: [
      {
        id: "IMG-006",
        type: "CT Abdomen Pre-contrast",
        url: "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?w=400",
        modality: "CT",
        imageCode: "CT-AB-001",
      },
      {
        id: "IMG-007",
        type: "CT Abdomen Post-contrast",
        url: "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?w=400",
        modality: "CT",
        imageCode: "CT-AB-002",
      },
    ],
  },
  {
    id: 4,
    examDate: "2025-11-03",
    examType: "Ultrasound Examination",
    images: [
      {
        id: "IMG-008",
        type: "Ultrasound Liver",
        url: "https://images.unsplash.com/photo-1579154204601-01588f351e67?w=400",
        modality: "Ultrasound",
        imageCode: "US-LV-001",
      },
      {
        id: "IMG-009",
        type: "Ultrasound Kidney",
        url: "https://images.unsplash.com/photo-1579154204601-01588f351e67?w=400",
        modality: "Ultrasound",
        imageCode: "US-KD-001",
      },
    ],
  },
  {
    id: 5,
    examDate: "2025-11-02",
    examType: "Spine X-Ray",
    images: [
      {
        id: "IMG-010",
        type: "X-Ray Spine AP",
        url: "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400",
        modality: "X-Ray",
        imageCode: "XR-SP-001",
      },
      {
        id: "IMG-011",
        type: "X-Ray Spine Lateral",
        url: "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400",
        modality: "X-Ray",
        imageCode: "XR-SP-002",
      },
    ],
  },
];

// Mock AI analysis data generator
export const generateAnalysisData = (patientDiagnosis) => ({
  aiConfidence: 94.5,
  diagnosis: patientDiagnosis || "Pending Analysis",
  findings: [
    {
      id: 1,
      text: "Abnormal density detected in upper right quadrant",
      severity: "high",
      confidence: 92,
    },
    {
      id: 2,
      text: "Minor calcification observed",
      severity: "medium",
      confidence: 87,
    },
    {
      id: 3,
      text: "Normal bone structure and alignment",
      severity: "low",
      confidence: 96,
    },
  ],
  metrics: [
    { label: "Lesion Size", value: "2.3 cm", status: "warning" },
    { label: "Density", value: "145 HU", status: "normal" },
    { label: "Volume", value: "12.5 cm³", status: "warning" },
    { label: "Growth Rate", value: "+5% (30d)", status: "critical" },
    { label: "Contrast Enhancement", value: "42 HU", status: "normal" },
    { label: "Attenuation", value: "38 HU", status: "normal" },
  ],
  recommendations: [
    "Follow-up imaging in 3 months",
    "Consult with oncology department",
    "Consider biopsy for detailed analysis",
    "Monitor patient symptoms closely",
  ],
});
