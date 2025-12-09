// Bilingual Medical Data
export const medicalTerms = {
  diseases: {
    tuberculosis: { vi: "Lao phổi", en: "Tuberculosis" },
    pneumonia: { vi: "Viêm phổi", en: "Pneumonia" },
    otherLungDisease: { vi: "Bệnh phổi khác", en: "Other Lung Disease" },
    noFinding: { vi: "Chưa phát hiện", en: "No Finding" },
    analyzing: { vi: "Đang phân tích", en: "Analyzing" },
    noAbnormality: {
      vi: "Không có phát hiện bất thường",
      en: "No Abnormality Found",
    },

    // Detailed diseases
    severePneumonia: { vi: "Viêm phổi nặng", en: "Severe Pneumonia" },
    pleuralEffusion: { vi: "Tràn dịch màng phổi", en: "Pleural Effusion" },
    acuteRespiratoryFailure: {
      vi: "Suy hô hấp cấp",
      en: "Acute Respiratory Failure",
    },
    chronicBronchitis: {
      vi: "Viêm phế quản mãn tính",
      en: "Chronic Bronchitis",
    },
    asthma: { vi: "Hen phế quản", en: "Asthma" },
    mildPneumonia: { vi: "Viêm phổi nhẹ", en: "Mild Pneumonia" },
    upperRespiratoryInfection: {
      vi: "Viêm đường hô hấp trên",
      en: "Upper Respiratory Infection",
    },
    commonCold: { vi: "Cảm cúm thông thường", en: "Common Cold" },
    respiratoryAllergy: {
      vi: "Dị ứng đường hô hấp",
      en: "Respiratory Allergy",
    },
  },

  examTypes: {
    chestExam: { vi: "Khám Ngực", en: "Chest Examination" },
    abdominalCT: { vi: "Chụp CT Bụng", en: "Abdominal CT" },
    chestXray: { vi: "X-Quang Ngực", en: "Chest X-Ray" },
    chestXray1: { vi: "X-Quang Ngực - Lần 1", en: "Chest X-Ray - 1st" },
    chestXray2: { vi: "X-Quang Ngực - Lần 2", en: "Chest X-Ray - 2nd" },
    ctPreContrast: {
      vi: "CT Bụng Trước Cản Quang",
      en: "CT Abdomen Pre-Contrast",
    },
    ctPostContrast: {
      vi: "CT Bụng Sau Cản Quang",
      en: "CT Abdomen Post-Contrast",
    },
  },

  findings: {
    consolidation: { vi: "Đông đặc phổi", en: "Lung Consolidation" },
    lungLesion: { vi: "Tổn thương phổi", en: "Lung Lesion" },
    calcification: { vi: "Vôi hóa", en: "Calcification" },
    noduleMass: { vi: "Khối u phổi", en: "Lung Nodule/Mass" },
    pleuralThickening: { vi: "Dày màng phổi", en: "Pleural Thickening" },
    pulmonaryFibrosis: { vi: "Xơ phổi", en: "Pulmonary Fibrosis" },
    aorticEnlargement: { vi: "Phình động mạch chủ", en: "Aortic Enlargement" },
    cardiomegaly: { vi: "Tim to", en: "Cardiomegaly" },
    infiltration: { vi: "Thâm nhiễm phổi", en: "Infiltration" },
  },

  modalities: {
    xray: { vi: "X-Quang", en: "X-Ray" },
    ct: { vi: "CT", en: "CT" },
    mri: { vi: "MRI", en: "MRI" },
    ultrasound: { vi: "Siêu âm", en: "Ultrasound" },
  },
};

// Helper function to get text by language
export const getMedicalText = (category, key, lang = "vi") => {
  if (!medicalTerms[category] || !medicalTerms[category][key]) return "";
  return (
    medicalTerms[category][key][lang] || medicalTerms[category][key].vi || ""
  );
};

// Bilingual medical images groups
export const getMedicalImagesGroups = (lang = "vi") => [
  {
    id: 1,
    examDate: "2025-11-10",
    examType: getMedicalText("examTypes", "chestExam", lang),
    images: [
      {
        id: "IMG-001",
        type: getMedicalText("examTypes", "chestXray1", lang),
        url: "/src/mock_data/patient_data/01_Tuberculosis/origin.png",
        modality: getMedicalText("modalities", "xray", lang),
        imageCode: "XR-CH-001",
      },
      {
        id: "IMG-002",
        type: getMedicalText("examTypes", "chestXray2", lang),
        url: "/src/mock_data/patient_data/02_pneumonia/origin.png",
        modality: getMedicalText("modalities", "xray", lang),
        imageCode: "XR-CH-002",
      },
    ],
  },
  {
    id: 2,
    examDate: "2025-11-05",
    examType: getMedicalText("examTypes", "abdominalCT", lang),
    images: [
      {
        id: "IMG-006",
        type: getMedicalText("examTypes", "ctPreContrast", lang),
        url: "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?w=400",
        modality: getMedicalText("modalities", "ct", lang),
        imageCode: "CT-AB-001",
      },
      {
        id: "IMG-007",
        type: getMedicalText("examTypes", "ctPostContrast", lang),
        url: "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?w=400",
        modality: getMedicalText("modalities", "ct", lang),
        imageCode: "CT-AB-002",
      },
    ],
  },
];

// Get suspected disease from path with language support
export const getSuspectedDiseaseFromPath = (imagePath, lang = "vi") => {
  if (!imagePath) return [];

  if (
    imagePath.includes("01_Tuberculosis") ||
    imagePath.includes("04_Turbeculosis")
  ) {
    return [
      {
        name: getMedicalText("diseases", "tuberculosis", lang),
        confidence: 94,
      },
    ];
  } else if (
    imagePath.includes("02_pneumonia") ||
    imagePath.includes("05_pneumonia") ||
    imagePath.includes("06_pneumonia")
  ) {
    return [
      { name: getMedicalText("diseases", "pneumonia", lang), confidence: 92 },
    ];
  } else if (imagePath.includes("03_Otherdisease")) {
    return [
      {
        name: getMedicalText("diseases", "otherLungDisease", lang),
        confidence: 88,
      },
    ];
  } else if (imagePath.includes("07_Nofinding")) {
    return [
      {
        name: getMedicalText("diseases", "noAbnormality", lang),
        confidence: 98,
      },
    ];
  }

  return [
    { name: getMedicalText("diseases", "analyzing", lang), confidence: 85 },
  ];
};

// Get finding image path
export const getFindingImagePath = (
  findingText,
  patientImagePath,
  lang = "vi"
) => {
  if (!patientImagePath) return null;

  const basePathMatch = patientImagePath.match(/(.+\/)origin\.png/);
  if (!basePathMatch) return null;

  const basePath = basePathMatch[1];

  // Map finding names to folder paths
  const findingPathMap = {
    [getMedicalText("findings", "consolidation", lang)]:
      "Consolidation/Untitled.jpeg",
    [getMedicalText("findings", "lungLesion", lang)]:
      "Lung Opacity/Untitled.jpeg",
    [getMedicalText("findings", "calcification", lang)]:
      "Calcification/Untitled.png",
    [getMedicalText("findings", "noduleMass", lang)]:
      "Nodule Mass/Untitled.jpeg",
    [getMedicalText("findings", "pleuralThickening", lang)]:
      "Pleural Thickening/Untitled.png",
    [getMedicalText("findings", "pulmonaryFibrosis", lang)]:
      "Pulmonary fibrosis/Untitled.jpeg",
    [getMedicalText("findings", "aorticEnlargement", lang)]:
      "Aortic enlargement/Untitled.png",
    [getMedicalText("findings", "cardiomegaly", lang)]:
      "Cardiomegaly/Pasted image.png",
    [getMedicalText("findings", "infiltration", lang)]:
      "Infiltration/Untitled.jpeg",
  };

  // Special handling for different folders
  if (basePath.includes("01_Tuberculosis")) {
    if (findingText === getMedicalText("findings", "lungLesion", lang)) {
      return `${basePath}Lung Opacity/Untitled.png`;
    }
  } else if (basePath.includes("02_pneumonia")) {
    if (findingText === getMedicalText("findings", "aorticEnlargement", lang)) {
      return `${basePath}Aortic enlargement/Untitled.jpeg`;
    }
  } else if (basePath.includes("03_Otherdisease")) {
    const consolidationKey = getMedicalText("findings", "consolidation", lang);
    if (findingText === consolidationKey) {
      return `${basePath}Pulmory Fibrosis/Untitled.jpeg`;
    }
  } else if (basePath.includes("05_pneumonia")) {
    if (findingText === getMedicalText("findings", "cardiomegaly", lang)) {
      return `${basePath}Cardiomegaly/011244ab511b20130d846f5f8f0c3866.jpeg`;
    }
  } else if (basePath.includes("06_pneumonia")) {
    const cardioKey = getMedicalText("findings", "cardiomegaly", lang);
    const aorticKey = getMedicalText("findings", "aorticEnlargement", lang);
    if (findingText === cardioKey) {
      return `${basePath}Cardiomegaly/Untitled.jpeg`;
    } else if (findingText === aorticKey) {
      return `${basePath}Aortic enlargement/Untitled.jpeg`;
    }
  }

  const imagePath = findingPathMap[findingText];
  return imagePath ? `${basePath}${imagePath}` : null;
};

export default medicalTerms;
