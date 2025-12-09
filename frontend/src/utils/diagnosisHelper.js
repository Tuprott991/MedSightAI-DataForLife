/**
 * Map diagnosis text (Vietnamese or English) to translation key
 * @param {string} diagnosis - Diagnosis text in Vietnamese or English
 * @returns {string} Translation key
 */
export const getDiagnosisKey = (diagnosis) => {
  const diagnosisMap = {
    // Vietnamese
    "Lao phổi": "diagnosis.tuberculosis",
    "Viêm phổi": "diagnosis.pneumonia",
    "Bệnh phổi khác": "diagnosis.otherLungDisease",
    "Chưa phát hiện": "diagnosis.noFinding",
    // English
    Tuberculosis: "diagnosis.tuberculosis",
    Pneumonia: "diagnosis.pneumonia",
    "Other Lung Disease": "diagnosis.otherLungDisease",
    "No Finding": "diagnosis.noFinding",
  };

  return diagnosisMap[diagnosis] || diagnosis;
};

/**
 * Get translated diagnosis text
 * @param {string} diagnosis - Vietnamese diagnosis text
 * @param {Function} t - Translation function from useTranslation
 * @returns {string} Translated diagnosis
 */
export const getTranslatedDiagnosis = (diagnosis, t) => {
  const key = getDiagnosisKey(diagnosis);
  // If key starts with 'diagnosis.', it's a translation key
  if (key.startsWith("diagnosis.")) {
    return t(key);
  }
  // Otherwise return as is
  return diagnosis;
};

/**
 * Get translated gender text
 * @param {string} gender - Vietnamese gender text ("Nam" or "Nữ")
 * @param {Function} t - Translation function from useTranslation
 * @returns {string} Translated gender
 */
export const getTranslatedGender = (gender, t) => {
  if (gender === "Nam") {
    return t("doctorDetail.patientInfo.male");
  } else if (gender === "Nữ") {
    return t("doctorDetail.patientInfo.female");
  }
  return gender;
};
