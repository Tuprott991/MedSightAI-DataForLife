# Full Patient Report API Documentation

## Overview
This API endpoint provides comprehensive patient information including all medical records, case details, AI analysis results, and generated reports in a single response.

## Endpoint

### GET /api/v1/reports/full/{case_id}

Get comprehensive patient information with report for a specific case.

**URL Parameters:**
- `case_id` (UUID, required): The unique identifier of the case

**Response Model:** `PatientFullReportResponse`

## Response Structure

The response includes four main sections:

### 1. Patient Information
- `patient_id` (UUID): Unique patient identifier
- `patient_name` (string): Full name of the patient
- `patient_age` (integer, optional): Age in years
- `patient_gender` (string, optional): Gender
- `blood_type` (string, optional): Blood type (A+, B-, O+, AB+, etc.)
- `status` (string, optional): Patient status (stable, improving, critical)
- `underlying_condition` (object, optional): Chronic conditions as JSON
  - Example: `{"hypertension": true, "diabetes": true, "asthma": false}`
- `phone_number` (string, optional): Contact number
- `patient_created_at` (datetime): Patient record creation timestamp

### 2. Case Information
- `case_id` (UUID): Unique case identifier
- `image_path` (string): S3 URL to the X-ray PNG image
- `processed_img_path` (string, optional): S3 URL to the DICOM file
- `case_timestamp` (datetime): When the case was created
- `diagnosis` (string, optional): Medical diagnosis
- `findings` (string, optional): Extended clinical notes

### 3. AI Analysis Results (if available)
- `ai_result_id` (UUID, optional): Unique AI result identifier
- `predicted_diagnosis` (string, optional): AI's predicted diagnosis
- `confident_score` (float, optional): Confidence score
- `bounding_box` (object, optional): Detection bounding boxes
  ```json
  {
    "detections": [
      {
        "bbox": [x1, y1, x2, y2],
        "concept": "Lung Opacity",
        "class_idx": 7,
        "probability": 0.85
      }
    ]
  }
  ```
- `concepts` (object, optional): Detected medical concepts
  ```json
  {
    "top_classes": [
      {
        "prob": 0.85,
        "concepts": "Lung Opacity",
        "class_idx": 7
      }
    ],
    "detected_concepts": ["Lung Opacity", "Pneumonia"]
  }
  ```
- `ai_result_created_at` (datetime, optional): AI analysis timestamp

### 4. Medical Report (if available)
- `report_id` (UUID, optional): Unique report identifier
- `model_report` (object, optional): AI-generated radiology report
  ```json
  {
    "MeSH": "Lung Opacity, Pulmonary fibrosis",
    "Image": "X-ray Chest PA",
    "Indication": "hypertension, diabetes",
    "Comparison": "None",
    "Findings": "The heart size is normal. The mediastinum and aorta are unremarkable...",
    "Impression": "There is evidence of Lung Opacity...",
    "raw_report_text": "MeSH: Lung Opacity..."
  }
  ```
- `doctor_report` (string, optional): Doctor's report/corrections
- `feedback_note` (string, optional): Feedback for model improvement
- `report_created_at` (datetime, optional): Report generation timestamp

## Example Request

```bash
curl -X GET "http://localhost:8000/api/v1/reports/full/01e58e5f-1c29-47c3-8b57-df985b2433c8"
```

## Example Response

```json
{
  "patient_id": "36127dea-feca-46de-ab38-f9dfb7dce969",
  "patient_name": "John Doe",
  "patient_age": 45,
  "patient_gender": "Male",
  "blood_type": "A+",
  "status": "stable",
  "underlying_condition": {
    "hypertension": true,
    "diabetes": true,
    "asthma": false
  },
  "phone_number": "+84912345678",
  "patient_created_at": "2025-01-01T10:00:00Z",
  
  "case_id": "01e58e5f-1c29-47c3-8b57-df985b2433c8",
  "image_path": "https://aithena.s3.ap-southeast-1.amazonaws.com/cases/patient-id.png",
  "processed_img_path": "https://aithena.s3.ap-southeast-1.amazonaws.com/cases/patient-id/image.dicom",
  "case_timestamp": "2025-12-12T10:00:00Z",
  "diagnosis": "Pneumonia",
  "findings": "Infiltrates visible in lower right lung",
  
  "ai_result_id": "ai-result-uuid",
  "predicted_diagnosis": "Pneumonia",
  "confident_score": null,
  "bounding_box": {
    "detections": [
      {
        "bbox": [1814, 1500, 2077, 2021],
        "concept": "Lung Opacity",
        "class_idx": 7,
        "probability": 0.7681617736816406
      }
    ]
  },
  "concepts": {
    "top_classes": [
      {
        "prob": 0.7681617736816406,
        "concepts": "Lung Opacity",
        "class_idx": 7
      }
    ],
    "detected_concepts": ["Lung Opacity"]
  },
  "ai_result_created_at": "2025-12-12T10:05:00Z",
  
  "report_id": "report-uuid",
  "model_report": {
    "MeSH": "Lung Opacity",
    "Image": "X-ray Chest PA",
    "Indication": "hypertension, diabetes",
    "Comparison": "None",
    "Findings": "The heart size is normal. The mediastinum and aorta are unremarkable. There is evidence of Lung Opacity in the lower lung zones.",
    "Impression": "There is evidence of Lung Opacity in the lower lung zones.",
    "raw_report_text": "MeSH: Lung Opacity\nimage: X-ray Chest PA..."
  },
  "doctor_report": null,
  "feedback_note": null,
  "report_created_at": "2025-12-12T10:10:00Z"
}
```

## Error Responses

### 404 Case Not Found
```json
{
  "detail": "Case not found"
}
```

### 404 Patient Not Found
```json
{
  "detail": "Patient not found"
}
```

## Use Cases

1. **Patient Portal**: Display complete patient medical records
2. **Doctor Dashboard**: View comprehensive case information for diagnosis
3. **Medical Report Export**: Generate PDF reports with all data
4. **Mobile App**: Show patient history and current case status
5. **Analytics**: Aggregate patient data for research

## Notes

- AI analysis results and medical report are **optional** - they will be `null` if not yet generated
- To generate AI analysis: Call `POST /api/v1/analysis/cam-inference/{case_id}`
- To generate medical report: Call `POST /api/v1/reports/generate` with `{"case_id": "..."}`
- Underlying conditions are stored as boolean flags in JSONB format
- All timestamps are in ISO 8601 format with timezone information

## Testing

Use the provided test script:
```bash
cd backend
python test_full_patient_report.py
```

The script will:
1. List available cases
2. Fetch comprehensive patient report
3. Display formatted summary with all sections

## Related Endpoints

- `POST /api/v1/analysis/cam-inference/{case_id}` - Run AI analysis
- `POST /api/v1/reports/generate` - Generate medical report
- `GET /api/v1/reports/{case_id}` - Get report only
- `PUT /api/v1/reports/{report_id}/doctor-report` - Update doctor's notes
