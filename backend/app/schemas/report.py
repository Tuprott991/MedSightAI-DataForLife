"""
Pydantic schemas for Report
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class ReportBase(BaseModel):
    """Base report schema"""
    case_id: UUID = Field(..., description="Case ID")


class ReportCreate(ReportBase):
    """Schema for creating a report"""
    model_report: Optional[Dict[str, Any]] = None
    doctor_report: Optional[str] = None
    feedback_note: Optional[str] = None


class ReportUpdate(BaseModel):
    """Schema for updating a report"""
    model_report: Optional[Dict[str, Any]] = None
    doctor_report: Optional[str] = None
    feedback_note: Optional[str] = None


class ReportResponse(ReportBase):
    """Schema for report response"""
    id: UUID
    model_report: Optional[Dict[str, Any]]
    doctor_report: Optional[str]
    feedback_note: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ReportGenerationRequest(BaseModel):
    """Schema for generating report from AI"""
    case_id: UUID
    patient_history: Optional[dict] = None
    ai_findings: Optional[dict] = None


class PatientFullReportResponse(BaseModel):
    """
    Comprehensive schema for patient information with report
    Includes: Patient info, Case details, AI results, and Generated report
    """
    # Patient Information
    patient_id: UUID
    patient_name: str
    patient_age: Optional[int]
    patient_gender: Optional[str]
    blood_type: Optional[str]
    status: Optional[str]
    underlying_condition: Optional[Dict[str, Any]]
    phone_number: Optional[str]
    patient_created_at: datetime
    
    # Case Information
    case_id: UUID
    image_path: str
    processed_img_path: Optional[str]
    case_timestamp: datetime
    diagnosis: Optional[str]
    findings: Optional[str]
    
    # AI Result Information
    ai_result_id: Optional[UUID]
    predicted_diagnosis: Optional[str]
    confident_score: Optional[float]
    bounding_box: Optional[Dict[str, Any]]
    concepts: Optional[Dict[str, Any]]
    ai_result_created_at: Optional[datetime]
    
    # Report Information
    report_id: Optional[UUID]
    model_report: Optional[Dict[str, Any]]
    doctor_report: Optional[str]
    feedback_note: Optional[str]
    report_created_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "patient_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "patient_name": "John Doe",
                "patient_age": 45,
                "patient_gender": "Male",
                "blood_type": "A+",
                "status": "stable",
                "underlying_condition": {"hypertension": True, "diabetes": False},
                "phone_number": "+84912345678",
                "patient_created_at": "2025-01-01T10:00:00Z",
                "case_id": "4fa85f64-5717-4562-b3fc-2c963f66afa7",
                "image_path": "https://bucket.s3.region.amazonaws.com/cases/patient-id.png",
                "processed_img_path": "https://bucket.s3.region.amazonaws.com/cases/patient-id/image.dicom",
                "case_timestamp": "2025-12-12T10:00:00Z",
                "diagnosis": "Pneumonia",
                "findings": "Infiltrates visible in lower right lung",
                "ai_result_id": "5fa85f64-5717-4562-b3fc-2c963f66afa8",
                "predicted_diagnosis": "Pneumonia",
                "confident_score": None,
                "bounding_box": {
                    "detections": [
                        {
                            "bbox": [100, 200, 300, 400],
                            "concept": "Lung Opacity",
                            "class_idx": 7,
                            "probability": 0.85
                        }
                    ]
                },
                "concepts": {
                    "top_classes": [
                        {
                            "prob": 0.85,
                            "concepts": "Lung Opacity",
                            "class_idx": 7
                        }
                    ],
                    "detected_concepts": ["Lung Opacity"]
                },
                "ai_result_created_at": "2025-12-12T10:05:00Z",
                "report_id": "6fa85f64-5717-4562-b3fc-2c963f66afa9",
                "model_report": {
                    "MeSH": "Lung Opacity",
                    "Image": "X-ray Chest PA",
                    "Indication": "hypertension",
                    "Comparison": "None",
                    "Findings": "Evidence of Lung Opacity in lower lung zones.",
                    "Impression": "Lung Opacity detected."
                },
                "doctor_report": None,
                "feedback_note": None,
                "report_created_at": "2025-12-12T10:10:00Z"
            }
        }
