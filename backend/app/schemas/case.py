"""
Pydantic schemas for Case
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class CaseBase(BaseModel):
    """Base case schema"""
    patient_id: UUID = Field(..., description="Patient ID")


class CaseCreate(CaseBase):
    """Schema for creating a case"""
    image_path: str = Field(..., description="Path to original image in S3")
    processed_img_path: Optional[str] = Field(None, description="Path to processed image in S3")
    similar_cases: Optional[List[str]] = Field(None, description="List of similar case IDs")
    similarity_scores: Optional[List[float]] = Field(None, description="Similarity scores for similar cases")
    diagnosis: Optional[str] = Field(None, description="Disease diagnosis (e.g., Pneumonia, TB)")
    findings: Optional[str] = Field(None, description="Extended notes and findings")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "image_path": "cases/patient-id/original/xray_001.jpg",
                "processed_img_path": "cases/patient-id/processed/xray_001.jpg",
                "similar_cases": ["case-id-1", "case-id-2"],
                "similarity_scores": [0.95, 0.87]
            }
        }


class CaseUpdate(BaseModel):
    """Schema for updating a case"""
    processed_img_path: Optional[str] = None
    similar_cases: Optional[List[str]] = None
    similarity_scores: Optional[List[float]] = None
    diagnosis: Optional[str] = None
    findings: Optional[str] = None


class CaseResponse(CaseBase):
    """Schema for case response"""
    id: UUID
    image_path: str
    processed_img_path: Optional[str]
    timestamp: datetime
    similar_cases: Optional[List[str]]
    similarity_scores: Optional[List[float]]
    diagnosis: Optional[str]
    findings: Optional[str]
    
    class Config:
        from_attributes = True


class CaseWithPatientResponse(CaseResponse):
    """Schema for case response with patient info"""
    patient_name: str
    patient_age: Optional[int]
    patient_gender: Optional[str]


class CaseListResponse(BaseModel):
    """Schema for paginated case list"""
    total: int
    page: int
    page_size: int
    cases: List[CaseResponse]
