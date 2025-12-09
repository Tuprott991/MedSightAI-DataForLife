"""
Pydantic schemas for Patient
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID


class PatientBase(BaseModel):
    """Base patient schema"""
    name: str = Field(..., description="Patient name")
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    history: Optional[Dict[str, Any]] = Field(None, description="Medical history with dates {date: {diagnosis, findings}}")
    blood_type: Optional[str] = Field(None, description="Blood type (A+, B-, O+, AB+, etc.)")
    status: Optional[str] = Field(None, description="Patient status: stable, improving, or critical")
    underlying_condition: Optional[Dict[str, Any]] = Field(None, description="Chronic or underlying conditions as JSON")


class PatientCreate(PatientBase):
    """Schema for creating a patient"""
    pass


class PatientUpdate(BaseModel):
    """Schema for updating a patient"""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    history: Optional[Dict[str, Any]] = None
    blood_type: Optional[str] = None
    status: Optional[str] = None
    underlying_condition: Optional[Dict[str, Any]] = None


class PatientResponse(PatientBase):
    """Schema for patient response"""
    id: UUID
    created_at: datetime
    blood_type: Optional[str] = None
    status: Optional[str] = None
    underlying_condition: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class PatientListResponse(BaseModel):
    """Schema for paginated patient list"""
    total: int
    page: int
    page_size: int
    patients: list[PatientResponse]


class LatestCaseInfo(BaseModel):
    """Schema for latest case information"""
    image_path: Optional[str] = None
    processed_img_path: Optional[str] = None
    timestamp: Optional[datetime] = None
    similar_cases: Optional[List[str]] = None
    similarity_scores: Optional[List[float]] = None
    diagnosis: Optional[str] = None
    findings: Optional[str] = None


class PatientInforResponse(BaseModel):
    """Schema for comprehensive patient information with latest case"""
    # Patient information
    id: UUID
    name: str
    age: Optional[int]
    gender: Optional[str]
    history: Optional[Dict[str, Any]]
    created_at: datetime
    blood_type: Optional[str]
    status: Optional[str]
    underlying_condition: Optional[Dict[str, Any]]
    
    # Latest case information
    latest_case: Optional[LatestCaseInfo] = None
    
    class Config:
        from_attributes = True
