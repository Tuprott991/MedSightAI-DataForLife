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
