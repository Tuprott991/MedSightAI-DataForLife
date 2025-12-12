"""
Pydantic schemas for AI Result
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID


class BoundingBox(BaseModel):
    """Bounding box schema"""
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: Optional[float] = None


class AIResultBase(BaseModel):
    """Base AI result schema"""
    case_id: UUID = Field(..., description="Case ID")
    predicted_diagnosis: Optional[str] = None
    confident_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    bounding_box: Optional[Dict[str, Any]] = Field(
        None, 
        description="Bounding box data: {detections: [{bbox, concept, class_idx, probability}]}"
    )
    concepts: Optional[Dict[str, Any]] = Field(
        None, 
        description="Concepts data: {top_classes: [{prob, concepts, class_idx}], detected_concepts: [...]}"
    )


class AIResultCreate(AIResultBase):
    """Schema for creating AI result"""
    pass


class AIResultUpdate(BaseModel):
    """Schema for updating AI result (for corrections)"""
    predicted_diagnosis: Optional[str] = None
    confident_score: Optional[float] = None
    bounding_box: Optional[Dict[str, Any]] = None
    concepts: Optional[Dict[str, Any]] = None


class AIResultResponse(AIResultBase):
    """Schema for AI result response"""
    id: UUID
    created_at: datetime
    concepts: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class AIAnalysisRequest(BaseModel):
    """Schema for requesting AI analysis"""
    case_id: UUID
    include_heatmap: bool = Field(default=True, description="Include Grad-CAM heatmap")
    include_concepts: bool = Field(default=True, description="Include concept-based analysis")


class AIAnalysisResponse(BaseModel):
    """Schema for complete AI analysis response"""
    case_id: UUID
    ai_result: AIResultResponse
    heatmap_path: Optional[str] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    similar_cases: Optional[List[str]] = None
    similarity_scores: Optional[List[float]] = None
