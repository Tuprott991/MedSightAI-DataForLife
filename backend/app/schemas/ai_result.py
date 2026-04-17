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


# ---------------------------------------------------------------------------
# YOLO Localization Schemas
# ---------------------------------------------------------------------------

class DetectionItem(BaseModel):
    """A single lesion detection from YOLO inference."""
    class_id: int
    class_name_en: str = Field(..., description="English class name (drawn on image)")
    class_name_vi: str = Field(..., description="Vietnamese class name (shown in UI)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    x1: int
    y1: int
    x2: int
    y2: int


class LocalizeResponse(BaseModel):
    """
    Response for POST /analysis/localize/{case_id}.

    Fresh inference:  annotated_image_b64 is populated, annotated_image_url is None.
    Cached result:    annotated_image_url is populated (S3 URL), annotated_image_b64 is None.
    """
    case_id: UUID
    detections: List[DetectionItem]
    annotated_image_url: Optional[str] = Field(
        None, description="S3 public URL of annotated image (available after background persist)"
    )
    annotated_image_b64: Optional[str] = Field(
        None, description="Base64-encoded JPEG of annotated image (fresh inference only)"
    )
    from_cache: bool = Field(False, description="True when served from DB + S3 cache")
    total_lesions: int = Field(0, description="Number of detected lesions")


class SimilarityCamResponse(BaseModel):
    """Response for saliency-based comparison between a query case and a retrieved case."""
    query_case_id: UUID
    similar_case_id: UUID
    method: str = Field(..., description="Saliency method used to generate the overlays")
    image_size: int = Field(..., description="Spatial size used for saliency inference")
    query_overlay_b64: str = Field(..., description="Base64-encoded PNG saliency overlay for the query image")
    similar_overlay_b64: str = Field(..., description="Base64-encoded PNG saliency overlay for the similar image")

