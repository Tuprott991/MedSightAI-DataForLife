"""
Common schemas and utilities
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    detail: Optional[Any] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database: str
    s3: str
    milvus: str


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = 1
    page_size: int = 20


class ImageUploadResponse(BaseModel):
    """Image upload response"""
    file_path: str
    file_size: int
    content_type: str


class SimilaritySearchRequest(BaseModel):
    """Similarity search request"""
    case_id: Optional[str] = None
    image_path: Optional[str] = None
    top_k: int = Field(5, ge=1, le=50)


class SimilarCaseDetail(BaseModel):
    """Display-ready similar case payload for frontend consumption."""
    case_id: str
    patient_id: str
    patient_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    status: Optional[str] = None
    diagnosis: Optional[str] = None
    image_path: Optional[str] = None
    processed_img_path: Optional[str] = None
    timestamp: Optional[datetime] = None


class SimilaritySearchResponse(BaseModel):
    """Similarity search response"""
    similar_case_ids: list[str]
    similarity_scores: list[float]
    case_details: list[SimilarCaseDetail] = Field(default_factory=list)
