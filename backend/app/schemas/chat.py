"""
Pydantic schemas for Chat (Education Mode)
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List, Literal
from datetime import datetime
from uuid import UUID


class ChatSessionCreate(BaseModel):
    """Schema for creating a chat session"""
    user_id: Optional[UUID] = Field(None, description="Student/user ID")
    case_id: Optional[UUID] = Field(None, description="Case ID for the current image")
    session_type: Literal["practice", "tutoring"] = "tutoring"


class ChatSessionResponse(BaseModel):
    """Schema for chat session response"""
    id: UUID
    user_id: UUID
    case_id: Optional[UUID]
    session_type: str
    score: Optional[float] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ChatMessageCreate(BaseModel):
    """Schema for creating a chat message"""
    session_id: UUID
    sender: Literal["ai", "user"]
    message: str


class ChatMessageResponse(BaseModel):
    """Schema for chat message response"""
    id: UUID
    session_id: UUID
    sender: str
    message: str
    timestamp: datetime
    
    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    """Schema for chat history"""
    session: ChatSessionResponse
    messages: List[ChatMessageResponse]


class ChatMessageRequest(BaseModel):
    """Request body for a MedGemma-backed chat turn."""
    message: str = Field(..., min_length=1, description="User message")
    image_url: Optional[str] = Field(None, description="Current image URL shown in the UI")
    mode: Literal["doctor", "student"] = "student"
    patient_context: Optional[Dict[str, Any]] = None
    current_annotations: Optional[List[Dict[str, Any]]] = None
    submitted_diagnosis: Optional[str] = None


class ChatTurnResponse(BaseModel):
    """Response for one persisted user/assistant chat turn."""
    session: ChatSessionResponse
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse


class ChatSessionResolveRequest(BaseModel):
    """Find or create an active session for a user and case."""
    user_id: Optional[UUID] = None
    case_id: Optional[UUID] = None
    session_type: Literal["practice", "tutoring"] = "tutoring"


class StudentSubmission(BaseModel):
    """Schema for student diagnosis submission"""
    session_id: UUID
    diagnosis: str
    bounding_boxes: List[dict]
    confidence: Optional[float] = None


class StudentScoreResponse(BaseModel):
    """Schema for student scoring response"""
    total_score: float
    bbox_accuracy: float
    diagnosis_accuracy: float
    explanation: str
    correct_answer: dict
    heatmap_path: Optional[str] = None
