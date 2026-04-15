"""
Pydantic schemas module
"""
from app.schemas.patient import (
    PatientBase, PatientCreate, PatientUpdate, 
    PatientResponse, PatientListResponse,
    PatientInforResponse, LatestCaseInfo, PatientInforListResponse
)
from app.schemas.case import (
    CaseBase, CaseCreate, CaseUpdate, 
    CaseResponse, CaseWithPatientResponse, CaseListResponse
)
from app.schemas.ai_result import (
    BoundingBox, AIResultBase, AIResultCreate, AIResultUpdate,
    AIResultResponse, AIAnalysisRequest, AIAnalysisResponse,
    DetectionItem, LocalizeResponse
)
from app.schemas.report import (
    ReportBase, ReportCreate, ReportUpdate, 
    ReportResponse, ReportGenerationRequest, PatientFullReportResponse
)
from app.schemas.chat import (
    ChatSessionCreate, ChatSessionResponse,
    ChatMessageCreate, ChatMessageResponse,
    ChatHistoryResponse, ChatMessageRequest, ChatTurnResponse,
    ChatSessionResolveRequest, StudentSubmission, StudentScoreResponse
)
from app.schemas.common import (
    MessageResponse, HealthResponse, PaginationParams,
    ImageUploadResponse, SimilaritySearchRequest, SimilaritySearchResponse
)

__all__ = [
    # Patient
    "PatientBase", "PatientCreate", "PatientUpdate",
    "PatientResponse", "PatientListResponse",
    "PatientInforResponse", "LatestCaseInfo",
    # Case
    "CaseBase", "CaseCreate", "CaseUpdate",
    "CaseResponse", "CaseWithPatientResponse", "CaseListResponse",
    # AI Result
    "BoundingBox", "AIResultBase", "AIResultCreate", "AIResultUpdate",
    "AIResultResponse", "AIAnalysisRequest", "AIAnalysisResponse",
    "DetectionItem", "LocalizeResponse",
    # Report
    "ReportBase", "ReportCreate", "ReportUpdate",
    "ReportResponse", "ReportGenerationRequest", "PatientFullReportResponse",
    # Chat
    "ChatSessionCreate", "ChatSessionResponse",
    "ChatMessageCreate", "ChatMessageResponse",
    "ChatHistoryResponse", "ChatMessageRequest", "ChatTurnResponse",
    "ChatSessionResolveRequest", "StudentSubmission", "StudentScoreResponse",
    # Common
    "MessageResponse", "HealthResponse", "PaginationParams",
    "ImageUploadResponse", "SimilaritySearchRequest", "SimilaritySearchResponse"
]
