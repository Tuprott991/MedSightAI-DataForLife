"""
API v1 endpoints module
"""
from fastapi import APIRouter
from app.api.v1.endpoints import (
    patients, cases, analysis, reports,
    education, similarity, health, disease_detection
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(patients.router, prefix="/patients", tags=["Patients"])
api_router.include_router(cases.router, prefix="/cases", tags=["Cases"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["AI Analysis"])
api_router.include_router(disease_detection.router, prefix="/disease-detection", tags=["Disease Detection"])
api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])
api_router.include_router(education.router, prefix="/education", tags=["Education Mode"])
api_router.include_router(similarity.router, prefix="/similarity", tags=["Similarity Search"])
