"""
Health check and utility endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config.database import get_db
from app.config.settings import settings
from app.config.zilliz import get_zilliz_headers
from app.schemas import HealthResponse, MessageResponse
from app.services import s3_service, zilliz_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Check health status of all services"""
    
    # Check database
    try:
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check S3
    try:
        s3_service.client.list_buckets()
        s3_status = "healthy"
    except Exception as e:
        s3_status = f"unhealthy: {str(e)}"
    
    # Check Zilliz Cloud / Milvus
    try:
        import requests
        url = f"{settings.ZILLIZ_CLOUD_URI}/v2/vectordb/collections/list"
        response = requests.post(url, headers=get_zilliz_headers(), json={}, timeout=5)
        if response.status_code == 200 and response.json().get("code") == 0:
            zilliz_status = "healthy"
        else:
            zilliz_status = f"unhealthy: {response.text}"
    except Exception as e:
        zilliz_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "database": db_status,
        "s3": s3_status,
        "zilliz": zilliz_status
    }


@router.get("/version")
async def get_version():
    """Get API version"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MedSight AI API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }
