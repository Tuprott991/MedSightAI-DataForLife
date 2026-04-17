"""
Zilliz Cloud Configuration
"""
from app.config.settings import settings


def get_zilliz_headers():
    """Get headers for Zilliz Cloud or Milvus REST API requests."""
    headers = {
        "Content-Type": "application/json",
    }
    if settings.ZILLIZ_CLOUD_API_KEY:
        headers["Authorization"] = f"Bearer {settings.ZILLIZ_CLOUD_API_KEY}"
    return headers


def get_zilliz_config():
    """Get Zilliz Cloud configuration"""
    return {
        "base_url": settings.ZILLIZ_CLOUD_URI,
        "collection_name": settings.ZILLIZ_COLLECTION_NAME,
        "img_dimension": settings.ZILLIZ_IMG_DIMENSION
    }
