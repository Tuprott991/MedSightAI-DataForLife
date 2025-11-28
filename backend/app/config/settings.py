"""
Application settings and configuration management
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # Application
    APP_NAME: str = "MedSight AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database - PostgreSQL (Neon)
    DATABASE_URL: str
    DB_ECHO: bool = False
    
    # AWS S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str
    
    # S3 Folder Structure
    # Patient folders
    S3_PATIENTS_PREFIX: str = "patients/"
    
    # Case folders
    S3_CASES_PREFIX: str = "cases/"
    S3_ORIGINAL_IMAGES_PREFIX: str = "original/"
    S3_PROCESSED_IMAGES_PREFIX: str = "processed/"
    S3_ANNOTATED_IMAGES_PREFIX: str = "annotated/"
    S3_SEGMENTATION_PREFIX: str = "segmentation/"
    S3_REPORTS_PREFIX: str = "reports/"
    
    # Education mode folders
    S3_EDUCATION_PREFIX: str = "education/"
    S3_STUDENT_UPLOADS_PREFIX: str = "student_uploads/"
    S3_STUDENT_ANNOTATIONS_PREFIX: str = "student_annotations/"
    S3_FEEDBACK_PREFIX: str = "feedback/"
    
    # Similar cases
    S3_SIMILAR_CASES_PREFIX: str = "similar_cases/"
    S3_THUMBNAILS_PREFIX: str = "thumbnails/"
    
    # Temporary and exports
    S3_TEMP_PREFIX: str = "temp/uploads/"
    S3_EXPORTS_PREFIX: str = "exports/"
    
    # Zilliz Cloud Vector Database
    ZILLIZ_CLOUD_URI: str
    ZILLIZ_CLOUD_API_KEY: str
    ZILLIZ_COLLECTION_NAME: str = "med_vector"
    ZILLIZ_TXT_DIMENSION: int = 1152
    ZILLIZ_IMG_DIMENSION: int = 1152
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI Model Paths (paths to other developers' code)
    MODEL_INFERENCE_PATH: str = "../MedSightAI"
    MEDGEMMA_PATH: str = "../medgemma"
    VINDR_DATASET_PATH: str = "../VindrDataset"
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".dcm"}
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
