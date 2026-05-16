"""
Application settings and configuration management
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"


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
    ZILLIZ_CLOUD_API_KEY: Optional[str] = None
    ZILLIZ_COLLECTION_NAME: str = "med_vector"
    ZILLIZ_IMG_DIMENSION: int = 1024
    ZILLIZ_VECTOR_FIELD_NAME: str = "img_emb"
    ZILLIZ_PRIMARY_FIELD_NAME: str = "primary_key"
    ZILLIZ_AUTO_ID: bool = False
    ZILLIZ_IMAGE_PATH_FIELD_NAME: str = "image_path"
    ZILLIZ_LABEL_FIELD_NAME: str = "label"
    
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
    RETRIEVAL_MODEL_PATH: str = str(
        BACKEND_DIR.parent / "retrieval_model" / "covid_convnextv2_seed_0_epoch_16_backbone.onnx"
    )
    SALIENCY_MODEL_CODE_PATH: str = str(
        BACKEND_DIR.parent.parent / "Image-Retrieval---Thesis-2026"
    )
    SALIENCY_MODEL_WEIGHTS_PATH: str = str(
        BACKEND_DIR.parent.parent / "Image-Retrieval---Thesis-2026" / "model.pth"
    )
    RETRIEVAL_RESIZE_SIZE: int = 384
    RETRIEVAL_IMAGE_SIZE: int = 384
    MEDGEMMA_MODEL_ID: str = "unsloth/medgemma-1.5-4b-it-bnb-4bit"
    MEDGEMMA_DEVICE: str = "cuda"
    MEDGEMMA_MAX_NEW_TOKENS: int = 900
    HF_TOKEN: Optional[str] = None

    # OpenAI GPT service
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_LLM_MODEL: str = "gpt-5.4-mini"
    OPENAI_TIMEOUT_SECONDS: int = 90
    OPENAI_MAX_RETRIES: int = 2
    OPENAI_CHAT_MAX_OUTPUT_TOKENS: int = 900
    OPENAI_REPORT_MAX_OUTPUT_TOKENS: int = 1800
    
    # Model Inference Service
    MODEL_INFERENCE_URL: str = "http://localhost:8001"  # URL of model_inference FastAPI service
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".dcm"}

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod", "false", "0", "no", "off"}:
                return False
            if normalized in {"debug", "development", "dev", "true", "1", "yes", "on"}:
                return True
        return value
    
    class Config:
        env_file = str(ENV_FILE)
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()
