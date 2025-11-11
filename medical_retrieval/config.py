"""
Configuration file for Medical Multimodal Retrieval System
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "indiana"
INDEX_DIR = BASE_DIR / "indexes"
CACHE_DIR = BASE_DIR / "cache"

# Create directories
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    # Choose one of the following models:
    
    # Option 1: PubMed CLIP (Fast, good for general medical)
    "pubmed_clip": {
        "model_name": "flaviagiammarino/pubmed-clip-vit-base-patch32",
        "model_type": "clip",
        "image_size": 224,
        "embedding_dim": 512,
        "batch_size": 32,
        "device": "cuda"  # or "cpu"
    },
    
    # Option 2: BiomedCLIP (Best for medical, recommended)
    "biomedclip": {
        "model_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "model_type": "clip",
        "image_size": 224,
        "embedding_dim": 512,
        "batch_size": 32,
        "device": "cuda"
    },
    
    # Option 3: SigLIP Base (For fine-tuning on medical data)
    "siglip": {
        "model_name": "google/siglip-base-patch16-224",
        "model_type": "siglip",
        "image_size": 224,
        "embedding_dim": 768,  # SigLIP uses 768-dim
        "batch_size": 32,
        "device": "cuda"
    },
    
    # Option 4: MedSigLIP-448 (BEST - Fine-tuned on VinDR-CXR, RECOMMENDED)
    "medsiglip": {
        "model_name": "aysangh/medsiglip-448-vindr-bin",
        "model_type": "siglip",
        "image_size": 448,  # Higher resolution = better quality
        "embedding_dim": 1152,  # Large SigLIP dimension
        "batch_size": 16,  # Smaller batch due to larger image size
        "device": "cuda"
    },
    
    # Active model (change this to switch models)
    "active_model": "medsiglip"  # Options: pubmed_clip, biomedclip, siglip, medsiglip
}

# Index configuration
INDEX_CONFIG = {
    "type": "IndexFlatL2",  # Exact search for medical accuracy
    "dimension": 1152,  # Will be overridden by model's embedding_dim
    "metric": "L2",
    "use_gpu": True,  # Set to False if no GPU available
    "gpu_id": 0
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    "top_k_initial": 100,  # Initial retrieval
    "top_k_final": 10,     # After reranking
    "similarity_threshold": 0.7,  # Minimum similarity score
    "enable_reranking": True
}

# Reranking weights for Indiana dataset
RERANKING_WEIGHTS = {
    "visual_similarity": 0.30,      # Image embedding similarity
    "findings_similarity": 0.25,    # Findings text match
    "impression_similarity": 0.20,  # Impression text match
    "mesh_overlap": 0.15,           # MeSH terms overlap
    "problems_overlap": 0.10        # Problems list overlap
}

# Database configuration
DB_CONFIG = {
    "type": "sqlite",  # Can switch to PostgreSQL for production
    "path": str(INDEX_DIR / "metadata.db"),
    "cache_embeddings": True
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "timeout": 30,
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "cors_origins": ["*"]
}

# Indiana dataset specific
INDIANA_CONFIG = {
    "csv_path": "Indiana_reports.csv",  # Adjust path
    "images_dir": "Indiana_images",     # Adjust path
    "projection_csv": "Indiana_projection.csv",
    "fields": {
        "uid": "uid",
        "mesh": "MeSH",
        "problems": "Problems", 
        "findings": "findings",
        "impression": "impression",
        "indication": "indication",
        "comparison": "comparison"
    }
}

# Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(BASE_DIR / "logs" / "app.log")
}
