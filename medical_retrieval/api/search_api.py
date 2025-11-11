"""
FastAPI endpoint for medical image retrieval
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from PIL import Image
import io
import numpy as np
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import MedCLIPEncoder
from models.reranker import MedicalReranker
from indexing.faiss_index import FAISSIndex
from indexing.database import MetadataDatabase
from data.preprocessor import TextPreprocessor
from config import (
    MODEL_CONFIG, INDEX_CONFIG, RETRIEVAL_CONFIG,
    RERANKING_WEIGHTS, API_CONFIG, INDEX_DIR
)

# Import model selector
import sys
sys.path.append(str(Path(__file__).parent.parent))
from model_selector import get_active_model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Multimodal Retrieval API",
    description="API for searching medical images using text and image queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (loaded once)
encoder = None
image_index = None
text_index = None
database = None
reranker = None
text_preprocessor = None


class SearchRequest(BaseModel):
    """Search request model"""
    query_text: Optional[str] = None
    top_k: int = 10
    enable_reranking: bool = True
    mesh_terms: Optional[List[str]] = None
    problems: Optional[List[str]] = None


class SearchResult(BaseModel):
    """Search result model"""
    uid: str
    filename: str
    image_path: str
    score: float
    findings: str
    impression: str
    mesh: List[str]
    problems: List[str]
    projection: str


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult]
    query_info: Dict
    total_results: int
    processing_time_ms: float


@app.on_event("startup")
async def startup_event():
    """Load models and indexes on startup"""
    global encoder, image_index, text_index, database, reranker, text_preprocessor
    
    logger.info("Loading models and indexes...")
    
    try:
        # Load encoder
        model_config = get_active_model_config()
        logger.info(f"Loading model: {model_config['active_model']}")
        logger.info(f"Model name: {model_config['model_name']}")
        
        encoder = MedCLIPEncoder(
            model_name=model_config['model_name'],
            model_type=model_config.get('model_type', 'clip'),
            device=model_config['device']
        )
        
        # Load indexes
        embedding_dim = encoder.get_embedding_dim()
        
        image_index = FAISSIndex(
            dimension=embedding_dim,
            index_type=INDEX_CONFIG['type'],
            use_gpu=INDEX_CONFIG['use_gpu'],
            gpu_id=INDEX_CONFIG['gpu_id']
        )
        image_index.load(str(INDEX_DIR / "image_index.faiss"))
        
        text_index = FAISSIndex(
            dimension=embedding_dim,
            index_type=INDEX_CONFIG['type'],
            use_gpu=INDEX_CONFIG['use_gpu'],
            gpu_id=INDEX_CONFIG['gpu_id']
        )
        text_index.load(str(INDEX_DIR / "text_index.faiss"))
        
        # Load database
        database = MetadataDatabase(str(INDEX_DIR / "metadata.db"))
        
        # Initialize reranker
        reranker = MedicalReranker(weights=RERANKING_WEIGHTS)
        
        # Initialize text preprocessor
        text_preprocessor = TextPreprocessor()
        
        logger.info("✓ Models and indexes loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Multimodal Retrieval API",
        "version": "1.0.0",
        "endpoints": {
            "/search": "Search by text or image",
            "/search/text": "Search by text only",
            "/search/image": "Search by image only",
            "/search/multimodal": "Search by both text and image",
            "/stats": "Get database statistics",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "encoder_loaded": encoder is not None,
        "indexes_loaded": image_index is not None and text_index is not None,
        "database_loaded": database is not None,
        "total_images": database.count() if database else 0
    }


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    if database is None:
        raise HTTPException(status_code=500, detail="Database not loaded")
    
    stats = database.get_statistics()
    stats['image_index_size'] = image_index.index.ntotal
    stats['text_index_size'] = text_index.index.ntotal
    
    return stats


@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(request: SearchRequest):
    """
    Search by text query only
    
    Args:
        request: Search request with query text
        
    Returns:
        Search results
    """
    import time
    start_time = time.time()
    
    if not request.query_text:
        raise HTTPException(status_code=400, detail="query_text is required")
    
    try:
        # Encode text query
        query_embedding = encoder.encode_text(request.query_text, normalize=True)[0]
        
        # Search in text index
        top_k_retrieval = RETRIEVAL_CONFIG['top_k_initial'] if request.enable_reranking else request.top_k
        result_ids, distances = text_index.search(
            query_embedding.reshape(1, -1),
            k=top_k_retrieval
        )
        
        # Get candidates from database
        candidates = database.get_many(result_ids[0])
        
        # Reranking
        if request.enable_reranking and len(candidates) > 0:
            # Encode all candidate texts
            candidate_embeddings = []
            for candidate in candidates:
                combined_text = text_preprocessor.combine_clinical_text(
                    findings=candidate['findings'],
                    impression=candidate['impression']
                )
                txt_emb = encoder.encode_text(combined_text, normalize=True)[0]
                
                candidate_embeddings.append({
                    'findings': txt_emb,
                    'impression': txt_emb,
                    'image': None
                })
            
            query_emb_dict = {
                'findings': query_embedding,
                'impression': query_embedding,
                'image': None
            }
            
            query_dict = {
                'mesh': request.mesh_terms or [],
                'problems': request.problems or []
            }
            
            ranked = reranker.rerank(query_dict, candidates, query_emb_dict, candidate_embeddings)
            candidates = [c for c, _ in ranked[:request.top_k]]
            scores = [s for _, s in ranked[:request.top_k]]
        else:
            scores = [1.0 / (1.0 + d) for d in distances[0][:request.top_k]]
        
        # Format results
        results = []
        for candidate, score in zip(candidates, scores):
            results.append(SearchResult(
                uid=candidate['uid'],
                filename=candidate['filename'],
                image_path=candidate['image_path'],
                score=float(score),
                findings=candidate['findings'] or "",
                impression=candidate['impression'] or "",
                mesh=candidate['mesh'],
                problems=candidate['problems'],
                projection=candidate['projection'] or ""
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=results,
            query_info={
                "query_text": request.query_text,
                "query_type": "text_only"
            },
            total_results=len(results),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    image: UploadFile = File(...),
    top_k: int = Form(10),
    enable_reranking: bool = Form(True)
):
    """
    Search by image query only
    
    Args:
        image: Uploaded image file
        top_k: Number of results to return
        enable_reranking: Whether to enable reranking
        
    Returns:
        Search results
    """
    import time
    start_time = time.time()
    
    try:
        # Load image
        image_bytes = await image.read()
        query_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Encode image query
        query_embedding = encoder.encode_image(query_image, normalize=True)[0]
        
        # Search in image index
        top_k_retrieval = RETRIEVAL_CONFIG['top_k_initial'] if enable_reranking else top_k
        result_ids, distances = image_index.search(
            query_embedding.reshape(1, -1),
            k=top_k_retrieval
        )
        
        # Get candidates from database
        candidates = database.get_many(result_ids[0])
        
        if enable_reranking and len(candidates) > 0:
            # Simple reranking by visual similarity
            scores = [1.0 / (1.0 + d) for d in distances[0]]
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            candidates = [c for c, _ in scored_candidates[:top_k]]
            scores = [s for _, s in scored_candidates[:top_k]]
        else:
            scores = [1.0 / (1.0 + d) for d in distances[0][:top_k]]
        
        # Format results
        results = []
        for candidate, score in zip(candidates, scores):
            results.append(SearchResult(
                uid=candidate['uid'],
                filename=candidate['filename'],
                image_path=candidate['image_path'],
                score=float(score),
                findings=candidate['findings'] or "",
                impression=candidate['impression'] or "",
                mesh=candidate['mesh'],
                problems=candidate['problems'],
                projection=candidate['projection'] or ""
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=results,
            query_info={
                "query_type": "image_only",
                "image_filename": image.filename
            },
            total_results=len(results),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in image search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/multimodal", response_model=SearchResponse)
async def search_multimodal(
    image: UploadFile = File(...),
    query_text: str = Form(...),
    top_k: int = Form(10),
    enable_reranking: bool = Form(True),
    image_weight: float = Form(0.5),
    text_weight: float = Form(0.5)
):
    """
    Search by both image and text
    
    Args:
        image: Uploaded image file
        query_text: Text query
        top_k: Number of results
        enable_reranking: Whether to enable reranking
        image_weight: Weight for image similarity
        text_weight: Weight for text similarity
        
    Returns:
        Search results
    """
    import time
    start_time = time.time()
    
    try:
        # Load and encode image
        image_bytes = await image.read()
        query_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_embedding = encoder.encode_image(query_image, normalize=True)[0]
        
        # Encode text
        text_embedding = encoder.encode_text(query_text, normalize=True)[0]
        
        # Search both indexes
        top_k_retrieval = RETRIEVAL_CONFIG['top_k_initial'] if enable_reranking else top_k
        
        img_ids, img_dists = image_index.search(image_embedding.reshape(1, -1), k=top_k_retrieval)
        txt_ids, txt_dists = text_index.search(text_embedding.reshape(1, -1), k=top_k_retrieval)
        
        # Combine results
        combined_scores = {}
        for uid, dist in zip(img_ids[0], img_dists[0]):
            combined_scores[uid] = image_weight * (1.0 / (1.0 + dist))
        
        for uid, dist in zip(txt_ids[0], txt_dists[0]):
            score = text_weight * (1.0 / (1.0 + dist))
            if uid in combined_scores:
                combined_scores[uid] += score
            else:
                combined_scores[uid] = score
        
        # Get top candidates
        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_uids = [uid for uid, _ in sorted_candidates[:top_k]]
        
        candidates = database.get_many(top_uids)
        scores = [combined_scores[c['uid']] for c in candidates]
        
        # Format results
        results = []
        for candidate, score in zip(candidates, scores):
            results.append(SearchResult(
                uid=candidate['uid'],
                filename=candidate['filename'],
                image_path=candidate['image_path'],
                score=float(score),
                findings=candidate['findings'] or "",
                impression=candidate['impression'] or "",
                mesh=candidate['mesh'],
                problems=candidate['problems'],
                projection=candidate['projection'] or ""
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=results,
            query_info={
                "query_text": query_text,
                "query_type": "multimodal",
                "image_filename": image.filename,
                "image_weight": image_weight,
                "text_weight": text_weight
            },
            total_results=len(results),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in multimodal search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        workers=1,  # Use 1 worker to avoid loading models multiple times
        log_level="info"
    )
