"""
Similarity Search API endpoints for image-only CBIR retrieval.
"""
from __future__ import annotations

import hashlib
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import case as crud_case
from app.schemas import SimilaritySearchRequest, SimilaritySearchResponse
from app.services import retrieval_embedding_service, s3_service, zilliz_service

router = APIRouter()


def _extract_s3_key(image_path: str) -> str:
    if image_path.startswith("http"):
        return "/".join(image_path.split("/")[3:])
    return image_path


def _convert_primary_keys_to_case_ids(primary_keys: list[int], db: Session) -> list[str]:
    case_ids: list[str] = []

    all_cases = db.query(crud_case.model).all()
    pk_to_case_id = {}
    for case in all_cases:
        case_id_str = str(case.id)
        hash_val = int(hashlib.sha256(case_id_str.encode()).hexdigest(), 16)
        primary_key = hash_val % (2**63 - 1)
        pk_to_case_id[primary_key] = case_id_str

    for primary_key in primary_keys:
        case_id = pk_to_case_id.get(primary_key)
        if case_id:
            case_ids.append(case_id)

    return case_ids


@router.post("/search", response_model=SimilaritySearchResponse)
async def search_similar_cases(
    request: SimilaritySearchRequest,
    db: Session = Depends(get_db),
):
    """
    Search for visually similar cases using image embeddings only.

    Supported modes:
    1. case_id: query with an existing case from the gallery
    2. image_path: query with an uploaded image already stored in S3
    """
    similarity_scores: list[float] = []
    similar_case_ids: list[str] = []

    if request.case_id:
        case = crud_case.get(db, UUID(request.case_id))
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        existing_embedding = zilliz_service.get_by_case_id(request.case_id)
        if existing_embedding and existing_embedding.get("img_emb"):
            image_embedding = existing_embedding["img_emb"]
        else:
            image_bytes = s3_service.download_file(_extract_s3_key(case.image_path))
            image_embedding = retrieval_embedding_service.generate_image_embedding(image_bytes)
            zilliz_service.upsert_embedding(str(case.id), image_embedding)

        primary_keys, similarity_scores = zilliz_service.search_similar_by_image(
            image_embedding,
            top_k=request.top_k,
            exclude_case_id=request.case_id,
        )
        similar_case_ids = _convert_primary_keys_to_case_ids(primary_keys, db)

        crud_case.update_similar_cases(
            db,
            case_id=UUID(request.case_id),
            similar_cases=similar_case_ids,
            similarity_scores=similarity_scores,
        )

    elif request.image_path:
        image_bytes = s3_service.download_file(_extract_s3_key(request.image_path))
        image_embedding = retrieval_embedding_service.generate_image_embedding(image_bytes)
        primary_keys, similarity_scores = zilliz_service.search_similar_by_image(
            image_embedding,
            top_k=request.top_k,
        )
        similar_case_ids = _convert_primary_keys_to_case_ids(primary_keys, db)

    else:
        raise HTTPException(status_code=400, detail="Must provide case_id or image_path")

    case_details = []
    for case_id_str in similar_case_ids:
        case = crud_case.get(db, UUID(case_id_str))
        if case:
            case_details.append(jsonable_encoder(case))

    return SimilaritySearchResponse(
        similar_case_ids=similar_case_ids,
        similarity_scores=similarity_scores,
        case_details=case_details,
    )


@router.post("/embed")
async def generate_embeddings(case_id: UUID, db: Session = Depends(get_db)):
    """
    Generate and store an image embedding for a case.
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    try:
        image_bytes = s3_service.download_file(_extract_s3_key(case.image_path))
        image_embedding = retrieval_embedding_service.generate_image_embedding(image_bytes)

        success = zilliz_service.upsert_embedding(
            case_id=str(case_id),
            img_embedding=image_embedding,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store embedding in Zilliz")

        model_info = retrieval_embedding_service.get_model_info()
        return {
            "status": "success",
            "message": "Image embedding generated and stored successfully",
            "case_id": str(case_id),
            "image_embedding_dim": len(image_embedding),
            "model": {
                "input_size": model_info["input_size"],
                "providers": model_info["providers"],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image embedding: {exc}",
        ) from exc
