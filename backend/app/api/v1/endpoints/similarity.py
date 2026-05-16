"""
Similarity Search API endpoints for image-only CBIR retrieval.
"""
from __future__ import annotations

import hashlib
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import case as crud_case, patient as crud_patient
from app.schemas import SimilarCaseDetail, SimilaritySearchRequest, SimilaritySearchResponse
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


def _build_case_details(case_ids: list[str], db: Session) -> list[SimilarCaseDetail]:
    case_details: list[SimilarCaseDetail] = []

    for case_id_str in case_ids:
        case = crud_case.get(db, UUID(case_id_str))
        if not case:
            continue

        patient = crud_patient.get(db, case.patient_id)
        if not patient:
            continue

        case_details.append(
            SimilarCaseDetail(
                case_id=str(case.id),
                patient_id=str(patient.id),
                patient_name=patient.name,
                age=patient.age,
                gender=patient.gender,
                status=patient.status,
                diagnosis=case.diagnosis,
                image_path=case.image_path,
                processed_img_path=case.processed_img_path,
                timestamp=case.timestamp,
            )
        )

    return case_details


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

        cached_case_ids = case.similar_cases or []
        cached_scores = case.similarity_scores or []

        if len(cached_case_ids) >= request.top_k and len(cached_scores) >= request.top_k:
            similar_case_ids = cached_case_ids[:request.top_k]
            similarity_scores = cached_scores[:request.top_k]
        else:
            existing_embedding = zilliz_service.get_by_case_id(request.case_id)
            vector_field_name = zilliz_service.vector_field_name
            if existing_embedding and existing_embedding.get(vector_field_name):
                image_embedding = existing_embedding[vector_field_name]
            else:
                image_bytes = s3_service.download_file(_extract_s3_key(case.image_path))
                image_embedding = retrieval_embedding_service.generate_image_embedding(image_bytes)
                zilliz_service.upsert_embedding(
                    str(case.id),
                    image_embedding,
                    image_path=case.image_path,
                    label=case.diagnosis or "unknown",
                )

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

    return SimilaritySearchResponse(
        similar_case_ids=similar_case_ids,
        similarity_scores=similarity_scores,
        case_details=_build_case_details(similar_case_ids, db),
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
            image_path=case.image_path,
            label=case.diagnosis or "unknown",
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
