from __future__ import annotations

import base64
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.v1.endpoints.analysis import _run_disease_detection_for_case
from app.config.database import get_db
from app.schemas import LocalizeResponse


router = APIRouter()


@router.post("/{case_id}", response_model=LocalizeResponse)
async def detect_disease_for_case(
    case_id: UUID,
    force_rerun: bool = Query(
        False, description="Force re-run even if cached result exists"
    ),
    conf_thres: float = Query(0.1, ge=0.0, le=1.0, description="Confidence threshold"),
    iou_thres: float = Query(0.45, ge=0.0, le=1.0, description="WBF IoU threshold"),
    db: Session = Depends(get_db),
) -> LocalizeResponse:
    response, _ = await _run_disease_detection_for_case(
        case_id=case_id,
        db=db,
        force_rerun=force_rerun,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        persist_immediately=True,
    )
    if response.annotated_image_b64 is not None:
        response.annotated_image_url = (
            f"data:image/jpeg;base64,{response.annotated_image_b64}"
        )
        response.annotated_image_b64 = None
    return response
