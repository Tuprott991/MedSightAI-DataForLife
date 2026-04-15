"""
AI Analysis API endpoints
"""
from __future__ import annotations

import base64
import io
import logging
from uuid import UUID

import requests
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.config.settings import settings
from app.core import case as crud_case, ai_result as crud_ai_result
from app.schemas import (
    AIAnalysisRequest, AIAnalysisResponse, AIResultResponse,
    AIResultCreate, MessageResponse,
    DetectionItem, LocalizeResponse,
)
from app.services import (
    ai_model_service, medsigclip_service,
    zilliz_service, s3_service,
)
from app.utils.single_image_infer import run_localization
from app.utils.s3_paths import S3PathBuilder

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: extract S3 key from a full S3 URL or raw key string
# ---------------------------------------------------------------------------

def _s3_key_from_url(image_path: str) -> str:
    """
    Extract the S3 object key from either a full URL or a bare key.

    Handles:
      https://bucket.s3.region.amazonaws.com/cases/...  -> cases/...
      cases/patient-id/original/xray.jpg                -> cases/...  (passthrough)
    """
    if image_path.startswith("http"):
        # e.g. https://bucket.s3.us-east-1.amazonaws.com/cases/...
        parts = image_path.split("/", 3)
        return parts[3] if len(parts) >= 4 else image_path
    return image_path


# ---------------------------------------------------------------------------
# Background task: persist annotated image + AI result to S3 / DB
# ---------------------------------------------------------------------------

def _persist_localization(
    case_id: UUID,
    detections: list[dict],
    annotated_bytes: bytes,
    db: Session,
) -> None:
    """
    Upload the annotated image to S3 and upsert the AIResult record.
    Runs as a FastAPI BackgroundTask.
    """
    try:
        # Build S3 key: cases/{case_id}/annotated/localized.jpg
        s3_key = S3PathBuilder.case_annotated_image(case_id, "localized.jpg")
        annotated_url = s3_service.upload_bytes(
            file_bytes=annotated_bytes,
            filename=s3_key,          # full key passed as filename
            prefix="",                # no additional prefix
            content_type="image/jpeg",
        )
        logger.info("[Localize] Annotated image uploaded to S3: %s", annotated_url)

        # Update Case.processed_img_path with the S3 URL (cache marker)
        case = crud_case.get(db, case_id)
        if case:
            crud_case.update(db, db_obj=case, obj_in={"processed_img_path": annotated_url})

        # Build JSONB payload for AIResult
        bounding_box_jsonb = {
            "detections": [
                {
                    "class_id": d["class_id"],
                    "class_name_en": d["class_name_en"],
                    "class_name_vi": d["class_name_vi"],
                    "confidence": d["confidence"],
                    "x1": d["x1"],
                    "y1": d["y1"],
                    "x2": d["x2"],
                    "y2": d["y2"],
                }
                for d in detections
            ]
        }

        # All unique lesion names as concepts list
        seen: set[int] = set()
        lesions = []
        for d in sorted(detections, key=lambda x: x["confidence"], reverse=True):
            if d["class_id"] not in seen:
                seen.add(d["class_id"])
                lesions.append({
                    "class_id": d["class_id"],
                    "class_name_en": d["class_name_en"],
                    "class_name_vi": d["class_name_vi"],
                    "confidence": d["confidence"],
                })
        concepts_jsonb = {"lesions": lesions}

        # Top diagnosis = highest-confidence unique lesion
        predicted_diagnosis = lesions[0]["class_name_en"] if lesions else "No finding"

        ai_result_data = {
            "case_id": case_id,
            "predicted_diagnosis": predicted_diagnosis,
            "confident_score": lesions[0]["confidence"] if lesions else None,
            "bounding_box": bounding_box_jsonb,
            "concepts": concepts_jsonb,
        }

        existing = crud_ai_result.get_by_case(db, case_id=case_id)
        if existing:
            crud_ai_result.update(db, db_obj=existing, obj_in=ai_result_data)
            logger.info("[Localize] Updated existing AIResult for case %s.", case_id)
        else:
            crud_ai_result.create(db, obj_in=ai_result_data)
            logger.info("[Localize] Created new AIResult for case %s.", case_id)

    except Exception as exc:
        logger.error("[Localize] Background persist failed for case %s: %s", case_id, exc, exc_info=True)


# ---------------------------------------------------------------------------
# POST /analysis/localize/{case_id}  — main new endpoint
# ---------------------------------------------------------------------------

@router.post("/localize/{case_id}", response_model=LocalizeResponse)
async def localize_case(
    case_id: UUID,
    background_tasks: BackgroundTasks,
    force_rerun: bool = Query(False, description="Force re-run even if cached result exists"),
    conf_thres: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou_thres: float = Query(0.45, ge=0.0, le=1.0, description="NMS IoU threshold"),
    db: Session = Depends(get_db),
) -> LocalizeResponse:
    """
    Run YOLOv5 chest-lesion localization on a case's X-ray image.

    **Cache-first logic:**
    - If `Case.processed_img_path` is set AND a matching `AIResult` row exists,
      returns the cached result immediately (`from_cache=true`).
    - Otherwise downloads the image from S3, runs YOLO inference, returns the
      result immediately (with base64 annotated image), and fires a background
      task to upload to S3 + persist to DB.

    Set `force_rerun=true` to skip the cache and re-run inference.
    """
    logger.info("[Localize] POST /localize/%s (force_rerun=%s)", case_id, force_rerun)

    # ── Fetch case ───────────────────────────────────────────────────────────
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not case.image_path:
        raise HTTPException(status_code=400, detail="Case has no image_path")

    # ── Cache check ──────────────────────────────────────────────────────────
    if not force_rerun and case.processed_img_path:
        ai_result = crud_ai_result.get_by_case(db, case_id=case_id)
        if ai_result and ai_result.bounding_box:
            detections_raw = ai_result.bounding_box.get("detections", [])
            detections = [DetectionItem(**d) for d in detections_raw]
            logger.info("[Localize] Cache hit for case %s — %d detection(s).", case_id, len(detections))
            return LocalizeResponse(
                case_id=case_id,
                detections=detections,
                annotated_image_url=case.processed_img_path,
                annotated_image_b64=None,
                from_cache=True,
                total_lesions=len(detections),
            )

    # ── Download original image from S3 ─────────────────────────────────────
    s3_key = _s3_key_from_url(case.image_path)
    logger.info("[Localize] Downloading image for case %s from S3 key: %s", case_id, s3_key)
    try:
        image_bytes = s3_service.download_file(s3_key)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Localize] S3 download failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to download case image: {exc}")

    # ── Run YOLO inference ───────────────────────────────────────────────────
    logger.info("[Localize] Running inference for case %s (%d bytes).", case_id, len(image_bytes))
    try:
        detections_raw, annotated_bytes = run_localization(
            image_bytes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
    except Exception as exc:
        logger.error("[Localize] Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    logger.info("[Localize] Inference complete — %d detection(s) for case %s.", len(detections_raw), case_id)

    # ── Build response (Path A — immediate to FE) ────────────────────────────
    detections = [DetectionItem(**d) for d in detections_raw]
    annotated_b64 = base64.b64encode(annotated_bytes).decode("ascii")

    # ── Fire background task (Path B — S3 + DB persist) ─────────────────────
    background_tasks.add_task(
        _persist_localization,
        case_id=case_id,
        detections=detections_raw,
        annotated_bytes=annotated_bytes,
        db=db,
    )

    return LocalizeResponse(
        case_id=case_id,
        detections=detections,
        annotated_image_url=None,
        annotated_image_b64=annotated_b64,
        from_cache=False,
        total_lesions=len(detections),
    )


# ---------------------------------------------------------------------------
# CAM-based inference (existing endpoint — kept intact)
# ---------------------------------------------------------------------------

@router.post("/cam-inference/{case_id}", response_model=AIResultResponse)
async def run_cam_inference(
    case_id: UUID,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold for detection"),
    db: Session = Depends(get_db),
):
    """
    Run CAM-based inference on a case's X-ray image

    This endpoint:
    1. Retrieves the PNG image from S3 using the case_id
    2. Calls the model_inference API (/api/v1/cam-inference/)
    3. Parses the response (top_classes with concepts and bboxes)
    4. Stores the results in ai_result table

    Args:
        case_id: UUID of the case to analyze
        threshold: Confidence threshold (default 0.5)
        db: Database session

    Returns:
        AI result with concepts and bounding boxes
    """
    logger.info(f"[CAM-INFERENCE] Starting inference for case_id: {case_id}, threshold: {threshold}")

    case = crud_case.get(db, case_id)
    if not case:
        logger.error(f"[CAM-INFERENCE] Case not found: {case_id}")
        raise HTTPException(status_code=404, detail="Case not found")

    if not case.image_path:
        logger.error(f"[CAM-INFERENCE] Case {case_id} has no image_path")
        raise HTTPException(status_code=400, detail="Case has no image")

    logger.info(f"[CAM-INFERENCE] Case image_path: {case.image_path}")

    s3_key = _s3_key_from_url(case.image_path)
    logger.info(f"[CAM-INFERENCE] Extracted S3 key: {s3_key}")

    try:
        logger.info(f"[CAM-INFERENCE] Downloading image from S3: {s3_key}")
        image_bytes = s3_service.download_file(s3_key)
        logger.info(f"[CAM-INFERENCE] Downloaded {len(image_bytes)} bytes from S3")

        files = {"file": ("image.png", io.BytesIO(image_bytes), "image/png")}
        params = {"threshold": threshold}

        model_api_url = f"{settings.MODEL_INFERENCE_URL}/api/v1/cam-inference/"
        logger.info(f"[CAM-INFERENCE] Calling model API: {model_api_url} with threshold={threshold}")

        response = requests.post(model_api_url, files=files, params=params, timeout=60)

        logger.info(f"[CAM-INFERENCE] Model API response status: {response.status_code}")
        logger.info(f"[CAM-INFERENCE] Model API response content: {response.text[:500]}")

        if response.status_code == 400:
            logger.warning(f"[CAM-INFERENCE] No abnormalities detected above threshold {threshold}")
            top_classes = []
            bboxes = []
        elif response.status_code != 200:
            response.raise_for_status()
        else:
            inference_result = response.json()
            logger.info(f"[CAM-INFERENCE] Parsed inference result keys: {inference_result.keys()}")
            top_classes = inference_result.get("top_classes", [])
            bboxes = inference_result.get("bboxes", [])
            logger.info(f"[CAM-INFERENCE] Extracted {len(top_classes)} top_classes and {len(bboxes)} bboxes")

        detected_concepts_list = [item["concepts"] for item in top_classes]
        logger.info(f"[CAM-INFERENCE] Detected concepts: {detected_concepts_list}")

        concepts_jsonb = {
            "top_classes": [
                {"prob": item["prob"], "concepts": item["concepts"], "class_idx": item["class_idx"]}
                for item in top_classes
            ],
            "detected_concepts": detected_concepts_list,
        }

        detections_list = []
        for idx, concept_item in enumerate(top_classes):
            if idx < len(bboxes) and bboxes[idx] is not None:
                detections_list.append({
                    "bbox": bboxes[idx],
                    "concept": concept_item["concepts"],
                    "class_idx": concept_item["class_idx"],
                    "probability": concept_item["prob"],
                })

        bounding_box_jsonb = {"detections": detections_list}

        if not detected_concepts_list:
            predicted_diagnosis = "No finding"
        elif case.diagnosis:
            predicted_diagnosis = case.diagnosis
        else:
            predicted_diagnosis = detected_concepts_list[0] if detected_concepts_list else "No finding"

        ai_result_data = {
            "case_id": case_id,
            "predicted_diagnosis": predicted_diagnosis,
            "confident_score": None,
            "bounding_box": bounding_box_jsonb,
            "concepts": concepts_jsonb,
        }

        existing_result = crud_ai_result.get_by_case(db, case_id=case_id)
        if existing_result:
            logger.info(f"[CAM-INFERENCE] Updating existing AI result for case {case_id}")
            ai_result = crud_ai_result.update(db, db_obj=existing_result, obj_in=ai_result_data)
        else:
            logger.info(f"[CAM-INFERENCE] Creating new AI result for case {case_id}")
            ai_result = crud_ai_result.create(db, obj_in=ai_result_data)

        logger.info(f"[CAM-INFERENCE] Successfully completed inference for case {case_id}")
        return ai_result

    except requests.exceptions.RequestException as e:
        logger.error(f"[CAM-INFERENCE] Request error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to call model inference API: {str(e)}")
    except Exception as e:
        logger.error(f"[CAM-INFERENCE] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing inference: {str(e)}")


# ---------------------------------------------------------------------------
# Remaining placeholder endpoints (kept for future implementation)
# ---------------------------------------------------------------------------

@router.post("/full-pipeline", response_model=AIAnalysisResponse)
async def run_full_analysis(
    request: AIAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Run complete AI analysis pipeline (not yet implemented).
    Use /localize/{case_id} for YOLO-based localization.
    """
    case = crud_case.get(db, request.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    raise NotImplementedError("Use /localize/{case_id} for production localization.")


@router.post("/preprocess")
async def preprocess_image(case_id: UUID, db: Session = Depends(get_db)):
    """Preprocess image for AI analysis (not yet implemented)."""
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    raise NotImplementedError("Connect to preprocessing module")


@router.post("/inference")
async def run_inference_endpoint(case_id: UUID, db: Session = Depends(get_db)):
    """Run AI model inference (not yet implemented)."""
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    raise NotImplementedError("Connect to inference module")


@router.get("/{case_id}/heatmap")
async def get_heatmap(case_id: UUID, db: Session = Depends(get_db)):
    """Generate Grad-CAM heatmap (not yet implemented)."""
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    raise NotImplementedError("Connect to Grad-CAM module")


@router.get("/{case_id}/concepts")
async def get_concepts(case_id: UUID, db: Session = Depends(get_db)):
    """Get concept-based analysis (not yet implemented)."""
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    ai_result = crud_ai_result.get_by_case(db, case_id=case_id)
    if not ai_result:
        raise HTTPException(status_code=404, detail="AI analysis not found")
    raise NotImplementedError("Connect to concept extraction module")


def process_similarity_search(case_id: UUID, image_path: str, db: Session):
    """Background task to process similarity search (not yet implemented)."""
    pass
