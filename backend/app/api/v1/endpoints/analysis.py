"""
AI Analysis API endpoints
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
import requests
import io
import logging

from app.config.database import get_db
from app.config.settings import settings
from app.core import case as crud_case, ai_result as crud_ai_result
from app.schemas import (
    AIAnalysisRequest, AIAnalysisResponse, AIResultResponse,
    AIResultCreate, MessageResponse
)
from app.services import (
    ai_model_service, medsigclip_service, 
    zilliz_service, s3_service
)

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/cam-inference/{case_id}", response_model=AIResultResponse)
async def run_cam_inference(
    case_id: UUID,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold for detection"),
    db: Session = Depends(get_db)
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
    
    # Get case from database
    case = crud_case.get(db, case_id)
    if not case:
        logger.error(f"[CAM-INFERENCE] Case not found: {case_id}")
        raise HTTPException(status_code=404, detail="Case not found")
    
    if not case.image_path:
        logger.error(f"[CAM-INFERENCE] Case {case_id} has no image_path")
        raise HTTPException(status_code=400, detail="Case has no image")
    
    logger.info(f"[CAM-INFERENCE] Case image_path: {case.image_path}")
    
    # Extract S3 key from image_path
    if case.image_path.startswith('http'):
        # URL format: https://bucket.s3.region.amazonaws.com/key
        parts = case.image_path.split('/')
        s3_key = '/'.join(parts[3:])
    else:
        s3_key = case.image_path
    
    logger.info(f"[CAM-INFERENCE] Extracted S3 key: {s3_key}")
    
    try:
        # Download image from S3
        logger.info(f"[CAM-INFERENCE] Downloading image from S3: {s3_key}")
        image_bytes = s3_service.download_file(s3_key)
        logger.info(f"[CAM-INFERENCE] Downloaded {len(image_bytes)} bytes from S3")
        
        # Prepare multipart upload for model_inference API
        files = {
            'file': ('image.png', io.BytesIO(image_bytes), 'image/png')
        }
        params = {
            'threshold': threshold
        }
        
        # Call model_inference API
        model_api_url = f"{settings.MODEL_INFERENCE_URL}/api/v1/cam-inference/"
        logger.info(f"[CAM-INFERENCE] Calling model API: {model_api_url} with threshold={threshold}")
        
        response = requests.post(
            model_api_url,
            files=files,
            params=params,
            timeout=60
        )
        
        logger.info(f"[CAM-INFERENCE] Model API response status: {response.status_code}")
        logger.info(f"[CAM-INFERENCE] Model API response content: {response.text[:500]}")
        
        # Handle response - even 400 status can have valid JSON data
        if response.status_code == 400:
            # Model returned no detections above threshold
            # This is valid - return empty result
            logger.warning(f"[CAM-INFERENCE] No abnormalities detected above threshold {threshold}")
            top_classes = []
            bboxes = []
        elif response.status_code != 200:
            # Other errors should be raised
            response.raise_for_status()
        else:
            # Parse successful response
            inference_result = response.json()
            logger.info(f"[CAM-INFERENCE] Parsed inference result keys: {inference_result.keys()}")
            # Extract data from response
            top_classes = inference_result.get('top_classes', [])
            bboxes = inference_result.get('bboxes', [])
            logger.info(f"[CAM-INFERENCE] Extracted {len(top_classes)} top_classes and {len(bboxes)} bboxes")
        
        # Build concepts JSONB structure
        # Format: {"top_classes": [...], "detected_concepts": [...]}
        detected_concepts_list = [item['concepts'] for item in top_classes]
        logger.info(f"[CAM-INFERENCE] Detected concepts: {detected_concepts_list}")
        
        concepts_jsonb = {
            "top_classes": [
                {
                    "prob": item['prob'],
                    "concepts": item['concepts'],
                    "class_idx": item['class_idx']
                }
                for item in top_classes
            ],
            "detected_concepts": detected_concepts_list
        }
        
        # Build bounding_box JSONB structure
        # Format: {"detections": [{"bbox": [...], "concept": "...", "class_idx": ..., "probability": ...}]}
        detections_list = []
        for idx, concept_item in enumerate(top_classes):
            if idx < len(bboxes) and bboxes[idx] is not None:
                detections_list.append({
                    "bbox": bboxes[idx],
                    "concept": concept_item['concepts'],
                    "class_idx": concept_item['class_idx'],
                    "probability": concept_item['prob']
                })
        
        bounding_box_jsonb = {
            "detections": detections_list
        }
        
        # Determine predicted diagnosis
        # If no abnormalities detected, set to "No finding"
        # Otherwise, use the most confident detection or keep existing case diagnosis
        if not detected_concepts_list:
            predicted_diagnosis = "No finding"
        elif case.diagnosis:
            predicted_diagnosis = case.diagnosis
        else:
            # Use the top detection concept if no diagnosis exists
            predicted_diagnosis = detected_concepts_list[0] if detected_concepts_list else "No finding"
        
        # Create AI result
        ai_result_data = {
            "case_id": case_id,
            "predicted_diagnosis": predicted_diagnosis,
            "confident_score": None,  # Not filled as per requirement
            "bounding_box": bounding_box_jsonb,
            "concepts": concepts_jsonb
        }
        
        # Check if AI result already exists for this case
        existing_result = crud_ai_result.get_by_case(db, case_id=case_id)
        
        if existing_result:
            # Update existing result
            logger.info(f"[CAM-INFERENCE] Updating existing AI result for case {case_id}")
            ai_result = crud_ai_result.update(
                db, 
                db_obj=existing_result, 
                obj_in=ai_result_data
            )
        else:
            # Create new result
            logger.info(f"[CAM-INFERENCE] Creating new AI result for case {case_id}")
            ai_result = crud_ai_result.create(db, obj_in=ai_result_data)
        
        logger.info(f"[CAM-INFERENCE] Successfully completed inference for case {case_id}")
        return ai_result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[CAM-INFERENCE] Request error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to call model inference API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"[CAM-INFERENCE] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing inference: {str(e)}"
        )


@router.post("/full-pipeline", response_model=AIAnalysisResponse)
async def run_full_analysis(
    request: AIAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Run complete AI analysis pipeline:
    1. Preprocess image
    2. Run inference
    3. Generate heatmap (Grad-CAM)
    4. Extract concepts
    5. Find similar cases
    6. Store results
    
    TODO: Implement by calling functions from ai_model_service
    """
    case = crud_case.get(db, request.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # TODO: Implement full pipeline
    # Step 1: Preprocess
    # processed_path = ai_model_service.preprocess_image(case.image_path)
    # crud_case.update(db, db_obj=case, obj_in={"processed_img_path": processed_path})
    
    # Step 2: Run inference
    # results = ai_model_service.run_inference(processed_path)
    
    # Step 3: Store AI results
    # ai_result_data = {
    #     "case_id": request.case_id,
    #     "predicted_diagnosis": results['predicted_diagnosis'],
    #     "confident_score": results['confidence_score'],
    #     "bounding_box": results['bounding_boxes']
    # }
    # ai_result = crud_ai_result.create(db, obj_in=ai_result_data)
    
    # Step 4: Generate heatmap if requested
    # heatmap_path = None
    # if request.include_heatmap:
    #     heatmap_path = ai_model_service.generate_gradcam(processed_path)
    
    # Step 5: Extract concepts if requested
    # concepts = None
    # if request.include_concepts:
    #     concepts = ai_model_service.extract_concepts(processed_path, results['bounding_boxes'])
    
    # Step 6: Generate embeddings and find similar cases (in background)
    # background_tasks.add_task(process_similarity_search, request.case_id, processed_path, db)
    
    raise NotImplementedError("Connect to AI model services")


@router.post("/preprocess")
async def preprocess_image(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Preprocess image for AI analysis
    
    TODO: Call ai_model_service.preprocess_image()
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Download from S3
    image_bytes = s3_service.download_file(case.image_path)
    
    # TODO: Preprocess
    # processed_path = ai_model_service.preprocess_image(image_bytes)
    
    # Update case with processed image path
    # crud_case.update(db, db_obj=case, obj_in={"processed_img_path": processed_path})
    
    raise NotImplementedError("Connect to preprocessing module")


@router.post("/inference")
async def run_inference(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Run AI model inference
    
    TODO: Call ai_model_service.run_inference()
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    if not case.processed_img_path:
        raise HTTPException(status_code=400, detail="Image not preprocessed yet")
    
    # TODO: Run inference
    # results = ai_model_service.run_inference(case.processed_img_path)
    
    # Store results
    # ai_result_data = {
    #     "case_id": case_id,
    #     "predicted_diagnosis": results['predicted_diagnosis'],
    #     "confident_score": results['confidence_score'],
    #     "bounding_box": results['bounding_boxes']
    # }
    # ai_result = crud_ai_result.create(db, obj_in=ai_result_data)
    
    raise NotImplementedError("Connect to inference module")


@router.get("/{case_id}/heatmap")
async def get_heatmap(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Generate and return Grad-CAM heatmap
    
    TODO: Call ai_model_service.generate_gradcam()
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # TODO: Generate heatmap
    # heatmap_path = ai_model_service.generate_gradcam(case.processed_img_path)
    # presigned_url = s3_service.get_presigned_url(heatmap_path)
    
    raise NotImplementedError("Connect to Grad-CAM module")


@router.get("/{case_id}/concepts")
async def get_concepts(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get concept-based analysis
    
    TODO: Call ai_model_service.extract_concepts()
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    ai_result = crud_ai_result.get_by_case(db, case_id=case_id)
    if not ai_result:
        raise HTTPException(status_code=404, detail="AI analysis not found")
    
    # TODO: Extract concepts
    # concepts = ai_model_service.extract_concepts(
    #     case.processed_img_path,
    #     ai_result.bounding_box
    # )
    
    raise NotImplementedError("Connect to concept extraction module")


def process_similarity_search(case_id: UUID, image_path: str, db: Session):
    """
    Background task to process similarity search
    """
    # TODO: Generate embeddings
    # image_emb, text_emb = medsigclip_service.generate_embeddings(image_path, description)
    
    # Store in Milvus
    # milvus_service.insert_embedding(str(case_id), image_emb, text_emb)
    
    # Search for similar cases
    # similar_ids, scores = milvus_service.search_similar_by_image(image_emb, top_k=5)
    
    # Update case with similar cases
    # crud_case.update_similar_cases(db, case_id=case_id, similar_cases=similar_ids, similarity_scores=scores)
    
    pass
