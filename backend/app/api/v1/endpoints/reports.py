"""
Report API endpoints
"""
from uuid import UUID
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import requests
import logging

from app.config.database import get_db
from app.core import report as crud_report, case as crud_case, patient as crud_patient, ai_result as crud_ai_result
from app.schemas import (
    ReportCreate, ReportUpdate, ReportResponse,
    ReportGenerationRequest, MessageResponse
)
from app.services import medgemma_service

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

# External report generation API
REPORT_GENERATION_API_URL = "https://noncoalescent-faucial-elli.ngrok-free.dev/generate-report"


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportGenerationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate medical report using external report generation API
    
    Flow:
    1. Get case, patient, and AI result data
    2. Extract image_url from case.image_path
    3. Extract indication from patient.underlying_condition (convert JSON to string)
    4. Extract bounding boxes from ai_result.bounding_box
    5. Call external API to generate report
    6. Store report in database
    
    Args:
        request: ReportGenerationRequest with case_id
        db: Database session
    
    Returns:
        ReportResponse with generated report data
    """
    logger.info(f"[REPORT-GEN] Starting report generation for case_id: {request.case_id}")
    
    # Get case from database
    case = crud_case.get(db, request.case_id)
    if not case:
        logger.error(f"[REPORT-GEN] Case not found: {request.case_id}")
        raise HTTPException(status_code=404, detail="Case not found")
    
    logger.info(f"[REPORT-GEN] Found case with image_path: {case.image_path}")
    
    # Get AI results
    ai_result = crud_ai_result.get_by_case(db, case_id=request.case_id)
    if not ai_result:
        logger.error(f"[REPORT-GEN] AI analysis not found for case: {request.case_id}")
        raise HTTPException(status_code=404, detail="AI analysis not found. Run CAM inference first")
    
    logger.info(f"[REPORT-GEN] Found AI result with {len(ai_result.bounding_box.get('detections', []))} detections")
    
    # Get patient info
    patient = crud_patient.get(db, case.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Extract image_url from case
    image_url = case.image_path
    if not image_url:
        raise HTTPException(status_code=400, detail="Case has no image")
    
    # Extract indication from patient.underlying_condition
    # Convert {"hypertension":true,"diabetes":true,"asthma":false} to "hypertension, diabetes"
    indication = "None"
    if patient.underlying_condition:
        conditions = []
        for condition, value in patient.underlying_condition.items():
            if value is True:
                conditions.append(condition)
        if conditions:
            indication = ", ".join(conditions)
    
    logger.info(f"[REPORT-GEN] Extracted indication: {indication}")
    
    # Extract bounding boxes from ai_result
    # Format: {"detections":[{"bbox":[x1,y1,x2,y2],"concept":"...","class_idx":...,"probability":...}]}
    bbox_list = []
    if ai_result.bounding_box and "detections" in ai_result.bounding_box:
        for detection in ai_result.bounding_box["detections"]:
            if "bbox" in detection and "concept" in detection:
                bbox_coords = detection["bbox"]
                if len(bbox_coords) == 4:
                    bbox_list.append({
                        "class_name": detection["concept"],
                        "x_min": int(bbox_coords[0]),
                        "y_min": int(bbox_coords[1]),
                        "x_max": int(bbox_coords[2]),
                        "y_max": int(bbox_coords[3])
                    })
    
    logger.info(f"[REPORT-GEN] Extracted {len(bbox_list)} bounding boxes")
    
    if not bbox_list:
        logger.error(f"[REPORT-GEN] No bounding boxes found in AI result")
        raise HTTPException(
            status_code=400, 
            detail="No bounding boxes found in AI result. Run CAM inference first"
        )
    
    # Prepare request payload for external API
    payload = {
        "image_url": image_url,
        "indication": indication,
        "bbox": bbox_list
    }
    
    try:
        # Call external report generation API
        logger.info(f"[REPORT-GEN] Calling external API: {REPORT_GENERATION_API_URL}")
        logger.info(f"[REPORT-GEN] Payload: image_url={image_url}, indication={indication}, bbox_count={len(bbox_list)}")
        
        response = requests.post(
            REPORT_GENERATION_API_URL,
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        logger.info(f"[REPORT-GEN] API response status: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"[REPORT-GEN] Received response with keys: {result.keys()}")
        
        # Extract radiology_report from response
        radiology_report = result.get("radiology_report")
        if not radiology_report:
            raise HTTPException(
                status_code=500,
                detail="Invalid response from report generation API: missing radiology_report"
            )
        
        # Check if report already exists for this case
        existing_report = crud_report.get_by_case(db, case_id=request.case_id)
        
        # Prepare report data
        report_data = {
            "case_id": request.case_id,
            "model_report": radiology_report,  # Store entire radiology_report object as JSONB
            "doctor_report": None,
            "feedback_note": None
        }
        
        if existing_report:
            # Update existing report
            logger.info(f"[REPORT-GEN] Updating existing report for case {request.case_id}")
            report = crud_report.update(
                db,
                db_obj=existing_report,
                obj_in=report_data
            )
        else:
            # Create new report
            logger.info(f"[REPORT-GEN] Creating new report for case {request.case_id}")
            report = crud_report.create(db, obj_in=report_data)
        
        logger.info(f"[REPORT-GEN] Successfully generated report for case {request.case_id}")
        return report
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[REPORT-GEN] Request error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to report generation API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"[REPORT-GEN] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/{case_id}", response_model=ReportResponse)
def get_report(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """Get report for a case"""
    report = crud_report.get_by_case(db, case_id=case_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.put("/{report_id}/doctor-report", response_model=ReportResponse)
def update_doctor_report(
    report_id: UUID,
    doctor_report: str,
    db: Session = Depends(get_db)
):
    """Update doctor's report section (Human-in-the-loop)"""
    report = crud_report.update_doctor_report(
        db,
        report_id=report_id,
        doctor_report=doctor_report
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.put("/{report_id}/feedback", response_model=ReportResponse)
def add_feedback(
    report_id: UUID,
    feedback_note: str,
    db: Session = Depends(get_db)
):
    """Add feedback note for model improvement"""
    report = crud_report.add_feedback(
        db,
        report_id=report_id,
        feedback_note=feedback_note
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.delete("/{report_id}", response_model=MessageResponse)
def delete_report(
    report_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a report"""
    report = crud_report.get(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    crud_report.delete(db, id=report_id)
    return {"message": "Report deleted successfully"}
