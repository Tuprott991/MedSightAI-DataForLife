"""
Report API endpoints
"""
from uuid import UUID
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging
from starlette.concurrency import run_in_threadpool

from app.config.database import get_db
from app.core import report as crud_report, case as crud_case, patient as crud_patient, ai_result as crud_ai_result
from app.schemas import (
    ReportCreate, ReportUpdate, ReportResponse,
    ReportGenerationRequest, MessageResponse, PatientFullReportResponse
)
from app.services import openai_llm_service

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

def _extract_indication(patient) -> str:
    if not patient.underlying_condition:
        return "None"

    conditions = [
        condition
        for condition, value in patient.underlying_condition.items()
        if value is True
    ]
    return ", ".join(conditions) if conditions else "None"


def _extract_bbox_list(ai_result) -> List[Dict[str, Any]]:
    bbox_list: List[Dict[str, Any]] = []
    if not ai_result or not ai_result.bounding_box:
        return bbox_list

    for detection in ai_result.bounding_box.get("detections", []):
        bbox_coords = detection.get("bbox")
        if not bbox_coords or len(bbox_coords) != 4:
            continue

        bbox_list.append({
            "class_name": detection.get("concept") or detection.get("class_name") or "Unknown",
            "x_min": int(bbox_coords[0]),
            "y_min": int(bbox_coords[1]),
            "x_max": int(bbox_coords[2]),
            "y_max": int(bbox_coords[3]),
            "probability": detection.get("probability") or detection.get("confidence"),
        })
    return bbox_list


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportGenerationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate medical report using OpenAI GPT vision
    
    Flow:
    1. Get case, patient, and AI result data
    2. Extract image_url from case.image_path
    3. Extract indication from patient.underlying_condition
    4. Extract bounding boxes from ai_result.bounding_box if present
    5. Call GPT vision/text service to generate a structured report
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
    
    # Get AI results if available. Report generation can still draft from the image.
    ai_result = crud_ai_result.get_by_case(db, case_id=request.case_id)
    if ai_result and ai_result.bounding_box:
        detection_count = len(ai_result.bounding_box.get("detections", []))
        logger.info(f"[REPORT-GEN] Found AI result with {detection_count} detections")
    else:
        logger.info(f"[REPORT-GEN] No AI result found for case {request.case_id}; generating report from image and context")
    
    # Get patient info
    patient = crud_patient.get(db, case.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Extract image_url from case
    image_url = case.image_path
    if not image_url:
        raise HTTPException(status_code=400, detail="Case has no image")
    
    indication = _extract_indication(patient)
    logger.info(f"[REPORT-GEN] Extracted indication: {indication}")

    bbox_list = _extract_bbox_list(ai_result)
    logger.info(f"[REPORT-GEN] Extracted {len(bbox_list)} bounding boxes")
    
    try:
        logger.info(f"[REPORT-GEN] Calling OpenAI report service for case {request.case_id}")

        radiology_report = await run_in_threadpool(
            openai_llm_service.generate_medical_report,
            image_url=image_url,
            patient_context={
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender,
                "blood_type": patient.blood_type,
                "status": patient.status,
                "underlying_condition": patient.underlying_condition,
                "indication": indication,
                "patient_history": request.patient_history,
            },
            case_context={
                "case_id": str(case.id),
                "image_path": case.image_path,
                "processed_img_path": case.processed_img_path,
                "timestamp": case.timestamp.isoformat() if case.timestamp else None,
                "diagnosis": case.diagnosis,
                "findings": case.findings,
                "ai_findings_from_request": request.ai_findings,
            },
            ai_context={
                "predicted_diagnosis": ai_result.predicted_diagnosis if ai_result else None,
                "confident_score": ai_result.confident_score if ai_result else None,
                "concepts": ai_result.concepts if ai_result else None,
                "raw_bounding_box": ai_result.bounding_box if ai_result else None,
            } if ai_result else None,
            detections=bbox_list,
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


@router.get("/full/{case_id}", response_model=PatientFullReportResponse)
def get_full_patient_report(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive patient information with report
    
    This endpoint returns:
    - Patient information (name, age, gender, blood type, status, underlying conditions, etc.)
    - Case details (image paths, diagnosis, findings, timestamp)
    - AI analysis results (predicted diagnosis, bounding boxes, concepts)
    - Generated medical report (model report, doctor report, feedback)
    
    Args:
        case_id: UUID of the case
        db: Database session
    
    Returns:
        PatientFullReportResponse with all patient and case information
    """
    logger.info(f"[FULL-REPORT] Fetching full patient report for case_id: {case_id}")
    
    # Get case
    case = crud_case.get(db, case_id)
    if not case:
        logger.error(f"[FULL-REPORT] Case not found: {case_id}")
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get patient
    patient = crud_patient.get(db, case.patient_id)
    if not patient:
        logger.error(f"[FULL-REPORT] Patient not found: {case.patient_id}")
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get AI result (optional - may not exist yet)
    ai_result = crud_ai_result.get_by_case(db, case_id=case_id)
    
    # Get report (optional - may not exist yet)
    report = crud_report.get_by_case(db, case_id=case_id)
    
    logger.info(f"[FULL-REPORT] Found: Patient={patient.name}, Case={case_id}, AI Result={ai_result is not None}, Report={report is not None}")
    
    # Build comprehensive response
    response_data = {
        # Patient Information
        "patient_id": patient.id,
        "patient_name": patient.name,
        "patient_age": patient.age,
        "patient_gender": patient.gender,
        "blood_type": patient.blood_type,
        "status": patient.status,
        "underlying_condition": patient.underlying_condition,
        "phone_number": patient.phone_number,
        "patient_created_at": patient.created_at,
        
        # Case Information
        "case_id": case.id,
        "image_path": case.image_path,
        "processed_img_path": case.processed_img_path,
        "case_timestamp": case.timestamp,
        "diagnosis": case.diagnosis,
        "findings": case.findings,
        
        # AI Result Information (if exists)
        "ai_result_id": ai_result.id if ai_result else None,
        "predicted_diagnosis": ai_result.predicted_diagnosis if ai_result else None,
        "confident_score": ai_result.confident_score if ai_result else None,
        "bounding_box": ai_result.bounding_box if ai_result else None,
        "concepts": ai_result.concepts if ai_result else None,
        "ai_result_created_at": ai_result.created_at if ai_result else None,
        
        # Report Information (if exists)
        "report_id": report.id if report else None,
        "model_report": report.model_report if report else None,
        "doctor_report": report.doctor_report if report else None,
        "feedback_note": report.feedback_note if report else None,
        "report_created_at": report.created_at if report else None,
    }
    
    logger.info(f"[FULL-REPORT] Successfully compiled full report for case {case_id}")
    
    return response_data
