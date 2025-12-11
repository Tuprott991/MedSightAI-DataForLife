"""
Case API endpoints
"""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.config.settings import settings
from app.core import case as crud_case, patient as crud_patient
from app.schemas import (
    CaseCreate, CaseUpdate, CaseResponse, CaseListResponse,
    CaseWithPatientResponse, MessageResponse, ImageUploadResponse
)
from app.services import s3_service

router = APIRouter()


@router.post("/", response_model=CaseResponse, status_code=201)
async def create_case(
    case_in: CaseCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new case with all case data
    
    Stores case information including image paths and similarity data.
    Images should already be uploaded to S3 before creating the case.
    
    Auto-updates patient history with diagnosis and findings from this case.
    """
    # Verify patient exists
    patient = crud_patient.get(db, case_in.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Create case with all provided data
    case = crud_case.create(db, obj_in=case_in.model_dump())
    
    # Auto-update patient history if diagnosis or findings provided
    if case_in.diagnosis or case_in.findings:
        from datetime import datetime
        
        # Get current history or initialize empty dict
        current_history = patient.history if patient.history else {}
        
        # Use today's date as key
        date_key = datetime.now().strftime("%m-%d-%Y")
        
        # Add new entry to history
        current_history[date_key] = {
            "diagnosis": case_in.diagnosis or "",
            "findings": case_in.findings or ""
        }
        
        # Update patient with new history
        patient.history = current_history
        db.commit()
        db.refresh(patient)
    
    return case


@router.post("/upload", response_model=CaseResponse, status_code=201)
async def create_case_with_file_upload(
    patient_id: UUID = Query(..., description="Patient ID"),
    file: UploadFile = File(..., description="X-ray image file (DICOM or PNG)"),
    db: Session = Depends(get_db)
):
    """
    Create a new case with file upload (multipart/form-data)
    
    If DICOM file is uploaded:
    - Stores DICOM at: s3://bucket/cases/{patient_id}/image.dicom
    - Converts and stores PNG at: s3://bucket/cases/{patient_id}.png
    - Returns PNG path in image_path field
    
    If PNG/JPG file is uploaded:
    - Stores at: s3://bucket/cases/{patient_id}.png
    """
    # Verify patient exists
    patient = crud_patient.get(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Validate file
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/dicom", "application/octet-stream"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG, PNG, and DICOM allowed")
    
    # Check if DICOM file
    is_dicom = file.content_type == "application/dicom" or file.filename.lower().endswith('.dcm') or file.filename.lower().endswith('.dicom')
    
    if is_dicom:
        # Upload DICOM and convert to PNG
        dicom_url, png_url = s3_service.upload_dicom_and_convert(
            file=file,
            patient_id=str(patient_id)
        )
        # Store PNG URL as the main image_path (for queries)
        image_path = png_url
        # Optionally store DICOM path in processed_img_path or a new field
        processed_img_path = dicom_url
    else:
        # Upload regular image (PNG/JPG) with standardized naming
        image_path = s3_service.upload_bytes(
            file_bytes=file.file.read(),
            filename=f"{patient_id}.png",
            prefix="cases/",
            content_type="image/png"
        )
        processed_img_path = None
    
    # Create case
    case_data = {
        "patient_id": patient_id,
        "image_path": image_path,  # Always PNG path
        "processed_img_path": processed_img_path  # DICOM path if uploaded
    }
    case = crud_case.create(db, obj_in=case_data)
    
    return case


@router.get("/", response_model=CaseListResponse)
def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    patient_id: UUID = Query(None),
    db: Session = Depends(get_db)
):
    """List all cases with optional patient filter"""
    skip = (page - 1) * page_size
    
    if patient_id:
        cases = crud_case.get_by_patient(db, patient_id=patient_id, skip=skip, limit=page_size)
        total = len(cases)  # Simplified
    else:
        cases = crud_case.get_multi(db, skip=skip, limit=page_size)
        total = crud_case.get_count(db)
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "cases": cases
    }


@router.get("/{case_id}", response_model=CaseResponse)
def get_case(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """Get case by ID"""
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.put("/{case_id}", response_model=CaseResponse)
def update_case(
    case_id: UUID,
    case_in: CaseUpdate,
    db: Session = Depends(get_db)
):
    """Update case information"""
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    update_data = case_in.model_dump(exclude_unset=True)
    case = crud_case.update(db, db_obj=case, obj_in=update_data)
    return case


@router.delete("/{case_id}", response_model=MessageResponse)
def delete_case(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a case"""
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Delete images from S3
    if case.image_path:
        s3_service.delete_file(case.image_path)
    if case.processed_img_path:
        s3_service.delete_file(case.processed_img_path)
    
    crud_case.delete(db, id=case_id)
    return {"message": "Case deleted successfully"}


@router.get("/{case_id}/image-url")
def get_case_image_url(
    case_id: UUID,
    image_type: str = Query("original", regex="^(original|processed)$"),
    db: Session = Depends(get_db)
):
    """Get presigned URL for case image"""
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    image_path = case.image_path if image_type == "original" else case.processed_img_path
    
    if not image_path:
        raise HTTPException(status_code=404, detail=f"{image_type.capitalize()} image not found")
    
    url = s3_service.get_presigned_url(image_path)
    return {"url": url, "expires_in": 3600}
