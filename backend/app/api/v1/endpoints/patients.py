"""
Patient API endpoints
"""
import os
from datetime import datetime
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import patient as crud_patient, case as crud_case
from app.models.models import Patient, Case
from app.schemas import (
    PatientCreate, PatientUpdate, PatientResponse, 
    PatientListResponse, PatientInforResponse, LatestCaseInfo, 
    PatientInforListResponse, MessageResponse
)
from app.services import s3_service

router = APIRouter()


@router.post("/", response_model=PatientResponse, status_code=201)
def create_patient(
    patient_in: PatientCreate,
    db: Session = Depends(get_db)
):
    """Create a new patient"""
    patient_data = patient_in.model_dump()
    patient = crud_patient.create(db, obj_in=patient_data)
    return patient


@router.post("/import-case", response_model=PatientInforResponse, status_code=201)
async def import_patient_case(
    name: str = Form(...),
    age: int | None = Form(None),
    gender: str | None = Form(None),
    phone_number: str | None = Form(None),
    diagnosis: str | None = Form(None),
    findings: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Create a patient and upload one new unprocessed case image in a single request.

    The image upload is handled entirely in the backend:
    - upload image bytes to S3
    - create patient
    - create case with processed_img_path = null

    No vector ingestion is performed here.
    """
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/dicom", "application/octet-stream"]
    filename_lower = (file.filename or "").lower()
    is_dicom = (
        file.content_type == "application/dicom"
        or filename_lower.endswith(".dcm")
        or filename_lower.endswith(".dicom")
    )

    if file.content_type not in allowed_types and not is_dicom:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG, PNG, and DICOM allowed")

    patient_payload = {
        "name": name,
        "age": age,
        "gender": gender,
        "phone_number": phone_number,
    }
    patient = crud_patient.create(db, obj_in=patient_payload)

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        if is_dicom:
            image_bytes = s3_service.dicom_to_png_bytes(file_bytes)
            image_filename = f"{patient.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
            content_type = "image/png"
        else:
            extension = os.path.splitext(file.filename or "")[1].lower() or ".png"
            image_bytes = file_bytes
            image_filename = f"{patient.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{extension}"
            content_type = file.content_type or "image/png"

        image_path = s3_service.upload_bytes(
            file_bytes=image_bytes,
            filename=image_filename,
            prefix="cases/",
            content_type=content_type,
        )

        case_payload = {
            "patient_id": patient.id,
            "image_path": image_path,
            "processed_img_path": None,
            "diagnosis": diagnosis,
            "findings": findings,
        }
        case = crud_case.create(db, obj_in=case_payload)

        if diagnosis or findings:
            history_date = datetime.now().strftime("%m-%d-%Y")
            current_history = patient.history if patient.history else {}
            current_history[history_date] = {
                "diagnosis": diagnosis or "",
                "findings": findings or "",
            }
            patient.history = current_history
            db.commit()
            db.refresh(patient)

        response_data = {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "history": patient.history,
            "created_at": patient.created_at,
            "blood_type": patient.blood_type,
            "status": patient.status,
            "underlying_condition": patient.underlying_condition,
            "phone_number": patient.phone_number,
            "fcm_token": patient.fcm_token,
            "latest_case": LatestCaseInfo(
                id=case.id,
                image_path=case.image_path,
                processed_img_path=case.processed_img_path,
                timestamp=case.timestamp,
                similar_cases=case.similar_cases,
                similarity_scores=case.similarity_scores,
                diagnosis=case.diagnosis,
                findings=case.findings,
            ),
        }
        return response_data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to import patient case: {exc}") from exc


@router.get("/", response_model=PatientListResponse)
def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query(None),
    db: Session = Depends(get_db)
):
    """List all patients with pagination"""
    skip = (page - 1) * page_size
    
    if search:
        patients = crud_patient.search_patients(db, query=search, skip=skip, limit=page_size)
        total = len(patients)  # Simplified; should count search results
    else:
        patients = crud_patient.get_multi(db, skip=skip, limit=page_size)
        total = crud_patient.get_count(db)
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "patients": patients
    }


@router.get("/list/infor", response_model=PatientInforListResponse)
def list_patients_with_infor(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query(None),
    processing_status: str | None = Query(None, pattern="^(processed|unprocessed)$"),
    db: Session = Depends(get_db)
):
    """List all patients with complete information including latest case data"""
    skip = (page - 1) * page_size
    latest_case_subquery = (
        db.query(
            Case.id.label("case_id"),
            Case.patient_id.label("case_patient_id"),
            Case.image_path.label("image_path"),
            Case.processed_img_path.label("processed_img_path"),
            Case.timestamp.label("timestamp"),
            Case.similar_cases.label("similar_cases"),
            Case.similarity_scores.label("similarity_scores"),
            Case.diagnosis.label("diagnosis"),
            Case.findings.label("findings"),
            func.row_number().over(
                partition_by=Case.patient_id,
                order_by=Case.timestamp.desc(),
            ).label("row_num"),
        )
        .subquery()
    )

    base_query = (
        db.query(
            Patient,
            latest_case_subquery.c.case_id,
            latest_case_subquery.c.image_path,
            latest_case_subquery.c.processed_img_path,
            latest_case_subquery.c.timestamp,
            latest_case_subquery.c.similar_cases,
            latest_case_subquery.c.similarity_scores,
            latest_case_subquery.c.diagnosis,
            latest_case_subquery.c.findings,
        )
        .outerjoin(
            latest_case_subquery,
            and_(
                latest_case_subquery.c.case_patient_id == Patient.id,
                latest_case_subquery.c.row_num == 1,
            ),
        )
    )

    if search:
        base_query = base_query.filter(Patient.name.ilike(f"%{search}%"))

    if processing_status == "processed":
        base_query = base_query.filter(latest_case_subquery.c.case_id.isnot(None))
        base_query = base_query.filter(latest_case_subquery.c.processed_img_path.isnot(None))
    elif processing_status == "unprocessed":
        base_query = base_query.filter(latest_case_subquery.c.case_id.isnot(None))
        base_query = base_query.filter(latest_case_subquery.c.processed_img_path.is_(None))

    total = base_query.with_entities(func.count(Patient.id)).scalar() or 0

    rows = (
        base_query
        .order_by(Patient.created_at.desc())
        .offset(skip)
        .limit(page_size)
        .all()
    )

    patients_with_infor = []
    for row in rows:
        patient = row[0]
        latest_case = None
        if row.case_id:
            latest_case = LatestCaseInfo(
                id=row.case_id,
                image_path=row.image_path,
                processed_img_path=row.processed_img_path,
                timestamp=row.timestamp,
                similar_cases=row.similar_cases,
                similarity_scores=row.similarity_scores,
                diagnosis=row.diagnosis,
                findings=row.findings,
            )

        patient_data = {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "history": patient.history,
            "created_at": patient.created_at,
            "blood_type": patient.blood_type,
            "status": patient.status,
            "underlying_condition": patient.underlying_condition,
            "phone_number": patient.phone_number,
            "fcm_token": patient.fcm_token,
            "latest_case": latest_case,
        }
        patients_with_infor.append(patient_data)
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "patients": patients_with_infor
    }


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: UUID,
    db: Session = Depends(get_db)
):
    """Get patient by ID"""
    patient = crud_patient.get(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: UUID,
    patient_in: PatientUpdate,
    db: Session = Depends(get_db)
):
    """Update patient information"""
    patient = crud_patient.get(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    update_data = patient_in.model_dump(exclude_unset=True)
    patient = crud_patient.update(db, db_obj=patient, obj_in=update_data)
    return patient


@router.get("/{patient_id}/infor", response_model=PatientInforResponse)
def get_patient_infor(
    patient_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive patient information including latest case data.
    Returns patient details + newest case information (image paths, diagnosis, findings, etc.)
    """
    patient = crud_patient.get(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get latest case for this patient
    latest_case = crud_patient.get_latest_case_for_patient(db, patient_id=str(patient_id))
    
    # Prepare response with patient info
    response_data = {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "history": patient.history,
        "created_at": patient.created_at,
        "blood_type": patient.blood_type,
        "status": patient.status,
        "underlying_condition": patient.underlying_condition,
        "phone_number": patient.phone_number,
        "fcm_token": patient.fcm_token,
        "latest_case": None
    }
    
    # Add latest case info if exists
    if latest_case:
        response_data["latest_case"] = LatestCaseInfo(
            id=latest_case.id,
            image_path=latest_case.image_path,
            processed_img_path=latest_case.processed_img_path,
            timestamp=latest_case.timestamp,
            similar_cases=latest_case.similar_cases,
            similarity_scores=latest_case.similarity_scores,
            diagnosis=latest_case.diagnosis,
            findings=latest_case.findings
        )
    
    return response_data


@router.delete("/{patient_id}", response_model=MessageResponse)
def delete_patient(
    patient_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a patient"""
    patient = crud_patient.get(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    crud_patient.delete(db, id=patient_id)
    return {"message": "Patient deleted successfully"}
