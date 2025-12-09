"""
Patient API endpoints
"""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import patient as crud_patient
from app.schemas import (
    PatientCreate, PatientUpdate, PatientResponse, 
    PatientListResponse, PatientInforResponse, LatestCaseInfo, MessageResponse
)

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
        "latest_case": None
    }
    
    # Add latest case info if exists
    if latest_case:
        response_data["latest_case"] = LatestCaseInfo(
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
