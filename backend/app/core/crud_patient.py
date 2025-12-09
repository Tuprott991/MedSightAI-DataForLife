"""
CRUD operations for Patient model
"""
from typing import Optional, List
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from app.core.crud import CRUDBase
from app.models.models import Patient, Case
from app.schemas.patient import PatientCreate, PatientUpdate


class CRUDPatient(CRUDBase[Patient]):
    """CRUD operations for Patient"""
    
    def get_by_name(self, db: Session, *, name: str) -> Optional[Patient]:
        """Get patient by name"""
        return db.query(Patient).filter(Patient.name == name).first()
    
    def search_patients(
        self, 
        db: Session, 
        *, 
        query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Patient]:
        """Search patients by name"""
        return (
            db.query(Patient)
            .filter(Patient.name.ilike(f"%{query}%"))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_patient_with_latest_case(self, db: Session, *, patient_id: str) -> Optional[Patient]:
        """Get patient with their latest case information"""
        return (
            db.query(Patient)
            .options(joinedload(Patient.cases))
            .filter(Patient.id == patient_id)
            .first()
        )
    
    def get_latest_case_for_patient(self, db: Session, *, patient_id: str) -> Optional[Case]:
        """Get the most recent case for a patient"""
        return (
            db.query(Case)
            .filter(Case.patient_id == patient_id)
            .order_by(desc(Case.timestamp))
            .first()
        )


# Create instance
patient = CRUDPatient(Patient)
