"""
SQLAlchemy database models
"""
from sqlalchemy import Column, String, Integer, DateTime, Float, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.config.database import Base
import uuid


class Patient(Base):
    """Patient table model"""
    __tablename__ = "patient"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    age = Column(Integer)
    gender = Column(Text)
    history = Column(JSONB)  # Stores medical history with dates {"date": {"diagnosis": ..., "findings": ...}}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    blood_type = Column(Text)  # e.g., A+, B-, O+, AB+
    status = Column(Text)  # stable, improving, critical
    underlying_condition = Column(JSONB)  # JSON object for chronic conditions
    
    # Relationships
    cases = relationship("Case", back_populates="patient", cascade="all, delete-orphan")


class Case(Base):
    """Cases table model"""
    __tablename__ = "cases"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patient.id", ondelete="CASCADE"), nullable=False)
    image_path = Column(Text, nullable=False)  # S3 path to original image
    processed_img_path = Column(Text)  # S3 path to preprocessed image
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    similar_cases = Column(JSON)  # Array of similar case IDs
    similarity_scores = Column(JSON)  # Array of similarity scores
    diagnosis = Column(Text)  # The disease diagnosis (e.g., Pneumonia, TB, etc.)
    findings = Column(Text)  # Extended notes and findings
    
    # Relationships
    patient = relationship("Patient", back_populates="cases")
    ai_results = relationship("AIResult", back_populates="case", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="case", cascade="all, delete-orphan")


class AIResult(Base):
    """AI Result table model"""
    __tablename__ = "ai_result"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    predicted_diagnosis = Column(Text)
    confident_score = Column(Float)
    bounding_box = Column(JSONB)  # Stores bounding box coordinates and labels
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="ai_results")


class Report(Base):
    """Report table model"""
    __tablename__ = "report"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    model_report = Column(Text)  # AI-generated report from MedGemma
    doctor_report = Column(Text)  # Doctor's report/corrections
    feedback_note = Column(Text)  # Feedback for model improvement
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="reports")


class ChatSession(Base):
    """Chat Session table model (for education mode)"""
    __tablename__ = "chat_session"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True))  # Student user ID
    image_path = Column(Text)  # Sample image for learning
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Chat Message table model"""
    __tablename__ = "chat_message"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_session.id", ondelete="CASCADE"), nullable=False)
    from_role = Column(Text, nullable=False)  # 'assistant' or 'user'
    message = Column(Text)
    submitted_image = Column(Text)  # Image with student's bounding box annotations
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
