"""
CRUD operations for Chat models
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from uuid import UUID
from app.core.crud import CRUDBase
from app.models.models import ChatSession, ChatMessage


class CRUDChatSession(CRUDBase[ChatSession]):
    """CRUD operations for ChatSession"""
    
    def get_by_user(
        self, 
        db: Session, 
        *, 
        user_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[ChatSession]:
        """Get all chat sessions for a user"""
        return (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.started_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_active_by_user_and_case(
        self,
        db: Session,
        *,
        user_id: UUID,
        case_id: UUID,
        session_type: str = "tutoring",
    ) -> Optional[ChatSession]:
        """Get the latest active chat session for a user and case."""
        return (
            db.query(ChatSession)
            .filter(
                ChatSession.user_id == user_id,
                ChatSession.case_id == case_id,
                ChatSession.session_type == session_type,
                ChatSession.ended_at.is_(None),
            )
            .order_by(ChatSession.started_at.desc())
            .first()
        )


class CRUDChatMessage(CRUDBase[ChatMessage]):
    """CRUD operations for ChatMessage"""
    
    def get_by_session(
        self, 
        db: Session, 
        *, 
        session_id: UUID
    ) -> List[ChatMessage]:
        """Get all messages for a session"""
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.asc())
            .all()
        )


# Create instances
chat_session = CRUDChatSession(ChatSession)
chat_message = CRUDChatMessage(ChatMessage)
