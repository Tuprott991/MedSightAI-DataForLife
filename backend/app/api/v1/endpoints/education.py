"""
Education Mode API endpoints
"""
import json
from uuid import UUID
from uuid import uuid4
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config.database import get_db
from app.core import (
    case as crud_case,
    chat_session as crud_chat_session,
    chat_message as crud_chat_message,
)
from app.schemas import (
    ChatSessionCreate, ChatSessionResponse,
    ChatMessageCreate, ChatMessageResponse,
    ChatHistoryResponse, ChatMessageRequest, ChatTurnResponse,
    ChatSessionResolveRequest, StudentSubmission, StudentScoreResponse,
    MessageResponse
)
from app.services import openai_llm_service, ai_model_service

router = APIRouter()


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _resolve_chat_image_url(session, message_in: ChatMessageRequest, db: Session) -> str:
    image_url = message_in.image_url
    if not image_url and session.case_id:
        case = crud_case.get(db, session.case_id)
        if case:
            image_url = case.image_path or case.processed_img_path
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url is required for GPT chat")
    return image_url


@router.get("/practice-cases")
async def get_practice_cases(
    disease_type: str = Query(None),
    difficulty: str = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Get practice cases for education mode
    Filtered by disease type and difficulty
    
    TODO: Implement case selection logic
    - Query cases from database
    - Filter by disease type
    - Optionally filter by difficulty level
    - Return unlabeled images for practice
    """
    raise NotImplementedError("Implement practice case selection")


@router.post("/submit-answer", response_model=StudentScoreResponse)
async def submit_student_answer(
    submission: StudentSubmission,
    db: Session = Depends(get_db)
):
    """
    Submit student's diagnosis and bounding boxes
    Calculate score and provide feedback
    
    TODO: Implement scoring algorithm
    1. Compare student's bounding boxes with ground truth (IoU metric)
    2. Compare diagnosis with correct answer
    3. Calculate overall score
    4. Generate detailed feedback using GPT
    """
    # Get session
    session = crud_chat_session.get(db, submission.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Get ground truth for the practice case
    # ground_truth = get_ground_truth(session.image_path)
    
    # TODO: Calculate bounding box accuracy (IoU)
    # bbox_accuracy = calculate_bbox_accuracy(submission.bounding_boxes, ground_truth['bboxes'])
    
    # TODO: Calculate diagnosis accuracy
    # diagnosis_accuracy = calculate_diagnosis_accuracy(submission.diagnosis, ground_truth['diagnosis'])
    
    # TODO: Generate feedback
    # feedback = openai_llm_service.generate_feedback(
    #     student_answer={
    #         'diagnosis': submission.diagnosis,
    #         'bounding_boxes': submission.bounding_boxes
    #     },
    #     correct_answer=ground_truth
    # )
    
    # TODO: Generate comparison heatmap
    # heatmap_path = generate_comparison_heatmap(...)
    
    raise NotImplementedError("Implement student scoring and feedback")


@router.post("/sessions", response_model=ChatSessionResponse, status_code=201)
async def create_chat_session(
    session_in: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """Create a new chat session for student learning"""
    session_data = session_in.model_dump()
    session_data["user_id"] = session_data["user_id"] or uuid4()
    session = crud_chat_session.create(db, obj_in=session_data)
    return session


@router.post("/sessions/resolve", response_model=ChatSessionResponse)
async def resolve_chat_session(
    session_in: ChatSessionResolveRequest,
    db: Session = Depends(get_db)
):
    """Find or create an active chat session for a user and case."""
    user_id = session_in.user_id or uuid4()

    if session_in.case_id:
        existing_session = crud_chat_session.get_active_by_user_and_case(
            db,
            user_id=user_id,
            case_id=session_in.case_id,
            session_type=session_in.session_type,
        )
        if existing_session:
            return existing_session

    session = crud_chat_session.create(
        db,
        obj_in={
            "user_id": user_id,
            "case_id": session_in.case_id,
            "session_type": session_in.session_type,
        },
    )
    return session


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(
    user_id: UUID = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all chat sessions for a user"""
    skip = (page - 1) * page_size
    sessions = crud_chat_session.get_by_user(db, user_id=user_id, skip=skip, limit=page_size)
    return sessions


@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Get chat history for a session"""
    session = crud_chat_session.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = crud_chat_message.get_by_session(db, session_id=session_id)
    
    return {
        "session": session,
        "messages": messages
    }


@router.post("/sessions/{session_id}/messages", response_model=ChatTurnResponse)
async def send_message(
    session_id: UUID,
    message_in: ChatMessageRequest,
    db: Session = Depends(get_db)
):
    """Send a persisted chat message and get a GPT image-grounded response."""
    session = crud_chat_session.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    image_url = _resolve_chat_image_url(session, message_in, db)
    
    # Save user message
    user_message_data = {
        "session_id": session_id,
        "sender": "user",
        "message": message_in.message
    }
    user_message = crud_chat_message.create(db, obj_in=user_message_data)
    
    history = crud_chat_message.get_by_session(db, session_id=session_id)
    
    try:
        ai_response = await run_in_threadpool(
            openai_llm_service.generate_chat_response,
            conversation_history=history,
            student_query=message_in.message,
            image_url=image_url,
            mode=message_in.mode,
            patient_context=message_in.patient_context,
            current_annotations=message_in.current_annotations,
            submitted_diagnosis=message_in.submitted_diagnosis,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GPT chat failed: {exc}") from exc
    
    ai_message = crud_chat_message.create(
        db,
        obj_in={
            "session_id": session_id,
            "sender": "ai",
            "message": ai_response,
        },
    )

    return {
        "session": session,
        "user_message": user_message,
        "assistant_message": ai_message,
    }


@router.post("/sessions/{session_id}/messages/stream")
def stream_message(
    session_id: UUID,
    message_in: ChatMessageRequest,
    db: Session = Depends(get_db)
):
    """Send a persisted chat message and stream a GPT image-grounded response."""
    session = crud_chat_session.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    image_url = _resolve_chat_image_url(session, message_in, db)

    user_message = crud_chat_message.create(
        db,
        obj_in={
            "session_id": session_id,
            "sender": "user",
            "message": message_in.message,
        },
    )
    history = crud_chat_message.get_by_session(db, session_id=session_id)

    def event_generator():
        response_parts: list[str] = []
        yield _sse_event(
            "user_message",
            {
                "session": jsonable_encoder(session),
                "user_message": jsonable_encoder(user_message),
            },
        )

        try:
            for delta in openai_llm_service.stream_chat_response(
                conversation_history=history,
                student_query=message_in.message,
                image_url=image_url,
                mode=message_in.mode,
                patient_context=message_in.patient_context,
                current_annotations=message_in.current_annotations,
                submitted_diagnosis=message_in.submitted_diagnosis,
            ):
                if not delta:
                    continue
                response_parts.append(delta)
                yield _sse_event("delta", {"delta": delta})

            ai_response = "".join(response_parts).strip()
            if not ai_response:
                raise RuntimeError("OpenAI returned an empty streamed response")

            ai_message = crud_chat_message.create(
                db,
                obj_in={
                    "session_id": session_id,
                    "sender": "ai",
                    "message": ai_response,
                },
            )
            yield _sse_event(
                "done",
                {
                    "session": jsonable_encoder(session),
                    "user_message": jsonable_encoder(user_message),
                    "assistant_message": jsonable_encoder(ai_message),
                },
            )
        except Exception as exc:
            yield _sse_event("error", {"detail": f"GPT chat stream failed: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
async def delete_chat_session(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a chat session"""
    session = crud_chat_session.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    crud_chat_session.delete(db, id=session_id)
    return {"message": "Chat session deleted successfully"}
