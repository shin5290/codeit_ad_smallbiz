import json
from fastapi.responses import StreamingResponse


from fastapi import APIRouter, Depends, File, Form, HTTPException, HTTPException, UploadFile
from sqlalchemy.orm import Session
from typing import Optional

import src.backend.process_db as process_db
import src.backend.services as services
import src.backend.schemas as schemas
from src.utils.logging import get_logger
from src.utils.session import resolve_session_id

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/message/stream")
async def chat_message_stream(
    message: str = Form(..., description="사용자 메시지"),
    session_id: Optional[str] = Form(None, description="세션 ID (선택)"),
    image: Optional[UploadFile] = File(None, description="업로드 이미지 (선택)"),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user_optional),
):
    """
    챗봇에 메시지 전송 (스트리밍)
    - consulting 응답을 SSE로 스트리밍
    """
    user_id = current_user.user_id if current_user else None

    async def event_stream():
        try:
            async for payload in services.handle_chat_message_stream(
                db=db,
                session_id=session_id,
                user_id=user_id,
                message=message,
                image=image,
            ):
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error(f"chat_message_stream failed: {exc}", exc_info=True)
            error_payload = {"type": "error", "message": "요청 처리 중 오류가 발생했습니다."}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/session", response_model=schemas.SessionResponse)
def get_chat_session(
    payload: schemas.SessionRequest,
    current_user=Depends(services.get_current_user),
    db: Session = Depends(process_db.get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    session_id = resolve_session_id(db, current_user, payload.session_id)
    return {"session_id": session_id}


@router.get("/history", response_model=schemas.HistoryPage)
def get_chat_history_page(
    limit: int = 15,
    cursor: Optional[int] = None,
    current_user=Depends(services.get_current_user),
    db: Session = Depends(process_db.get_db),
):
    """
    유저의 대화 히스토리 페이지 조회
    - limit: 한 페이지에 불러올 아이템 수  
    - cursor: 다음 페이지를 불러오기 위한 커서 ID (없으면 첫 페이지)
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    items, next_cursor = process_db.get_user_history_page(
        db=db,
        user_id=current_user.user_id,
        cursor_id=cursor,
        limit=limit,
    )
    return {"items": items, "next_cursor": next_cursor}

@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 20,
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """대화 히스토리 조회(챗봇용)"""
    messages = process_db.get_chat_history_by_session(db, session_id, limit=limit)

    return {
        "session_id": session_id,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
                "image_id": msg.image_id,
            }
            for msg in messages
        ]
    }


@router.get("/generation/{session_id}")
async def get_generation_history(
    session_id: str,
    limit: int = 10,
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """세션의 광고 생성 이력 조회"""
    generations = process_db.get_generation_history_by_session(db, session_id, limit=limit)

    return {
        "session_id": session_id,
        "generations": [
            {
                "id": gen.id,
                "content_type": gen.content_type,
                "output_text": gen.output_text,
                "style": gen.style,
                "aspect_ratio": gen.aspect_ratio,
                "timestamp": gen.created_at.isoformat(),
                "image": gen.output_image.file_directory if gen.output_image else None,
            }
            for gen in generations
        ]
    }