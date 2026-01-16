"""
RAG 챗봇 API 라우터
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session

from src.backend import process_db, services, schemas
from src.backend.task import get_task


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/message")
async def chat_message(
    background_tasks: BackgroundTasks,
    message: str = Form(..., description="사용자 메시지"),
    session_id: Optional[str] = Form(None, description="세션 ID (선택)"),
    image: Optional[UploadFile] = File(None, description="업로드 이미지 (선택)"),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """
    챗봇에 메시지 전송

    **흐름:**
    1. 사용자 메시지 + 이미지(선택) 수신
    2. RAG 챗봇이 Intent 분석 (generation/modification/consulting)
    3. Intent별 분기:
       - consulting: LLM 상담 응답 반환
       - generation/modification: 생성 파이프라인 시작 + task_id 반환
    """
    user_id = current_user.user_id if current_user else None

    result = await services.handle_chat_message(
        db=db,
        session_id=session_id,
        user_id=user_id,
        message=message,
        image=image,
        background_tasks=background_tasks,
    )

    return result


@router.post("/generate")
async def chat_generate(
    background_tasks: BackgroundTasks,
    session_id: str = Form(..., description="세션 ID"),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """
    챗봇을 통해 수집된 정보로 광고 생성

    **사용 시점:**
    - `/chat/message` 응답에서 `ready_to_generate: true`를 받은 후 호출
    """
    user_id = current_user.user_id if current_user else None
    task_id = str(uuid.uuid4())

    background_tasks.add_task(
        services.handle_chat_generate,
        db=db,
        session_id=session_id,
        user_id=user_id,
        task_id=task_id,
    )

    return {"task_id": task_id}


@router.get("/workflow/{session_id}")
async def get_workflow_state(
    session_id: str,
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """현재 워크플로우 상태 조회 (디버깅/UI 표시용)"""
    from src.backend.chatbot import get_chatbot

    chatbot = get_chatbot()
    workflow_state = chatbot.get_workflow_state(session_id)

    return {
        "session_id": session_id,
        "ad_type": workflow_state.ad_type,
        "business_type": workflow_state.business_type,
        "user_input": workflow_state.user_input,
        "style": workflow_state.style,
        "aspect_ratio": workflow_state.aspect_ratio,
        "is_complete": workflow_state.is_complete,
        "missing_info": workflow_state.get_missing_info(),
    }


@router.post("/workflow/{session_id}/reset")
async def reset_workflow(
    session_id: str,
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """워크플로우 상태 초기화 (새로운 광고 시작)"""
    from src.backend.chatbot import get_chatbot

    chatbot = get_chatbot()
    chatbot.reset_workflow(session_id)

    return {"message": "워크플로우가 초기화되었습니다."}


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 20,
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """대화 히스토리 조회"""
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


# =====================================================
# Phase 3: 수정/컨펌 플로우
# =====================================================

@router.post("/revise")
async def revise_content(
    background_tasks: BackgroundTasks,
    session_id: str = Form(..., description="세션 ID"),
    revision_request: str = Form(..., description="수정 요청 내용"),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """
    생성된 광고 수정 요청

    **파라미터:**
    - `session_id`: 세션 ID
    - `revision_request`: 수정 요청 내용 (예: "더 밝게", "애니메이션 스타일로", "16:9 비율로")
    """
    user_id = current_user.user_id if current_user else None
    task_id = str(uuid.uuid4())

    background_tasks.add_task(
        services.handle_chat_revise,
        db=db,
        session_id=session_id,
        user_id=user_id,
        revision_request=revision_request,
        task_id=task_id,
    )

    return {"task_id": task_id}


@router.post("/confirm")
async def confirm_content(
    session_id: str = Form(..., description="세션 ID"),
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """최종 광고 확정"""
    result = await services.handle_chat_confirm(
        db=db,
        session_id=session_id,
    )

    return result


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
                "is_confirmed": gen.is_confirmed,
                "revision_number": gen.revision_number,
                "revision_of_id": gen.revision_of_id,
                "timestamp": gen.created_at.isoformat(),
                "image": gen.output_image.file_directory if gen.output_image else None,
            }
            for gen in generations
        ]
    }
