import os, sys, uuid, asyncio, hashlib, logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import Depends, FastAPI, HTTPException, Response, status, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import src.backend.process_db as process_db
import src.backend.schemas as schemas
import src.backend.services as services
import src.backend.task as task_service
from src.utils.image import get_image_file_response
from src.utils.session import resolve_session_id, normalize_session_id, ensure_chat_session

# RAG 챗봇 라우터 import
from src.backend.routers import chat

# 로깅 설정
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # 모듈명 추출 (src.backend.services -> services)
        name_parts = record.name.split('.')
        module_name = name_parts[-1] if name_parts else record.name
        record.module_name = module_name
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter('[%(asctime)s]  %(levelname)-7s [%(module_name)s] %(message)s', datefmt='%y/%m/%d %H:%M:%S'))
logging.root.handlers = []
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    process_db.init_db()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 챗봇 라우터 등록
app.include_router(chat.router)

@app.get("/")
async def read_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "src", "frontend", "test.html")
    return FileResponse(file_path, headers={"Cache-Control": "no-store"})

@app.get("/auth/me")
def me(current_user = Depends(services.get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    return {
        "name": current_user.name,
    }


@app.post("/auth/signup", response_model=schemas.UserResponse)
def signup(user: schemas.SignupRequest, db: Session = Depends(process_db.get_db)):
    try:
        return services.register_user(db, user)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.post("/auth/login", response_model=schemas.TokenResponse)
def login(
    db: Session = Depends(process_db.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    token = services.authenticate_user(db, form_data.username, form_data.password)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/logout")
def logout():
    return {"ok": True}

@app.put("/auth/user", response_model=schemas.UserResponse)
def update_user(
    user_data: schemas.UpdateUserRequest,
    current_user=Depends(services.get_current_user),
    db: Session = Depends(process_db.get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    return services.update_user(db, current_user, user_data)

@app.delete("/auth/user", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    delete_data: schemas.DeleteUserRequest,
    current_user=Depends(services.get_current_user),
    db: Session = Depends(process_db.get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    services.delete_user(db, current_user, delete_data.login_pw)
    return Response(status_code=status.HTTP_204_NO_CONTENT)




@app.post("/generate")
async def generate_advertisement(
    background_tasks: BackgroundTasks,
    input_text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    generation_type: Optional[str] = Form(None),  # text, image
    style: Optional[str] = Form(None),  # ultra_realistic, semi_realistic, anime
    aspect_ratio: Optional[str] = Form(None),  # 1:1, 16:9, 9:16, 4:3
    db: Session = Depends(process_db.get_db),
    current_user=Depends(services.get_current_user),
):
    """
    광고 생성 요청 엔드포인트
    - 즉시 task_id 반환
    - 백그라운드에서 파이프라인 실행
    - /tasks/{task_id}로 진행 상황 조회
    """
    task_id = str(uuid.uuid4())
    user_id = current_user.user_id if current_user else None

    logger.info(
        f"[/generate] Request received - "
        f"task_id={task_id}, "
        f"input_text={input_text[:50] if len(input_text) > 50 else input_text}, "
        f"generation_type={generation_type}, "
        f"has_image={image is not None}"
    )
    if image:
        logger.info(
            f"[/generate] Image details - "
            f"filename={image.filename}, "
            f"content_type={image.content_type}, "
            f"size={image.size if hasattr(image, 'size') else 'unknown'}"
        )

    try:
        session_key = normalize_session_id(session_id)
        session_key = ensure_chat_session(db, session_key, user_id)
        logger.info(f"[/generate] Session: {session_key}, User: {user_id}, Task: {task_id}")

        # Task 생성 (즉시)
        task_service.create_task(task_id)

        # 백그라운드 작업으로 파이프라인 실행
        background_tasks.add_task(
            services.handle_generate_pipeline,
            db=db,
            input_text=input_text,
            session_id=session_key,
            user_id=user_id,
            image=image,
            task_id=task_id,
            create_task_entry=False,  # 이미 생성했으므로 False
            generation_type=generation_type,
            style=style,
            aspect_ratio=aspect_ratio,
        )

        logger.info(f"[/generate] Task {task_id} created and queued for background processing")

        # 즉시 task_id 반환
        return {
            "task_id": task_id,
            "session_id": session_key,
            "status": "processing"
        }

    except Exception as exc:
        logger.error(f"[/generate] Failed to create task {task_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"요청 처리 실패: {exc}")


@app.post("/chat/session", response_model=schemas.SessionResponse)
def get_chat_session(
    payload: schemas.SessionRequest,
    current_user=Depends(services.get_current_user),
    db: Session = Depends(process_db.get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    session_id = resolve_session_id(db, current_user, payload.session_id)
    return {"session_id": session_id}


@app.get("/chat/history", response_model=schemas.HistoryPage)
def get_chat_history_page(
    limit: int = 15,
    cursor: Optional[int] = None,
    current_user=Depends(services.get_current_user),
    db: Session = Depends(process_db.get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    items, next_cursor = process_db.get_user_history_page(
        db=db,
        user_id=current_user.user_id,
        cursor_id=cursor,
        limit=limit,
    )
    return {"items": items, "next_cursor": next_cursor}



@app.get("/images/{file_hash}")
def get_image(file_hash: str, db: Session = Depends(process_db.get_db)):
    """
    “파일 경로”를 “URL”로 바꿔주는 이미지 서빙
    """
    return get_image_file_response(db, file_hash)


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, response: Response):
    """
    작업 상태 조회
    - 진행률, 상태, 결과 반환
    - 프론트엔드에서 polling하여 진행 상황 확인
    """
    response.headers["Cache-Control"] = "no-store"
    task = task_service.get_task(task_id)

    if not task:
        logger.warning(f"[GET /tasks/{task_id}] Task not found in storage")
        raise HTTPException(status_code=404, detail="Task not found")

    task_dict = task.to_dict()

    # 완료/실패 상태일 때만 로깅 (polling 노이즈 최소화)
    if task.status in [task_service.TaskStatus.DONE, task_service.TaskStatus.FAILED]:
        logger.info(
            f"[GET /tasks/{task_id}] Status query - "
            f"status={task.status}, progress={task.progress}%"
        )

    return task_dict


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    완료된 작업 삭제 (선택적)
    """
    deleted = task_service.delete_task(task_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {"message": "Task deleted"}
