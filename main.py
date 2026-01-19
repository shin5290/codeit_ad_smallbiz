import os, logging, json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Response, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from uvicorn.logging import AccessFormatter, DefaultFormatter

import src.backend.process_db as process_db
import src.backend.schemas as schemas
import src.backend.services as services
from src.utils.image import get_image_file_response
from src.utils.session import resolve_session_id

# 로깅 설정
handler = logging.StreamHandler()
handler.setFormatter(DefaultFormatter("%(levelprefix)s %(asctime)s.%(msecs)03d - %(message)s", use_colors=True, datefmt='%Y-%m-%d %H:%M:%S'))

logging.root.handlers = []
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)

access_handler = logging.StreamHandler()
access_handler.setFormatter(AccessFormatter("%(levelprefix)s %(asctime)s.%(msecs)03d - \"%(request_line)s\" %(status_code)s", datefmt="%Y-%m-%d %H:%M:%S"))
access_logger = logging.getLogger("uvicorn.access")
access_logger.handlers = [access_handler]
access_logger.propagate = False

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

# 정적 파일 서빙 설정
app.mount(
    "/static",
    StaticFiles(directory="src/frontend/static"),
    name="static",
)


@app.get("/")
async def read_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "src", "frontend", "test.html")
    return FileResponse(file_path, headers={"Cache-Control": "no-store"})



# -----------------------------
# 인증 및 사용자 관리
# -----------------------------

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


@app.post("/auth/login", response_model=schemas.AuthResponse)
def login(
    response: Response,
    db: Session = Depends(process_db.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    services.authenticate_user(db, form_data.username, form_data.password, response)
    return {"ok": True}


@app.post("/auth/logout", response_model=schemas.AuthResponse)
def logout(response: Response):
    response.delete_cookie("access_token", path="/")
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




# -----------------------------
# 챗봇 및 대화 관리
# -----------------------------

@app.post("/chat/message/stream")
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

@app.get("/chat/history/{session_id}")
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


@app.get("/chat/generation/{session_id}")
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



# -----------------------------
# 이미지 서빙
# -----------------------------

@app.get("/images/{file_hash}")
def get_image(file_hash: str, db: Session = Depends(process_db.get_db)):
    """
    “파일 경로”를 “URL”로 바꿔주는 이미지 서빙
    """
    return get_image_file_response(db, file_hash)

