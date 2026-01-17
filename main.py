import os, logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from uvicorn.logging import AccessFormatter, DefaultFormatter

import src.backend.process_db as process_db
import src.backend.schemas as schemas
import src.backend.services as services
import src.backend.task as task_service
from src.utils.image import get_image_file_response

# RAG 챗봇 라우터 import
from src.backend.routers import chat

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
