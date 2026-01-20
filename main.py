import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session
from uvicorn.logging import AccessFormatter, DefaultFormatter

import src.backend.process_db as process_db
import src.backend.services as services
from src.backend.routers import admin, auth, chat
from src.utils.image import get_image_file_response
from src.utils.logging import setup_logging, get_logger
from src.generation.image_generation.preload import preload_models


# 로깅 설정
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB 초기화
    process_db.init_db()

    # 이미지 생성 모델 preload (GPU에 미리 올려둠)
    logger.info("Starting model preload...")
    preload_models(device="cuda")
    logger.info("Server startup complete!")

    yield
    

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
app.mount(
    "/static",
    StaticFiles(directory="src/frontend/static"),
    name="static",
)
current_dir = os.path.dirname(os.path.abspath(__file__))
app.include_router(admin.router) # 관리자 기능

@app.get("/")
async def read_index():
    file_path = os.path.join(current_dir, "src", "frontend", "test.html")
    return FileResponse(file_path, headers={"Cache-Control": "no-store"})


@app.get("/admin")
async def read_admin(current_user=Depends(services.get_current_user_optional)):
    if not current_user or not getattr(current_user, "is_admin", False):
        return RedirectResponse(url="/", status_code=302)
    file_path = os.path.join(current_dir, "src", "frontend", "admin.html")
    return FileResponse(file_path, headers={"Cache-Control": "no-store"})


app.include_router(auth.router) # 인증 및 사용자 관리
app.include_router(chat.router) # 챗봇 및 대화 관리


# 이미지 서빙
@app.get("/images/{file_hash}")
def get_image(file_hash: str, db: Session = Depends(process_db.get_db)):
    """
    “파일 경로”를 “URL”로 바꿔주는 이미지 서빙
    """
    return get_image_file_response(db, file_hash)
