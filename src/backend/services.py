import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError

import process_db, schemas
from utils.auth_utils import verify_password, decode_token, create_access_token
from utils.config import settings

try:
    import jwt
except ImportError:  # pragma: no cover - optional dependency
    jwt = None

logger = logging.getLogger(__name__)


def register_user(db, signup: schemas.SignupRequest):
    """
    회원가입 서비스
    - 중복 아이디 체크
    - 비밀번호 해싱 및 저장
    - 사용자 정보 반환
    """
    if process_db.check_duplicate_id(db, signup.login_id):
        raise HTTPException(400, "이미 사용 중인 아이디입니다")

    user = process_db.create_user(
        db=db,
        login_id=signup.login_id,
        login_pw=signup.login_pw,
        name=signup.name,
    )
    return user


def authenticate_user(db, login: schemas.LoginRequest):
    """
    사용자 인증 서비스
    - 아이디 존재 여부 확인
    - 비밀번호 검증
    - 세션 토큰 생성 및 반환
    """
    user = process_db.get_user_by_login_id(db, login.login_id)
    if not user:
        raise HTTPException(400, "아이디 또는 비밀번호가 일치하지 않습니다.")

    if not verify_password(login.login_pw, user.login_pw):
        raise HTTPException(400, "아이디 또는 비밀번호가 일치하지 않습니다.")

    return {
        "access_token": create_access_token(user.user_id),
        "user_id": user.user_id,
        "name": user.name,
        "session_token": _generate_session_token(user.user_id),
    }


def update_user(db: Session, current_user, update: schemas.UpdateUserRequest):
    """
    회원 정보 수정 서비스
    - 이름 변경
    - 비밀번호 변경(현재 비밀번호 검증 필요)
    """
    if update.new_password:
        if not update.current_password or not verify_password(update.current_password, current_user.login_pw):
            raise HTTPException(400, "비밀번호가 올바르지 않습니다.")

    updated = process_db.update_user(
        db=db,
        user=current_user,
        name=update.name if update.name is not None else None,
        new_login_pw=update.new_password if update.new_password else None,
    )
    return updated


def delete_user(db: Session, current_user, login_pw: str):
    """
    회원 삭제 서비스
    - 현재 비밀번호 검증 후 삭제
    """
    if not verify_password(login_pw, current_user.login_pw):
        raise HTTPException(400, "비밀번호가 올바르지 않습니다.")
    process_db.delete_user(db, current_user)



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")  # 클라이언트는 Bearer 헤더로 전달

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(process_db.get_db)):
    """
    JWT token으로 현재 로그인 유저 정보 가져오기
    """
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
    except JWTError:
        raise HTTPException(status_code=401, detail="토큰 오류")
    
    user = process_db.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    
    return user



async def create_advertisement(
        task_id: str, 
        user_id: Optional[int], 
        input_text: str, 
        aspect_ratio: str, 
        task_storage, 
        session_id: Optional[str] = None,
        input_image_paths: Optional[list[str]] = None,
    ):
    task_storage[task_id]["status"] = "processing"
    task_storage[task_id]["progress"] = 10

    db = process_db.SessionLocal()
    chat_session = None

    try:
        if session_id:
            chat_session = process_db.get_chat_session(db, session_id)

        if not chat_session:
            chat_session = process_db.create_chat_session(db, user_id=user_id, session_id=session_id)

        session_key = chat_session.session_id

        task_storage[task_id]["progress"] = 25
        image_prompt = await _generate_image_prompt(input_text)

        task_storage[task_id]["progress"] = 45
        ad_copy = await _generate_ad_copy(input_text)

        task_storage[task_id]["progress"] = 65
        image_url = _build_placeholder_image(input_text, aspect_ratio)

        task_storage[task_id]["progress"] = 80
        process_db.save_generation_history(
            db,
            {
                "session_id": session_key,
                "content_type": "image",
                "input_text": input_text,
                "output_text": ad_copy,
                "output_url": image_url,
                "generation_method": "t2i",
                "style": "concept",
                "seed": 0,
                "aspect_ratio": aspect_ratio,
            },
        )

        process_db.save_chat_message(
            db,
            {
                "session_id": session_key,
                "role": "user",
                "content": input_text,
            },
        )
        process_db.save_chat_message(
            db,
            {
                "session_id": session_key,
                "role": "assistant",
                "content": ad_copy,
                "gen_image_url": image_url,
            },
        )

        task_storage[task_id]["progress"] = 100
        task_storage[task_id]["status"] = "completed"
        task_storage[task_id]["result"] = {
            "ad_copy": ad_copy,
            "session_id": session_key,
            "image_prompt": image_prompt,
            "images": [
            {
                "image_hash": "placeholder",    # 지금은 output_url 기반이라 해시 의미 없음. 아래 개선 참고
                "file_path": image_url,         # 지금은 data:... 이므로 그대로
                "role": "output",
                "position": 0,
            }
        ],
        }
    except Exception as exc:
        logger.error("Task %s 실패: %s", task_id, exc, exc_info=True)
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["error"] = str(exc)
    finally:
        db.close()


def get_generation_history(db, user_id: int, limit: int = 20):
    return process_db.get_generation_history(db, user_id=user_id, limit=limit)


def get_chat_sessions(db, user_id: int, limit: int = 20):
    return process_db.get_chat_sessions(db, user_id=user_id, limit=limit)


def get_chat_history(db, user_id: int, session_id: str):
    return process_db.get_chat_history(db, user_id=user_id, session_id=session_id)


def _generate_session_token(user_id: int) -> str:
    if jwt and getattr(settings, "SECRET_KEY", None):
        payload = {"user_id": user_id, "exp": datetime.utcnow() + timedelta(days=7)}
        return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    return str(uuid.uuid4())


async def _generate_image_prompt(user_input: str) -> str:
    try:
        from src.generation.text_generation.prompt_manager import PromptTemplateManager

        manager = PromptTemplateManager()
        prompt = manager.generate_image_prompt(user_input)
        if isinstance(prompt, dict):
            return prompt.get("positive") or user_input
        return str(prompt)
    except Exception as exc:
        logger.info("프롬프트 생성 실패, fallback 사용: %s", exc)
        return f"high quality, {user_input}"


async def _generate_ad_copy(user_input: str) -> str:
    try:
        from src.generation.text_generation.text_generator import TextGenerator

        generator = TextGenerator()
        return generator.generate_ad_copy(user_input)
    except Exception as exc:
        logger.info("문구 생성 실패, fallback 사용: %s", exc)
        return f"{user_input[:18]} 특별 혜택"


def _build_placeholder_image(user_input: str, aspect_ratio: str) -> str:
    width, height = (600, 600)
    if aspect_ratio == "4:3":
        width, height = 640, 480
    elif aspect_ratio == "3:4":
        width, height = 480, 640

    safe_text = quote(user_input[:40] or "AD")
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect width='100%' height='100%' fill='#f4f5fb'/>"
        f"<text x='50%' y='45%' dominant-baseline='middle' text-anchor='middle' "
        f"font-family='Arial' font-size='20' fill='#333'>Placeholder</text>"
        f"<text x='50%' y='55%' dominant-baseline='middle' text-anchor='middle' "
        f"font-family='Arial' font-size='14' fill='#555'>{safe_text}</text>"
        "</svg>"
    )
    return f"data:image/svg+xml;utf8,{svg}"
