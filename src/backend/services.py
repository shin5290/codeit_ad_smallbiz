import logging, os
from dataclasses import dataclass
from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session
from typing import List, Optional

from src.backend import process_db, schemas
from src.utils.security import verify_password, create_access_token, decode_token
from src.utils.session import normalize_session_id, ensure_chat_session
from src.utils.image import save_uploaded_images
from src.utils.config import PROJECT_ROOT


logger = logging.getLogger(__name__)
task_storage = {}

# 클라이언트는 Bearer 헤더로 전달
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False) 

def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(process_db.get_db),
):
    """
    JWT token으로 현재 로그인 유저 정보 가져오기
    - 토큰이 없으면 None 반환
    """
    if not token:
        return None

    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="토큰 오류")

    user = process_db.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")

    return user

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


def authenticate_user(db, login_id: str, login_pw: str) -> str:
    """
    사용자 인증(로그인) 서비스
    - 아이디 존재 여부 확인, 비밀번호 검증
    - JWT access 토큰 생성 및 반환
    """
    user = process_db.get_user_by_login_id(db, login_id)
    if not user or not verify_password(login_pw, user.login_pw):
        raise HTTPException(400, "아이디 또는 비밀번호가 일치하지 않습니다.")
    return create_access_token(str(user.user_id))


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



@dataclass
class IngestResult:
    session_id: str
    chat_history_id: int
    
async def ingest_user_message(
    *,
    db,
    input_text: str,
    session_id: Optional[str],
    user_id: Optional[int],
    images: List[UploadFile],
) -> IngestResult:
    """
    입력 수집/저장 레이어
    - 세션 확보/생성/유저귀속
    - 유저 텍스트 저장
    - 업로드 이미지 디스크 저장 + DB 연결
    - 반환: session_key, chat_row_id(또는 chat_row), saved_payloads(필요하면)
    """

    session_id = normalize_session_id(session_id) # 프론트에서 받아온 값 정규화

    # 1) 세션 확보/생성/귀속
    session_key = ensure_chat_session(db, session_id, user_id)

    # 2) 텍스트 DB 저장
    chat_row = process_db.save_chat_message(
        db,
        {"session_id": session_key, "role": "user", "content": input_text},
    )

    # 3) 이미지 디스크 저장 + DB 저장
    if images:
        base_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        payloads = await save_uploaded_images(images=images, base_dir=base_dir)

        if payloads:
            process_db.attach_images_to_chat(
                db=db,
                chat_history_id=chat_row.id,
                images=payloads,
                role="input",
            )

    return IngestResult(session_id=session_key, chat_history_id=chat_row.id)
