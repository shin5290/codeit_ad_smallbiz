import hashlib
import uuid
from typing import Dict, List, Optional, TypedDict

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

import models
from utils.auth_utils import hash_password
from utils.config import settings


engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# -----------------------------
# DB init / session
# -----------------------------
def init_db():
    models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# User
# -----------------------------
def get_user_by_login_id(db: Session, login_id: str):
    return db.query(models.User).filter(models.User.login_id == login_id).first()


def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.user_id == user_id).first()


def check_duplicate_id(db: Session, login_id: str) -> bool:
    return db.query(models.User).filter(models.User.login_id == login_id).count() > 0


def create_user(db: Session, login_id: str, login_pw: str, name: str):
    hashed_pw = hash_password(login_pw)
    db_user = models.User(login_id=login_id, login_pw=hashed_pw, name=name)
    db.add(db_user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user: models.User, name: Optional[str] = None, new_login_pw: Optional[str] = None):
    if name is not None:
        user.name = name
    if new_login_pw is not None:
        user.login_pw = hash_password(new_login_pw)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user: models.User):
    db.delete(user)
    db.commit()


# -----------------------------
# Chat Session
# -----------------------------
def create_chat_session(db: Session, user_id: Optional[int] = None, session_id: Optional[str] = None):
    session_key = session_id or str(uuid.uuid4())
    chat_session = models.ChatSession(session_id=session_key, user_id=user_id)
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


def get_chat_session(db: Session, session_id: str):
    return db.query(models.ChatSession).filter(models.ChatSession.session_id == session_id).first()


# -----------------------------
# Image: bytes-hash 기반
# -----------------------------
def calculate_file_hash_from_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_or_create_image_from_bytes(
    db: Session,
    file_bytes: bytes,
    file_directory: str,
) -> models.ImageMatching:
    """
    - file_bytes: 실제 파일 내용(bytes)
    - file_directory: 디스크 경로(서버 내부 경로)
    """
    file_hash = calculate_file_hash_from_bytes(file_bytes)

    existing = (
        db.query(models.ImageMatching)
        .filter(models.ImageMatching.file_hash == file_hash)
        .first()
    )
    if existing:
        # NOTE: 같은 파일이 다른 경로로 저장되는 경우가 있을 수 있는데,
        #       지금은 최초 경로를 유지한다. 필요하면 여기서 갱신 정책을 정할 수 있음.
        return existing

    image = models.ImageMatching(
        file_hash=file_hash,
        file_directory=file_directory,
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


class ImagePayload(TypedDict):
    bytes: bytes
    path: str


def attach_images_to_chat(
    db: Session,
    chat_history_id: int,
    images: List[ImagePayload],
    role: str = "input",
):
    """
    chat_history_id에 이미지 여러 장을 role(input/output)로 연결
    images: [{"bytes": <file bytes>, "path": <disk path>}, ...]
    """
    for idx, img in enumerate(images):
        image_row = get_or_create_image_from_bytes(
            db=db,
            file_bytes=img["bytes"],
            file_directory=img["path"],
        )
        db.add(
            models.HistoryImage(
                chat_history_id=chat_history_id,
                image_id=image_row.id,
                role=role,
                position=idx,
            )
        )
    db.commit()


def attach_images_to_generation(
    db: Session,
    generation_history_id: int,
    images: List[ImagePayload],
    role: str = "output",
):
    """
    generation_history_id에 이미지 여러 장을 role(output)로 연결
    images: [{"bytes": <file bytes>, "path": <disk path>}, ...]
    """
    for idx, img in enumerate(images):
        image_row = get_or_create_image_from_bytes(
            db=db,
            file_bytes=img["bytes"],
            file_directory=img["path"],
        )
        db.add(
            models.HistoryImage(
                generation_history_id=generation_history_id,
                image_id=image_row.id,
                role=role,
                position=idx,
            )
        )
    db.commit()


# -----------------------------
# Chat History 저장/조회
# -----------------------------
def save_chat_message(db: Session, data: Dict):
    """
    텍스트 메시지만 저장.
    이미지 연결은 attach_images_to_chat(...)로 별도 처리 (bytes 필요).
    """
    chat = models.ChatHistory(
        session_id=data["session_id"],
        role=data["role"],
        content=data["content"],
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def get_chat_sessions(db: Session, user_id: int, limit: int = 20):
    sessions = (
        db.query(models.ChatSession)
        .filter(models.ChatSession.user_id == user_id)
        .order_by(models.ChatSession.created_at.desc())
        .limit(limit)
        .all()
    )

    summaries = []
    for session in sessions:
        first_message = (
            db.query(models.ChatHistory.content)
            .filter(
                models.ChatHistory.session_id == session.session_id,
                models.ChatHistory.role == "user",
            )
            .order_by(models.ChatHistory.id.asc())
            .first()
        )
        summaries.append(
            {
                "session_id": session.session_id,
                "first_message": first_message[0] if first_message else None,
                "created_at": session.created_at,
            }
        )
    return summaries


def get_chat_history(db: Session, user_id: int, session_id: str):
    session = (
        db.query(models.ChatSession)
        .filter(models.ChatSession.session_id == session_id, models.ChatSession.user_id == user_id)
        .first()
    )
    if not session:
        return []

    messages = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.session_id == session_id)
        .order_by(models.ChatHistory.id.asc())
        .all()
    )

    history = []
    for message in messages:
        images = []
        for link in message.history_images:
            # role은 지금 단계에서 input/output만 사용한다고 했으니 그대로 내려줌
            if link.image and link.role in ("input", "output"):
                images.append(
                    {
                        "image_hash": link.image.file_hash,
                        "file_path": link.image.file_directory,  # 서버 내부 경로
                        "role": link.role,
                        "position": link.position,
                    }
                )

        images.sort(key=lambda x: (x["position"] is None, x["position"] or 0))

        history.append(
            {
                "role": message.role,
                "content": message.content,
                "images": images,
                "created_at": message.created_at,
            }
        )

    return history


# -----------------------------
# Generation History 저장/조회
# -----------------------------
def save_generation_history(db: Session, data: Dict):
    """
    generation_history row만 저장.
    이미지 연결은 attach_images_to_generation(...)로 별도 처리 (bytes 필요).
    """
    history = models.GenerationHistory(
        session_id=data["session_id"],
        content_type=data["content_type"],
        input_text=data.get("input_text"),
        output_text=data.get("output_text"),
        generation_method=data.get("generation_method"),
        style=data.get("style"),
        industry=data.get("industry"),
        seed=data.get("seed"),
        aspect_ratio=data.get("aspect_ratio"),
    )

    db.add(history)
    db.commit()
    db.refresh(history)
    return history


def get_generation_history(db: Session, user_id: int, limit: int = 20) -> List[Dict]:
    histories = (
        db.query(models.GenerationHistory)
        .join(models.ChatSession, models.GenerationHistory.session_id == models.ChatSession.session_id)
        .filter(models.ChatSession.user_id == user_id)
        .order_by(models.GenerationHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    results = []
    for history in histories:
        output_path = None
        output_hash = None

        # output 이미지 1개만 대표로 뽑음(필요하면 리스트로 확장 가능)
        for image_link in history.history_images:
            if image_link.role == "output" and image_link.image:
                output_path = image_link.image.file_directory
                output_hash = image_link.image.file_hash
                break

        results.append(
            {
                "id": history.id,
                "output_url": output_path,   # 기존 키 유지(프론트가 쓰면)
                "output_hash": output_hash,  # 필요하면 사용
                "created_at": history.created_at,
                "style": history.style,
                "aspect_ratio": history.aspect_ratio,
            }
        )
    return results
