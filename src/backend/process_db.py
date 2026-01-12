from typing import Dict, List, Optional, TypedDict
import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker, selectinload

from src.backend import models
from src.utils.security import hash_password
from src.utils.config import settings

logger = logging.getLogger(__name__)


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
def create_chat_session(db: Session, session_id: str, user_id: Optional[int] = None):
    """
    chat_session 생성
    """
    chat_session = models.ChatSession(session_id=session_id, user_id=user_id)
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session

def get_chat_session(db: Session, session_id: str):
    """
    session_id로 chat session 가져오기
    """
    return (
        db.query(models.ChatSession)
        .filter(models.ChatSession.session_id == session_id)
        .first()
    )

def get_latest_session_by_user_id(db: Session, user_id: int):
    """
    유저의 가장 최신 session 가져오기
    """
    return (
        db.query(models.ChatSession)
        .filter(models.ChatSession.user_id == user_id)
        .order_by(models.ChatSession.created_at.desc())
        .first()
    )

def get_session_summaries(db: Session, user_id: int, limit: int = 20):
    """
    유저 세션 목록(List)
    """
    sessions = (
        db.query(models.ChatSession)
        .filter(models.ChatSession.user_id == user_id)
        .order_by(models.ChatSession.created_at.desc())
        .limit(limit)
        .all()
    )

    results = []
    for session in sessions:
        last_message = (
            db.query(models.ChatHistory)
            .filter(models.ChatHistory.session_id == session.session_id)
            .order_by(models.ChatHistory.created_at.desc())
            .first()
        )

        results.append({
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_message": last_message.content if last_message else None,
            "last_message_at": last_message.created_at if last_message else None,
        })

    return results


def user_owns_session(db: Session, user_id: int, session_id: str) -> bool:
    """
    세션 소유권 체크
    """
    return (
        db.query(models.ChatSession)
        .filter(
            models.ChatSession.session_id == session_id,
            models.ChatSession.user_id == user_id,
        )
        .first()
        is not None
    )

def attach_session_to_user(db: Session, session_id: str, user_id: int) -> Optional[models.ChatSession]:
    """
    게스트 session_id를 로그인 user_id에 귀속시킴.
    - 이미 다른 유저에게 귀속된 세션이면 건드리지 않음(보안).
    - user_id가 None(게스트)이면 user_id로 업데이트.
    """
    s = get_chat_session(db, session_id)
    if not s:
        return None

    # 이미 다른 유저 소유면 탈취 방지
    if s.user_id is not None and s.user_id != user_id:
        return None

    # 게스트 -> 유저 귀속
    if s.user_id is None:
        s.user_id = user_id
        db.add(s)
        db.commit()
        db.refresh(s)

    return s


# -----------------------------
# Image: bytes-hash 기반
# -----------------------------
def save_image_from_hash(
    db: Session,
    file_hash: str,
    file_directory: str,
) -> models.ImageMatching:
    """
    이미지 저장(중복X)
    - file_hash: 실제 파일 내용의 해시값
    - file_directory: 디스크 경로(서버 내부 경로)
    """
    logger.info(f"save_image_from_hash: file_hash={file_hash}, file_directory={file_directory}")

    existing = (
        db.query(models.ImageMatching)
        .filter(models.ImageMatching.file_hash == file_hash)
        .first()
    )
    if existing:
        # NOTE: 같은 파일이 다른 경로로 저장되는 경우가 있을 수 있는데,
        #       지금은 최초 경로를 유지한다. 필요하면 여기서 갱신 정책을 정할 수 있음.
        logger.info(f"save_image_from_hash: existing image found with id={existing.id}")
        return existing

    image = models.ImageMatching(
        file_hash=file_hash,
        file_directory=file_directory,
    )
    db.add(image)
    db.commit()
    db.refresh(image)

    logger.info(f"save_image_from_hash: new image saved with id={image.id}")
    return image


def get_image_by_hash(db: Session, file_hash: str) -> models.ImageMatching | None:
    """
    이미지 조회
    """
    return (
        db.query(models.ImageMatching)
        .filter(models.ImageMatching.file_hash == file_hash)
        .first()
    )


class ImagePayload(TypedDict):
    file_hash: str
    file_directory: str


def attach_image_to_chat(
    db: Session,
    chat_history_id: int,
    image: ImagePayload,
):
    """
    chat_history_id에 단일 이미지 연결
    image: {"file_hash": <file hash>, "file_directory": <file_directory>}
    """
    image_row = save_image_from_hash(
        db=db,
        file_hash=image["file_hash"],
        file_directory=image["file_directory"],
    )
    (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.id == chat_history_id)
        .update({"image_id": image_row.id})
    )
    db.commit()
    return image_row

# -----------------------------
# Chat History 저장/조회
# -----------------------------
def save_chat_message(db: Session, data: Dict):
    """
    텍스트 메시지 저장.
    image_id가 있으면 함께 저장.
    """
    logger.info(f"save_chat_message: session_id={data['session_id']}, role={data['role']}, image_id={data.get('image_id')}")

    chat = models.ChatHistory(
        session_id=data["session_id"],
        role=data["role"],
        content=data["content"],
        image_id=data.get("image_id"),
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)

    logger.info(f"save_chat_message: chat message saved with id={chat.id}")
    return chat

def _to_history_dicts(messages: list[models.ChatHistory]):
    """
    history 변환 로직
    """
    history = []
    for message in messages:
        image = None
        if message.image:
            role = "input" if message.role == "user" else "output"
            image = {
                "image_hash": message.image.file_hash,
                "file_directory": f"/images/{message.image.file_hash}",
                "role": role,
            }

        history.append({
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "image": image,
            "created_at": message.created_at,
        })
    return history


def _collect_user_history_images(
    db: Session,
    user_id: int,
    role: Optional[str] = None,
) -> List[Dict]:
    allowed_roles = ("input", "output")
    if role is not None and role not in allowed_roles:
        return []

    results = []

    chat_q = (
        db.query(models.ChatHistory, models.ImageMatching)
        .join(models.ImageMatching, models.ImageMatching.id == models.ChatHistory.image_id)
        .join(models.ChatSession, models.ChatSession.session_id == models.ChatHistory.session_id)
        .filter(models.ChatSession.user_id == user_id)
    )
    if role == "input":
        chat_q = chat_q.filter(models.ChatHistory.role == "user")
    elif role == "output":
        chat_q = chat_q.filter(models.ChatHistory.role == "assistant")

    for chat_history, image in chat_q.all():
        role_value = "input" if chat_history.role == "user" else "output"
        results.append({
            "history_id": chat_history.id,
            "image_id": image.id,
            "role": role_value,
            "created_at": chat_history.created_at,
            "image": image,
        })

    if role is None or role == "input":
        gen_in_q = (
            db.query(models.GenerationHistory, models.ImageMatching)
            .join(models.ImageMatching, models.ImageMatching.id == models.GenerationHistory.input_image_id)
            .join(models.ChatSession, models.ChatSession.session_id == models.GenerationHistory.session_id)
            .filter(models.ChatSession.user_id == user_id)
        )
        for gen_history, image in gen_in_q.all():
            results.append({
                "history_id": gen_history.id,
                "image_id": image.id,
                "role": "input",
                "created_at": gen_history.created_at,
                "image": image,
            })

    if role is None or role == "output":
        gen_out_q = (
            db.query(models.GenerationHistory, models.ImageMatching)
            .join(models.ImageMatching, models.ImageMatching.id == models.GenerationHistory.output_image_id)
            .join(models.ChatSession, models.ChatSession.session_id == models.GenerationHistory.session_id)
            .filter(models.ChatSession.user_id == user_id)
        )
        for gen_history, image in gen_out_q.all():
            results.append({
                "history_id": gen_history.id,
                "image_id": image.id,
                "role": "output",
                "created_at": gen_history.created_at,
                "image": image,
            })

    deduped = {}
    for entry in results:
        key = (entry["image_id"], entry["role"])
        if key not in deduped or entry["created_at"] > deduped[key]["created_at"]:
            deduped[key] = entry

    return list(deduped.values())




def get_user_history_page(db, user_id, cursor_id=None, limit=15):
    """
    유저별 히스토리 페이징
    정렬: created_at desc, id desc
    cursor 있으면 ChatHistory.id < cursor_id
    가져온 뒤 프론트 편하게 하려고 reverse 해서 오래된→최신 순으로 반환
    """
    q = (
        db.query(models.ChatHistory)
        .join(models.ChatSession, models.ChatSession.session_id == models.ChatHistory.session_id)
        .filter(models.ChatSession.user_id == user_id)
        .options(
            selectinload(models.ChatHistory.image)
        )
        .order_by(models.ChatHistory.created_at.desc(), models.ChatHistory.id.desc())
    )

    if cursor_id:
        q = q.filter(models.ChatHistory.id < cursor_id)

    messages_desc = q.limit(limit).all()
    if not messages_desc:
        return [], None

    next_cursor = messages_desc[-1].id if len(messages_desc) == limit else None
    messages = list(reversed(messages_desc))
    history = _to_history_dicts(messages)

    return history, next_cursor

def get_session_history_page(db, session_id, cursor_id=None, limit=20):
    """
    세션별 히스토리 페이징
    """
    q = (
        db.query(models.ChatHistory)
        .options(
            selectinload(models.ChatHistory.image)
        )
        .filter(models.ChatHistory.session_id == session_id)
        .order_by(models.ChatHistory.created_at.desc(), models.ChatHistory.id.desc())
    )

    if cursor_id:
        q = q.filter(models.ChatHistory.id < cursor_id)

    messages_desc = q.limit(limit).all()
    if not messages_desc:
        return [], None

    next_cursor = messages_desc[-1].id if len(messages_desc) == limit else None
    messages = list(reversed(messages_desc))
    history = _to_history_dicts(messages)

    return history, next_cursor


# -----------------------------
# Generation History 저장
# -----------------------------
def save_generation_history(db: Session, data: Dict):
    """
    GenerationHistory 저장
    """
    gen_history = models.GenerationHistory(
        session_id=data["session_id"],
        content_type=data["content_type"],
        input_text=data.get("input_text"),
        output_text=data.get("output_text"),
        prompt=data.get("prompt"),  # 이미지 생성용 프롬프트
        input_image_id=data.get("input_image_id"),
        output_image_id=data.get("output_image_id"),
        generation_method=data.get("generation_method"),
        style=data.get("style"),
        industry=data.get("industry"),
        seed=data.get("seed"),
        aspect_ratio=data.get("aspect_ratio"),
    )
    db.add(gen_history)
    db.commit()
    db.refresh(gen_history)
    return gen_history
