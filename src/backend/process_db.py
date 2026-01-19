from typing import Dict, List, Optional
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

# -----------------------------
# Chat History 저장/조회
# -----------------------------
def save_chat_message(
        db: Session, 
        session_id: str,
        role: str,
        content: str,
        image_id: Optional[int] = None,
    ):
    """
    텍스트 메시지 저장.
    image_id가 있으면 함께 저장.
    """
    logger.info(f"save_chat_message: session_id={session_id}, role={role}, image_id={image_id}")


    chat = models.ChatHistory(
        session_id=session_id,
        role=role,
        content=content,
        image_id=image_id,
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

def get_chat_history_by_session(
    db: Session,
    session_id: str,
    limit: Optional[int] = 10,
) -> List[models.ChatHistory]:
    """
    세션별 대화 이력 조회
    """
    query = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.session_id == session_id)
        .order_by(models.ChatHistory.created_at.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    return query.all()


# -----------------------------
# Generation History 저장/조회
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


def get_generation_history_by_session(
    db: Session,
    session_id: str,
    limit: Optional[int] = 5,
) -> List[models.GenerationHistory]:
    """
    세션별 생성 이력 조회
    """
    query = (
        db.query(models.GenerationHistory)
        .filter(models.GenerationHistory.session_id == session_id)
        .order_by(models.GenerationHistory.created_at.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    return query.all()


def get_latest_generation(
    db: Session,
    session_id: str,
) -> Optional[models.GenerationHistory]:
    """
    세션의 가장 최근 생성 이력 조회
    """
    return (
        db.query(models.GenerationHistory)
        .filter(models.GenerationHistory.session_id == session_id)
        .order_by(models.GenerationHistory.created_at.desc())
        .first()
    )


def get_generation_by_session_and_id(
    db: Session,
    session_id: str,
    generation_id: int,
) -> Optional[models.GenerationHistory]:
    """
    세션ID와 생성ID로 특정 생성 이력 조회
    """
    return (
        db.query(models.GenerationHistory)
        .filter(models.GenerationHistory.session_id == session_id)
        .filter(models.GenerationHistory.id == generation_id)
        .first()
    )
