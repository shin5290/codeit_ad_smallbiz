from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import create_engine, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker, selectinload

from src.backend import models
from src.utils.security import hash_password
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

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
    is_admin = login_id == "admin"
    db_user = models.User(
        login_id=login_id,
        login_pw=hashed_pw,
        name=name,
        is_admin=is_admin,
    )
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


def list_users(
    db: Session,
    *,
    limit: int = 50,
    offset: int = 0,
    user_id: Optional[int] = None,
    login_id: Optional[str] = None,
    name: Optional[str] = None,
    is_admin: Optional[bool] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
):
    """
    유저 목록 조회 (관리자용)
    """
    query = db.query(models.User)
    if user_id is not None:
        query = query.filter(models.User.user_id == user_id)
    if login_id:
        query = query.filter(models.User.login_id.ilike(f"%{login_id}%"))
    if name:
        query = query.filter(models.User.name.ilike(f"%{name}%"))
    if is_admin is not None:
        query = query.filter(models.User.is_admin.is_(is_admin))
    if start_at is not None:
        query = query.filter(models.User.created_at >= start_at)
    if end_at is not None:
        query = query.filter(models.User.created_at < end_at)

    query = query.order_by(models.User.created_at.desc(), models.User.user_id.desc())
    total = query.count()
    users = query.offset(offset).limit(limit).all()
    return users, total


def delete_users_by_ids(
    db: Session,
    *,
    user_ids: List[int],
    exclude_admin: bool = True,
    exclude_user_id: Optional[int] = None,
):
    """
    복수 유저 삭제 (관리자용)
    """
    if not user_ids:
        return [], []

    query = db.query(models.User).filter(models.User.user_id.in_(user_ids))
    if exclude_admin:
        query = query.filter(models.User.is_admin.is_(False))
    if exclude_user_id is not None:
        query = query.filter(models.User.user_id != exclude_user_id)

    users = query.all()
    deleted_ids = [u.user_id for u in users]
    for user in users:
        db.delete(user)
    db.commit()

    skipped_ids = [uid for uid in user_ids if uid not in deleted_ids]
    return deleted_ids, skipped_ids


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
    logger.debug(
        "save_chat_message: session_id=%s, role=%s, image_id=%s",
        session_id,
        role,
        image_id,
    )


    chat = models.ChatHistory(
        session_id=session_id,
        role=role,
        content=content,
        image_id=image_id,
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)

    logger.debug("save_chat_message: chat message saved with id=%s", chat.id)
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
        strength=data.get("strength"),
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


# -----------------------------
# Admin 조회 기능
# -----------------------------

def get_admin_generation_page(
    db: Session,
    *,
    limit: int = 5,
    offset: int = 0,
    user_id: Optional[int] = None,
    login_id: Optional[str] = None,
    session_id: Optional[str] = None,
    content_type: Optional[str] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
):
    """
    관리자용 생성 이력 조회 (필터/페이징)
    """
    query = (
        db.query(models.GenerationHistory)
        .join(models.ChatSession, models.ChatSession.session_id == models.GenerationHistory.session_id)
        .outerjoin(models.User, models.User.user_id == models.ChatSession.user_id)
        .options(
            selectinload(models.GenerationHistory.input_image),
            selectinload(models.GenerationHistory.output_image),
            selectinload(models.GenerationHistory.session).selectinload(models.ChatSession.user),
        )
    )

    if session_id:
        query = query.filter(models.GenerationHistory.session_id == session_id)
    if content_type:
        query = query.filter(models.GenerationHistory.content_type == content_type)
    if user_id is not None:
        query = query.filter(models.ChatSession.user_id == user_id)
    if login_id:
        query = query.filter(models.User.login_id.ilike(f"%{login_id}%"))
    if start_at is not None:
        query = query.filter(models.GenerationHistory.created_at >= start_at)
    if end_at is not None:
        query = query.filter(models.GenerationHistory.created_at < end_at)

    total = query.count()
    items = (
        query.order_by(models.GenerationHistory.created_at.desc(), models.GenerationHistory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return items, total


def get_latest_user_input_before(
    db: Session,
    *,
    session_id: str,
    before_at: Optional[datetime],
):
    """
    세션 내 마지막 사용자 메시지 조회 (생성 시점 기준)
    """
    query = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.session_id == session_id)
        .filter(models.ChatHistory.role == "user")
        .order_by(models.ChatHistory.created_at.desc(), models.ChatHistory.id.desc())
    )
    if before_at is not None:
        query = query.filter(models.ChatHistory.created_at <= before_at)
    return query.first()


def get_admin_session_page(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    query: Optional[str] = None,
    user_id: Optional[int] = None,
    login_id: Optional[str] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
):
    """
    관리자용 세션 목록 조회 (필터/페이징)
    """
    message_subq = (
        db.query(
            models.ChatHistory.session_id.label("session_id"),
            func.count(models.ChatHistory.id).label("message_count"),
            func.max(models.ChatHistory.created_at).label("last_message_at"),
        )
        .group_by(models.ChatHistory.session_id)
        .subquery()
    )

    generation_subq = (
        db.query(
            models.GenerationHistory.session_id.label("session_id"),
            func.count(models.GenerationHistory.id).label("generation_count"),
        )
        .group_by(models.GenerationHistory.session_id)
        .subquery()
    )

    query_builder = (
        db.query(
            models.ChatSession.session_id,
            models.ChatSession.user_id,
            models.ChatSession.created_at,
            models.User.login_id,
            func.coalesce(message_subq.c.message_count, 0).label("message_count"),
            message_subq.c.last_message_at.label("last_message_at"),
            func.coalesce(generation_subq.c.generation_count, 0).label("generation_count"),
        )
        .outerjoin(message_subq, models.ChatSession.session_id == message_subq.c.session_id)
        .outerjoin(generation_subq, models.ChatSession.session_id == generation_subq.c.session_id)
        .outerjoin(models.User, models.User.user_id == models.ChatSession.user_id)
    )

    if query:
        query_builder = query_builder.filter(models.ChatSession.session_id.ilike(f"%{query}%"))
    if user_id is not None:
        query_builder = query_builder.filter(models.ChatSession.user_id == user_id)
    if login_id:
        query_builder = query_builder.filter(models.User.login_id.ilike(f"%{login_id}%"))
        
    # 세션 조회 정렬 및 날짜 필터는 요청에 따라 '생성일(created_at)' 기준으로 수행
    if start_at is not None:
        query_builder = query_builder.filter(models.ChatSession.created_at >= start_at)
    if end_at is not None:
        query_builder = query_builder.filter(models.ChatSession.created_at < end_at)

    total = query_builder.count()
    items = (
        query_builder.order_by(models.ChatSession.created_at.desc(), models.ChatSession.session_id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return items, total


def get_admin_session_overview(db: Session, *, session_id: str):
    """
    관리자용 세션 요약 정보
    """
    row = (
        db.query(
            models.ChatSession.session_id,
            models.ChatSession.user_id,
            models.ChatSession.created_at,
            models.User.login_id,
            func.count(models.ChatHistory.id).label("message_count"),
            func.max(models.ChatHistory.created_at).label("last_message_at"),
        )
        .outerjoin(models.ChatHistory, models.ChatHistory.session_id == models.ChatSession.session_id)
        .outerjoin(models.User, models.User.user_id == models.ChatSession.user_id)
        .filter(models.ChatSession.session_id == session_id)
        .group_by(
            models.ChatSession.session_id,
            models.ChatSession.user_id,
            models.ChatSession.created_at,
            models.User.login_id,
        )
        .first()
    )
    return row


def get_admin_session_messages(
    db: Session,
    *,
    session_id: str,
    limit: int = 200,
    offset: int = 0,
    query: Optional[str] = None,
    role: Optional[str] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    has_image: Optional[bool] = None,
):
    """
    관리자용 세션 메시지 조회 (페이징)
    """
    query_builder = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.session_id == session_id)
    )
    
    if query:
        query_builder = query_builder.filter(models.ChatHistory.content.ilike(f"%{query}%"))

    if role:
        query_builder = query_builder.filter(models.ChatHistory.role == role)

    if start_at is not None:
        query_builder = query_builder.filter(models.ChatHistory.created_at >= start_at)
    if end_at is not None:
        query_builder = query_builder.filter(models.ChatHistory.created_at < end_at)

    if has_image is True:
        query_builder = query_builder.filter(models.ChatHistory.image_id.isnot(None))
    elif has_image is False:
        query_builder = query_builder.filter(models.ChatHistory.image_id.is_(None))
        
    query_builder = query_builder.options(selectinload(models.ChatHistory.image)).order_by(
        models.ChatHistory.created_at.desc(), models.ChatHistory.id.desc()
    )
    
    total = query_builder.count()
    items = query_builder.offset(offset).limit(limit).all()
    return items, total


def search_admin_messages(
    db: Session,
    *,
    query: Optional[str] = None,
    level: Optional[str] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    관리자용 채팅 메시지 검색 (필터/페이징)
    """
    query_builder = (
        db.query(
            models.ChatHistory,
            models.ChatSession.user_id,
            models.User.login_id,
        )
        .join(models.ChatSession, models.ChatSession.session_id == models.ChatHistory.session_id)
        .outerjoin(models.User, models.User.user_id == models.ChatSession.user_id)
    )

    if query:
        query_builder = query_builder.filter(models.ChatHistory.content.ilike(f"%{query}%"))

    if level:
        level_key = level.strip().lower()
        keyword_map = {
            "error": ["error", "failed", "exception", "traceback", "permission denied", "undefined"],
            "warn": ["warn", "warning", "deprecated"],
            "warning": ["warn", "warning", "deprecated"],
            "critical": ["critical", "fatal", "panic"],
        }
        keywords = keyword_map.get(level_key)
        if keywords:
            filters = [models.ChatHistory.content.ilike(f"%{kw}%") for kw in keywords]
            query_builder = query_builder.filter(or_(*filters))

    if start_at is not None:
        query_builder = query_builder.filter(models.ChatHistory.created_at >= start_at)
    if end_at is not None:
        query_builder = query_builder.filter(models.ChatHistory.created_at < end_at)

    total = query_builder.count()
    items = (
        query_builder.order_by(models.ChatHistory.created_at.desc(), models.ChatHistory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return items, total
