from typing import Optional
import uuid
from sqlalchemy.orm import Session

import backend.process_db as process_db

def normalize_session_id(session_id: Optional[str]) -> str:
    if not session_id or str(session_id) in ("undefined", "null", "Null", "None"):
        return str(uuid.uuid4())
    return session_id


def resolve_session_id(db: Session, current_user, session_id: Optional[str]) -> Optional[str]:
    """
    세션 귀속/조회
    - session_id가 있으면 로그인 유저에 귀속 시도
    - 없으면 최신 session_id 조회
    """
    if session_id:
        attached = process_db.attach_session_to_user(db, session_id, current_user.user_id)
        return attached.session_id if attached else None

    latest = process_db.get_latest_session_by_user_id(db, current_user.user_id)
    
    return latest.session_id if latest else None

def ensure_chat_session(
    db: Session,
    session_id: Optional[str],
    user_id: Optional[int],
) -> str:
    """
    - session_id 있으면 조회
    - 없거나 DB에 없으면 생성
    - user_id 있으면 귀속
    - 반드시 ChatSession을 반환
    """
    chat_session = process_db.get_chat_session(db, session_id)
    
    if not chat_session:
        chat_session = process_db.create_chat_session(db, session_id=session_id, user_id=user_id)
    elif user_id and chat_session.user_id is None:
        process_db.attach_session_to_user(db, session_id, user_id)

    return chat_session.session_id
