from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey,
    DateTime, Boolean, func, Index, CheckConstraint
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


# =====================================================
# User
# =====================================================
class User(Base):
    """
    유저 정보 (계정 단위)
    - 유저 삭제 시에도 세션/이력은 남김 (SET NULL)
    """
    __tablename__ = "user"

    user_id = Column(Integer, primary_key=True, index=True)
    login_id = Column(String(50), unique=True, nullable=False, index=True)
    login_pw = Column(String(255), nullable=False)
    name = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    sessions = relationship(
        "ChatSession",
        back_populates="user",
        cascade="save-update, merge",
        passive_deletes=True,
    )


# =====================================================
# Chat Session
# =====================================================
class ChatSession(Base):
    """
    채팅 세션
    - 절대 삭제하지 않음
    - is_active / closed_at 으로만 상태 관리
    """
    __tablename__ = "chat_session"

    session_id = Column(String(100), primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("user.user_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = Column(DateTime, server_default=func.now())
    is_active = Column(Boolean, default=True)
    closed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")

    chat_histories = relationship(
        "ChatHistory",
        back_populates="session",
        cascade="save-update, merge",
        passive_deletes=True,
    )

    generation_histories = relationship(
        "GenerationHistory",
        back_populates="session",
        cascade="save-update, merge",
        passive_deletes=True,
    )


# =====================================================
# Chat History
# =====================================================
class ChatHistory(Base):
    """
    채팅 메시지 단위 로그
    - 이미지 직접 FK 없음
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        ForeignKey("chat_session.session_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)

    created_at = Column(DateTime, server_default=func.now())

    session = relationship("ChatSession", back_populates="chat_histories")

    history_images = relationship(
        "HistoryImage",
        back_populates="chat_history",
        cascade="save-update, merge",
        passive_deletes=True,
    )


# =====================================================
# Generation History
# =====================================================
class GenerationHistory(Base):
    """
    광고/이미지/텍스트 생성 이벤트 로그
    - 이미지는 HistoryImage 통해 연결
    """
    __tablename__ = "generation_history"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        ForeignKey("chat_session.session_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    content_type = Column(String(20), nullable=False)  # image / text / both

    input_text = Column(Text)
    output_text = Column(Text)

    generation_method = Column(String(50))
    style = Column(String(50))
    industry = Column(String(50))
    seed = Column(Integer)
    aspect_ratio = Column(String(10))
    created_at = Column(DateTime, server_default=func.now())

    session = relationship("ChatSession", back_populates="generation_histories")

    history_images = relationship(
        "HistoryImage",
        back_populates="generation_history",
        cascade="save-update, merge",
        passive_deletes=True,
    )


# =====================================================
# Image Matching
# =====================================================
class ImageMatching(Base):
    """
    이미지 파일 레지스트리 (캐시 테이블)
    - file_hash 기준 단일 row
    """
    __tablename__ = "image_matching"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String(128), nullable=False, unique=True, index=True)
    file_directory = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    history_images = relationship(
        "HistoryImage",
        back_populates="image",
        passive_deletes=True,
    )


# =====================================================
# HistoryImage (공용 연결 테이블)
# =====================================================
class HistoryImage(Base):
    """
    ChatHistory / GenerationHistory 공용 이미지 연결 테이블
    - 이미지 역할(role) 관리
    - 둘 중 하나의 FK만 채워져야 함
    """

    __tablename__ = "history_image"

    id = Column(Integer, primary_key=True)

    chat_history_id = Column(
        Integer,
        ForeignKey("chat_history.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    generation_history_id = Column(
        Integer,
        ForeignKey("generation_history.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    image_id = Column(
        Integer,
        ForeignKey("image_matching.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    role = Column(String(20), nullable=False)  # input / output / reference / mask
    position = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "(chat_history_id IS NOT NULL AND generation_history_id IS NULL) OR "
            "(chat_history_id IS NULL AND generation_history_id IS NOT NULL)",
            name="ck_history_image_owner",
        ),
    )

    chat_history = relationship("ChatHistory", back_populates="history_images")
    generation_history = relationship("GenerationHistory", back_populates="history_images")
    image = relationship("ImageMatching", back_populates="history_images")


# =====================================================
# Indexes
# =====================================================
Index("idx_chat_session_user", ChatSession.user_id)
Index(
    "idx_generation_session_created",
    GenerationHistory.session_id,
    GenerationHistory.created_at,
)
Index(
    "idx_history_image_chat_role",
    HistoryImage.chat_history_id,
    HistoryImage.role,
)
Index(
    "idx_history_image_generation_role",
    HistoryImage.generation_history_id,
    HistoryImage.role,
)
