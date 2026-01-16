from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey,
    DateTime, Boolean, func, Index
)
from sqlalchemy.orm import relationship, declarative_base
from pgvector.sqlalchemy import Vector

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
    - 단일 이미지 FK
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
    image_id = Column(
        Integer,
        ForeignKey("image_matching.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # pgvector: 임베딩 벡터 (OpenAI text-embedding-3-small: 1536 차원)
    embedding = Column(Vector(1536), nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    session = relationship("ChatSession", back_populates="chat_histories")

    image = relationship("ImageMatching", back_populates="chat_histories")


# =====================================================
# Generation History
# =====================================================
class GenerationHistory(Base):
    """
    광고/이미지/텍스트 생성 이벤트 로그
    """
    __tablename__ = "generation_history"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        ForeignKey("chat_session.session_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    content_type = Column(String(20), nullable=False)  # image / text

    input_text = Column(Text)
    output_text = Column(Text)
    prompt = Column(Text)  # 이미지 생성용 프롬프트

    input_image_id = Column(
        Integer,
        ForeignKey("image_matching.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    output_image_id = Column(
        Integer,
        ForeignKey("image_matching.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    generation_method = Column(String(50))  # control_type (canny, depth, openpose)
    style = Column(String(50))
    industry = Column(String(50))
    seed = Column(Integer)
    aspect_ratio = Column(String(10))
    created_at = Column(DateTime, server_default=func.now())

    # Revision 관련 필드
    is_confirmed = Column(Boolean, default=False)  # 최종 확정 여부
    revision_of_id = Column(
        Integer,
        ForeignKey('generation_history.id'),
        nullable=True
    )  # 이전 버전 ID (self FK)
    revision_number = Column(Integer, default=0)  # 수정 버전 번호

    session = relationship("ChatSession", back_populates="generation_histories")
    input_image = relationship(
        "ImageMatching",
        back_populates="generation_input_histories",
        foreign_keys=[input_image_id],
    )
    output_image = relationship(
        "ImageMatching",
        back_populates="generation_output_histories",
        foreign_keys=[output_image_id],
    )

    # Revision 관계 정의
    revision_of = relationship(
        "GenerationHistory",
        remote_side=[id],
        backref="revisions"
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

    chat_histories = relationship(
        "ChatHistory",
        back_populates="image",
        passive_deletes=True,
    )
    generation_input_histories = relationship(
        "GenerationHistory",
        back_populates="input_image",
        foreign_keys="GenerationHistory.input_image_id",
        passive_deletes=True,
    )
    generation_output_histories = relationship(
        "GenerationHistory",
        back_populates="output_image",
        foreign_keys="GenerationHistory.output_image_id",
        passive_deletes=True,
    )


# =====================================================
# Indexes
# =====================================================
Index("idx_chat_session_user", ChatSession.user_id)
Index(
    "idx_generation_session_created",
    GenerationHistory.session_id,
    GenerationHistory.created_at,
)
Index("idx_chat_history_image", ChatHistory.image_id)
Index("idx_generation_input_image", GenerationHistory.input_image_id)
Index("idx_generation_output_image", GenerationHistory.output_image_id)
