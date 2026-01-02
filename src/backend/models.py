"""
- 해시 알고리즘: 이미지를 저장할 때 hashlib 등을 이용해 고유한 해시(예: MD5, SHA-256)를 생성하는 로직이 필요
- 파일 삭제: 만약 GenerationHistory에서 데이터를 지워도 이미지 파일 자체는 남겨둘 건지(캐시 용도),아니면 같이 지울 건지 결정
지금 구조는 매칭 테이블이 따로 있어 이력이 지워져도 파일 정보는 남습니다.
"""
from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey,
    DateTime, func
)
from sqlalchemy.orm import relationship, declarative_base


Base = declarative_base()

class User(Base):
    """
    유저 정보: 로그인 및 세션/기록의 최상위 부모
    """
    __tablename__ = "user"

    user_id = Column(Integer, primary_key=True, index=True)
    login_id = Column(String(50), unique=True, nullable=False, index=True)
    login_pw = Column(String(255), nullable=False)
    name = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # 관계 설정 (Cascade 적용: 유저 삭제 시 관련 데이터 처리)
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    generations = relationship("GenerationHistory", back_populates="user")


class ChatSession(Base):
    """
    채팅 세션: 유저의 '대화 방' 단위 관리
    """
    __tablename__ = "chat_session"

    session_id = Column(String(100), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="sessions")
    chat_histories = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")
    generation_histories = relationship("GenerationHistory", back_populates="session")


class ChatHistory(Base):
    """
    채팅 메시지 개별 기록
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("chat_session.session_id"), nullable=False)
    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    
    # 이미지 매칭 테이블과의 연결 
    input_image_hash = Column(String(128), ForeignKey("input_image_matching.file_hash"), nullable=True)
    gen_image_url = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    session = relationship("ChatSession", back_populates="chat_histories")
    input_image = relationship("InputImageMatching", back_populates="chat_histories")


class GenerationHistory(Base):
    """
    이미지/텍스트 광고 생성 결과물 기록
    """
    __tablename__ = "generation_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=True)
    session_id = Column(String(100), ForeignKey("chat_session.session_id"), nullable=False)
    
    content_type = Column(String(20), nullable=False)  # image / text
    input_text = Column(Text)
    input_image_url = Column(Text)
    
    # 생성된 이미지의 해시값 (매칭 테이블 연결)
    output_image_hash = Column(String(128), ForeignKey("gen_image_matching.file_hash"), nullable=True)
    output_url = Column(Text)
    
    generation_method = Column(String(50))
    style = Column(String(50))
    industry = Column(String(50))
    seed = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="generations")
    session = relationship("ChatSession", back_populates="generation_histories")
    gen_image_info = relationship("GenImageMatching", back_populates="generations")


class GenImageMatching(Base):
    """
    생성된 이미지 파일 경로 매칭 테이블
    """
    __tablename__ = "gen_image_matching"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String(128), unique=True, nullable=False, index=True)
    file_directory = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # GenerationHistory와 연결
    generations = relationship("GenerationHistory", back_populates="gen_image_info")


class InputImageMatching(Base):
    """
    사용자가 업로드한 이미지 파일 경로 매칭 테이블
    """
    __tablename__ = "input_image_matching"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String(128), unique=True, nullable=False, index=True)
    file_directory = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # ChatHistory와 연결 (업로드한 이미지는 채팅 메시지에 포함되므로)
    chat_histories = relationship("ChatHistory", back_populates="input_image")