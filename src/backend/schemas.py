from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# 회원가입 요청
class SignupRequest(BaseModel):
    login_id: str = Field(min_length=3, max_length=20, pattern=r"^[a-zA-Z0-9]+$")
    login_pw: str = Field(min_length=4, max_length=30)
    name: str = Field(min_length=1, max_length=50)

# 회원가입 응답
class UserResponse(BaseModel):
    user_id: int
    login_id: str
    name: str
    created_at: datetime

# 로그인 요청
class LoginRequest(BaseModel):
    login_id: str
    login_pw: str

# 로그인 토큰 응답
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# 세션 요청/응답
class SessionRequest(BaseModel):
    session_id: Optional[str] = None
class SessionResponse(BaseModel):
    session_id: Optional[str] = None

# 회원수정 요청
class UpdateUserRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=50)
    current_password: Optional[str] = Field(default=None, min_length=4, max_length=30)
    new_password: Optional[str] = Field(default=None, min_length=4, max_length=30)

    @model_validator(mode="after")
    def validate_update(self):
        if self.name is None and self.new_password is None:
            raise ValueError("변경할 항목이 없습니다.")
        if self.new_password and not self.current_password:
            raise ValueError("비밀번호를 변경하려면 현재 비밀번호가 필요합니다.")
        return self

# 회원삭제 요청
class DeleteUserRequest(BaseModel):
    login_pw: str = Field(min_length=4, max_length=30)





# 이미지 파일 서빙
class ImageItem(BaseModel):
    image_hash: str
    file_directory: str 
    role: Literal["input", "output"]
    position: Optional[int] = None

# 채팅 기록 
class ChatMessage(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    images: List[ImageItem] = []   # 여러 장 지원
    created_at: datetime


class HistoryPage(BaseModel):
    items: List[ChatMessage]
    next_cursor: Optional[int] = None
