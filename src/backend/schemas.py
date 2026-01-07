from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SignupRequest(BaseModel):
    login_id: str = Field(min_length=3, max_length=20, pattern=r"^[a-zA-Z0-9]+$")
    login_pw: str = Field(min_length=4, max_length=30)
    name: str = Field(min_length=1, max_length=50)


class LoginRequest(BaseModel):
    login_id: str
    login_pw: str


class UpdateUserRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=50)
    current_password: Optional[str] = Field(default=None, min_length=4, max_length=30)
    new_password: Optional[str] = Field(default=None, min_length=4, max_length=30)

    @model_validator(mode="after")
    def validate_update(self):
        if self.name is None and self.new_password is None:
            raise ValueError("변경할 항목이 없습니다.")
        if self.new_password and not self.current_password:
            raise ValueError("비밀번호를 변경하려면 current_password가 필요합니다.")
        return self


class DeleteUserRequest(BaseModel):
    login_pw: str = Field(min_length=4, max_length=30)


class UserResponse(BaseModel):
    user_id: int
    login_id: str
    name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AuthResponse(BaseModel):
    access_token: str
    user_id: int
    name: str
    session_token: str

class ImageItem(BaseModel):
    image_hash: str
    file_path: str  # DB의 image_matching.file_directory
    role: Literal["input", "output"]
    position: Optional[int] = None

class GenerateRequest(BaseModel):
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    input_text: str = Field(min_length=1, max_length=500)
    aspect_ratio: Literal["1:1", "4:3", "3:4"]


class TaskResult(BaseModel):
    ad_copy: Optional[str] = None
    session_id: Optional[str] = None
    images: List[ImageItem] = []


class TaskStatus(BaseModel):
    status: Literal["pending", "processing", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    result: Optional[TaskResult] = None
    error: Optional[str] = None


class GenerationHistoryItem(BaseModel):
    id: int
    output_url: Optional[str]
    created_at: datetime
    style: Optional[str]
    aspect_ratio: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class ChatSessionSummary(BaseModel):
    session_id: str
    first_message: Optional[str]
    created_at: datetime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    images: List[ImageItem] = []   # 여러 장 지원
    created_at: datetime
