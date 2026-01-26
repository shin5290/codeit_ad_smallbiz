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

# 로그인/로그아웃 응답
class AuthResponse(BaseModel):
    ok: bool = True

# 세션 요청/응답
class SessionRequest(BaseModel):
    session_id: Optional[str] = None
class SessionResponse(BaseModel):
    session_id: Optional[str] = None

# 회원수정 요청
class UpdateUserRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=50)
    current_password: str = Field(min_length=4, max_length=30)  # 필수
    new_password: Optional[str] = Field(default=None, min_length=4, max_length=30)

    @model_validator(mode="after")
    def validate_update(self):
        if self.name is None and self.new_password is None:
            raise ValueError("변경할 항목이 없습니다.")
        return self

# 회원삭제 요청
class DeleteUserRequest(BaseModel):
    login_pw: str = Field(min_length=4, max_length=30)





# 이미지 파일 서빙
class ImageItem(BaseModel):
    image_hash: str
    file_directory: str 
    role: Literal["input", "output"]

# 채팅 기록 
class ChatMessage(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    image: Optional[ImageItem] = None
    created_at: datetime


class HistoryPage(BaseModel):
    items: List[ChatMessage]
    next_cursor: Optional[int] = None


# -----------------------------
# Admin
# -----------------------------
class AdminUserItem(BaseModel):
    user_id: int
    login_id: str
    name: str
    created_at: datetime
    is_admin: bool


class AdminUserListResponse(BaseModel):
    users: List[AdminUserItem]
    total: int


class AdminDeleteUsersRequest(BaseModel):
    user_ids: List[int]


class AdminDeleteUsersResponse(BaseModel):
    deleted_ids: List[int]
    skipped_ids: List[int]


class AdminImageRef(BaseModel):
    file_hash: str


class AdminGenerationItem(BaseModel):
    id: int
    session_id: str
    user_id: Optional[int] = None
    login_id: Optional[str] = None
    name: Optional[str] = None
    content_type: str
    input_text: Optional[str] = None
    refined_input_text: Optional[str] = None
    output_text: Optional[str] = None
    prompt: Optional[str] = None
    input_image: Optional[AdminImageRef] = None
    output_image: Optional[AdminImageRef] = None
    generation_method: Optional[str] = None
    style: Optional[str] = None
    industry: Optional[str] = None
    seed: Optional[int] = None
    strength: Optional[float] = None
    aspect_ratio: Optional[str] = None
    created_at: datetime


class AdminGenerationPage(BaseModel):
    items: List[AdminGenerationItem]
    total: int
    page: int
    limit: int


class AdminSessionItem(BaseModel):
    session_id: str
    user_id: Optional[int] = None
    login_id: Optional[str] = None
    created_at: datetime
    last_message_at: Optional[datetime] = None
    message_count: int
    generation_count: int = 0


class AdminSessionPage(BaseModel):
    items: List[AdminSessionItem]
    total: int
    limit: int
    offset: int


class AdminSessionMessage(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    image: Optional[AdminImageRef] = None


class AdminGenerationSummary(BaseModel):
    id: int
    content_type: str
    output_text: Optional[str] = None
    output_image: Optional[AdminImageRef] = None
    created_at: datetime
    task_id: Optional[str] = None


class AdminLogHint(BaseModel):
    date: str
    file: str


class AdminSessionDetail(BaseModel):
    session_id: str
    user_id: Optional[int] = None
    login_id: Optional[str] = None
    created_at: datetime
    last_message_at: Optional[datetime] = None
    message_count: int
    run_id: Optional[str] = None
    log_hint: Optional[AdminLogHint] = None
    messages: List[AdminSessionMessage]
    generations: List[AdminGenerationSummary]


class AdminMessageItem(BaseModel):
    id: int
    session_id: str
    user_id: Optional[int] = None
    login_id: Optional[str] = None
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class AdminMessagePage(BaseModel):
    items: List[AdminMessageItem]
    total: int
    limit: int
    offset: int


class AdminLogDatesResponse(BaseModel):
    dates: List[str]


class AdminLogFileItem(BaseModel):
    name: str
    size_bytes: int
    modified_at: datetime


class AdminLogFilesResponse(BaseModel):
    files: List[AdminLogFileItem]


class AdminLogTailResponse(BaseModel):
    lines: List[str]
