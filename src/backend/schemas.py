from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

# 회원가입 요청 시 받을 데이터
class UserCreate(BaseModel):
    login_id: str
    login_pw: str
    name: str

# API 응답으로 내보낼 데이터 (비밀번호 제외)
class UserResponse(BaseModel):
    user_id: int
    login_id: str
    name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True) # SQLAlchemy 객체를 읽기 위함

# 로그인 요청
class UserLogin(BaseModel):
    login_id: str
    login_pw: str