import datetime
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError

from config import settings
import backend.process_db as process_db

# bcrypt 알고리즘을 사용하도록 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 클라이언트는 Bearer 헤더로 전달
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)  

def hash_password(password: str) -> str:
    """비밀번호를 해시화(암호화)"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """입력된 비밀번호와 DB의 해시값을 비교"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int, extra: dict | None = None) -> str:
    """
    JWT access token 생성
    """
    now = datetime.datetime.utcnow()
    payload = {
        "sub": str(user_id), 
        "iat": now, 
        "exp": now + datetime.timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
        }
    if extra: payload.update(extra)

    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGO)

def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGO])


def get_current_user(token: str | None = Depends(oauth2_scheme), db: Session = Depends(process_db.get_db)):
    """
    JWT token으로 현재 로그인 유저 정보 가져오기
    - 토큰이 없으면 None 반환
    """
    if not token:
        return None

    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
    except JWTError:
        raise HTTPException(status_code=401, detail="토큰 오류")

    user = process_db.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자")
    
    return user