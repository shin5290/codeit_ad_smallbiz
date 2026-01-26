import datetime
from passlib.context import CryptContext
from jose import jwt

from src.utils.config import settings

# bcrypt 알고리즘을 사용하도록 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") 
_MAX_BCRYPT_PASSWORD_BYTES = 72

def hash_password(password: str) -> str:
    """비밀번호를 해시화(암호화)"""
    if len(password.encode("utf-8")) > _MAX_BCRYPT_PASSWORD_BYTES:
        raise ValueError("비밀번호는 72바이트 이하로 입력해주세요.")
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """입력된 비밀번호와 DB의 해시값을 비교"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except ValueError:
        # bcrypt backend rejects >72 byte passwords; treat as mismatch.
        return False


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
