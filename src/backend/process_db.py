from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import models, schemas
from utils.auth_utils import hash_password, verify_password
from utils.config import settings

# 엔진 생성 
engine = create_engine(settings.DATABASE_URL)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 테이블 생성 함수 (main.py 실행 시 호출)
def init_db():
    from models import Base
    Base.metadata.create_all(bind=engine)

# FastAPI에서 사용할 DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 유저 조회
def get_user_by_login_id(db: Session, login_id: str):
    return db.query(models.User).filter(models.User.login_id == login_id).first()

# 회원가입
def create_user(db: Session, user: schemas.UserCreate):
    # 비밀번호 암호화 진행
    hashed_pw = hash_password(user.login_pw)

    db_user = models.User(
        login_id=user.login_id,
        login_pw=hashed_pw, 
        name=user.name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user



# 로그인 검증
def authenticate_user(db: Session, user_data: schemas.UserLogin):
    user = get_user_by_login_id(db, user_data.login_id)
    if not user:
        return None
    
    if not verify_password(user_data.login_pw, user.login_pw):
        return None
        
    return user