import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# 현재 config.py 파일의 위치를 기준으로 절대 경로 계산
# __file__은 이 코드가 실행되는 파일의 위치를 가리킵니다.
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/utils
project_root = os.path.abspath(os.path.join(current_dir, "../../")) # 최상위 루트
env_path = os.path.join(project_root, ".env")

# 디버깅용: 서버 실행 시 터미널에 경로가 출력
#print(f"[*] Loading .env from: {env_path}")
if not os.path.exists(env_path):
    print("[!] Warning: .env file not found at this path!")

class Settings(BaseSettings):
    DATABASE_URL: str 
    OPENAI_API_KEY: Optional[str] = None
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGO: str="HS256"
    JWT_EXPIRE_MINUTES: int=60

    # env_file_encoding 추가
    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding='utf-8',
        extra='ignore' # .env에 다른 변수가 있어도 무시
    )

settings = Settings()