import os,sys
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

def _find_project_root(start_path: str) -> str:
    """
    현재 파일 위치에서 상위로 올라가며 src/ 디렉토리를 가진 루트를 찾는다.
    """
    current = os.path.abspath(start_path)
    if os.path.isfile(current):
        current = os.path.dirname(current)

    while True:
        if os.path.isdir(os.path.join(current, "src")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("프로젝트 루트를 찾을 수 없습니다.")
        current = parent

PROJECT_ROOT = _find_project_root(__file__)
env_path = os.path.join(PROJECT_ROOT, ".env")

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
