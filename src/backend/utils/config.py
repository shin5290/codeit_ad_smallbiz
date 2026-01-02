from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import Optional

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_path = os.path.join(base_dir, ".env")

class Settings(BaseSettings):
    DATABASE_URL: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=env_path)

settings = Settings()