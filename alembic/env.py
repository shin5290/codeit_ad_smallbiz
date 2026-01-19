"""
Alembic 환경 설정

이 파일은 Alembic 마이그레이션 실행 시 사용됩니다.
models.py의 Base 메타데이터와 DATABASE_URL을 연결합니다.
"""

import sys
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# models.py에서 Base 가져오기
from src.backend.models import Base

# settings에서 DATABASE_URL 가져오기
from src.utils.config import settings

# Alembic Config 객체
config = context.config

# 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MetaData 설정 (autogenerate 지원)
target_metadata = Base.metadata

# DATABASE_URL 설정
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def run_migrations_offline() -> None:
    """
    오프라인 모드에서 마이그레이션 실행

    실제 DB 연결 없이 SQL 스크립트만 생성합니다.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    온라인 모드에서 마이그레이션 실행

    실제 DB에 연결하여 마이그레이션을 적용합니다.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
