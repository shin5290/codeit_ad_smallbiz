"""Add pgvector embedding column to chat_history

Revision ID: 002
Revises: 001
Create Date: 2026-01-16

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    """
    pgvector 확장 설치 및 chat_history 테이블에 embedding 컬럼 추가
    """
    # 1. pgvector 확장 확인 (superuser 권한 필요 시 수동 설치)
    from sqlalchemy import text
    conn = op.get_bind()
    result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
    if result.fetchone() is None:
        try:
            op.execute('CREATE EXTENSION IF NOT EXISTS vector')
        except Exception as e:
            print("\n" + "="*60)
            print("⚠️  pgvector 확장을 설치할 수 없습니다.")
            print("   DBA에게 다음 명령을 실행해달라고 요청하세요:")
            print("   psql -d <database> -c 'CREATE EXTENSION vector;'")
            print("="*60 + "\n")
            raise e

    # 2. chat_history 테이블에 embedding 컬럼 추가
    # OpenAI text-embedding-3-small 모델은 1536차원 벡터 생성
    op.add_column(
        'chat_history',
        sa.Column('embedding', Vector(1536), nullable=True)
    )

    # 3. 벡터 유사도 검색을 위한 인덱스 생성
    # IVFFlat 인덱스: 빠른 근사 최근접 이웃 검색
    # lists 파라미터: 클러스터 수 (일반적으로 행 수 / 1000 권장)
    # vector_cosine_ops: 코사인 유사도 기반 검색
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_embedding
        ON chat_history
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade():
    """
    pgvector 관련 변경 사항 롤백
    """
    # 1. 인덱스 삭제
    op.execute('DROP INDEX IF EXISTS idx_chat_history_embedding')

    # 2. embedding 컬럼 삭제
    op.drop_column('chat_history', 'embedding')

    # 3. pgvector 확장 삭제 (주의: 다른 테이블에서 사용 중일 수 있음)
    # op.execute('DROP EXTENSION IF EXISTS vector')
