"""GenerationHistory에 revision 관련 컬럼 추가

Revision ID: 001
Revises: None
Create Date: 2025-01-16

추가되는 컬럼:
- is_confirmed: Boolean (최종 확정 여부)
- revision_of_id: Integer FK (이전 버전 ID, self FK)
- revision_number: Integer (수정 버전 번호)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    GenerationHistory 테이블에 revision 관련 컬럼 추가
    """
    # is_confirmed 컬럼 추가
    op.add_column(
        'generation_history',
        sa.Column('is_confirmed', sa.Boolean(), nullable=True, default=False)
    )

    # revision_of_id 컬럼 추가 (self FK)
    op.add_column(
        'generation_history',
        sa.Column('revision_of_id', sa.Integer(), nullable=True)
    )

    # revision_number 컬럼 추가
    op.add_column(
        'generation_history',
        sa.Column('revision_number', sa.Integer(), nullable=True, default=0)
    )

    # 기존 데이터에 기본값 설정
    op.execute("UPDATE generation_history SET is_confirmed = FALSE WHERE is_confirmed IS NULL")
    op.execute("UPDATE generation_history SET revision_number = 0 WHERE revision_number IS NULL")

    # revision_of_id에 대한 외래 키 제약 조건 추가
    op.create_foreign_key(
        'fk_generation_history_revision_of',
        'generation_history',
        'generation_history',
        ['revision_of_id'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    """
    revision 관련 컬럼 제거 (롤백용)
    """
    # 외래 키 제약 조건 먼저 제거
    op.drop_constraint(
        'fk_generation_history_revision_of',
        'generation_history',
        type_='foreignkey'
    )

    # 컬럼 제거
    op.drop_column('generation_history', 'revision_number')
    op.drop_column('generation_history', 'revision_of_id')
    op.drop_column('generation_history', 'is_confirmed')
