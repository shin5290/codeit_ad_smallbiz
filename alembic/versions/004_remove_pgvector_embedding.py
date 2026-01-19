"""Remove embedding column from chat_history

Revision ID: 004
Revises: 003
Create Date: 2026-01-17
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop embedding index/column if present."""
    op.execute("DROP INDEX IF EXISTS idx_chat_history_embedding")
    op.execute("ALTER TABLE chat_history DROP COLUMN IF EXISTS embedding")


def downgrade() -> None:
    """No-op: embedding column removed."""
    pass
