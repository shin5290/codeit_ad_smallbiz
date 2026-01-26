"""Add is_admin flag to user

Revision ID: 006
Revises: 005
Create Date: 2026-01-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add is_admin column and promote admin account."""
    op.add_column(
        "user",
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.execute("UPDATE \"user\" SET is_admin = TRUE WHERE login_id = 'admin'")
    op.alter_column("user", "is_admin", server_default=None)


def downgrade() -> None:
    """Drop is_admin column."""
    op.drop_column("user", "is_admin")
