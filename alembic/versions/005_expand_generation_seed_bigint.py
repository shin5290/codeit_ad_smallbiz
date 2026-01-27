"""Expand generation_history.seed to bigint and add strength

Revision ID: 005
Revises: 004
Create Date: 2026-01-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Expand seed column to bigint and add strength."""
    op.alter_column(
        "generation_history",
        "seed",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=True,
    )
    op.add_column(
        "generation_history",
        sa.Column("strength", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Revert seed column to int and drop strength."""
    op.drop_column("generation_history", "strength")
    op.alter_column(
        "generation_history",
        "seed",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
