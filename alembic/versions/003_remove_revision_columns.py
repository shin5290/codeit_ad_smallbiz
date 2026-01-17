"""Remove revision/confirm columns from generation_history

Revision ID: 003
Revises: 002
Create Date: 2026-01-17
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop revision/confirm columns."""
    op.drop_constraint(
        "fk_generation_history_revision_of",
        "generation_history",
        type_="foreignkey",
    )
    op.drop_column("generation_history", "revision_number")
    op.drop_column("generation_history", "revision_of_id")
    op.drop_column("generation_history", "is_confirmed")


def downgrade() -> None:
    """Re-add revision/confirm columns (rollback)."""
    op.add_column(
        "generation_history",
        sa.Column("is_confirmed", sa.Boolean(), nullable=True, default=False),
    )
    op.add_column(
        "generation_history",
        sa.Column("revision_of_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "generation_history",
        sa.Column("revision_number", sa.Integer(), nullable=True, default=0),
    )
    op.create_foreign_key(
        "fk_generation_history_revision_of",
        "generation_history",
        "generation_history",
        ["revision_of_id"],
        ["id"],
        ondelete="SET NULL",
    )
