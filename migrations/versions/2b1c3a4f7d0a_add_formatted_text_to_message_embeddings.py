"""add formatted_text to message_embeddings

Revision ID: 2b1c3a4f7d0a
Revises: 5ebd7bdda103
Create Date: 2025-11-01 11:05:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "2b1c3a4f7d0a"
down_revision = "5ebd7bdda103"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "message_embeddings",
        sa.Column("formatted_text", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("message_embeddings", "formatted_text")

