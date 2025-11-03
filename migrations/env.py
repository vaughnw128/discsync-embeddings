from __future__ import annotations

import os
import logging
from typing import Optional

from alembic import context
from sqlalchemy import Connection, create_engine, pool
from sqlmodel import SQLModel

# Import models so tables are registered on SQLModel.metadata
from discsync_embeddings.core import sqlmodels as _models  # noqa: F401

# Interpret the config file for Python logging.
# Do not configure logging via alembic.ini to avoid duplicate handlers.
config = context.config
logging.getLogger("alembic").disabled = True

# Use SQLModel metadata for autogenerate
target_metadata = SQLModel.metadata


def _database_url() -> Optional[str]:
    """Return a PostgreSQL database URL for migrations.

    Prefers DATABASE_URL_MIGRATIONS, then MIGRATIONS_DATABASE_URL,
    then DATABASE_URL. Only PostgreSQL URLs are accepted.
    """

    raw_url = (
        os.environ.get("DATABASE_URL_MIGRATIONS")
        or os.environ.get("MIGRATIONS_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
    )
    if not raw_url or not raw_url.strip():
        return None

    url = raw_url.strip()

    # Normalize postgres alias
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    if not url.startswith("postgresql"):
        raise RuntimeError(
            "Only PostgreSQL is supported for migrations. Set a PostgreSQL DATABASE_URL (or DATABASE_URL_MIGRATIONS)."
        )

    return url


def _to_sync_driver(url: str) -> str:
    """Return a sync-driver SQLAlchemy URL for Alembic engines.

    Converts async drivers to their sync equivalents and makes the driver
    explicit for PostgreSQL.
    """

    if url.startswith("postgresql+asyncpg"):
        return url.replace("postgresql+asyncpg", "postgresql+psycopg", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode against PostgreSQL."""

    url = _database_url()
    if not url:
        raise RuntimeError("PostgreSQL DATABASE_URL is required for Alembic migrations")

    # Use sync driver in offline as well for consistency
    url = _to_sync_driver(url)

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using a synchronous engine."""

    url = _database_url()
    if not url:
        raise RuntimeError("PostgreSQL DATABASE_URL is required for Alembic migrations")

    sync_url = _to_sync_driver(url)
    connectable = create_engine(sync_url, poolclass=pool.NullPool, future=True)

    with connectable.connect() as connection:
        do_run_migrations(connection)


def run_migrations() -> None:
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()


run_migrations()
