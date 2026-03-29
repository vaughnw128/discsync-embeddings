# built-in
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

# external
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Import models for metadata
from discsync_embeddings.core import sqlmodels as _models  # noqa: F401

_engine: Optional[AsyncEngine] = None
_Session: Optional[async_sessionmaker[AsyncSession]] = None


def _normalize_async_url(url: str) -> str:
    """Ensure a PostgreSQL URL uses the asyncpg dialect.

    CNPG secrets provide ``postgresql://...`` but SQLAlchemy's async engine
    requires ``postgresql+asyncpg://...``.
    """

    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def database_url() -> Optional[str]:
    url = os.environ.get("DATABASE_URL", None)
    if url is not None:
        url = _normalize_async_url(url)
    return url


def get_engine() -> Optional[AsyncEngine]:
    """Get or create the async database engine."""

    global _engine, _Session
    if _engine is not None:
        return _engine

    url = database_url()
    if url is None:
        return None

    _engine = create_async_engine(url, echo=False, pool_pre_ping=True)
    _Session = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Provide a transactional scope around a series of operations."""

    global _Session
    if _Session is None:
        eng = get_engine()
        if eng is None:
            raise RuntimeError("Database is not configured")
    assert _Session is not None
    session = _Session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
