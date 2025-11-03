# built-in
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

# external
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel

# Import models so metadata is populated
from discsync_embeddings.core import sqlmodels as _models  # noqa: F401

_engine: Optional[AsyncEngine] = None
_Session: Optional[async_sessionmaker[AsyncSession]] = None


def database_url() -> Optional[str]:
    if (os.environ.get("DEV_MODE") or "").strip().lower() == "true":
        return "sqlite+aiosqlite:///./dev.db"

    return os.environ.get("DATABASE_URL", None)


def get_engine() -> Optional[AsyncEngine]:
    global _engine, _Session
    if _engine is not None:
        return _engine

    url = database_url()
    if url is None:
        return None

    # Always keep SQL echo off; use profiler/loggers if needed
    _engine = create_async_engine(url, echo=False, pool_pre_ping=True)

    if url.startswith("sqlite+"):

        @event.listens_for(_engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _):  # type: ignore[no-redef]
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

    _Session = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
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


async def init_dev_sqlite_schema() -> None:
    url = database_url()
    if url is None or not url.startswith("sqlite+"):
        return
    eng = get_engine()
    if eng is None:
        return
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
