# built-in
from contextlib import asynccontextmanager
from importlib.metadata import version
import asyncio

# external
from fastapi import FastAPI
from starlette.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# project
from discsync_embeddings.helpers.logging import logger
from discsync_embeddings.core.db import init_dev_sqlite_schema
from discsync_embeddings.app.routers import build_api_router
from discsync_embeddings.core.embeddings import worker_loop, Embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    logger.info("Starting discsync-embeddings")

    # Initialize SQLite schema in dev mode (no-op for PostgreSQL)
    await init_dev_sqlite_schema()

    # Preload embedder
    await asyncio.to_thread(Embedder.get)

    # Start background embeddings worker
    worker_task = asyncio.create_task(worker_loop())

    try:
        yield
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        logger.info("Shutting down discsync-embeddings")


middleware = [
    Middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(
    title="discsync-embeddings",
    lifespan=lifespan,
    version=version("discsync-embeddings"),
    middleware=middleware,
)

app.include_router(build_api_router())
