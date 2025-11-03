# built-in
import asyncio
import os
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple, cast

# external
from qdrant_client import QdrantClient
from sqlalchemy import select, case
from sqlalchemy.ext.asyncio import AsyncSession
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# project
from discsync_embeddings.core.db import get_session
from discsync_embeddings.core.sqlmodels import Message, MessageEmbedding
from discsync_embeddings.core.message import (
    build_chunk,
    message_fingerprint,
)
from discsync_embeddings.helpers.logging import logger

# Configuration (simple direct env lookups)
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
QDRANT_COLLECTION: str = os.environ.get(
    "QDRANT_COLLECTION", "discsync-embeddings"
)
EMBED_BATCH: int = int(os.environ.get("EMBED_BATCH", "64"))
EMBED_INTERVAL_SEC: float = float(os.environ.get("EMBED_INTERVAL_SEC", "5"))
EMBED_RETRIES: int = int(os.environ.get("EMBED_RETRIES", "3"))
EMBED_RETRY_BACKOFF: float = float(
    os.environ.get("EMBED_RETRY_BACKOFF", "1.5")
)
QDRANT_URL: Optional[str] = os.environ.get("QDRANT_URL")
QDRANT_API_KEY: Optional[str] = os.environ.get("QDRANT_API_KEY")
QDRANT_TIMEOUT: int = int(os.environ.get("QDRANT_TIMEOUT", "30"))


def get_qdrant_client() -> QdrantClient:
    url = (QDRANT_URL or "").strip() or None
    if url is not None:
        client = QdrantClient(
            url=url,
            port=443,
            timeout=QDRANT_TIMEOUT,
        )
        logger.info(client.info())
        return client
    return QdrantClient(
        host="localhost",
        api_key=QDRANT_API_KEY,
        port=443,
        timeout=QDRANT_TIMEOUT,
    )


class _LlamaSvc:
    _embed: Optional[HuggingFaceEmbedding] = None
    _index: Optional[VectorStoreIndex] = None

    @classmethod
    def get(cls) -> Tuple[HuggingFaceEmbedding, VectorStoreIndex]:
        if cls._embed is None:
            t0 = time.monotonic()
            logger.info("Loading HF embed model: %s", EMBEDDING_MODEL)
            cls._embed = HuggingFaceEmbedding(
                model_name=EMBEDDING_MODEL,
                device="cpu",
            )
            logger.info("HF model ready in %.2fs", time.monotonic() - t0)
        if cls._index is None:
            t1 = time.monotonic()
            logger.info("Initializing Qdrant store: %s", QDRANT_COLLECTION)
            store = QdrantVectorStore(
                client=get_qdrant_client(),
                collection_name=QDRANT_COLLECTION,
            )
            logger.info("Qdrant store ready in %.2fs", time.monotonic() - t1)
            t2 = time.monotonic()
            logger.info("Building VectorStoreIndex")
            cls._index = VectorStoreIndex.from_vector_store(
                vector_store=store,
                embed_model=cls._embed,
            )
            logger.info(
                "VectorStoreIndex ready in %.2fs", time.monotonic() - t2
            )
        assert cls._embed is not None and cls._index is not None
        return cls._embed, cls._index


async def _insert_with_retries(
    index: VectorStoreIndex, doc: Document
) -> None:
    attempts = EMBED_RETRIES
    delay = EMBED_RETRY_BACKOFF
    for i in range(attempts):
        try:
            await asyncio.to_thread(index.insert, doc)
            return
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1:
                raise
            logger.warning("insert retry %s/%s: %s", i + 1, attempts, exc)
            await asyncio.sleep(delay * (2**i))


async def _process_message(session: AsyncSession, m: Message) -> None:
    fp = message_fingerprint(m)
    existing = await session.get(MessageEmbedding, m.id)
    if (
        existing is not None
        and existing.content_sha256 == fp
        and existing.status == "completed"
    ):
        return

    if existing is None:
        existing = MessageEmbedding(message_id=m.id, status="pending")
        session.add(existing)
        await session.flush()

    existing.status = "tokenizing"
    await session.flush()

    text, payload = build_chunk(m)
    logger.debug("Message %s text length: %s", m.id, len(text))
    if not text.strip():
        existing.status = "skipped"
        existing.error = "empty"
        existing.content_sha256 = fp
        existing.formatted_text = ""
        await session.flush()
        return

    _, index = await asyncio.to_thread(_LlamaSvc.get)
    logger.info("Start embedding message %s", m.id)

    # Persist the formatted text used for embedding
    existing.formatted_text = text

    doc = Document(text=text, metadata=payload, doc_id=str(m.id))

    # LlamaIndex insert with retries to avoid transient timeouts.
    await _insert_with_retries(index, doc)

    existing.qdrant_collection = QDRANT_COLLECTION
    existing.qdrant_point_ids = [str(m.id)]
    existing.model_name = EMBEDDING_MODEL
    existing.token_count = len(text.split())
    existing.chunk_count = 1
    existing.status = "completed"
    existing.error = None
    existing.content_sha256 = fp
    existing.embedded_at = datetime.utcnow()
    await session.flush()


async def process_pending_messages(limit: int = 100) -> int:
    processed = 0
    async with get_session() as session:
        msg_tbl = Message.__table__
        emb_tbl = MessageEmbedding.__table__

        priority = case(
            (emb_tbl.c.status == "failed", 0),
            (emb_tbl.c.status == "pending", 1),
            else_=2,
        )

        stmt = (
            select(Message)
            .outerjoin(emb_tbl, emb_tbl.c.message_id == msg_tbl.c.id)
            .where(
                (emb_tbl.c.message_id.is_(None))
                | (emb_tbl.c.status.in_(["pending", "failed"]))
            )
            .order_by(priority.asc(), msg_tbl.c.created_at_ms.asc())
            .limit(limit)
        )
        rs = await session.execute(stmt)
        msgs: List[Message] = list(rs.scalars().all())
        for m in msgs:
            try:
                await _process_message(session, m)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding failed for %s: %s", m.id, exc)
                emb = await session.get(MessageEmbedding, m.id)
                if emb is None:
                    emb = MessageEmbedding(message_id=m.id)
                    session.add(emb)
                    await session.flush()
                emb.status = "failed"
                emb.error = str(exc)[:500]
                await session.flush()
    return processed


async def worker_loop(interval_sec: float = 5.0) -> None:
    if interval_sec <= 0:
        interval_sec = EMBED_INTERVAL_SEC
    while True:
        try:
            n = await process_pending_messages(limit=EMBED_BATCH)
            if n == 0:
                await asyncio.sleep(interval_sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Worker error: %s", exc)
            await asyncio.sleep(interval_sec)
