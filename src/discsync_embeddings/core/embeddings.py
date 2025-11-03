# built-in
import asyncio
import os
import time
from datetime import datetime, UTC
from typing import List, Optional, Tuple, Any
from threading import Lock
from uuid import NAMESPACE_URL, uuid5

# external
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sqlalchemy import select, case
from sqlalchemy.ext.asyncio import AsyncSession
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# project
from discsync_embeddings.core.db import get_session
from discsync_embeddings.core.sqlmodels import Message, MessageEmbedding
from discsync_embeddings.core.message import (
    build_chunk_async,
    message_fingerprint,
)
from discsync_embeddings.helpers.logging import logger

# Configuration
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "discsync-embeddings")
EMBED_BATCH: int = int(os.environ.get("EMBED_BATCH", "64"))
EMBED_INTERVAL_SEC: float = float(os.environ.get("EMBED_INTERVAL_SEC", "5"))
EMBED_RETRIES: int = int(os.environ.get("EMBED_RETRIES", "3"))
EMBED_RETRY_BACKOFF: float = float(os.environ.get("EMBED_RETRY_BACKOFF", "1.5"))
QDRANT_URL: Optional[str] = os.environ.get("QDRANT_URL")
QDRANT_API_KEY: Optional[str] = os.environ.get("QDRANT_API_KEY")
QDRANT_TIMEOUT: int = int(os.environ.get("QDRANT_TIMEOUT", "30"))
EMBED_DEVICE: str = os.environ.get("EMBED_DEVICE", "cpu").lower()


def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client based on environment configuration."""

    url = (QDRANT_URL or "").strip() or None
    if url is not None:
        return QdrantClient(
            url=url,
            port=443,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
        )
    # Local/default connection
    logger.info("Connecting to local Qdrant at %s:%s", "localhost", 443)
    return QdrantClient(
        host="localhost",
        port=443,
        timeout=QDRANT_TIMEOUT,
    )


def _priority_case(emb_tbl):
    """Return a SQLAlchemy case expression for embedding priority."""

    return case(
        {"failed": 0, "pending": 1},
        value=emb_tbl.c.status,
        else_=2,
    )


def _pending_messages_stmt(limit: int):
    """Return a SQLAlchemy statement to fetch pending messages for embedding."""

    msg_tbl = Message.__table__
    emb_tbl = MessageEmbedding.__table__
    onclause: Any = emb_tbl.c.message_id == msg_tbl.c.id
    return (
        select(Message)
        .outerjoin(emb_tbl, onclause)
        .where((emb_tbl.c.message_id.is_(None)) | (emb_tbl.c.status.in_(["pending", "failed"])))
        .order_by(_priority_case(emb_tbl).asc(), msg_tbl.c.created_at_ms.asc())
        .limit(limit)
    )


async def _fetch_pending_messages(session: AsyncSession, limit: int) -> List[Message]:
    """Fetch pending messages for embedding."""

    rs = await session.execute(_pending_messages_stmt(limit))
    return list(rs.scalars().all())


async def _get_or_create_embedding(session: AsyncSession, msg_id: int) -> MessageEmbedding:
    """Get or create a MessageEmbedding row for a message ID."""

    emb = await session.get(MessageEmbedding, msg_id)
    if emb is None:
        emb = MessageEmbedding(message_id=msg_id, status="pending")
        session.add(emb)
        await session.flush()
    return emb


async def _mark_failed(session: AsyncSession, msg_id: int, error: str) -> None:
    """Mark a message embedding as failed with an error message."""

    emb = await session.get(MessageEmbedding, msg_id)
    if emb is None:
        emb = MessageEmbedding(message_id=msg_id)
        session.add(emb)
        await session.flush()
    emb.status = "failed"
    emb.error = error[:500]
    await session.flush()


def _resolve_device() -> str:
    """Resolve embedding device from EMBED_DEVICE env var.

    Supports: 'cpu', 'cuda', 'mps', or 'auto' (prefers CUDA if available).
    Falls back to 'cpu' if torch is unavailable or CUDA is not available.
    """

    dev = EMBED_DEVICE.strip().lower() if EMBED_DEVICE else "cpu"
    if dev == "auto":
        try:
            import torch  # type: ignore

            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"
    return dev if dev in {"cpu", "cuda", "mps"} else "cpu"


def _point_id_for_message(msg_id: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"discsync:message:{msg_id}"))


class Embedder:
    _embed: Optional[HuggingFaceEmbedding] = None
    _index: Optional[VectorStoreIndex] = None
    _lock: Lock = Lock()
    _device: Optional[str] = None

    @classmethod
    def reset(cls) -> None:
        """Reset the embedder and index (for testing)."""

        with cls._lock:
            cls._embed = None
            cls._index = None
            cls._device = None

    @classmethod
    def get(cls) -> Tuple[HuggingFaceEmbedding, VectorStoreIndex]:
        """Get or initialize the embedder and index."""

        if cls._embed is not None and cls._index is not None:
            return cls._embed, cls._index
        with cls._lock:
            if cls._embed is None:
                t0 = time.monotonic()
                device = _resolve_device()
                cls._device = device
                logger.info("Loading HF embed model: %s on %s", EMBEDDING_MODEL, device)
                cls._embed = HuggingFaceEmbedding(
                    model_name=EMBEDDING_MODEL,
                    device=device,
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
                logger.info("VectorStoreIndex ready in %.2fs", time.monotonic() - t2)

            # Extra CUDA diagnostics
            if cls._device == "cuda":
                try:
                    import torch  # type: ignore

                    cuda_ver = getattr(torch.version, "cuda", None)
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                    logger.info("CUDA runtime=%s gpu=%s", cuda_ver, gpu_name or "unknown")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("CUDA diagnostics unavailable: %s", exc)
            logger.info(
                "Embedder ready: model=%s device=%s collection=%s",
                EMBEDDING_MODEL,
                cls._device,
                QDRANT_COLLECTION,
            )
        assert cls._embed is not None and cls._index is not None
        return cls._embed, cls._index

    @classmethod
    async def embed_batch(cls, session: AsyncSession, messages: List[Message]) -> int:
        """Embed a batch of messages."""

        if not messages:
            return 0

        # Ensure services are initialized
        embed_model, _ = await asyncio.to_thread(cls.get)

        # Prepare rows and texts
        rows: List[MessageEmbedding] = []
        texts: List[str] = []
        payloads: List[dict] = []
        ids: List[str] = []
        fps: List[str] = []
        dedup_skipped = 0
        empty_skipped = 0
        for m in messages:
            fp = message_fingerprint(m)
            existing = await session.get(MessageEmbedding, m.id)
            if existing is not None and existing.content_sha256 == fp and existing.status == "completed":
                dedup_skipped += 1
                continue
            emb_row = await _get_or_create_embedding(session, m.id)
            emb_row.status = "tokenizing"

            # Use async builder to resolve names from DB
            text, payload = await build_chunk_async(session, m)
            if not text.strip():
                emb_row.status = "skipped"
                emb_row.error = "empty"
                emb_row.content_sha256 = fp
                emb_row.formatted_text = ""
                empty_skipped += 1
                continue
            rows.append(emb_row)
            texts.append(text)
            payloads.append(payload)
            ids.append(_point_id_for_message(m.id))
            fps.append(fp)
        await session.flush()
        if not rows:
            logger.debug(
                "Batch had no embeddable messages (dedup=%s empty=%s)",
                dedup_skipped,
                empty_skipped,
            )
            return 0

        t_embed_start = time.monotonic()
        try:
            vectors = await asyncio.to_thread(embed_model.get_text_embedding_batch, texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch embed failed, falling back per-item: %s", exc)
            vectors = []
            for t in texts:
                vec = await asyncio.to_thread(embed_model.get_text_embedding, t)
                vectors.append(vec)
        t_embed = time.monotonic() - t_embed_start

        # Bulk upsert to Qdrant
        client = get_qdrant_client()
        points = [PointStruct(id=pid, vector=vec, payload=payload) for pid, vec, payload in zip(ids, vectors, payloads)]
        t_upsert_start = time.monotonic()
        await asyncio.to_thread(
            client.upsert,
            collection_name=QDRANT_COLLECTION,
            points=points,
        )
        t_upsert = time.monotonic() - t_upsert_start

        # Update rows
        for emb_row, text, pid, fp in zip(rows, texts, ids, fps):
            emb_row.formatted_text = text
            emb_row.qdrant_collection = QDRANT_COLLECTION
            emb_row.qdrant_point_ids = [pid]
            emb_row.model_name = EMBEDDING_MODEL
            emb_row.token_count = len(text.split())
            emb_row.chunk_count = 1
            emb_row.status = "completed"
            emb_row.error = None
            emb_row.content_sha256 = fp
            emb_row.embedded_at = datetime.now(UTC).replace(tzinfo=None)
        await session.flush()
        total = len(rows)
        eff_time = t_embed + t_upsert
        mps_total = (total / eff_time) if eff_time > 0 else 0.0
        # Single concise timing log
        logger.info(
            "timing: count=%d embed=%.3fs upsert=%.3fs total=%.3fs mps=%.2f",
            total,
            t_embed,
            t_upsert,
            eff_time,
            mps_total,
        )
        return total

    @classmethod
    async def embed_message(cls, session: AsyncSession, m: Message) -> None:
        """Embed a single message."""

        fp = message_fingerprint(m)
        existing = await session.get(MessageEmbedding, m.id)
        if existing is not None and existing.content_sha256 == fp and existing.status == "completed":
            return

        emb_row = await _get_or_create_embedding(session, m.id)
        emb_row.status = "tokenizing"
        await session.flush()

        text, payload = await build_chunk_async(session, m)
        # Reduced noisy logs; keep length check only
        if not text.strip():
            emb_row.status = "skipped"
            emb_row.error = "empty"
            emb_row.content_sha256 = fp
            emb_row.formatted_text = ""
            await session.flush()
            return

        # Ensure embedder/index are initialized
        _, index = await asyncio.to_thread(cls.get)

        emb_row.formatted_text = text
        pid = _point_id_for_message(m.id)
        doc = Document(text=text, metadata=payload, doc_id=pid)
        await insert_to_qdrant(index, doc)

        emb_row.qdrant_collection = QDRANT_COLLECTION
        emb_row.qdrant_point_ids = [pid]
        emb_row.model_name = EMBEDDING_MODEL
        emb_row.token_count = len(text.split())
        emb_row.chunk_count = 1
        emb_row.status = "completed"
        emb_row.error = None
        emb_row.content_sha256 = fp
        emb_row.embedded_at = datetime.now(UTC).replace(tzinfo=None)
        await session.flush()


async def insert_to_qdrant(index: VectorStoreIndex, doc: Document) -> None:
    """Insert a document into Qdrant with retries."""

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


async def process_pending_messages(limit: int = 100) -> int:
    processed = 0
    async with get_session() as session:
        msgs = await _fetch_pending_messages(session, limit)
        logger.debug("Fetched %s pending messages", len(msgs))
        # Chunk messages for compute-batched embedding
        batch_size = int(os.environ.get("EMBED_COMPUTE_BATCH", "64"))
        for i in range(0, len(msgs), batch_size):
            chunk = msgs[i : i + batch_size]
            logger.debug("Processing chunk %s-%s (size=%s)", i, i + len(chunk) - 1, len(chunk))
            try:
                processed += await Embedder.embed_batch(session, chunk)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embed batch failed, trying item-wise: %s", exc)
                for m in chunk:
                    try:
                        await Embedder.embed_message(session, m)
                        processed += 1
                    except Exception as exc2:  # noqa: BLE001
                        logger.warning("Embedding failed for %s: %s", m.id, exc2)
                        await _mark_failed(session, m.id, str(exc2))
    return processed


async def worker_loop(interval_sec: float = 5.0) -> None:
    """Background worker loop to process pending messages for embedding."""

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
