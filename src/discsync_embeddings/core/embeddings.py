# built-in
import asyncio
import os
import time
from datetime import datetime, UTC
from typing import List, Optional, Any
from threading import Lock

# external
from qdrant_client import QdrantClient
from sqlalchemy import select, case
from sqlalchemy.ext.asyncio import AsyncSession
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import TextNode, MetadataMode
import torch

# project
from discsync_embeddings.core.db import get_session
from discsync_embeddings.core.sqlmodels import Message, MessageEmbedding
from discsync_embeddings.core.message import (
    message_fingerprint,
    EmbeddableMessage,
)
from discsync_embeddings.helpers.logging import logger

# Configuration

# Embedding settings
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_FALLBACK_MODEL: str = os.environ.get("EMBEDDING_FALLBACK_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH: int = int(os.environ.get("EMBED_BATCH", "10"))
EMBED_RETRIES: int = int(os.environ.get("EMBED_RETRIES", "3"))
EMBED_RETRY_BACKOFF: float = float(os.environ.get("EMBED_RETRY_BACKOFF", "1.5"))
EMBED_DEVICE: str = os.environ.get("EMBED_DEVICE", "cpu").lower()

# Qdrant settings
QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "discsync-embeddings")
QDRANT_HOSTNAME: Optional[str] = os.environ.get("QDRANT_HOSTNAME")
QDRANT_PORT: Optional[int] = os.environ.get("QDRANT_PORT", 443)
QDRANT_API_KEY: Optional[str] = os.environ.get("QDRANT_API_KEY")
QDRANT_TIMEOUT: int = int(os.environ.get("QDRANT_TIMEOUT", "30"))


def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client based on environment configuration."""

    hostname = (QDRANT_HOSTNAME or "").strip() or None
    if hostname is not None:
        logger.info(f"Connecting to local Qdrant at {hostname}:{QDRANT_PORT}")
        return QdrantClient(
            url=hostname,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
        )

    # Local/default connection
    logger.info(f"Connecting to local Qdrant at localhost:{QDRANT_PORT}")
    return QdrantClient(
        host="localhost",
        port=QDRANT_PORT,
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


async def update_embed_table(session: AsyncSession, msg_id: int) -> MessageEmbedding:
    """Get or create a MessageEmbedding row for a message ID."""

    emb = await session.get(MessageEmbedding, msg_id)
    if emb is None:
        emb = MessageEmbedding(message_id=msg_id, status="pending")
        session.add(emb)
        await session.flush()
    return emb


def get_embed_device() -> str:
    """Resolve embedding device from EMBED_DEVICE env var.

    Supports: 'cpu', 'cuda', 'mps', or 'auto' (prefers CUDA if available).
    Falls back to 'cpu' if torch is unavailable or CUDA is not available.
    """

    embed_device = os.environ.get("EMBED_DEVICE", "cpu").strip().lower()
    match embed_device:
        case "cuda":
            return "cuda"
        case "mps":
            return "mps"
        case "auto":
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        case _:
            return "cpu"


class Embedder:
    _lock: Lock = Lock()
    hf_embedding: Optional[HuggingFaceEmbedding] = None
    vector_index: Optional[VectorStoreIndex] = None
    store: Optional[QdrantVectorStore] = None

    def __init__(self):
        """Initialize the embedder."""

        # Load model
        device = get_embed_device()
        logger.info(f"Loading HF embed model: {EMBEDDING_MODEL} on {device}")
        self.hf_embedding = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            device=device,
            trust_remote_code=True,
        )

        # Load Qdrant store after ensuring collection schema.
        self.store = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=QDRANT_COLLECTION,
            batch_size=EMBED_BATCH,
        )

        # Initialize index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.store,
            embed_model=self.hf_embedding,
        )

    def _ensure_fallback_model(self) -> None:
        """Switch to a known SentenceTransformers model if current fails."""

        current = getattr(self.hf_embedding, "model_name", EMBEDDING_MODEL)
        if current == EMBEDDING_FALLBACK_MODEL:
            return
        device = get_embed_device()
        logger.warning(
            f"Falling back embedding model from {current} to {EMBEDDING_FALLBACK_MODEL} due to incompatibility",
        )
        self.hf_embedding = HuggingFaceEmbedding(
            model_name=EMBEDDING_FALLBACK_MODEL,
            device=device,
            trust_remote_code=False,
        )

    async def embed_batch(self, messages: List[Message]) -> int:
        """Embed a batch of messages."""

        if not messages:
            return 0

        time_tokenize_start = time.monotonic()

        # Bulk check which message IDs already have embeddings
        msg_ids = [m.id for m in messages]
        async with get_session() as session:
            emb_tbl = MessageEmbedding.__table__
            existing_ids_result = await session.execute(
                select(emb_tbl.c.message_id).where(emb_tbl.c.message_id.in_(msg_ids))
            )
            existing_ids = set(existing_ids_result.scalars().all())

        # Filter messages to embed
        messages_to_embed = [m for m in messages if m.id not in existing_ids]
        if not messages_to_embed:
            logger.debug("All messages in batch already embedded")
            return 0

        # Compute fingerprints up-front
        fps = {m.id: message_fingerprint(m) for m in messages_to_embed}

        # Build TextNodes concurrently
        embeddable_messages = [EmbeddableMessage(m) for m in messages_to_embed]
        nodes: List[TextNode] = await asyncio.gather(*(emb.to_text_node() for emb in embeddable_messages))

        # Update embed table rows in bulk, single flush
        async with get_session() as session:
            emb_tbl = MessageEmbedding.__table__

            # Ensure rows exist for all messages to embed
            # Fetch any existing rows (should be none, but safe)
            existing_rows_result = await session.execute(
                select(MessageEmbedding).where(emb_tbl.c.message_id.in_([m.id for m in messages_to_embed]))
            )
            rows_by_id = {r.message_id: r for r in existing_rows_result.scalars().all()}

            # Create missing rows
            for m in messages_to_embed:
                if m.id not in rows_by_id:
                    row = MessageEmbedding(message_id=m.id, status="pending")
                    session.add(row)
                    rows_by_id[m.id] = row

            # Update rows with tokenize status and formatted text
            for node in nodes:
                msg_id = int(node.metadata.get("message_id"))
                row = rows_by_id[msg_id]
                row.status = "tokenizing"
                row.content_sha256 = fps[msg_id]
                row.formatted_text = node.text

            await session.flush()

            time_tokenize_end = time.monotonic() - time_tokenize_start

            # Proceed to embedding
            time_embed_start = time.monotonic()
            texts = [n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes]
            try:
                vectors = await asyncio.to_thread(
                    self.hf_embedding.get_text_embedding_batch, texts
                )
            except Exception as err:  # noqa: BLE001
                msg = str(err)
                if "position_ids" in msg and "required by model" in msg:
                    self._ensure_fallback_model()
                    vectors = await asyncio.to_thread(
                        self.hf_embedding.get_text_embedding_batch, texts
                    )
                else:
                    raise

            for n, v in zip(nodes, vectors):
                n.embedding = v
            time_embed_end = time.monotonic() - time_embed_start

            # Upsert nodes into Qdrant via the vector store
            time_upsert_start = time.monotonic()
            await asyncio.to_thread(self.store.add, nodes)
            time_upsert_end = time.monotonic() - time_upsert_start

            # Finalize embed rows
            used_model = getattr(self.hf_embedding, "model_name", EMBEDDING_MODEL)
            for node in nodes:
                msg_id = int(node.metadata.get("message_id"))
                emb_row = rows_by_id[msg_id]
                content_only = node.get_content(metadata_mode=MetadataMode.NONE)
                emb_row.qdrant_collection = QDRANT_COLLECTION
                emb_row.qdrant_point_ids = [node.id_]
                emb_row.model_name = used_model
                emb_row.token_count = len(content_only.split())
                emb_row.chunk_count = 1
                emb_row.status = "completed"
                emb_row.error = None
                emb_row.embedded_at = datetime.now(UTC).replace(tzinfo=None)
            await session.flush()

        total_time = time_embed_end + time_upsert_end + time_tokenize_end
        messages_per_second = (len(nodes) / total_time) if total_time > 0 else 0.0
        logger.info(
            f"timing: count={len(nodes)} tokenize={time_tokenize_end:.3f}s "
            f"embed={time_embed_end:.3f}s upsert={time_upsert_end:.3f}s "
            f"total={total_time:.3f}s mps={messages_per_second:.2f}"
        )
        return len(nodes)

    async def process_pending_messages(self, limit: int = 100) -> None:
        async with get_session() as session:
            response = await session.execute(_pending_messages_stmt(limit))
            messages = list(response.scalars().all())

            logger.debug(f"Fetched {len(messages)} pending messages")

            try:
                await self.embed_batch(messages)
            except Exception as err:  # noqa: BLE001
                logger.warning(f"Embed batch failed: {err}")


async def worker_loop(interval_sec: float = 5.0) -> None:
    """Background worker loop to process pending messages for embedding."""

    embedder = Embedder()

    while True:
        try:
            await embedder.process_pending_messages(limit=EMBED_BATCH)
        except Exception as err:  # noqa: BLE001
            logger.warning(f"Worker error: {err}")
            await asyncio.sleep(interval_sec)
