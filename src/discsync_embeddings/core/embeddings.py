# built-in
import asyncio
import os
import time
from datetime import datetime, UTC
from typing import List, Any

# external
from sqlalchemy import select, case
from sqlalchemy.ext.asyncio import AsyncSession

# project
from discsync_embeddings.core.db import get_session
from discsync_embeddings.core.qdrant import get_qdrant, QdrantService
from discsync_embeddings.core.sqlmodels import Message, MessageEmbedding
from discsync_embeddings.core.message import (
    message_fingerprint,
    EmbeddableMessage,
)
from discsync_embeddings.helpers.logging import logger

# Embedding settings
EMBED_BATCH: int = int(os.environ.get("EMBED_BATCH", "100"))


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


async def embed_batch(qdrant: QdrantService, messages: List[Message]) -> int:
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
    built_embeddable_messages: List[EmbeddableMessage] = await asyncio.gather(
        *(emb.build() for emb in embeddable_messages)
    )

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
        for message in built_embeddable_messages:
            msg_id = int(message.metadata.get("message_id"))
            row = rows_by_id[msg_id]
            row.status = "tokenizing"
            row.content_sha256 = fps[msg_id]
            row.formatted_text = message.cleaned_text

        await session.flush()

        time_tokenize_end = time.monotonic() - time_tokenize_start

        # Proceed to embedding
        time_embed_start = time.monotonic()

        await qdrant.embed_messages(built_embeddable_messages)

        time_embed_end = time.monotonic() - time_embed_start

        # Finalize embed rows
        for message in built_embeddable_messages:
            msg_id = int(message.metadata.get("message_id"))
            emb_row = rows_by_id[msg_id]
            emb_row.qdrant_collection = qdrant.collection
            emb_row.qdrant_point_ids = [message.pid]
            emb_row.model_name = qdrant.model_name
            emb_row.chunk_count = 1
            emb_row.status = "completed"
            emb_row.error = None
            emb_row.embedded_at = datetime.now(UTC).replace(tzinfo=None)
        await session.flush()

    total_time = time_embed_end + time_tokenize_end
    messages_per_second = (len(built_embeddable_messages) / total_time) if total_time > 0 else 0.0
    logger.info(
        f"timing: count={len(built_embeddable_messages)} tokenize={time_tokenize_end:.3f}s "
        f"embed={time_embed_end:.3f}s"
        f"total={total_time:.3f}s mps={messages_per_second:.2f}"
    )
    return len(built_embeddable_messages)


async def process_pending_messages(qdrant: QdrantService, limit: int = 100) -> None:
    async with get_session() as session:
        response = await session.execute(_pending_messages_stmt(limit))
        messages = list(response.scalars().all())

        logger.debug(f"Fetched {len(messages)} pending messages")

        try:
            await embed_batch(qdrant, messages)
        except Exception as err:  # noqa: BLE001
            logger.warning(f"Embed batch failed: {err}")


async def worker_loop(interval_sec: float = 5.0) -> None:
    """Background worker loop to process pending messages for embedding."""

    qdrant_service = await get_qdrant()

    while True:
        try:
            await process_pending_messages(qdrant=qdrant_service, limit=EMBED_BATCH)
        except Exception as err:  # noqa: BLE001
            logger.warning(f"Worker error: {err}")
            await asyncio.sleep(interval_sec)
