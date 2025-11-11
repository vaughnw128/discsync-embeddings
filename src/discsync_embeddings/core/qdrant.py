# built-in
import os
from typing import List, Optional
import asyncio

# external
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from discsync_embeddings.core.message import EmbeddableMessage

# project
from discsync_embeddings.helpers.logging import logger

# Embedding settings
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")

# Qdrant settings
QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "discsync-embeddings")
QDRANT_HOSTNAME: Optional[str] = os.environ.get("QDRANT_HOSTNAME")
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "443"))
QDRANT_API_KEY: Optional[str] = os.environ.get("QDRANT_API_KEY")
QDRANT_TIMEOUT: int = int(os.environ.get("QDRANT_TIMEOUT", "30"))


def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client based on environment configuration."""

    hostname = (QDRANT_HOSTNAME or "").strip() or None
    if hostname is not None:
        logger.info(f"Connecting to Qdrant at {hostname}:{QDRANT_PORT}")
        return QdrantClient(
            url=hostname,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
            prefer_grpc=False,
        )

    # Local/default connection
    logger.info(f"Connecting to local Qdrant at localhost:{QDRANT_PORT}")
    return QdrantClient(
        host="localhost",
        port=QDRANT_PORT,
        timeout=QDRANT_TIMEOUT,
    )


class QdrantService:
    """Service for interacting with Qdrant vector store."""

    embeddings: OpenAIEmbeddings
    vector_store: QdrantVectorStore
    model_name: str = EMBEDDING_MODEL
    collection: str = QDRANT_COLLECTION

    async def setup(self) -> "QdrantService":
        logger.info("Setting up QdrantService...")

        self.embeddings = OpenAIEmbeddings(
            base_url="http://localhost:1234/v1",
            dimensions=1024,
            model=EMBEDDING_MODEL,
            api_key="n/a",
            timeout=60,
            check_embedding_ctx_length=False,
        )

        client = get_qdrant_client()

        # Try to create the collection if it doesn't exist
        try:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        except Exception:  # noqa: BLE001
            pass  # Already exists

        self.vector_store = QdrantVectorStore(
            client=client,
            embedding=self.embeddings,
            collection_name=QDRANT_COLLECTION,
            validate_collection_config=False,
        )

        logger.info("QdrantService setup complete.")

        return self

    async def embed_messages(self, messages: List[EmbeddableMessage]) -> List[str]:
        """Embed and store messages in Qdrant.

        Uses add_texts with explicit string coercion to avoid accidental None
        values or non-string types triggering 400 errors from the embedding
        backend (expects string or array of strings in 'input').
        """

        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []

        for message in messages:
            cleaned = message.cleaned_text if isinstance(message.cleaned_text, str) else ""
            texts.append(cleaned)
            metadatas.append(message.metadata or {})
            ids.append(message.pid or "")

        if texts:
            sample_count = min(3, len(texts))
            logger.debug(
                "Embedding batch sample types: "
                + ", ".join(f"{type(t).__name__}:{repr(t)[:40]}" for t in texts[:sample_count])
            )

        await asyncio.to_thread(
            self.vector_store.add_texts,
            texts,
            metadatas=metadatas,
            ids=ids,
        )
        return ids


_service: Optional[QdrantService] = None


async def get_qdrant() -> QdrantService:
    global _service
    if _service is not None:
        return _service
    _service = QdrantService()
    await _service.setup()
    return _service
