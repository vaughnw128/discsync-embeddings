# built-in
from typing import Optional, List, Dict, Any
from datetime import datetime

# external
from sqlalchemy import Index, Column
from sqlalchemy.types import JSON, BigInteger, Text
from sqlmodel import Field, SQLModel


class Guild(SQLModel, table=True):
    __tablename__ = "guilds"

    id: int = Field(primary_key=True, sa_type=BigInteger())
    name: Optional[str] = Field(default=None)
    owner_id: Optional[int] = Field(default=None, sa_type=BigInteger())
    icon: Optional[str] = Field(default=None)
    features: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON()))
    raw_json: Dict[str, Any] = Field(sa_column=Column(JSON()))


class Channel(SQLModel, table=True):
    __tablename__ = "channels"

    id: int = Field(primary_key=True, sa_type=BigInteger())
    guild_id: Optional[int] = Field(default=None, index=True, sa_type=BigInteger())
    name: Optional[str] = Field(default=None)
    kind: Optional[str] = Field(default=None)
    topic: Optional[str] = Field(default=None)
    nsfw: bool = Field(default=False)
    parent_id: Optional[int] = Field(default=None, sa_type=BigInteger())
    position: Optional[int] = Field(default=None)
    raw_json: Dict[str, Any] = Field(sa_column=Column(JSON()))


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int = Field(primary_key=True, sa_type=BigInteger())
    username: Optional[str] = Field(default=None)
    global_name: Optional[str] = Field(default=None)
    discriminator: Optional[str] = Field(default=None)
    avatar: Optional[str] = Field(default=None)
    bot: bool = Field(default=False)
    raw_json: Dict[str, Any] = Field(sa_column=Column(JSON()))


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: int = Field(primary_key=True, sa_type=BigInteger())
    guild_id: Optional[int] = Field(default=None, index=True, sa_type=BigInteger())
    channel_id: int = Field(index=True, sa_type=BigInteger())
    author_id: int = Field(index=True, sa_type=BigInteger())
    created_at_ms: int = Field(index=True, sa_type=BigInteger())
    edited_at_ms: Optional[int] = Field(default=None, sa_type=BigInteger())
    content: Optional[str] = Field(default=None)
    tts: bool = Field(default=False)
    pinned: bool = Field(default=False)
    kind: Optional[str] = Field(default=None)
    flags: Optional[int] = Field(default=None, sa_type=BigInteger())

    mentions: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON()))
    attachments: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON()))
    embeds: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON()))
    components: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON()))
    reactions: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON()))
    message_reference: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON()))
    raw_json: Dict[str, Any] = Field(sa_column=Column(JSON()))

    __table_args__ = (
        Index(
            "idx_messages_channel_id_created",
            "channel_id",
            "created_at_ms",
        ),
        Index(
            "idx_messages_channel_id_id",
            "channel_id",
            "id",
        ),
        Index(
            "idx_messages_author_id",
            "author_id",
        ),
    )


class MessageEmbedding(SQLModel, table=True):
    __tablename__ = "message_embeddings"

    # One-to-one with messages.id
    message_id: int = Field(primary_key=True, foreign_key="messages.id", sa_type=BigInteger())

    # Processing metadata
    embedded_at: Optional[datetime] = Field(default=None, index=True)
    model_name: Optional[str] = Field(default=None)
    chunk_count: Optional[int] = Field(default=None)

    # Store formatted chunk text used for embedding
    formatted_text: Optional[str] = Field(default=None, sa_type=Text())

    # Destination info
    qdrant_collection: Optional[str] = Field(default=None)
    # Store point ids as JSON array for portability
    qdrant_point_ids: Optional[List[str]] = Field(default=None, sa_column=Column(JSON()))

    # Status tracking
    status: Optional[str] = Field(default=None, index=True)
    error: Optional[str] = Field(default=None)

    # Optional hash of content to avoid re-embedding if unchanged
    content_sha256: Optional[str] = Field(default=None, index=True)

    __table_args__ = (
        Index(
            "idx_msg_embed_status_embedded",
            "status",
            "embedded_at",
        ),
    )
