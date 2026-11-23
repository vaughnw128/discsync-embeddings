# built-in
from datetime import datetime
from typing import Any

# external
from sqlalchemy import Column, Index
from sqlalchemy.types import JSON, BigInteger, Text
from sqlmodel import Field, SQLModel


class Guild(SQLModel, table=True):
    __tablename__ = "guilds"

    id: int = Field(primary_key=True, sa_type=BigInteger())
    name: str | None = Field(default=None)
    owner_id: int | None = Field(default=None, sa_type=BigInteger())
    icon: str | None = Field(default=None)
    features: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    raw_json: dict[str, Any] = Field(sa_column=Column(JSON()))


class Channel(SQLModel, table=True):
    __tablename__ = "channels"

    id: int = Field(primary_key=True, sa_type=BigInteger())
    guild_id: int | None = Field(default=None, index=True, sa_type=BigInteger())
    name: str | None = Field(default=None)
    kind: str | None = Field(default=None)
    topic: str | None = Field(default=None)
    nsfw: bool = Field(default=False)
    parent_id: int | None = Field(default=None, sa_type=BigInteger())
    position: int | None = Field(default=None)
    raw_json: dict[str, Any] = Field(sa_column=Column(JSON()))


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int = Field(primary_key=True, sa_type=BigInteger())
    username: str | None = Field(default=None)
    global_name: str | None = Field(default=None)
    discriminator: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    bot: bool = Field(default=False)
    raw_json: dict[str, Any] = Field(sa_column=Column(JSON()))


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: int = Field(primary_key=True, sa_type=BigInteger())
    guild_id: int | None = Field(default=None, index=True, sa_type=BigInteger())
    channel_id: int = Field(index=True, sa_type=BigInteger())
    author_id: int = Field(index=True, sa_type=BigInteger())
    created_at_ms: int = Field(index=True, sa_type=BigInteger())
    edited_at_ms: int | None = Field(default=None, sa_type=BigInteger())
    content: str | None = Field(default=None)
    tts: bool = Field(default=False)
    pinned: bool = Field(default=False)
    kind: str | None = Field(default=None)
    flags: int | None = Field(default=None, sa_type=BigInteger())

    mentions: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    attachments: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    embeds: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    components: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    reactions: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    message_reference: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    raw_json: dict[str, Any] = Field(sa_column=Column(JSON()))

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

    message_id: int = Field(primary_key=True, foreign_key="messages.id", sa_type=BigInteger())
    embedded_at: datetime | None = Field(default=None, index=True)
    model_name: str | None = Field(default=None)
    chunk_count: int | None = Field(default=None)
    formatted_text: str | None = Field(default=None, sa_type=Text())
    qdrant_collection: str | None = Field(default=None)
    qdrant_point_ids: list[str] | None = Field(default=None, sa_column=Column(JSON()))
    status: str | None = Field(default=None, index=True)
    error: str | None = Field(default=None)
    content_sha256: str | None = Field(default=None, index=True)

    __table_args__ = (
        Index(
            "idx_msg_embed_status_embedded",
            "status",
            "embedded_at",
        ),
    )
