# built-in
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from collections.abc import Buffer
from uuid import uuid5, NAMESPACE_URL

from llama_index.core.schema import TextNode

from discsync_embeddings.core.db import get_session

# project
from discsync_embeddings.core.sqlmodels import Message, User, Channel
from sqlalchemy import select

# In-memory caches for names
_author_name_cache: Dict[int, str] = {}
_channel_name_cache: Dict[int, str] = {}

# Normalization patterns
_RE_EMOJI_CUSTOM = re.compile(r"<a?:([A-Za-z0-9_~]+):\d+>")
_RE_MENTION_USER = re.compile(r"<@!?(\d+)>")
_RE_MENTION_ROLE = re.compile(r"<@&(\d+)>")
_RE_MENTION_CHANNEL = re.compile(r"<#(\d+)>")


def _normalize_once(text: str) -> str:
    """Apply Discord-specific normalization passes without trimming length."""

    text = _RE_EMOJI_CUSTOM.sub(r"[custom:\1]", text)
    text = _RE_MENTION_USER.sub(r"@user:\1", text)
    text = _RE_MENTION_ROLE.sub(r"@role:\1", text)
    text = _RE_MENTION_CHANNEL.sub(r"#channel:\1", text)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces."""

    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def normalize_text(text: Optional[str], *, limit: int | None = None) -> str:
    """Normalize Discord markup and collapse whitespace; optional truncation."""

    if text is None:
        return ""
    out = _collapse_whitespace(_normalize_once(text))
    if isinstance(limit, int) and limit > 0:
        return out[:limit]
    return out


def _pick_user_display_name(user: Optional[User], fallback: str) -> str:
    """Get the display name for a user, falling back to the user ID."""

    if user is None:
        return fallback
    if getattr(user, "global_name", None):
        return str(user.global_name)
    if getattr(user, "username", None):
        return str(user.username)
    return fallback


async def prefetch_names(author_ids: set[int], channel_ids: set[int]) -> None:
    """Warm author and channel name caches for the provided IDs in bulk."""

    if not author_ids and not channel_ids:
        return

    async with get_session() as session:
        if author_ids:
            result = await session.execute(select(User).where(User.id.in_(list(author_ids))))
            for user in result.scalars():
                _author_name_cache[user.id] = _pick_user_display_name(user, str(user.id))
        if channel_ids:
            result = await session.execute(select(Channel).where(Channel.id.in_(list(channel_ids))))
            for channel in result.scalars():
                _channel_name_cache[channel.id] = channel.name if getattr(channel, "name", None) else str(channel.id)


async def get_author_name(author_id: int) -> str:
    """Get the author's display name (with simple in-memory caching)."""

    cached = _author_name_cache.get(author_id)
    if cached is not None:
        return cached
    async with get_session() as session:
        user = await session.get(User, author_id)
    username = _pick_user_display_name(user, str(author_id))
    _author_name_cache[author_id] = username
    return username


async def get_channel_name(channel_id: int) -> str:
    """Get a channel name (with simple in-memory caching)."""

    cached = _channel_name_cache.get(channel_id)
    if cached is not None:
        return cached
    async with get_session() as session:
        channel = await session.get(Channel, channel_id)
    channel_name = channel.name if channel and getattr(channel, "name", None) else str(channel_id)
    _channel_name_cache[channel_id] = channel_name
    return channel_name


def get_point_id(msg_id: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"discsync:message:{msg_id}"))


def message_fingerprint(m: Message) -> str:
    """Compute a fingerprint for a message based on its content and metadata."""

    blob: Dict[str, Any] = {
        "id": m.id,
        "content": m.content,
        "edited_at_ms": m.edited_at_ms,
        "attachments": m.attachments,
        "embeds": m.embeds,
        "components": m.components,
        "reactions": m.reactions,
        "message_reference": m.message_reference,
    }
    data = json.dumps(blob, sort_keys=True, ensure_ascii=False, default=str)
    buf = cast(Buffer, data.encode("utf-8"))
    return hashlib.sha256(buf).hexdigest()


@dataclass(slots=True)
class EmbeddableMessage:
    """Build an embeddable message from a Message object."""

    message: Message

    async def build_text(self) -> str:
        """Build message text"""

        m = self.message
        channel_name = await get_channel_name(self.message.channel_id)
        author_name = await get_author_name(self.message.author_id)
        parts: List[str] = [
            f"@{author_name} in #{channel_name}:",
            normalize_text(m.content, limit=4000),
        ]

        if m.reactions:
            parts.append("[has reactions]")

        if m.message_reference:
            ref_author_raw = m.message_reference.get("author", {}).get("id")
            try:
                ref_author_id = int(ref_author_raw) if ref_author_raw is not None else None
            except (TypeError, ValueError):
                ref_author_id = None
            if ref_author_id is not None:
                original_author_name = await get_author_name(ref_author_id)
                parts.append(f"[replying to @{original_author_name}]")

        if m.attachments:
            parts.append("[shared attachments]")

        return " ".join(parts)  # Spaces work better!

    async def get_metadata(self) -> Dict[str, Any]:
        m = self.message

        author_name = await get_author_name(self.message.author_id)
        channel_name = await get_channel_name(self.message.channel_id)

        metadata: Dict[str, Any] = {
            "message_id": m.id,
            "guild_id": m.guild_id,
            "channel_id": m.channel_id,
            "author_id": m.author_id,
            "created_at_ms": m.created_at_ms,
            "author_name": author_name,
            "channel_name": channel_name,
        }

        return metadata

    async def to_text_node(self) -> TextNode:
        """Convert to a TextNode for embedding."""

        text = await self.build_text()
        metadata = await self.get_metadata()
        pid = get_point_id(self.message.id)
        return TextNode(text=text, metadata=metadata, id_=pid)
