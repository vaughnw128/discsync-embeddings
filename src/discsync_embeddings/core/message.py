# built-in
import asyncio
import hashlib
import json
import re
from re import Match, Pattern
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast, Callable, Coroutine
from collections.abc import Buffer
from uuid import uuid5, NAMESPACE_URL

# external
import emoji

# project
from discsync_embeddings.core.db import get_session
from discsync_embeddings.core.sqlmodels import Message, User, Channel

_author_name_cache: Dict[int, str] = {}
_channel_name_cache: Dict[int, str] = {}

# Normalization patterns
_RE_EMOJI_CUSTOM = re.compile(r"<a?:([A-Za-z0-9_~]+):\d+>")
_RE_MENTION_USER = re.compile(r"<@!?(\d+)>")
_RE_MENTION_ROLE = re.compile(r"<@&(\d+)>")
_RE_MENTION_CHANNEL = re.compile(r"<#(\d+)>")


async def _async_re_sub(
    pattern: Pattern[str],
    repl_coro: Callable[[Match[str]], Coroutine[Any, Any, str]],
    text: str,
) -> str:
    """Asynchronous version of re.sub where the replacement function is a coroutine."""
    matches: List[Match[str]] = list(pattern.finditer(text))
    if not matches:
        return text

    # Start all replacement coroutines concurrently.
    replacements = await asyncio.gather(*(repl_coro(m) for m in matches))

    parts: List[str] = []
    last_end = 0
    for match, replacement in zip(matches, replacements):
        start, end = match.span()
        parts.append(text[last_end:start])
        parts.append(replacement)
        last_end = end
    parts.append(text[last_end:])

    return "".join(parts)


def _collapse_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces."""

    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _replace_emojis(text: str) -> str:
    """Replace emojis in text.

    - Custom Discord emojis like `<:name:id>` or `<a:name:id>` become
      `[emoji:name]`.
    - Runs of Unicode emoji collapse to a single `[emoji]` token.
    """

    out = _RE_EMOJI_CUSTOM.sub(r"[emoji:\1]", text)
    out = emoji.demojize(out)
    return out


async def _replace_mentions(text: str) -> str:
    """Replace mentions in text.

    - User mentions like `<@id>` become `@username`.
    - Channel mentions like `<#id>` become `#channelname`.
    - Role mentions like `<@&id>` are changed to @role.
    """

    async def _replace_user_mention(match: re.Match[str]) -> str:
        user_id_raw = match.group(1)
        try:
            user_id = int(user_id_raw)
        except (TypeError, ValueError):
            return match.group(0)

        author_name = await get_author_name(user_id)
        return f"@{author_name}"

    async def _replace_channel_mention(match: re.Match[str]) -> str:
        channel_id_raw = match.group(1)
        try:
            channel_id = int(channel_id_raw)
        except (TypeError, ValueError):
            return match.group(0)

        channel_name = await get_channel_name(channel_id)
        return f"#{channel_name}"

    out = await _async_re_sub(_RE_MENTION_USER, _replace_user_mention, text)
    out = await _async_re_sub(_RE_MENTION_CHANNEL, _replace_channel_mention, out)
    out = _RE_MENTION_ROLE.sub("@role", out)
    return out


async def get_author_name(author_id: int) -> str:
    """Get the author's display name (with simple in-memory caching)."""

    cached = _author_name_cache.get(author_id)
    if cached is not None:
        return cached
    async with get_session() as session:
        user = await session.get(User, author_id)
    try:
        username = user.global_name
    except Exception:  # noqa: BLE001
        username = "bot"
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


async def clean_message(content: Optional[str]) -> str:
    """Clean message content for embedding."""

    if content is None:
        return ""

    text = _collapse_whitespace(content)
    text = _replace_emojis(text)
    text = text.encode("ascii", errors="ignore").decode()  # remove non printable characters
    text = await _replace_mentions(text)
    return text


@dataclass(slots=True)
class EmbeddableMessage:
    """Build an embeddable message from a Message object."""

    message: Message
    cleaned_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    pid: Optional[str] = None

    async def build_text(self) -> str:
        """Build message text"""

        m = self.message
        channel_name = await get_channel_name(self.message.channel_id)
        author_name = await get_author_name(self.message.author_id)
        content = await clean_message(m.content)
        parts: List[str] = [
            f"@{author_name} in #{channel_name}:",
            content,
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

    async def build(self) -> "EmbeddableMessage":
        self.cleaned_text = await self.build_text()
        self.metadata = await self.get_metadata()
        self.pid = get_point_id(self.message.id)

        return self
