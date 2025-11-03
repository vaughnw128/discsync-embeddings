# built-in
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast
from collections.abc import Buffer
from collections import OrderedDict

# external
from sqlalchemy.ext.asyncio import AsyncSession

# project
from discsync_embeddings.core.sqlmodels import Message, User, Channel

# Normalization patterns
_RE_EMOJI_CUSTOM = re.compile(r"<a?:([A-Za-z0-9_~]+):\d+>")
_RE_MENTION_USER = re.compile(r"<@!?(\d+)>")
_RE_MENTION_ROLE = re.compile(r"<@&(\d+)>")
_RE_MENTION_CHANNEL = re.compile(r"<#(\d+)>")

# Simple LRU caches for names
_USER_NAME_CACHE: OrderedDict[int, Optional[str]] = OrderedDict()
_CHANNEL_NAME_CACHE: OrderedDict[int, Optional[str]] = OrderedDict()
_CACHE_MAX: int = 5000


def _cache_get(cache: "OrderedDict[int, Optional[str]]", key: int) -> Optional[str] | None:
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _cache_put(cache: "OrderedDict[int, Optional[str]]", key: int, value: Optional[str]) -> None:
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _CACHE_MAX:
        cache.popitem(last=False)


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


def summarize_reactions(
    reactions: Optional[List[Dict[str, Any]]],
) -> str:
    """Summarize reactions with counts."""

    if not reactions or not isinstance(reactions, list):
        return ""
    try:
        pairs: List[str] = []
        for r in reactions:
            emoji = r.get("emoji") or {}
            name = emoji.get("name")
            emoji_id = emoji.get("id")
            try:
                icount = int(r.get("count")) if r.get("count") is not None else 0
            except Exception:
                icount = 0
            if icount <= 0:
                continue

            # handle custom as [custom:name];
            is_custom = emoji_id not in (None, "", 0)
            if is_custom:
                label = f"[custom:{name}]" if name else "[custom]"
            else:
                label = str(name) if name else ""
            if label:
                pairs.append(f"{label} x{icount}")
        return f"reactions: {'; '.join(pairs)}." if pairs else ""
    except Exception:
        return ""


def summarize_attachments(
    attachments: Optional[List[Dict[str, Any]]],
) -> str:
    """Summarize attachment types with counts."""

    if not attachments or not isinstance(attachments, list):
        return ""
    try:
        counts: Dict[str, int] = {}
        filenames: List[str] = []
        for a in attachments:
            ctype = a.get("content_type") or a.get("type")
            if ctype:
                kind = str(ctype).split("/")[0]
                counts[kind] = counts.get(kind, 0) + 1
            fname = a.get("filename") or a.get("name")
            if fname:
                filenames.append(normalize_text(str(fname), limit=80))
        parts: List[str] = []
        if counts:
            items = sorted(counts.items())[:5]
            parts.append("attachments: " + ", ".join(f"{k} x{v}" for k, v in items))
        if filenames:
            parts.append("files: " + ", ".join(filenames[:3]))
        return ". ".join(parts) + "." if parts else ""
    except Exception:
        return ""


def _get_author_name(m: Message) -> Optional[str]:
    """Get the author's display name from raw_json."""

    try:
        author_obj = (m.raw_json or {}).get("author", {})
        name = author_obj.get("global_name") or author_obj.get("username")
        return str(name) if name else None
    except Exception:
        return None


def _get_channel_name(m: Message) -> Optional[str]:
    """Get the channel name from raw_json."""

    try:
        ch = (m.raw_json or {}).get("channel") or {}
        name = ch.get("name")
        return str(name) if name else None
    except Exception:
        return None


async def resolve_author_name(session: AsyncSession, user_id: Optional[int], fallback: Optional[str]) -> Optional[str]:
    if user_id is None:
        return fallback
    cached = _cache_get(_USER_NAME_CACHE, user_id)
    if cached is not None:
        return cached
    try:
        user = await session.get(User, user_id)
        name: Optional[str] = None
        if user is not None:
            name = user.global_name or user.username  # type: ignore[attr-defined]
            name = str(name) if name else None
        if name is None:
            name = fallback
        _cache_put(_USER_NAME_CACHE, user_id, name)
        return name
    except Exception:
        return fallback


async def resolve_channel_name(
    session: AsyncSession, channel_id: Optional[int], fallback: Optional[str]
) -> Optional[str]:
    if channel_id is None:
        return fallback
    cached = _cache_get(_CHANNEL_NAME_CACHE, channel_id)
    if cached is not None:
        return cached
    try:
        ch = await session.get(Channel, channel_id)
        name: Optional[str] = None
        if ch is not None:
            name = getattr(ch, "name", None)
            name = str(name) if name else None
        if name is None:
            name = fallback
        _cache_put(_CHANNEL_NAME_CACHE, channel_id, name)
        return name
    except Exception:
        return fallback


def summarize_embeds(embeds: Optional[List[Dict[str, Any]]]) -> str:
    """Summarize embed titles."""

    if not embeds or not isinstance(embeds, list):
        return ""
    try:
        items: List[str] = []
        for e in embeds:
            raw = e.get("title") or e.get("site_name") or e.get("url")
            if raw:
                label = normalize_text(str(raw), limit=60)
                if label:
                    items.append(label)
        return f"links: {'; '.join(items[:3])}." if items else ""
    except Exception:
        return ""


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
class MessageChunkBuilder:
    """Build a text chunk and metadata payload for a message."""

    message: Message
    author_name: Optional[str] = None
    channel_name: Optional[str] = None
    body_override: Optional[str] = None
    context_override: Optional[str] = None

    def meta(self) -> str:
        """Message metadata."""

        m = self.message
        author = self.author_name or _get_author_name(m)
        ch_name = self.channel_name or _get_channel_name(m)
        # Always include a channel label; fallback to id if no name
        if ch_name:
            ch_label = f"#{normalize_text(ch_name, limit=60)}"
        else:
            ch_label = f"#channel:{m.channel_id}"

        parts: List[str] = []
        if author:
            parts.append(f"@{normalize_text(author, limit=60)}")
        if ch_label:
            if parts:
                parts.append(f"in {ch_label}")
            else:
                parts.append(ch_label)
        return f"meta: {' '.join(parts)}" if parts else ""

    def reply_context(self) -> str:
        if self.context_override:
            return self.context_override

        # One-line parent quote if present in raw_json.referenced_message
        m = self.message
        quoted = _get_parent_content(m)
        if quoted:
            snippet = normalize_text(quoted, limit=200)
            return f"context: {snippet}"
        rid = _get_reply_id(m)
        return f"context: reply-to {rid}" if rid else ""

    def body(self) -> str:
        if self.body_override is not None:
            return self.body_override
        return normalize_text(self.message.content)

    def summaries(self) -> List[str]:
        return [
            summarize_reactions(self.message.reactions),
            summarize_attachments(self.message.attachments),
            summarize_embeds(self.message.embeds),
        ]

    def _build_summary_block(self) -> str:
        """Create a single summary string joined by ' | '."""
        return " | ".join(s for s in self.summaries() if s)

    def payload(self) -> Dict[str, Any]:
        m = self.message
        return {
            "message_id": m.id,
            "guild_id": m.guild_id,
            "channel_id": m.channel_id,
            "author_id": m.author_id,
            "created_at_ms": m.created_at_ms,
        }

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the passage text and metadata payload for a message."""

        meta_line = self.meta()
        body_str = self.body()
        passage_part = f"passage: {body_str}" if body_str else ""
        ctx_line = self.reply_context()
        summary_str = self._build_summary_block()

        # Desired order: meta -> passage -> context -> summaries
        parts: List[str] = []
        if meta_line:
            parts.append(meta_line)
        if passage_part:
            parts.append(passage_part)
        if ctx_line:
            parts.append(ctx_line)
        if summary_str:
            parts.append(summary_str)

        text = " ".join(p for p in parts if p)
        return text, self.payload()


async def build_chunk_async(session: AsyncSession, m: Message) -> Tuple[str, Dict[str, Any]]:
    """Async variant that resolves user/channel names and reply author."""

    auth_fallback = _get_author_name(m)
    chan_fallback = _get_channel_name(m)
    author_name = await resolve_author_name(session, m.author_id, auth_fallback)
    channel_name = await resolve_channel_name(session, m.channel_id, chan_fallback)

    body_text = await _resolve_mentions_in_text(session, m.content, m.mentions)

    parent_uid, parent_name_hint = _get_parent_author(m)
    parent_name = await resolve_author_name(session, parent_uid, parent_name_hint) if parent_uid else parent_name_hint
    parent_content = _get_parent_content(m)
    context_line: Optional[str] = None
    if parent_content:
        parent_body = await _resolve_mentions_in_text(
            session, parent_content, (m.raw_json or {}).get("referenced_message", {}).get("mentions")
        )
        if parent_name:
            context_line = f"context: @{parent_name}: {parent_body}"
        else:
            context_line = f"context: {parent_body}"
    else:
        rid = _get_reply_id(m)
        if rid and parent_name:
            context_line = f"context: reply-to @{parent_name}"
        elif rid:
            context_line = f"context: reply-to {rid}"

    builder = MessageChunkBuilder(
        message=m,
        author_name=author_name,
        channel_name=channel_name,
        body_override=body_text,
        context_override=context_line,
    )
    return builder.build()


def build_chunk(m: Message) -> Tuple[str, Dict[str, Any]]:
    """Build a text chunk and metadata payload for a message."""

    builder = MessageChunkBuilder(message=m)
    return builder.build()


def _get_reply_id(m: Message) -> Optional[str]:
    """Get the replied-to message ID from message_reference."""

    ref = m.message_reference or {}
    if isinstance(ref, dict):
        rid = ref.get("message_id") or ref.get("id")
        return str(rid) if rid else None
    return None


def _get_parent_content(m: Message) -> Optional[str]:
    """Get the content of the parent message if available in raw_json."""

    try:
        parent = (m.raw_json or {}).get("referenced_message") or {}
        txt = parent.get("content")
        return str(txt) if txt else None
    except Exception:
        return None


def _get_parent_author(m: Message) -> Tuple[Optional[int], Optional[str]]:
    """Get parent author's id and display name from raw_json if present."""

    try:
        parent = (m.raw_json or {}).get("referenced_message") or {}
        author = parent.get("author") or {}
        uid = author.get("id")
        name = author.get("global_name") or author.get("username")
        uid_int = int(uid) if uid is not None else None
        return uid_int, (str(name) if name else None)
    except Exception:
        return None, None


def _extract_mentions_map(mentions: Optional[Dict[str, Any]]) -> Dict[int, str]:
    """Build a map of user_id -> display name from mentions JSON if present."""

    result: Dict[int, str] = {}
    if not mentions:
        return result
    try:
        candidates = []
        if isinstance(mentions, list):
            candidates = mentions
        elif isinstance(mentions, dict):
            users_list = mentions.get("users")
            if isinstance(users_list, list):
                candidates = users_list
            else:
                for v in mentions.values():
                    if isinstance(v, list):
                        candidates.extend([x for x in v if isinstance(x, dict)])
        for u in candidates:
            uid = u.get("id")
            if uid is None:
                continue
            try:
                uid_int = int(uid)
            except Exception:
                continue
            name = u.get("global_name") or u.get("username")
            if name:
                result[uid_int] = str(name)
    except Exception:
        return result
    return result


async def _resolve_mentions_in_text(
    session: AsyncSession, text: Optional[str], mentions: Optional[Dict[str, Any]]
) -> str:
    """Replace <@id> tokens with @username using mentions JSON and DB."""

    src = text or ""
    if not src:
        return ""
    ids: List[int] = []
    for m_ in _RE_MENTION_USER.finditer(src):
        try:
            ids.append(int(m_.group(1)))
        except Exception:
            continue
    if not ids:
        return normalize_text(src)

    ids_set = list(dict.fromkeys(ids))
    name_map = _extract_mentions_map(mentions)

    resolved: Dict[int, Optional[str]] = {}
    for uid in ids_set:
        if uid in name_map:
            resolved[uid] = name_map[uid]
        else:
            resolved[uid] = await resolve_author_name(session, uid, None)

    out_parts: List[str] = []
    last = 0
    for m_ in _RE_MENTION_USER.finditer(src):
        uid_s = m_.group(1)
        try:
            uid = int(uid_s)
        except Exception:
            uid = None
        out_parts.append(src[last : m_.start()])
        if uid is not None and resolved.get(uid):
            out_parts.append(f"@{resolved[uid]}")
        else:
            out_parts.append(m_.group(0))
        last = m_.end()
    out_parts.append(src[last:])

    return normalize_text("".join(out_parts))
