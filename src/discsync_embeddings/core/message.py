# built-in
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast
from collections.abc import Buffer

# project
from discsync_embeddings.core.sqlmodels import Message


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text[:4000]


def summarize_reactions(
    reactions: Optional[List[Dict[str, Any]]],
) -> str:
    if not reactions or not isinstance(reactions, list):
        return ""
    try:
        pairs: List[str] = []
        for r in reactions:
            emoji = (r.get("emoji") or {}).get("name")
            count = r.get("count")
            if emoji and count:
                pairs.append(f":{emoji}:x{count}")
        return f" Reactions: {' '.join(pairs)}." if pairs else ""
    except Exception:
        return ""


def summarize_attachments(
    attachments: Optional[List[Dict[str, Any]]],
) -> str:
    if not attachments or not isinstance(attachments, list):
        return ""
    try:
        kinds: List[str] = []
        for a in attachments:
            ctype = a.get("content_type") or a.get("type")
            if ctype:
                kinds.append(str(ctype).split("/")[0])
        if kinds:
            top = sorted(kinds)[:5]
            return f" Attachments: {', '.join(top)}."
        return ""
    except Exception:
        return ""


def summarize_embeds(embeds: Optional[List[Dict[str, Any]]]) -> str:
    if not embeds or not isinstance(embeds, list):
        return ""
    try:
        titles: List[str] = []
        for e in embeds:
            t = e.get("title") or e.get("site_name") or e.get("url")
            if t:
                titles.append(str(t)[:60])
        if titles:
            return f" Links: {', '.join(titles[:3])}."
        return ""
    except Exception:
        return ""


def message_fingerprint(m: Message) -> str:
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


@dataclass
class MessageChunkBuilder:
    message: Message

    def header(self) -> str:
        m = self.message
        return (
            f"[{m.created_at_ms}] guild={m.guild_id} "
            f"channel={m.channel_id} author={m.author_id} msg={m.id}"
        )

    def body(self) -> str:
        return clean_text(self.message.content)

    def summaries(self) -> List[str]:
        m = self.message
        return [
            summarize_reactions(m.reactions),
            summarize_attachments(m.attachments),
            summarize_embeds(m.embeds),
        ]

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
        parts: List[str] = [self.header()]
        b = self.body()
        if b:
            parts.append(b)
        parts.extend(s for s in self.summaries() if s)
        return " ".join(p for p in parts if p), self.payload()


def build_chunk(m: Message) -> Tuple[str, Dict[str, Any]]:
    builder = MessageChunkBuilder(message=m)
    return builder.build()
