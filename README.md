# Discsync Embeddings

Discord message embeddings to work with the [discsync](github.com/vaughnw128/discsync) message syncing service

## Overview

### Simple flow
- Background worker fetches pending messages in batches of 64 from the discsync db.
- For each message, it builds a single-line `formatted_text`:
  - meta + passage + context + summaries (in that order)
  - Resolves `@username` and `#channel` from the database; cleans Discord markup.
- Embeds the batch using a HuggingFace embedding model.
- Upserts vectors into Qdrant with stable UUIDv5 point IDs (derived from message id).
- Records the embedding result in `message_embeddings` (status, formatted_text,
  model info, token counts, Qdrant collection/IDs, timestamp).
- Repeats continuously; logs one concise timing line per batch with embed time,
  upsert time, and accurate messages-per-second.

Handles retries, rate limits, and transient errors gracefully, recovering automatically and restarting
from checkpoints in the database.

# Technical Details

## Embedded text (formatted_text) schema

The service embeds a single, human-readable line per message called
`formatted_text`. It has a defined schema that is not deviated from to ensure downstream
models have consistent data.

Order and segments
- meta: Always first when present
  - Format: `meta: @<author> in #<channel>`
  - Fallbacks:
    - If author name unknown: omit the `@<author>` part
    - If channel name unknown: `#channel:<channel_id>`
  - Examples:
    - `meta: @alice in #general`
    - `meta: #channel:1250000000000000000`
- passage: Only when body text exists
  - Format: `passage: <message-body>`
  - Mentions are resolved: `<@123>` → `@alice` when known; otherwise `@user:123`
  - Custom emojis: `<:name:id>` → `[custom:name]`
  - Role mentions: `<@&456>` → `@role:456`
  - Channel mentions: `<#789>` → `#channel:789`
- context: When reply/parent information is available
  - If parent content is available:
    - With parent author: `context: @<parent-author>: <one-line parent snippet>`
    - Without parent author: `context: <one-line parent snippet>`
  - If only a reply reference is available:
    - With parent author: `context: reply-to @<parent-author>`
    - Without author: `context: reply-to <parent_message_id>`
  - Mentions inside the context snippet are resolved the same as the passage
- summaries: Optional, single block joined by ` | ` separators when present
  - Possible parts:
    - Reactions: `reactions: <label> x<count>; <label> x<count>.`
      - Custom emoji labels render as `[custom:<name>]`, Unicode emoji use their symbol
    - Attachments: `attachments: <kind> x<count>, ...[. files: fname1, fname2, fname3]`
      - Kinds are aggregated by top-level media type (e.g., `image`, `video`)
      - Up to 3 filenames listed
    - Links: `links: <title-or-site-or-url>; <...>; <...>.`
      - Up to 3 items

Canonical shape
- Concatenated with single spaces between segments.
- Summary sub-parts are joined by ` | ` inside the single summaries block.
- Each summary sub-part ends with a period.

Notes
- token and text are already normalized (single spaces, sanitized Discord markup)
- ID is a Discord snowflake rendered as digits
- The meta, context, and summaries segments are optional; passage appears only
  when body text is non-empty.

Examples
- Basic with mention
  - Input content: `hey <@123> welcome!`
  - Output: `meta: @alice in #general passage: hey @alice welcome!`

- With reply context and reactions
  - Output:
    `meta: @alice in #general passage: thanks! context: @bob: you’re welcome `
    `reactions: 🔥 x3; ✅ x1.`

- With attachments and links
  - Output:
    `meta: @alice in #showcase passage: new demo attached `
    `attachments: image x2. files: demo.png, chart.png | links: Project Page; Docs.`
