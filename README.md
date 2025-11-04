# Discsync Embeddings

Discord message embeddings to work with the [discsync](github.com/vaughnw128/discsync) message syncing service

## Overview

### Simple flow
- Background worker fetches pending messages in batches (EMBED_BATCH,
  default 512) from the discsync DB.
- For each message, it builds a single-line `formatted_text`:
  - `@<author> in #<channel>: <normalized-body> [replying to @<parent>] [has reactions] [shared attachments]`
  - Resolves author/channel names from the database by ID and caches them.
  - Normalizes Discord markup in the body (mentions, emojis, roles, channels),
    and collapses whitespace to single spaces.
- Embeds the batch using a HuggingFace embedding model in compute sub-batches
  (EMBED_COMPUTE_BATCH, default 128) for throughput and memory efficiency.
- Upserts vectors into Qdrant with stable UUIDv5 point IDs (derived from
  message id).
- Records the embedding result in `message_embeddings` (status, formatted_text,
  model info, token counts, Qdrant collection/IDs, timestamp).
- Repeats continuously; logs one concise timing line per batch with tokenize
  time, embed time, upsert time, and messages-per-second.

Handles retries, rate limits, and transient errors gracefully, recovering
automatically and restarting from checkpoints in the database.

# Technical Details

## Embedded text (formatted_text) shape

The service embeds a single, human-readable line per message called
`formatted_text`. It uses a simplified, consistent structure:

- Prefix: `@<author> in #<channel>:`
  - Author and channel names are resolved from the database by ID. If a name
    is unknown, the numeric ID is used (e.g., `@123`, `#123`).
- Body: normalized message content (single line)
  - Mentions: `<@123>` → `@user:123` when the body mention cannot be resolved
  - Custom emojis: `<:name:id>` → `[custom:name]`
  - Role mentions: `<@&456>` → `@role:456`
  - Channel mentions: `<#789>` → `#channel:789`
  - Whitespace collapsed to single spaces, CR/LF removed
- Optional markers appended as space-separated tokens:
  - `[`replying to @<parent-author>`]` when a reply reference is present
  - `[`has reactions`]` when reactions exist
  - `[`shared attachments`]` when attachments exist

Notes
- The entire `formatted_text` is a single line, concatenated with single
  spaces.
- IDs are Discord snowflakes rendered as digits when names are unavailable.
- Only the markers that apply are included.

### Examples
- Basic with mention
  - Input content: `hey <@123> welcome!`
  - Output: `@alice in #general: hey @user welcome!`

- With reply and reactions
  - Output: `@alice in #general: thanks! [replying to @bob] [has reactions]`

- With attachments
  - Output: `@alice in #showcase: new demo attached [shared attachments]`