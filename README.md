# Discsync Embeddings

Discord message embeddings to work with the [discsync](github.com/vaughnw128/discsync) message syncing service. Simplifies message
objects into single lines, with repeatable and verifiable metadata to ensure correct syncing and idempotency.

## Overview

- Single-line formatted_text per message
- Author/channel names resolved by ID via DB with in-memory cache
- Batch embedding pipeline with timing logs and Qdrant vector store
- Idempotent upsert using stable point IDs and content fingerprints
- Supports SQLite (dev) and Postgres; FastAPI health endpoint for monitoring

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
  - `[replying to @<parent-author>]` when a reply reference is present
  - `[has reactions]` when reactions exist
  - `[shared attachments]` when attachments exist

### Examples
- Basic with mention
  - Input content: `hey <@123> welcome!`
  - Output: `@alice in #general: hey @user:123 welcome!`

- With reply and reactions
  - Output: `@alice in #general: thanks! [replying to @bob] [has reactions]`

- With attachments
  - Output: `@alice in #showcase: new demo attached [shared attachments]`

## Quick start (uv)

1. Set up prerequisites:
  - Qdrant running locally (default port 6333)
  - Python 3.12+ and uv installed

2. Install dependencies via UV:

```cmd
uv sync
```

3. Run in dev mode (SQLite) against local Qdrant:

```cmd
export QDRANT_HOSTNAME=http://localhost
export QDRANT_PORT=6333
uv run uvicorn discsync_embeddings.main:app --host 0.0.0.0 --port 8080 --reload
```

4. Using Postgres instead of SQLite:

```cmd
export DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
uv run uvicorn discsync_embeddings.main:app --host 0.0.0.0 --port 8080
```
