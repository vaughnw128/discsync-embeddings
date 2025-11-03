# syntax=docker/dockerfile:1.4
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY src /app/src
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml
COPY migrations /app/migrations
COPY alembic.ini /app/alembic.ini

WORKDIR /app

RUN uv sync --frozen --no-cache

# Run the application.
CMD ["/app/.venv/bin/uvicorn", "discsync_embeddings.main:app", "--host", "0.0.0.0", "--port", "8080"]
