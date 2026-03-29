# Builder
FROM cgr.dev/chainguard/python:latest-dev AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-cache --no-dev

# Runtime
FROM cgr.dev/chainguard/python:latest AS runtime

WORKDIR /app

COPY --from=builder /app /app

ENTRYPOINT ["/app/.venv/bin/uvicorn", "discsync_embeddings.main:app", "--host", "0.0.0.0", "--port", "8080"]
