# Builder
FROM cgr.dev/chainguard/python:latest-dev AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY --chown=nonroot:nonroot pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev --no-install-project

COPY --chown=nonroot:nonroot . .
RUN uv pip install --no-editable .

# Runtime
FROM cgr.dev/chainguard/python:latest-dev AS runtime

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "uvicorn", "discsync_embeddings.main:app", "--host", "0.0.0.0", "--port", "8080"]
