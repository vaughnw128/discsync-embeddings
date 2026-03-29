FROM cgr.dev/chainguard/python:latest-dev AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/app/venv

COPY --chown=nonroot:nonroot pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev --no-install-project

COPY --chown=nonroot:nonroot . .
RUN uv sync --frozen --no-cache --no-dev --no-editable

RUN cp -r /app/venv/lib/python*/site-packages /app/site-packages

FROM cgr.dev/chainguard/python:latest AS runtime

WORKDIR /app

COPY --from=builder /app/site-packages /app/site-packages

ENV PYTHONPATH="/app/site-packages"

ENTRYPOINT ["python", "-m", "uvicorn", "discsync_embeddings.main:app", "--host", "0.0.0.0", "--port", "8080"]
