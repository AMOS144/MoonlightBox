FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra linux-ml

COPY backend ./backend
RUN chmod +x /app/backend/docker-entrypoint.sh

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "moonlightbox.api:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
