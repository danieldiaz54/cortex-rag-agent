# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
COPY ui/ ./ui/
COPY migrations/ ./migrations/

RUN python -m compileall -q src

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CORTEX_ENV=production \
    CORTEX_HOST=0.0.0.0 \
    CORTEX_PORT=8090 \
    CORTEX_SQLITE_PATH=/app/.data/cortex.sqlite3

RUN groupadd --system cortex && \
    useradd --system --gid cortex --create-home --home-dir /app cortex && \
    mkdir -p /app/.data && \
    chown -R cortex:cortex /app

WORKDIR /app
COPY --from=builder --chown=cortex:cortex /build/pyproject.toml /build/requirements.txt ./
COPY --from=builder --chown=cortex:cortex /build/src ./src
COPY --from=builder --chown=cortex:cortex /build/ui ./ui
COPY --from=builder --chown=cortex:cortex /build/migrations ./migrations

USER cortex

EXPOSE 8090

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/api/v1/health', timeout=3)" || exit 1

CMD ["python", "-m", "src.interfaces.http.server"]
