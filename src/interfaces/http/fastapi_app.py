"""Optional FastAPI ASGI wrapper mounting CortexApplicationService for container deployments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.application.services import CortexApplicationService
from src.interfaces.http.server import UI_INDEX_PATH, map_exception_to_http


def create_fastapi_app(service: CortexApplicationService | None = None) -> Any:
    """Create a FastAPI ASGI application wired to CortexApplicationService when FastAPI is installed."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is not installed. Use `python -m src.interfaces.http.server` for the "
            "zero-dependency standard library HTTP/SSE server."
        ) from exc

    app_service = service or CortexApplicationService(auto_seed=True)
    app = FastAPI(
        title="Cortex Hybrid RAG & Multi-Agent API",
        version="1.0.0",
        description="Hybrid BM25 + Vector + RRF Multi-Agent RAG with Citation Guardrails.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/ui", response_class=HTMLResponse)
    async def serve_ui() -> HTMLResponse:
        if Path(UI_INDEX_PATH).exists():
            return HTMLResponse(Path(UI_INDEX_PATH).read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Cortex Hybrid RAG Multi-Agent Workbench</h1>")

    @app.get("/api/v1/health")
    async def health() -> JSONResponse:
        return JSONResponse(app_service.get_health())

    @app.get("/api/v1/chunks")
    async def list_chunks() -> JSONResponse:
        chunks = app_service.list_chunks()
        return JSONResponse({"chunks": chunks, "total": len(chunks)})

    @app.post("/api/v1/documents/ingest")
    async def ingest_document(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            result = app_service.ingest_document(payload)
            return JSONResponse(result, status_code=200)
        except Exception as exc:
            code, err = map_exception_to_http(exc)
            return JSONResponse(err, status_code=code)

    @app.post("/api/v1/rag/query")
    async def rag_query(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            query = payload.get("query", "")
            top_k = payload.get("top_k", 4)
            if top_k is None:
                top_k = 4
            result = app_service.execute_query(query=query, top_k=top_k)
            return JSONResponse(result, status_code=200)
        except Exception as exc:
            code, err = map_exception_to_http(exc)
            return JSONResponse(err, status_code=code)

    @app.post("/api/v1/rag/stream")
    async def rag_stream(request: Request) -> Any:
        try:
            payload = await request.json()
            query = payload.get("query", "")
            top_k = payload.get("top_k", 4)
            if top_k is None:
                top_k = 4
            events = app_service.stream_query_events(query=query, top_k=top_k)
        except Exception as exc:
            code, err = map_exception_to_http(exc)
            return JSONResponse(err, status_code=code)

        def event_generator() -> Any:
            for event_name, event_data in events:
                yield f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/api/v1/evals/run")
    async def run_evals(request: Request) -> JSONResponse:
        try:
            body_bytes = await request.body()
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            enforce_gate = payload.get("enforce_gate", True)
            result = app_service.run_eval_suite(enforce_gate=enforce_gate)
            return JSONResponse(result, status_code=200)
        except Exception as exc:
            code, err = map_exception_to_http(exc)
            return JSONResponse(err, status_code=code)

    @app.get("/api/v1/evals/ledger")
    async def list_eval_ledger() -> JSONResponse:
        ledger = app_service.list_eval_ledger()
        return JSONResponse({"ledger": ledger, "runs": ledger, "total": len(ledger)})

    @app.get("/api/v1/audit/queries")
    async def list_query_audits() -> JSONResponse:
        audits = app_service.list_query_audits()
        return JSONResponse({"audits": audits, "queries": audits, "total": len(audits)})

    return app
