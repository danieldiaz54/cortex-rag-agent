"""Zero-dependency HTTP/1.1 REST + SSE Server built on Python 3.12 standard library."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.application.services import CortexApplicationService
from src.domain.exceptions import (
    DomainError,
    EmptyRetrievalError,
    EvalRegressionError,
    InvalidCitationError,
    InvalidDocumentError,
    InvalidQueryError,
    UngroundedClaimError,
)

MAX_JSON_BODY_BYTES = 64 * 1024  # 64 KB strict payload ceiling
REPO_ROOT = Path(__file__).resolve().parents[3]
UI_INDEX_PATH = REPO_ROOT / "ui" / "index.html"


class PayloadTooLargeError(ValueError):
    """Raised when an HTTP request body exceeds MAX_JSON_BODY_BYTES (64 KB)."""


class MalformedJsonError(ValueError):
    """Raised when an HTTP request body is not valid JSON or not a JSON object."""


def map_exception_to_http(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Map domain and protocol exceptions to semantic HTTP status codes without stack traces."""
    if isinstance(
        exc,
        (
            InvalidQueryError,
            InvalidDocumentError,
            MalformedJsonError,
            PayloadTooLargeError,
        ),
    ):
        return (
            400,
            {
                "error": {
                    "code": exc.__class__.__name__,
                    "message": str(exc),
                }
            },
        )

    if isinstance(
        exc,
        (
            EmptyRetrievalError,
            InvalidCitationError,
            UngroundedClaimError,
            EvalRegressionError,
            DomainError,
        ),
    ):
        return (
            422,
            {
                "error": {
                    "code": exc.__class__.__name__,
                    "message": str(exc),
                }
            },
        )

    return (
        500,
        {
            "error": {
                "code": "InternalServerError",
                "message": "An unexpected internal server error occurred.",
            }
        },
    )


class CortexThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer holding a reference to CortexApplicationService."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        service: CortexApplicationService,
    ) -> None:
        self.service = service
        super().__init__(server_address, RequestHandlerClass)


class CortexHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 REST and SSE request handler for Cortex RAG API v1."""

    protocol_version = "HTTP/1.1"

    @property
    def service(self) -> CortexApplicationService:
        server = self.server
        if isinstance(server, CortexThreadingHTTPServer):
            return server.service
        raise RuntimeError("Server is not configured with CortexApplicationService.")

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy stderr access logs during automated test runs unless enabled."""
        if os.environ.get("CORTEX_ACCESS_LOG", "0") == "1":
            super().log_message(format, *args)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, X-Request-ID")
        self.send_header("X-Content-Type-Options", "nosniff")

    def _send_json(self, status_code: int, payload: dict[str, Any] | list[Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def _send_html(self, status_code: int, html_content: str) -> None:
        raw = html_content.encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def _send_not_found(self, path: str) -> None:
        self._send_json(
            404,
            {
                "error": {
                    "code": "NotFoundError",
                    "message": f"Route '{path}' was not found.",
                }
            },
        )

    def _read_json_body(self, *, allow_empty: bool = False) -> dict[str, Any]:
        content_length_raw = self.headers.get("Content-Length")
        if content_length_raw is None:
            if allow_empty:
                return {}
            raise MalformedJsonError("Missing Content-Length header or JSON request body.")

        try:
            content_length = int(content_length_raw)
        except ValueError as exc:
            raise MalformedJsonError("Invalid Content-Length header.") from exc

        if content_length < 0:
            raise MalformedJsonError("Negative Content-Length header is invalid.")

        if content_length > MAX_JSON_BODY_BYTES:
            raise PayloadTooLargeError(
                f"Request body of {content_length} bytes exceeds maximum limit of {MAX_JSON_BODY_BYTES} bytes (64 KB)."
            )

        if content_length == 0:
            if allow_empty:
                return {}
            raise MalformedJsonError("Request body cannot be empty.")

        raw_bytes = self.rfile.read(content_length)
        if len(raw_bytes) > MAX_JSON_BODY_BYTES:
            raise PayloadTooLargeError("Request body exceeds 64 KB limit.")

        try:
            decoded = raw_bytes.decode("utf-8")
            parsed = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedJsonError("Request body is not valid UTF-8 JSON.") from exc

        if not isinstance(parsed, dict):
            raise MalformedJsonError("JSON request payload must be an object.")

        return parsed

    def _handle_sse_stream(self, query: str, top_k: int) -> None:
        # Execute and validate before sending 200 SSE headers so domain errors map to 400/422 JSON
        events = self.service.stream_query_events(query=query, top_k=top_k)

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        for event_name, event_data in events:
            frame = f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/") or "/"

        try:
            if path in ("/", "/ui", "/index.html"):
                if UI_INDEX_PATH.exists():
                    html_text = UI_INDEX_PATH.read_text(encoding="utf-8")
                else:
                    html_text = (
                        "<!DOCTYPE html><html><head><title>Cortex RAG Workbench</title></head>"
                        "<body><h1>Cortex Hybrid RAG Multi-Agent Workbench</h1></body></html>"
                    )
                self._send_html(200, html_text)
                return

            if path == "/api/v1/health":
                self._send_json(200, self.service.get_health())
                return

            if path == "/api/v1/chunks":
                chunks = self.service.list_chunks()
                self._send_json(200, {"chunks": chunks, "total": len(chunks)})
                return

            if path == "/api/v1/evals/ledger":
                ledger = self.service.list_eval_ledger()
                self._send_json(
                    200,
                    {
                        "ledger": ledger,
                        "runs": ledger,
                        "total": len(ledger),
                    },
                )
                return

            if path == "/api/v1/audit/queries":
                audits = self.service.list_query_audits()
                self._send_json(
                    200,
                    {
                        "audits": audits,
                        "queries": audits,
                        "total": len(audits),
                    },
                )
                return

            if path == "/api/v1/rag/stream":
                qs = parse_qs(parsed_url.query, keep_blank_values=True)
                query = qs.get("query", [""])[0]
                top_k_raw = qs.get("top_k", ["4"])[0]
                try:
                    top_k = int(top_k_raw)
                except ValueError as exc:
                    raise InvalidQueryError("Parameter 'top_k' must be an integer.") from exc
                self._handle_sse_stream(query=query, top_k=top_k)
                return

            self._send_not_found(path)
        except Exception as exc:
            status_code, err_body = map_exception_to_http(exc)
            self._send_json(status_code, err_body)

    def do_POST(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/") or "/"

        try:
            if path == "/api/v1/documents/ingest":
                payload = self._read_json_body(allow_empty=False)
                result = self.service.ingest_document(payload)
                self._send_json(200, result)
                return

            if path == "/api/v1/rag/query":
                payload = self._read_json_body(allow_empty=False)
                query = payload.get("query", "")
                top_k = payload.get("top_k", 4)
                if top_k is None:
                    top_k = 4
                result = self.service.execute_query(query=query, top_k=top_k)
                self._send_json(200, result)
                return

            if path == "/api/v1/rag/stream":
                payload = self._read_json_body(allow_empty=False)
                query = payload.get("query", "")
                top_k = payload.get("top_k", 4)
                if top_k is None:
                    top_k = 4
                self._handle_sse_stream(query=query, top_k=top_k)
                return

            if path == "/api/v1/evals/run":
                payload = self._read_json_body(allow_empty=True)
                enforce_gate = payload.get("enforce_gate", True)
                top_k = payload.get("top_k")
                result = self.service.run_eval_suite(
                    enforce_gate=enforce_gate,
                    top_k=top_k,
                )
                self._send_json(200, result)
                return

            self._send_not_found(path)
        except Exception as exc:
            status_code, err_body = map_exception_to_http(exc)
            self._send_json(status_code, err_body)


def create_http_server(
    host: str = "127.0.0.1",
    port: int = 8090,
    service: CortexApplicationService | None = None,
) -> ThreadingHTTPServer:
    """Create and configure a ThreadingHTTPServer bound to `(host, port)`."""
    if service is None:
        sqlite_path = os.environ.get("CORTEX_SQLITE_PATH", ":memory:") or ":memory:"
        service = CortexApplicationService(db_path=sqlite_path, auto_seed=True)

    return CortexThreadingHTTPServer(
        (host, port),
        CortexHTTPRequestHandler,
        service=service,
    )


def main() -> None:
    """Start the Cortex HTTP + SSE server from environment configuration."""
    host = os.environ.get("CORTEX_HOST", "0.0.0.0")
    port = int(os.environ.get("CORTEX_PORT", "8090"))
    sqlite_path = os.environ.get("CORTEX_SQLITE_PATH", "./.data/cortex_dev.sqlite3")
    service = CortexApplicationService(db_path=sqlite_path, auto_seed=True)
    server = create_http_server(host=host, port=port, service=service)
    bound_host, bound_port = server.server_address[:2]
    print(f"Cortex Hybrid RAG Server listening on http://{bound_host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
