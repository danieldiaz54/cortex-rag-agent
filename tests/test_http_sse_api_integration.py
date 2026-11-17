"""End-to-end HTTP REST and Server-Sent Events (SSE) API integration tests."""

from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.application.services import CortexApplicationService
from src.infrastructure.persistence.sqlite_store import SQLiteHybridStore
from src.interfaces.http.server import create_http_server


class TestHttpSseApiIntegration(unittest.TestCase):
    """Verify HTTP/1.1 REST endpoints, SSE streaming, semantic error codes, and zero stack traces."""

    def setUp(self) -> None:
        self.store = SQLiteHybridStore(db_path=":memory:")
        self.service = CortexApplicationService(store=self.store, auto_seed=True)
        self.server = create_http_server("127.0.0.1", 0, service=self.service)
        host, port = self.server.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.store.close()

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> tuple[int, dict]:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8")
                return resp.status, json.loads(body)
        except HTTPError as err:
            body = err.read().decode("utf-8")
            return err.code, json.loads(body)

    def test_ui_workbench_health_and_chunks_endpoints(self) -> None:
        # GET / -> 200 HTML containing Cortex RAG Agent UI
        req = Request(f"{self.base_url}/", method="GET")
        with urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            self.assertIn("Cortex", html)

        # GET /api/v1/health -> 200 OK
        status, health = self._request_json("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["corpus_chunks"], 8)

        # GET /api/v1/chunks -> 200 OK with 8 seeded chunks
        status, chunks_body = self._request_json("GET", "/api/v1/chunks")
        self.assertEqual(status, 200)
        self.assertEqual(chunks_body["total"], 8)
        self.assertEqual(len(chunks_body["chunks"]), 8)

    def test_document_ingest_valid_and_invalid_payloads(self) -> None:
        # Valid chunk ingestion
        status, body = self._request_json(
            "POST",
            "/api/v1/documents/ingest",
            {
                "chunk_id": "chunk-custom-009",
                "doc_id": "DOC-SEC-09",
                "title": "SPIFFE/SPIRE Mutual TLS Identity Attestation",
                "section": "security",
                "content": "SPIFFE workload identities rotate X.509 SVID certificates every 30 minutes.",
                "tags": ["spiffe", "mtls", "zero-trust"],
            },
        )
        self.assertIn(status, (200, 201))
        self.assertEqual(body["status"], "ingested")
        self.assertEqual(body["chunk_id"], "chunk-custom-009")
        self.assertEqual(body["corpus_size"], 9)

        # Invalid chunk ingestion -> 400 Bad Request
        status_err, err_body = self._request_json(
            "POST",
            "/api/v1/documents/ingest",
            {
                "chunk_id": "",
                "doc_id": "DOC-SEC-09",
                "title": "",
                "content": "",
            },
        )
        self.assertEqual(status_err, 400)
        self.assertIn("error", err_body)
        self.assertNotIn("Traceback", json.dumps(err_body))

    def test_rag_query_success_validation_and_empty_corpus_422(self) -> None:
        # 200 OK with verified citations and agent step traces
        status, body = self._request_json(
            "POST",
            "/api/v1/rag/query",
            {
                "query": "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers and how is it resolved?",
                "top_k": 4,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["intent"], "LEXICAL_EXACT")
        self.assertGreaterEqual(body["faithfulness_score"], 0.80)
        self.assertTrue(len(body["citations"]) >= 1)
        self.assertEqual(body["citations"][0]["chunk_id"], "chunk-err-409")
        self.assertEqual(len(body["traces"]), 4)
        self.assertTrue(body["audit_id"].startswith("aud-"))

        # 400 Bad Request on empty query
        status_400, err_400 = self._request_json(
            "POST",
            "/api/v1/rag/query",
            {"query": "   ", "top_k": 4},
        )
        self.assertEqual(status_400, 400)
        self.assertEqual(err_400["error"]["code"], "InvalidQueryError")
        self.assertNotIn("Traceback", json.dumps(err_400))

        # 422 Unprocessable Entity when querying an empty store (zero stack traces)
        empty_store = SQLiteHybridStore(db_path=":memory:")
        empty_service = CortexApplicationService(store=empty_store, auto_seed=False)
        empty_server = create_http_server("127.0.0.1", 0, service=empty_service)
        e_host, e_port = empty_server.server_address[:2]
        e_thread = threading.Thread(target=empty_server.serve_forever, daemon=True)
        e_thread.start()
        try:
            req = Request(
                f"http://{e_host}:{e_port}/api/v1/rag/query",
                data=json.dumps({"query": "ERR-409-OPTIMISTIC-LOCK", "top_k": 3}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as ctx:
                urlopen(req, timeout=10)
            self.assertEqual(ctx.exception.code, 422)
            raw_err = ctx.exception.read().decode("utf-8")
            parsed_err = json.loads(raw_err)
            self.assertEqual(parsed_err["error"]["code"], "EmptyRetrievalError")
            self.assertNotIn("Traceback", raw_err)
            self.assertNotIn("File \"", raw_err)
        finally:
            empty_server.shutdown()
            empty_server.server_close()
            empty_store.close()

    def test_sse_streaming_endpoint_emits_trace_token_citations_and_done(self) -> None:
        stream_url = (
            f"{self.base_url}/api/v1/rag/stream"
            "?q=ERR-409-OPTIMISTIC-LOCK&query=ERR-409-OPTIMISTIC-LOCK&top_k=4"
        )
        req = Request(stream_url, headers={"Accept": "text/event-stream"}, method="GET")
        with urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
            stream_text = resp.read().decode("utf-8")

        self.assertIn("event: trace", stream_text)
        self.assertIn("event: token", stream_text)
        self.assertIn("event: citations", stream_text)
        self.assertIn("event: done", stream_text)
        self.assertIn("chunk-err-409", stream_text)

    def test_evals_run_and_append_only_ledger_endpoints(self) -> None:
        # Execute a RAG query first so query audit log has an entry
        self._request_json(
            "POST",
            "/api/v1/rag/query",
            {"query": "Why does AGA-8 raise DivergenceException at 230 bar?", "top_k": 4},
        )

        # POST /api/v1/evals/run -> 200 OK
        status_eval, eval_report = self._request_json(
            "POST",
            "/api/v1/evals/run",
            {"enforce_gate": True},
        )
        self.assertEqual(status_eval, 200)
        self.assertTrue(eval_report["passed_gate"])
        self.assertEqual(eval_report["total_cases"], 5)
        self.assertEqual(eval_report["passed_cases"], 5)

        # GET /api/v1/evals/ledger -> 200 OK
        status_ledger, ledger_body = self._request_json("GET", "/api/v1/evals/ledger")
        self.assertEqual(status_ledger, 200)
        self.assertEqual(ledger_body["total"], 1)
        self.assertEqual(ledger_body["ledger"][0]["run_id"], eval_report["run_id"])

        # GET /api/v1/audit/queries -> 200 OK
        status_audits, audits_body = self._request_json("GET", "/api/v1/audit/queries")
        self.assertEqual(status_audits, 200)
        self.assertGreaterEqual(audits_body["total"], 1)
        self.assertTrue(audits_body["audits"][0]["audit_id"].startswith("aud-"))


if __name__ == "__main__":
    unittest.main()
