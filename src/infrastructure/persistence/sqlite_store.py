"""SQLite Hybrid Vector/Lexical Store and Append-Only Audit/Evaluation Ledgers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
import uuid

from src.domain.entities import DocumentChunk, EvalSuiteReport, RagResponse
from src.domain.retrieval import compute_deterministic_embedding


class AppendOnlyLedgerError(RuntimeError):
    """Raised when an UPDATE or DELETE is attempted against an append-only ledger table."""


class SQLiteHybridStore:
    """Local/Dev Hybrid Corpus Store with Database-Enforced Append-Only Ledgers."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        if db_path != ":memory:":
            parent = Path(db_path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create hybrid corpus table, append-only ledger tables, and immutability triggers."""
        ddl_script = """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id       TEXT PRIMARY KEY,
            doc_id         TEXT NOT NULL,
            title          TEXT NOT NULL,
            section        TEXT NOT NULL,
            content        TEXT NOT NULL,
            tags_json      TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS query_audit_log (
            audit_id           TEXT PRIMARY KEY,
            query              TEXT NOT NULL,
            intent             TEXT NOT NULL,
            answer             TEXT NOT NULL,
            citations_json     TEXT NOT NULL,
            faithfulness_score REAL NOT NULL,
            latency_ms         REAL NOT NULL,
            created_at         TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_runs_ledger (
            run_id                 TEXT PRIMARY KEY,
            total_cases            INTEGER NOT NULL,
            passed_cases           INTEGER NOT NULL,
            avg_faithfulness       REAL NOT NULL,
            avg_citation_precision REAL NOT NULL,
            avg_context_recall     REAL NOT NULL,
            avg_latency_ms         REAL NOT NULL,
            passed_gate            INTEGER NOT NULL,
            results_json           TEXT NOT NULL,
            created_at             TEXT NOT NULL
        );

        CREATE TRIGGER IF NOT EXISTS trg_query_audit_no_update
        BEFORE UPDATE ON query_audit_log
        BEGIN
            SELECT RAISE(ABORT, 'APPEND_ONLY_LEDGER_VIOLATION: Ledger records are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_query_audit_no_delete
        BEFORE DELETE ON query_audit_log
        BEGIN
            SELECT RAISE(ABORT, 'APPEND_ONLY_LEDGER_VIOLATION: Ledger records cannot be deleted');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_eval_ledger_no_update
        BEFORE UPDATE ON eval_runs_ledger
        BEGIN
            SELECT RAISE(ABORT, 'APPEND_ONLY_LEDGER_VIOLATION: Ledger records are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_eval_ledger_no_delete
        BEFORE DELETE ON eval_runs_ledger
        BEGIN
            SELECT RAISE(ABORT, 'APPEND_ONLY_LEDGER_VIOLATION: Ledger records cannot be deleted');
        END;
        """
        with self._lock:
            self._conn.executescript(ddl_script)
            self._conn.commit()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Persist a DocumentChunk, computing its deterministic 64-d embedding if absent."""
        if not chunk.embedding:
            rich_text = f"{chunk.title} {chunk.section} {' '.join(chunk.tags)} {chunk.content}"
            embedding = compute_deterministic_embedding(rich_text)
            enriched_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                section=chunk.section,
                content=chunk.content,
                tags=chunk.tags,
                embedding=embedding,
            )
        else:
            enriched_chunk = chunk

        tags_json = json.dumps(list(enriched_chunk.tags))
        embedding_json = json.dumps(list(enriched_chunk.embedding))
        now_iso = self._utc_now_iso()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO document_chunks (
                    chunk_id, doc_id, title, section, content, tags_json, embedding_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    doc_id         = excluded.doc_id,
                    title          = excluded.title,
                    section        = excluded.section,
                    content        = excluded.content,
                    tags_json      = excluded.tags_json,
                    embedding_json = excluded.embedding_json,
                    created_at     = excluded.created_at
                """,
                (
                    enriched_chunk.chunk_id,
                    enriched_chunk.doc_id,
                    enriched_chunk.title,
                    enriched_chunk.section,
                    enriched_chunk.content,
                    tags_json,
                    embedding_json,
                    now_iso,
                ),
            )
            self._conn.commit()

        return enriched_chunk

    def list_chunks(self) -> list[DocumentChunk]:
        """Return all stored DocumentChunk entities ordered deterministically by chunk_id."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT chunk_id, doc_id, title, section, content, tags_json, embedding_json
                FROM document_chunks
                ORDER BY chunk_id ASC
                """
            ).fetchall()

        chunks: list[DocumentChunk] = []
        for row in rows:
            tags_raw = json.loads(row["tags_json"])
            emb_raw = json.loads(row["embedding_json"])
            chunks.append(
                DocumentChunk(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    title=str(row["title"]),
                    section=str(row["section"]),
                    content=str(row["content"]),
                    tags=tuple(str(t) for t in tags_raw),
                    embedding=tuple(float(v) for v in emb_raw),
                )
            )
        return chunks

    def append_query_audit(self, response: RagResponse) -> str:
        """Append an immutable RAG execution audit record and return its audit_id."""
        audit_id = f"aud-{uuid.uuid4().hex[:12]}"
        citations_payload = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "section": c.section,
                "excerpt": c.excerpt,
                "rrf_score": c.rrf_score,
            }
            for c in response.citations
        ]
        citations_json = json.dumps(citations_payload)
        intent_val = response.intent.value if hasattr(response.intent, "value") else str(response.intent)
        now_iso = self._utc_now_iso()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO query_audit_log (
                    audit_id, query, intent, answer, citations_json,
                    faithfulness_score, latency_ms, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    response.query,
                    intent_val,
                    response.answer,
                    citations_json,
                    float(response.faithfulness_score),
                    float(response.total_latency_ms),
                    now_iso,
                ),
            )
            self._conn.commit()

        return audit_id

    def list_query_audits(self, limit: int = 50) -> list[dict[str, Any]]:
        """List immutable query audit log entries ordered newest-first."""
        safe_limit = max(1, int(limit))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT audit_id, query, intent, answer, citations_json,
                       faithfulness_score, latency_ms, created_at
                FROM query_audit_log
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            citations_raw = json.loads(row["citations_json"])
            results.append(
                {
                    "audit_id": str(row["audit_id"]),
                    "query": str(row["query"]),
                    "intent": str(row["intent"]),
                    "answer": str(row["answer"]),
                    "citations": citations_raw,
                    "citations_json": str(row["citations_json"]),
                    "faithfulness_score": float(row["faithfulness_score"]),
                    "latency_ms": float(row["latency_ms"]),
                    "created_at": str(row["created_at"]),
                }
            )
        return results

    def append_eval_report(self, report: EvalSuiteReport) -> None:
        """Append an immutable evaluation harness run report to `eval_runs_ledger`."""
        serialized_results: list[dict[str, Any]] = []
        for item in report.results:
            item_dict = asdict(item)
            if hasattr(item.intent, "value"):
                item_dict["intent"] = item.intent.value
            serialized_results.append(item_dict)

        results_json = json.dumps(serialized_results)
        now_iso = self._utc_now_iso()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO eval_runs_ledger (
                    run_id, total_cases, passed_cases, avg_faithfulness,
                    avg_citation_precision, avg_context_recall, avg_latency_ms,
                    passed_gate, results_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    int(report.total_cases),
                    int(report.passed_cases),
                    float(report.avg_faithfulness),
                    float(report.avg_citation_precision),
                    float(report.avg_context_recall),
                    float(report.avg_latency_ms),
                    1 if report.passed_gate else 0,
                    results_json,
                    now_iso,
                ),
            )
            self._conn.commit()

    def list_eval_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        """List immutable evaluation ledger entries ordered newest-first."""
        safe_limit = max(1, int(limit))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT run_id, total_cases, passed_cases, avg_faithfulness,
                       avg_citation_precision, avg_context_recall, avg_latency_ms,
                       passed_gate, results_json, created_at
                FROM eval_runs_ledger
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        reports: list[dict[str, Any]] = []
        for row in rows:
            results_raw = json.loads(row["results_json"])
            reports.append(
                {
                    "run_id": str(row["run_id"]),
                    "total_cases": int(row["total_cases"]),
                    "passed_cases": int(row["passed_cases"]),
                    "avg_faithfulness": float(row["avg_faithfulness"]),
                    "avg_citation_precision": float(row["avg_citation_precision"]),
                    "avg_context_recall": float(row["avg_context_recall"]),
                    "avg_latency_ms": float(row["avg_latency_ms"]),
                    "passed_gate": bool(row["passed_gate"]),
                    "results": results_raw,
                    "results_json": str(row["results_json"]),
                    "created_at": str(row["created_at"]),
                }
            )
        return reports

    def execute_raw_mutation_for_test(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Execute a raw SQL statement and translate ledger trigger aborts to AppendOnlyLedgerError."""
        try:
            with self._lock:
                self._conn.execute(sql, params)
                self._conn.commit()
        except (sqlite3.IntegrityError, sqlite3.DatabaseError) as exc:
            if "APPEND_ONLY_LEDGER_VIOLATION" in str(exc):
                raise AppendOnlyLedgerError(str(exc)) from exc
            raise

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
