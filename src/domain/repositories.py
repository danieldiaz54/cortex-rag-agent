"""Domain repository port protocols for inverted persistence dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from src.domain.entities import DocumentChunk, EvalSuiteReport, RagResponse


@runtime_checkable
class ChunkRepositoryProtocol(Protocol):
    """Port interface for storing and retrieving DocumentChunk items."""

    def list_chunks(self) -> Sequence[DocumentChunk]:
        """Return all stored document chunks in deterministic order."""
        ...

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        """Insert or update a DocumentChunk in the persistence store."""
        ...


@runtime_checkable
class LedgerRepositoryProtocol(Protocol):
    """Port interface for append-only audit logging of RAG queries and evaluation reports."""

    def append_query_audit(self, response: RagResponse) -> str:
        """Append an immutable audit record for a RagResponse and return its hash or audit ID."""
        ...

    def append_eval_report(self, report: EvalSuiteReport) -> None:
        """Persist an evaluation suite report into the audit ledger."""
        ...

    def list_eval_reports(self) -> list[dict[str, Any]]:
        """List stored evaluation suite reports."""
        ...
