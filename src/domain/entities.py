"""Immutable domain entities and value objects for Cortex Hybrid RAG."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.domain.exceptions import InvalidDocumentError, InvalidQueryError


class QueryIntent(str, Enum):
    """Classification of query retrieval intent for dynamic hybrid weighting."""

    LEXICAL_EXACT = "LEXICAL_EXACT"
    SEMANTIC_CONCEPTUAL = "SEMANTIC_CONCEPTUAL"
    HYBRID_ANALYTICAL = "HYBRID_ANALYTICAL"


@dataclass(frozen=True)
class DocumentChunk:
    """Immutable, self-validating knowledge base document chunk."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    content: str
    tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise InvalidDocumentError("DocumentChunk.chunk_id must be a non-empty string.")
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise InvalidDocumentError("DocumentChunk.doc_id must be a non-empty string.")
        if not isinstance(self.title, str) or not self.title.strip():
            raise InvalidDocumentError("DocumentChunk.title must be a non-empty string.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise InvalidDocumentError("DocumentChunk.content must be a non-empty string.")
        if not isinstance(self.section, str) or not self.section.strip():
            object.__setattr__(self, "section", "general")
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(str(t) for t in self.tags))
        if not isinstance(self.embedding, tuple):
            object.__setattr__(self, "embedding", tuple(float(x) for x in self.embedding))


@dataclass(frozen=True)
class RetrievalCandidate:
    """Scored candidate chunk produced by Hybrid BM25 + Dense Vector + RRF fusion."""

    chunk: DocumentChunk
    bm25_score: float
    vector_score: float
    bm25_rank: int
    vector_rank: int
    rrf_score: float


@dataclass(frozen=True)
class Citation:
    """Verified inline citation linking a synthesized claim to a retrieved chunk."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    excerpt: str
    rrf_score: float


@dataclass(frozen=True)
class AgentStepTrace:
    """Execution telemetry trace for a single agent step in the RAG pipeline."""

    agent_name: str
    action: str
    detail: str
    duration_ms: float


@dataclass(frozen=True)
class RagResponse:
    """Complete verified output from the Multi-Agent Hybrid RAG pipeline."""

    query: str
    intent: QueryIntent
    answer: str
    citations: tuple[Citation, ...]
    candidates: tuple[RetrievalCandidate, ...]
    traces: tuple[AgentStepTrace, ...]
    faithfulness_score: float
    total_latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.citations, tuple):
            object.__setattr__(self, "citations", tuple(self.citations))
        if not isinstance(self.candidates, tuple):
            object.__setattr__(self, "candidates", tuple(self.candidates))
        if not isinstance(self.traces, tuple):
            object.__setattr__(self, "traces", tuple(self.traces))


@dataclass(frozen=True)
class EvalCase:
    """Single golden evaluation test case for the RAG evaluation harness."""

    case_id: str
    query: str
    expected_chunk_ids: tuple[str, ...]
    expected_keywords: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.case_id.strip():
            raise InvalidQueryError("EvalCase.case_id must be a non-empty string.")
        if not self.query or not self.query.strip():
            raise InvalidQueryError("EvalCase.query must be a non-empty string.")
        if not isinstance(self.expected_chunk_ids, tuple):
            object.__setattr__(self, "expected_chunk_ids", tuple(self.expected_chunk_ids))
        if not isinstance(self.expected_keywords, tuple):
            object.__setattr__(self, "expected_keywords", tuple(self.expected_keywords))


@dataclass(frozen=True)
class EvalCaseResult:
    """Evaluation metrics for an individual golden test case."""

    case_id: str
    query: str
    intent: QueryIntent
    faithfulness: float
    citation_precision: float
    context_recall: float
    keyword_hit_rate: float
    latency_ms: float
    passed: bool


@dataclass(frozen=True)
class EvalSuiteReport:
    """Aggregated evaluation report across a golden dataset suite."""

    run_id: str
    total_cases: int
    passed_cases: int
    avg_faithfulness: float
    avg_citation_precision: float
    avg_context_recall: float
    avg_latency_ms: float
    results: tuple[EvalCaseResult, ...]
    passed_gate: bool

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple):
            object.__setattr__(self, "results", tuple(self.results))
