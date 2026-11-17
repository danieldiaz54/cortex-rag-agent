"""Pure Domain Layer for Cortex Hybrid RAG Multi-Agent Core."""

from src.domain.agents import (
    CitationGuardrailAgent,
    GuardrailValidationResult,
    HybridRetrieverAgent,
    MultiAgentRagPipeline,
    RouteDecision,
    RouterAgent,
    SynthesizerAgent,
)
from src.domain.entities import (
    AgentStepTrace,
    Citation,
    DocumentChunk,
    EvalCase,
    EvalCaseResult,
    EvalSuiteReport,
    QueryIntent,
    RagResponse,
    RetrievalCandidate,
)
from src.domain.eval_harness import RagEvalHarness
from src.domain.exceptions import (
    DomainError,
    EmptyRetrievalError,
    EvalRegressionError,
    InvalidCitationError,
    InvalidDocumentError,
    InvalidQueryError,
    UngroundedClaimError,
)
from src.domain.repositories import (
    ChunkRepositoryProtocol,
    LedgerRepositoryProtocol,
)
from src.domain.retrieval import (
    BM25Scorer,
    HybridSearchEngine,
    compute_deterministic_embedding,
    cosine_similarity,
    tokenize,
)

__all__ = [
    "AgentStepTrace",
    "BM25Scorer",
    "ChunkRepositoryProtocol",
    "Citation",
    "CitationGuardrailAgent",
    "DocumentChunk",
    "DomainError",
    "EmptyRetrievalError",
    "EvalCase",
    "EvalCaseResult",
    "EvalRegressionError",
    "EvalSuiteReport",
    "GuardrailValidationResult",
    "HybridRetrieverAgent",
    "HybridSearchEngine",
    "InvalidCitationError",
    "InvalidDocumentError",
    "InvalidQueryError",
    "LedgerRepositoryProtocol",
    "MultiAgentRagPipeline",
    "QueryIntent",
    "RagEvalHarness",
    "RagResponse",
    "RetrievalCandidate",
    "RouteDecision",
    "RouterAgent",
    "SynthesizerAgent",
    "UngroundedClaimError",
    "compute_deterministic_embedding",
    "cosine_similarity",
    "tokenize",
]
