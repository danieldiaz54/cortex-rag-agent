"""Pure domain exceptions for the Cortex Hybrid RAG Multi-Agent core."""

from __future__ import annotations


class DomainError(Exception):
    """Base exception for all domain-level invariant and rule violations."""

    def __init__(self, message: str, *, code: str = "DOMAIN_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class InvalidQueryError(DomainError):
    """Raised when a user query is empty, whitespace-only, or exceeds maximum length."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="INVALID_QUERY")


class InvalidDocumentError(DomainError):
    """Raised when a DocumentChunk has invalid or empty required fields."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="INVALID_DOCUMENT")


class EmptyRetrievalError(DomainError):
    """Raised when the document corpus is empty or no retrieval candidates match."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="EMPTY_RETRIEVAL")


class InvalidCitationError(DomainError):
    """Raised when a synthesized answer references a [cite:id] not in retrieved chunks."""

    def __init__(self, message: str, *, invalid_ids: tuple[str, ...] = ()) -> None:
        super().__init__(message, code="INVALID_CITATION")
        self.invalid_ids = invalid_ids


class UngroundedClaimError(DomainError):
    """Raised when an answer lacks citations or fails the faithfulness grounding threshold."""

    def __init__(self, message: str, *, faithfulness_score: float = 0.0) -> None:
        super().__init__(message, code="UNGROUNDED_CLAIM")
        self.faithfulness_score = faithfulness_score


class EvalRegressionError(DomainError):
    """Raised when an evaluation suite run falls below minimum quality gate thresholds."""

    def __init__(
        self,
        message: str,
        *,
        avg_faithfulness: float = 0.0,
        avg_context_recall: float = 0.0,
        avg_citation_precision: float = 0.0,
    ) -> None:
        super().__init__(message, code="EVAL_REGRESSION")
        self.avg_faithfulness = avg_faithfulness
        self.avg_context_recall = avg_context_recall
        self.avg_citation_precision = avg_citation_precision
