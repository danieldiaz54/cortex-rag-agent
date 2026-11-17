"""Golden Dataset Evaluation Harness for Multi-Agent Hybrid RAG quality gates."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import time

from src.domain.agents import MultiAgentRagPipeline
from src.domain.entities import (
    DocumentChunk,
    EvalCase,
    EvalCaseResult,
    EvalSuiteReport,
)
from src.domain.exceptions import (
    EmptyRetrievalError,
    EvalRegressionError,
    InvalidQueryError,
)
from src.domain.retrieval import _stem_token, tokenize


class RagEvalHarness:
    """Evaluates golden test cases for faithfulness, citation precision, context recall, and latency."""

    def __init__(
        self,
        pipeline: MultiAgentRagPipeline | None = None,
        *,
        chunks: Sequence[DocumentChunk] | None = None,
        min_faithfulness: float = 0.70,
        min_citation_precision: float = 0.70,
        min_context_recall: float = 0.75,
        top_k: int = 5,
    ) -> None:
        self.pipeline = pipeline or MultiAgentRagPipeline()
        self._default_chunks: tuple[DocumentChunk, ...] = tuple(chunks) if chunks is not None else ()
        self.min_faithfulness = min_faithfulness
        self.min_citation_precision = min_citation_precision
        self.min_context_recall = min_context_recall
        self.top_k = top_k

    @staticmethod
    def _compute_keyword_hit_rate(answer: str, expected_keywords: Sequence[str]) -> float:
        if not expected_keywords:
            return 1.0
        answer_lower = answer.lower()
        answer_tokens = set(tokenize(answer_lower))
        answer_stems = {_stem_token(t) for t in answer_tokens}

        hits = 0
        for kw in expected_keywords:
            kw_clean = kw.strip().lower()
            if not kw_clean:
                hits += 1
                continue
            if kw_clean in answer_lower:
                hits += 1
                continue
            kw_tokens = tokenize(kw_clean)
            if kw_tokens and all(
                (t in answer_tokens or _stem_token(t) in answer_stems) for t in kw_tokens
            ):
                hits += 1

        return round(hits / len(expected_keywords), 6)

    def evaluate_case(
        self,
        case: EvalCase,
        chunks: Sequence[DocumentChunk] | None = None,
        *,
        top_k: int | None = None,
    ) -> EvalCaseResult:
        """Run a single EvalCase through the pipeline and compute quality metrics."""
        target_chunks = tuple(chunks) if chunks is not None else self._default_chunks
        if not target_chunks:
            raise EmptyRetrievalError("Cannot run evaluation case against an empty document corpus.")

        k_val = top_k if top_k is not None else self.top_k
        t0 = time.perf_counter()
        response = self.pipeline.run(case.query, target_chunks, top_k=k_val)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        retrieved_ids = {cand.chunk.chunk_id for cand in response.candidates}
        cited_ids = {cit.chunk_id for cit in response.citations}
        expected_ids = set(case.expected_chunk_ids)

        if not expected_ids:
            context_recall = 1.0
            citation_precision = 1.0 if cited_ids else 0.0
        else:
            context_recall = round(len(expected_ids & retrieved_ids) / len(expected_ids), 6)
            if not cited_ids:
                citation_precision = 0.0
            else:
                citation_precision = round(len(cited_ids & expected_ids) / len(cited_ids), 6)

        keyword_hit_rate = self._compute_keyword_hit_rate(response.answer, case.expected_keywords)
        faithfulness = round(response.faithfulness_score, 6)

        case_passed = (
            faithfulness >= self.min_faithfulness
            and citation_precision >= self.min_citation_precision
            and context_recall >= self.min_context_recall
        )

        return EvalCaseResult(
            case_id=case.case_id,
            query=case.query,
            intent=response.intent,
            faithfulness=faithfulness,
            citation_precision=citation_precision,
            context_recall=context_recall,
            keyword_hit_rate=keyword_hit_rate,
            latency_ms=latency_ms,
            passed=case_passed,
        )

    def run_suite(
        self,
        cases: Sequence[EvalCase],
        chunks: Sequence[DocumentChunk] | None = None,
        *,
        run_id: str | None = None,
        top_k: int | None = None,
        enforce_gate: bool = True,
        min_faithfulness: float | None = None,
        min_citation_precision: float | None = None,
        min_context_recall: float | None = None,
    ) -> EvalSuiteReport:
        """Execute a sequence of EvalCases, aggregate metrics, and optionally enforce regression gate."""
        if not cases:
            raise InvalidQueryError("Evaluation suite must contain at least one EvalCase.")

        target_chunks = tuple(chunks) if chunks is not None else self._default_chunks
        if not target_chunks:
            raise EmptyRetrievalError("Cannot run evaluation suite against an empty document corpus.")

        req_faithfulness = (
            min_faithfulness if min_faithfulness is not None else self.min_faithfulness
        )
        req_precision = (
            min_citation_precision
            if min_citation_precision is not None
            else self.min_citation_precision
        )
        req_recall = (
            min_context_recall if min_context_recall is not None else self.min_context_recall
        )

        results: list[EvalCaseResult] = []
        for case in cases:
            res = self.evaluate_case(case, target_chunks, top_k=top_k)
            results.append(res)

        total = len(results)
        passed_cases = sum(1 for r in results if r.passed)
        avg_faithfulness = round(sum(r.faithfulness for r in results) / total, 6)
        avg_citation_precision = round(sum(r.citation_precision for r in results) / total, 6)
        avg_context_recall = round(sum(r.context_recall for r in results) / total, 6)
        avg_latency_ms = round(sum(r.latency_ms for r in results) / total, 4)

        passed_gate = (
            avg_faithfulness >= req_faithfulness
            and avg_citation_precision >= req_precision
            and avg_context_recall >= req_recall
        )

        if not run_id:
            digest = hashlib.sha256(
                f"{total}:{avg_faithfulness}:{avg_citation_precision}:{avg_context_recall}:{time.time_ns()}".encode(
                    "utf-8"
                )
            ).hexdigest()[:12]
            run_id = f"eval-{digest}"

        report = EvalSuiteReport(
            run_id=run_id,
            total_cases=total,
            passed_cases=passed_cases,
            avg_faithfulness=avg_faithfulness,
            avg_citation_precision=avg_citation_precision,
            avg_context_recall=avg_context_recall,
            avg_latency_ms=avg_latency_ms,
            results=tuple(results),
            passed_gate=passed_gate,
        )

        if enforce_gate and not report.passed_gate:
            raise EvalRegressionError(
                f"Evaluation quality gate failed (run_id={report.run_id}): "
                f"avg_faithfulness={avg_faithfulness:.4f} (min={req_faithfulness:.4f}), "
                f"avg_citation_precision={avg_citation_precision:.4f} (min={req_precision:.4f}), "
                f"avg_context_recall={avg_context_recall:.4f} (min={req_recall:.4f}).",
                avg_faithfulness=avg_faithfulness,
                avg_context_recall=avg_context_recall,
                avg_citation_precision=avg_citation_precision,
            )

        return report

    def evaluate_suite(
        self,
        cases: Sequence[EvalCase],
        chunks: Sequence[DocumentChunk] | None = None,
        *,
        run_id: str | None = None,
        top_k: int | None = None,
        enforce_gate: bool = True,
        min_faithfulness: float | None = None,
        min_citation_precision: float | None = None,
        min_context_recall: float | None = None,
    ) -> EvalSuiteReport:
        """Alias for run_suite()."""
        return self.run_suite(
            cases,
            chunks,
            run_id=run_id,
            top_k=top_k,
            enforce_gate=enforce_gate,
            min_faithfulness=min_faithfulness,
            min_citation_precision=min_citation_precision,
            min_context_recall=min_context_recall,
        )

    def evaluate(
        self,
        cases: Sequence[EvalCase],
        chunks: Sequence[DocumentChunk] | None = None,
        *,
        run_id: str | None = None,
        top_k: int | None = None,
        enforce_gate: bool = True,
        min_faithfulness: float | None = None,
        min_citation_precision: float | None = None,
        min_context_recall: float | None = None,
    ) -> EvalSuiteReport:
        """Alias for run_suite()."""
        return self.run_suite(
            cases,
            chunks,
            run_id=run_id,
            top_k=top_k,
            enforce_gate=enforce_gate,
            min_faithfulness=min_faithfulness,
            min_citation_precision=min_citation_precision,
            min_context_recall=min_context_recall,
        )
