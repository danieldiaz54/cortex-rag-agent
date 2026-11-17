"""Multi-Agent Pipeline: RouterAgent, HybridRetrieverAgent, SynthesizerAgent, CitationGuardrailAgent."""

from __future__ import annotations

from collections.abc import Sequence
import re
import time
from typing import NamedTuple

from src.domain.entities import (
    AgentStepTrace,
    Citation,
    DocumentChunk,
    QueryIntent,
    RagResponse,
    RetrievalCandidate,
)
from src.domain.exceptions import (
    EmptyRetrievalError,
    InvalidCitationError,
    InvalidQueryError,
    UngroundedClaimError,
)
from src.domain.retrieval import (
     _STOPWORDS,
    _stem_token,
    HybridSearchEngine,
    tokenize,
)

_CITATION_PATTERN = re.compile(r"\[cite:([a-zA-Z0-9_.:-]+)\]")

_EXACT_CODE_PATTERN = re.compile(
    r"(?:\b(?:err|aga|rfc|iso|cve|http|api|sha|ieee|nist|ansi)[-_]?\d+[a-z0-9_-]*\b"
    r"|--[a-z0-9_-]+"
    r"|\b(?:k1|rrf_k|top_k|b)\s*=\s*\d+(?:\.\d+)?"
    r"|\b\d+\.\d+\b)",
    re.IGNORECASE,
)

_ACRONYM_OR_NUM_PATTERN = re.compile(r"(?:\b[A-Z]{2,8}[0-9]*\b|\b\d+\b)")

_IGNORED_UPPER_WORDS: frozenset[str] = frozenset(
    {"HOW", "WHY", "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "THE", "AND", "FOR", "WITH", "RAG"}
)

_CONCEPTUAL_TERMS: frozenset[str] = frozenset(
    {
        "how",
        "why",
        "architecture",
        "architectural",
        "concept",
        "conceptual",
        "explain",
        "design",
        "principle",
        "principles",
        "overview",
        "philosophy",
        "mechanism",
        "workflow",
        "purpose",
        "role",
        "strategy",
        "pattern",
        "model",
        "prevent",
        "ensure",
    }
)

_ANALYTICAL_TERMS: frozenset[str] = frozenset(
    {
        "compare",
        "comparison",
        "versus",
        "vs",
        "tradeoff",
        "trade-off",
        "tradeoffs",
        "analyze",
        "analysis",
        "analytical",
        "combine",
        "combining",
        "fusion",
        "balance",
        "between",
        "interaction",
        "difference",
        "differences",
        "relationship",
        "impact",
        "evaluate",
        "hybrid",
    }
)


class RouteDecision(NamedTuple):
    """Output of RouterAgent containing intent and hybrid RRF weights."""

    intent: QueryIntent
    bm25_weight: float
    vector_weight: float


class GuardrailValidationResult(NamedTuple):
    """Output of CitationGuardrailAgent containing faithfulness score and verified citations."""

    faithfulness_score: float
    citations: tuple[Citation, ...]


class RouterAgent:
    """Classifies user queries into LEXICAL_EXACT, SEMANTIC_CONCEPTUAL, or HYBRID_ANALYTICAL."""

    MAX_QUERY_LENGTH = 2000

    WEIGHT_MAP: dict[QueryIntent, tuple[float, float]] = {
        QueryIntent.LEXICAL_EXACT: (0.75, 0.25),
        QueryIntent.SEMANTIC_CONCEPTUAL: (0.35, 0.65),
        QueryIntent.HYBRID_ANALYTICAL: (0.50, 0.50),
    }

    def route(self, query: str) -> RouteDecision:
        """Determine QueryIntent and (bm25_weight, vector_weight) for a query."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("Query cannot be empty or whitespace.")
        cleaned = query.strip()
        if len(cleaned) > self.MAX_QUERY_LENGTH:
            raise InvalidQueryError(
                f"Query length {len(cleaned)} exceeds maximum allowed {self.MAX_QUERY_LENGTH}."
            )

        tokens = tokenize(cleaned)
        if not tokens:
            raise InvalidQueryError("Query must contain at least one valid alphanumeric token.")

        token_set = set(tokens)
        has_exact_code = bool(_EXACT_CODE_PATTERN.search(cleaned))
        has_analytical = bool(token_set & _ANALYTICAL_TERMS)
        has_conceptual = bool(token_set & _CONCEPTUAL_TERMS)

        raw_acronyms = [
            m.group(0)
            for m in _ACRONYM_OR_NUM_PATTERN.finditer(cleaned)
            if m.group(0) not in _IGNORED_UPPER_WORDS
        ]
        has_lexical_markers = has_exact_code or len(raw_acronyms) > 0

        if has_exact_code and not has_analytical:
            intent = QueryIntent.LEXICAL_EXACT
        elif has_analytical or (has_lexical_markers and has_conceptual):
            intent = QueryIntent.HYBRID_ANALYTICAL
        elif has_lexical_markers and not has_conceptual:
            intent = QueryIntent.LEXICAL_EXACT
        elif has_conceptual:
            intent = QueryIntent.SEMANTIC_CONCEPTUAL
        else:
            intent = QueryIntent.HYBRID_ANALYTICAL

        bm25_w, vec_w = self.WEIGHT_MAP[intent]
        return RouteDecision(intent=intent, bm25_weight=bm25_w, vector_weight=vec_w)

    def classify(self, query: str) -> QueryIntent:
        """Return only the QueryIntent classification for `query`."""
        return self.route(query).intent


class HybridRetrieverAgent:
    """Agent wrapper around HybridSearchEngine for dynamic weighted RRF retrieval."""

    def __init__(self, search_engine: HybridSearchEngine | None = None) -> None:
        self.search_engine = search_engine or HybridSearchEngine()

    def retrieve(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
        *,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> tuple[RetrievalCandidate, ...]:
        """Execute weighted hybrid retrieval over the document corpus."""
        return self.search_engine.search(
            query,
            chunks,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )


class SynthesizerAgent:
    """Synthesizes a grounded technical answer with inline [cite:<chunk_id>] provenance markers."""

    def __init__(self, *, max_cited_chunks: int = 2) -> None:
        self.max_cited_chunks = max(1, max_cited_chunks)

    @staticmethod
    def _select_relevant_sentences(query: str, chunk: DocumentChunk) -> str:
        query_terms = {
            _stem_token(t) for t in tokenize(query) if t not in _STOPWORDS
        } | set(tokenize(query))

        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", chunk.content.strip())
            if s.strip()
        ]
        if not raw_sentences:
            return chunk.content.strip()

        if len(chunk.content) <= 500 or len(raw_sentences) <= 3:
            joined = " ".join(raw_sentences)
        else:
            scored: list[tuple[int, int, str]] = []
            for idx, sent in enumerate(raw_sentences):
                sent_tokens = tokenize(sent)
                sent_stems = {_stem_token(t) for t in sent_tokens} | set(sent_tokens)
                overlap = len(query_terms & sent_stems)
                scored.append((overlap, -idx, sent))
            scored.sort(reverse=True)
            chosen = sorted(scored[:3], key=lambda item: -item[1])
            joined = " ".join(item[2] for item in chosen)

        joined = joined.rstrip(". ")
        return f"{chunk.title} ({chunk.section}): {joined}."

    def synthesize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> str:
        """Construct a grounded technical synthesis with explicit [cite:<chunk_id>] markers."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("Cannot synthesize an answer for an empty query.")
        if not candidates:
            raise EmptyRetrievalError("Cannot synthesize an answer without retrieval candidates.")

        top_candidate = candidates[0]
        selected: list[RetrievalCandidate] = [top_candidate]

        # Include a second candidate only when it is nearly tied in hybrid relevance
        for cand in candidates[1 : self.max_cited_chunks]:
            rrf_ratio = cand.rrf_score / top_candidate.rrf_score if top_candidate.rrf_score > 0 else 0.0
            has_lexical_support = (
                top_candidate.bm25_score <= 0.0
                or cand.bm25_score >= top_candidate.bm25_score * 0.75
            )
            has_vector_support = (
                top_candidate.vector_score <= 0.0
                or cand.vector_score >= top_candidate.vector_score * 0.80
            )
            if rrf_ratio >= 0.95 and has_lexical_support and has_vector_support:
                selected.append(cand)

        statements: list[str] = []
        for cand in selected:
            claim_text = self._select_relevant_sentences(query, cand.chunk)
            statements.append(f"{claim_text} [cite:{cand.chunk.chunk_id}]")

        return " ".join(statements)


class CitationGuardrailAgent:
    """Verifies inline [cite:<chunk_id>] markers and enforces lexical/term grounding faithfulness."""

    def __init__(self, *, min_faithfulness: float = 0.65) -> None:
        if not (0.0 <= min_faithfulness <= 1.0):
            raise ValueError("min_faithfulness must be between 0.0 and 1.0.")
        self.min_faithfulness = min_faithfulness

    @staticmethod
    def extract_citation_ids(answer: str) -> tuple[str, ...]:
        """Extract ordered unique chunk_ids referenced via [cite:<chunk_id>] in `answer`."""
        if not answer:
            return ()
        matches = _CITATION_PATTERN.findall(answer)
        return tuple(dict.fromkeys(matches))

    @staticmethod
    def _chunk_grounding_vocab(chunk: DocumentChunk) -> set[str]:
        raw = tokenize(f"{chunk.title} {chunk.section} {' '.join(chunk.tags)} {chunk.content}")
        vocab: set[str] = set()
        for tok in raw:
            vocab.add(tok)
            vocab.add(_stem_token(tok))
            if "-" in tok or "_" in tok:
                for part in re.split(r"[-_]+", tok):
                    if part:
                        vocab.add(part)
                        vocab.add(_stem_token(part))
        return vocab

    def compute_faithfulness(
        self,
        answer: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> float:
        """Compute faithfulness grounding score in [0.0, 1.0] of `answer` against cited chunks."""
        cited_ids = self.extract_citation_ids(answer)
        if not cited_ids or not candidates:
            return 0.0

        candidate_map = {c.chunk.chunk_id: c for c in candidates}
        cited_chunks = [candidate_map[cid].chunk for cid in cited_ids if cid in candidate_map]
        if not cited_chunks:
            return 0.0

        combined_vocab: set[str] = set()
        for chunk in cited_chunks:
            combined_vocab |= self._chunk_grounding_vocab(chunk)

        stripped_answer = _CITATION_PATTERN.sub("", answer).strip()
        answer_tokens = [t for t in tokenize(stripped_answer) if t not in _STOPWORDS]
        if not answer_tokens:
            return 0.0

        grounded_count = sum(
            1
            for tok in answer_tokens
            if tok in combined_vocab or _stem_token(tok) in combined_vocab
        )
        token_grounding = grounded_count / len(answer_tokens)

        # Also evaluate segment-level grounding to catch ungrounded clauses appended to valid ones
        segments = [
            seg.strip()
            for seg in re.split(r"\[cite:[a-zA-Z0-9_.:-]+\]|\n+", answer)
            if seg.strip()
        ]
        segment_scores: list[float] = []
        for seg in segments:
            seg_tokens = [t for t in tokenize(seg) if t not in _STOPWORDS]
            if not seg_tokens:
                continue
            seg_hits = sum(
                1
                for tok in seg_tokens
                if tok in combined_vocab or _stem_token(tok) in combined_vocab
            )
            segment_scores.append(seg_hits / len(seg_tokens))

        if not segment_scores:
            return 0.0

        min_seg_score = min(segment_scores)
        avg_seg_score = sum(segment_scores) / len(segment_scores)
        blended = 0.55 * token_grounding + 0.30 * avg_seg_score + 0.15 * min_seg_score
        return round(max(0.0, min(1.0, blended)), 6)

    def validate(
        self,
        answer: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        min_faithfulness: float | None = None,
    ) -> GuardrailValidationResult:
        """Validate citations and grounding faithfulness, raising typed domain errors on failure."""
        if not isinstance(answer, str) or not answer.strip():
            raise UngroundedClaimError("Synthesized answer is empty.", faithfulness_score=0.0)

        cited_ids = self.extract_citation_ids(answer)
        if not cited_ids:
            raise UngroundedClaimError(
                "Answer contains factual claims without any [cite:<chunk_id>] markers.",
                faithfulness_score=0.0,
            )

        candidate_map = {c.chunk.chunk_id: c for c in candidates}
        invalid_ids = tuple(cid for cid in cited_ids if cid not in candidate_map)
        if invalid_ids:
            raise InvalidCitationError(
                f"Answer references non-existent or unretrieved chunk IDs: {', '.join(invalid_ids)}",
                invalid_ids=invalid_ids,
            )

        threshold = min_faithfulness if min_faithfulness is not None else self.min_faithfulness
        faithfulness = self.compute_faithfulness(answer, candidates)
        if faithfulness < threshold:
            raise UngroundedClaimError(
                f"Answer faithfulness score {faithfulness:.4f} is below minimum threshold {threshold:.4f}.",
                faithfulness_score=faithfulness,
            )

        citations: list[Citation] = []
        for cid in cited_ids:
            cand = candidate_map[cid]
            excerpt = cand.chunk.content[:180].strip()
            if len(cand.chunk.content) > 180:
                excerpt += "..."
            citations.append(
                Citation(
                    chunk_id=cand.chunk.chunk_id,
                    doc_id=cand.chunk.doc_id,
                    title=cand.chunk.title,
                    section=cand.chunk.section,
                    excerpt=excerpt,
                    rrf_score=cand.rrf_score,
                )
            )

        return GuardrailValidationResult(
            faithfulness_score=faithfulness,
            citations=tuple(citations),
        )

    def verify(
        self,
        answer: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        min_faithfulness: float | None = None,
    ) -> GuardrailValidationResult:
        """Alias for validate()."""
        return self.validate(answer, candidates, min_faithfulness=min_faithfulness)


class MultiAgentRagPipeline:
    """Orchestrates RouterAgent -> HybridSearchEngine -> SynthesizerAgent -> CitationGuardrailAgent."""

    def __init__(
        self,
        *,
        router: RouterAgent | None = None,
        search_engine: HybridSearchEngine | None = None,
        retriever: HybridRetrieverAgent | None = None,
        synthesizer: SynthesizerAgent | None = None,
        guardrail: CitationGuardrailAgent | None = None,
        min_faithfulness: float = 0.65,
    ) -> None:
        self.router = router or RouterAgent()
        self.search_engine = search_engine or (
            retriever.search_engine if retriever is not None else HybridSearchEngine()
        )
        self.retriever = retriever or HybridRetrieverAgent(self.search_engine)
        self.synthesizer = synthesizer or SynthesizerAgent()
        self.guardrail = guardrail or CitationGuardrailAgent(min_faithfulness=min_faithfulness)

    def run(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
        *,
        top_k: int = 5,
    ) -> RagResponse:
        """Execute the complete multi-agent RAG workflow and return a verified RagResponse."""
        pipeline_start = time.perf_counter()
        traces: list[AgentStepTrace] = []

        # Step 1: RouterAgent
        t0 = time.perf_counter()
        decision = self.router.route(query)
        t_router = round((time.perf_counter() - t0) * 1000.0, 4)
        traces.append(
            AgentStepTrace(
                agent_name="RouterAgent",
                action="classify_intent",
                detail=(
                    f"intent={decision.intent.value} "
                    f"(bm25_weight={decision.bm25_weight:.2f}, vector_weight={decision.vector_weight:.2f})"
                ),
                duration_ms=t_router,
            )
        )

        # Step 2: HybridRetrieverAgent / HybridSearchEngine
        t1 = time.perf_counter()
        candidates = self.retriever.retrieve(
            query,
            chunks,
            top_k=top_k,
            bm25_weight=decision.bm25_weight,
            vector_weight=decision.vector_weight,
        )
        t_retrieval = round((time.perf_counter() - t1) * 1000.0, 4)
        traces.append(
            AgentStepTrace(
                agent_name="HybridRetrieverAgent",
                action="hybrid_rrf_search",
                detail=(
                    f"retrieved={len(candidates)} "
                    f"top_chunk={candidates[0].chunk.chunk_id} "
                    f"top_rrf={candidates[0].rrf_score:.6f}"
                ),
                duration_ms=t_retrieval,
            )
        )

        # Step 3: SynthesizerAgent
        t2 = time.perf_counter()
        answer = self.synthesizer.synthesize(query, candidates)
        t_synth = round((time.perf_counter() - t2) * 1000.0, 4)
        extracted_ids = self.guardrail.extract_citation_ids(answer)
        traces.append(
            AgentStepTrace(
                agent_name="SynthesizerAgent",
                action="grounded_synthesis",
                detail=f"synthesized_chars={len(answer)} cited_chunks={len(extracted_ids)}",
                duration_ms=t_synth,
            )
        )

        # Step 4: CitationGuardrailAgent
        t3 = time.perf_counter()
        validation = self.guardrail.validate(answer, candidates)
        t_guard = round((time.perf_counter() - t3) * 1000.0, 4)
        traces.append(
            AgentStepTrace(
                agent_name="CitationGuardrailAgent",
                action="verify_citations_and_grounding",
                detail=(
                    f"verified_citations={len(validation.citations)} "
                    f"faithfulness={validation.faithfulness_score:.4f}"
                ),
                duration_ms=t_guard,
            )
        )

        total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000.0, 4)
        return RagResponse(
            query=query.strip(),
            intent=decision.intent,
            answer=answer,
            citations=validation.citations,
            candidates=candidates,
            traces=tuple(traces),
            faithfulness_score=validation.faithfulness_score,
            total_latency_ms=total_latency_ms,
        )

    def execute(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
        *,
        top_k: int = 5,
    ) -> RagResponse:
        """Alias for run()."""
        return self.run(query, chunks, top_k=top_k)
