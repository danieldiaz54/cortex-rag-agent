"""Unit tests for Domain Entities, Hybrid Retrieval, Multi-Agent Pipeline, Eval Harness, and AST Purity."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest

from src.domain.agents import (
    CitationGuardrailAgent,
    HybridRetrieverAgent,
    MultiAgentRagPipeline,
    RouterAgent,
    SynthesizerAgent,
)
from src.domain.entities import (
    DocumentChunk,
    EvalCase,
    QueryIntent,
    RetrievalCandidate,
)
from src.domain.eval_harness import RagEvalHarness
from src.domain.exceptions import (
    EmptyRetrievalError,
    EvalRegressionError,
    InvalidCitationError,
    InvalidDocumentError,
    InvalidQueryError,
    UngroundedClaimError,
)
from src.domain.retrieval import (
    BM25Scorer,
    HybridSearchEngine,
    compute_deterministic_embedding,
    cosine_similarity,
    tokenize,
)
from src.infrastructure.persistence.seed_corpus import (
    DEFAULT_TECHNICAL_CHUNKS,
    build_default_eval_cases,
)


def _build_corpus_chunks() -> tuple[DocumentChunk, ...]:
    """Construct pure domain DocumentChunk objects from the default technical corpus."""
    chunks: list[DocumentChunk] = []
    for raw in DEFAULT_TECHNICAL_CHUNKS:
        rich_text = (
            f"{raw['title']} {raw['section']} "
            f"{' '.join(raw.get('tags', ())) } {raw['content']}"
        )
        emb = compute_deterministic_embedding(rich_text)
        chunks.append(
            DocumentChunk(
                chunk_id=str(raw["chunk_id"]),
                doc_id=str(raw["doc_id"]),
                title=str(raw["title"]),
                section=str(raw["section"]),
                content=str(raw["content"]),
                tags=tuple(str(t) for t in raw.get("tags", ())),
                embedding=emb,
            )
        )
    return tuple(chunks)


class TestDomainEntitiesAndRetrieval(unittest.TestCase):
    """Verify DocumentChunk invariants, BM25, deterministic embeddings, cosine similarity, and RRF."""

    def test_document_chunk_self_validation_and_defaults(self) -> None:
        chunk = DocumentChunk(
            chunk_id="chunk-001",
            doc_id="DOC-01",
            title="Distributed Consensus Protocol",
            section="",
            content="Raft requires a majority quorum for log replication.",
            tags=["raft", "quorum"],  # type: ignore[arg-type]
        )
        self.assertEqual(chunk.section, "general")
        self.assertIsInstance(chunk.tags, tuple)
        self.assertEqual(chunk.tags, ("raft", "quorum"))

        with self.assertRaises(InvalidDocumentError):
            DocumentChunk(
                chunk_id="   ",
                doc_id="DOC-01",
                title="Valid Title",
                section="sec",
                content="Valid content",
            )

        with self.assertRaises(InvalidDocumentError):
            DocumentChunk(
                chunk_id="chunk-001",
                doc_id="",
                title="Valid Title",
                section="sec",
                content="Valid content",
            )

        with self.assertRaises(InvalidDocumentError):
            DocumentChunk(
                chunk_id="chunk-001",
                doc_id="DOC-01",
                title="",
                section="sec",
                content="Valid content",
            )

        with self.assertRaises(InvalidDocumentError):
            DocumentChunk(
                chunk_id="chunk-001",
                doc_id="DOC-01",
                title="Valid Title",
                section="sec",
                content="   ",
            )

    def test_deterministic_embedding_unit_l2_norm_and_cosine_similarity(self) -> None:
        vec_a = compute_deterministic_embedding(
            "Cortex Hybrid RAG combines BM25 and pgvector Reciprocal Rank Fusion"
        )
        vec_a_repeat = compute_deterministic_embedding(
            "Cortex Hybrid RAG combines BM25 and pgvector Reciprocal Rank Fusion"
        )
        vec_b = compute_deterministic_embedding(
            "BM25 lexical scoring and pgvector dense embeddings with RRF rank fusion"
        )
        vec_empty = compute_deterministic_embedding("   ")

        self.assertEqual(len(vec_a), 64)
        self.assertEqual(vec_a, vec_a_repeat)
        self.assertTrue(all(v == 0.0 for v in vec_empty))

        l2_norm = math.sqrt(sum(x * x for x in vec_a))
        self.assertAlmostEqual(l2_norm, 1.0, places=5)

        sim_self = cosine_similarity(vec_a, vec_a)
        sim_related = cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(sim_self, 1.0, places=6)
        self.assertGreater(sim_related, 0.35)

        with self.assertRaises(ValueError):
            compute_deterministic_embedding("test", dim=0)

        with self.assertRaises(ValueError):
            cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))

    def test_bm25_scorer_and_hybrid_search_rrf_ranking(self) -> None:
        corpus = _build_corpus_chunks()
        scorer = BM25Scorer(corpus)
        scores = scorer.score_chunks("ERR-409-OPTIMISTIC-LOCK idempotency backoff")
        self.assertEqual(len(scores), len(corpus))

        err_409_idx = next(
            i for i, c in enumerate(corpus) if c.chunk_id == "chunk-err-409"
        )
        self.assertEqual(max(scores), scores[err_409_idx])
        self.assertGreater(scores[err_409_idx], 0.0)

        engine = HybridSearchEngine(rrf_k=60)
        candidates = engine.search(
            "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers?",
            corpus,
            top_k=3,
            bm25_weight=0.75,
            vector_weight=0.25,
        )
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0].chunk.chunk_id, "chunk-err-409")
        self.assertEqual(candidates[0].bm25_rank, 1)
        self.assertGreaterEqual(candidates[0].rrf_score, candidates[1].rrf_score)
        self.assertGreaterEqual(candidates[1].rrf_score, candidates[2].rrf_score)

        with self.assertRaises(InvalidQueryError):
            engine.search("   ", corpus)

        with self.assertRaises(InvalidQueryError):
            engine.search("x" * 2005, corpus)

        with self.assertRaises(InvalidQueryError):
            engine.search("valid query", corpus, top_k=0)

        with self.assertRaises(EmptyRetrievalError):
            engine.search("valid query", ())


class TestMultiAgentPipelineAndGuardrails(unittest.TestCase):
    """Verify RouterAgent, SynthesizerAgent, CitationGuardrailAgent, and RagEvalHarness."""

    def setUp(self) -> None:
        self.corpus = _build_corpus_chunks()

    def test_router_agent_intent_classification(self) -> None:
        router = RouterAgent()

        d_err = router.route("How to remediate ERR-409-OPTIMISTIC-LOCK in ledgers?")
        self.assertEqual(d_err.intent, QueryIntent.LEXICAL_EXACT)
        self.assertEqual((d_err.bm25_weight, d_err.vector_weight), (0.75, 0.25))

        d_aga = router.route("AGA-8 230 bar DivergenceException in custody transfer")
        self.assertEqual(d_aga.intent, QueryIntent.LEXICAL_EXACT)
        self.assertEqual((d_aga.bm25_weight, d_aga.vector_weight), (0.75, 0.25))

        d_sem = router.route(
            "How does the architectural design prevent split-brain writes during network partitions?"
        )
        self.assertEqual(d_sem.intent, QueryIntent.SEMANTIC_CONCEPTUAL)
        self.assertEqual((d_sem.bm25_weight, d_sem.vector_weight), (0.35, 0.65))

        d_hyb = router.route(
            "Compare and analyze the tradeoff between BM25 lexical scoring and vector fusion"
        )
        self.assertEqual(d_hyb.intent, QueryIntent.HYBRID_ANALYTICAL)
        self.assertEqual((d_hyb.bm25_weight, d_hyb.vector_weight), (0.50, 0.50))

        with self.assertRaises(InvalidQueryError):
            router.route("")

        with self.assertRaises(InvalidQueryError):
            router.route("a" * 2001)

    def test_citation_guardrail_agent_enforces_grounding_and_valid_citations(self) -> None:
        guardrail = CitationGuardrailAgent(min_faithfulness=0.65)
        target_chunk = next(c for c in self.corpus if c.chunk_id == "chunk-err-409")
        candidate = RetrievalCandidate(
            chunk=target_chunk,
            bm25_score=10.5,
            vector_score=0.82,
            bm25_rank=1,
            vector_rank=1,
            rrf_score=0.01639344,
        )

        # 1. Missing citations -> UngroundedClaimError
        with self.assertRaises(UngroundedClaimError):
            guardrail.validate(
                "Error code ERR-409-OPTIMISTIC-LOCK requires SHA-256 idempotency verification.",
                (candidate,),
            )

        # 2. Non-existent citation ID -> InvalidCitationError
        with self.assertRaises(InvalidCitationError) as ctx:
            guardrail.validate(
                "Error code ERR-409-OPTIMISTIC-LOCK requires idempotency [cite:non-existent].",
                (candidate,),
            )
        self.assertIn("non-existent", ctx.exception.invalid_ids)

        # 3. Hallucinated ungrounded claim with valid citation tag -> UngroundedClaimError
        with self.assertRaises(UngroundedClaimError):
            guardrail.validate(
                "Quantum photosynthesis in orbital satellites uses banana smoothies "
                "for telemetry encryption [cite:chunk-err-409].",
                (candidate,),
            )

        # 4. Properly grounded claim -> passes and returns verified Citation
        grounded_answer = (
            "Error code ERR-409-OPTIMISTIC-LOCK occurs when concurrent ledger writers "
            "conflict on aggregate sequence numbers, requiring SHA-256 idempotency key "
            "verification and exponential jitter backoff. [cite:chunk-err-409]"
        )
        validation = guardrail.validate(grounded_answer, (candidate,))
        self.assertGreaterEqual(validation.faithfulness_score, 0.85)
        self.assertEqual(len(validation.citations), 1)
        self.assertEqual(validation.citations[0].chunk_id, "chunk-err-409")

    def test_multi_agent_pipeline_and_eval_harness_quality_gates(self) -> None:
        pipeline = MultiAgentRagPipeline(
            router=RouterAgent(),
            retriever=HybridRetrieverAgent(),
            synthesizer=SynthesizerAgent(),
            guardrail=CitationGuardrailAgent(min_faithfulness=0.65),
        )
        response = pipeline.run(
            "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers and how is it resolved?",
            self.corpus,
            top_k=4,
        )
        self.assertEqual(response.intent, QueryIntent.LEXICAL_EXACT)
        self.assertGreaterEqual(response.faithfulness_score, 0.80)
        self.assertEqual(len(response.traces), 4)
        self.assertTrue(any(c.chunk_id == "chunk-err-409" for c in response.citations))

        # Golden Dataset Evaluation Suite pass
        harness = RagEvalHarness(pipeline=pipeline, chunks=self.corpus)
        golden_cases = build_default_eval_cases()
        report = harness.run_suite(golden_cases, enforce_gate=True)
        self.assertTrue(report.passed_gate)
        self.assertEqual(report.total_cases, 5)
        self.assertEqual(report.passed_cases, 5)
        self.assertGreaterEqual(report.avg_faithfulness, 0.80)
        self.assertEqual(report.avg_context_recall, 1.0)
        self.assertEqual(report.avg_citation_precision, 1.0)

        # Unreachable faithfulness threshold -> EvalRegressionError
        with self.assertRaises(EvalRegressionError):
            harness.run_suite(
                golden_cases,
                enforce_gate=True,
                min_faithfulness=1.05,
            )

        # Mismatched expected chunk IDs -> EvalRegressionError
        bad_cases = (
            EvalCase(
                case_id="eval-bad-001",
                query="How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers?",
                expected_chunk_ids=("chunk-nonexistent-999",),
                expected_keywords=("ERR-409-OPTIMISTIC-LOCK",),
            ),
        )
        with self.assertRaises(EvalRegressionError):
            harness.run_suite(bad_cases, enforce_gate=True)


class TestDomainPurityAST(unittest.TestCase):
    """Enforce .agents/rules/domain-purity.md across all files in src/domain/ via AST inspection."""

    FORBIDDEN_PREFIXES: tuple[str, ...] = (
        "sqlite3",
        "http",
        "fastapi",
        "urllib",
        "sqlalchemy",
        "psycopg",
        "redis",
        "src.infrastructure",
        "src.interfaces",
        "src.application",
    )

    def test_domain_layer_has_zero_infrastructure_or_interface_imports(self) -> None:
        domain_dir = Path(__file__).resolve().parents[1] / "src" / "domain"
        py_files = sorted(domain_dir.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 5)

        violations: list[str] = []
        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod_name = alias.name
                        if any(
                            mod_name == prefix or mod_name.startswith(f"{prefix}.")
                            for prefix in self.FORBIDDEN_PREFIXES
                        ):
                            violations.append(f"{py_file.name}: import {mod_name}")
                elif isinstance(node, ast.ImportFrom):
                    mod_name = node.module or ""
                    if any(
                        mod_name == prefix or mod_name.startswith(f"{prefix}.")
                        for prefix in self.FORBIDDEN_PREFIXES
                    ):
                        violations.append(f"{py_file.name}: from {mod_name} import ...")

        self.assertEqual(
            violations,
            [],
            f"Domain purity violations detected in src/domain/: {violations}",
        )


if __name__ == "__main__":
    unittest.main()
