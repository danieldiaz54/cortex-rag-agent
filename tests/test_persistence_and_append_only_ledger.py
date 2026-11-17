"""Integration tests for SQLiteHybridStore persistence and Append-Only Ledger immutability."""

from __future__ import annotations

import math
import unittest

from src.domain.agents import MultiAgentRagPipeline
from src.domain.entities import DocumentChunk
from src.domain.eval_harness import RagEvalHarness
from src.domain.repositories import ChunkRepositoryProtocol, LedgerRepositoryProtocol
from src.infrastructure.persistence.seed_corpus import (
    build_default_eval_cases,
    seed_default_corpus,
)
from src.infrastructure.persistence.sqlite_store import (
    AppendOnlyLedgerError,
    SQLiteHybridStore,
)


class TestPersistenceAndAppendOnlyLedger(unittest.TestCase):
    """Verify SQLiteHybridStore chunk operations, deterministic embeddings, and ledger triggers."""

    def setUp(self) -> None:
        self.store = SQLiteHybridStore(db_path=":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_repository_protocol_conformance_and_chunk_upsert(self) -> None:
        self.assertIsInstance(self.store, ChunkRepositoryProtocol)
        self.assertIsInstance(self.store, LedgerRepositoryProtocol)

        raw_chunk = DocumentChunk(
            chunk_id="chunk-test-101",
            doc_id="DOC-TEST-101",
            title="Zero-Trust mTLS Certificate Rotation",
            section="security",
            content="SPIFFE workload identities rotate short-lived X.509 SVIDs every 30 minutes.",
            tags=("mtls", "spiffe", "zero-trust"),
        )
        enriched = self.store.upsert_chunk(raw_chunk)
        self.assertEqual(len(enriched.embedding), 64)
        norm = math.sqrt(sum(v * v for v in enriched.embedding))
        self.assertAlmostEqual(norm, 1.0, places=5)

        stored_chunks = self.store.list_chunks()
        self.assertEqual(len(stored_chunks), 1)
        self.assertEqual(stored_chunks[0].chunk_id, "chunk-test-101")
        self.assertEqual(stored_chunks[0].embedding, enriched.embedding)

        # Upsert update on conflict
        updated_chunk = DocumentChunk(
            chunk_id="chunk-test-101",
            doc_id="DOC-TEST-101",
            title="Updated Zero-Trust mTLS Rotation Policy",
            section="security",
            content="SPIFFE workload identities rotate X.509 SVIDs every 15 minutes.",
            tags=("mtls", "spiffe"),
        )
        self.store.upsert_chunk(updated_chunk)
        refreshed = self.store.list_chunks()
        self.assertEqual(len(refreshed), 1)
        self.assertEqual(refreshed[0].title, "Updated Zero-Trust mTLS Rotation Policy")

    def test_seed_default_corpus_populates_eight_technical_chunks(self) -> None:
        seeded_count = seed_default_corpus(self.store)
        self.assertEqual(seeded_count, 8)
        chunks = self.store.list_chunks()
        self.assertEqual(len(chunks), 8)
        for chunk in chunks:
            self.assertEqual(len(chunk.embedding), 64)

    def test_append_only_query_audit_and_eval_ledger_immutability_triggers(self) -> None:
        seed_default_corpus(self.store)
        chunks = self.store.list_chunks()
        pipeline = MultiAgentRagPipeline()

        # 1. Append Query Audit Record
        rag_response = pipeline.run(
            "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers?",
            chunks,
            top_k=4,
        )
        audit_id = self.store.append_query_audit(rag_response)
        self.assertTrue(audit_id.startswith("aud-"))

        audits = self.store.list_query_audits()
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["audit_id"], audit_id)

        # Verify BEFORE UPDATE trigger on query_audit_log
        with self.assertRaises(AppendOnlyLedgerError):
            self.store.execute_raw_mutation_for_test(
                "UPDATE query_audit_log SET answer = 'tampered' WHERE audit_id = ?",
                (audit_id,),
            )

        # Verify BEFORE DELETE trigger on query_audit_log
        with self.assertRaises(AppendOnlyLedgerError):
            self.store.execute_raw_mutation_for_test(
                "DELETE FROM query_audit_log WHERE audit_id = ?",
                (audit_id,),
            )

        # Record must remain intact and untampered
        audits_after = self.store.list_query_audits()
        self.assertEqual(len(audits_after), 1)
        self.assertEqual(audits_after[0]["answer"], rag_response.answer)

        # 2. Append Evaluation Suite Report
        harness = RagEvalHarness(pipeline=pipeline, chunks=chunks)
        report = harness.run_suite(build_default_eval_cases(), enforce_gate=True)
        self.store.append_eval_report(report)

        eval_reports = self.store.list_eval_reports()
        self.assertEqual(len(eval_reports), 1)
        self.assertEqual(eval_reports[0]["run_id"], report.run_id)
        self.assertTrue(eval_reports[0]["passed_gate"])

        # Verify BEFORE UPDATE trigger on eval_runs_ledger
        with self.assertRaises(AppendOnlyLedgerError):
            self.store.execute_raw_mutation_for_test(
                "UPDATE eval_runs_ledger SET passed_cases = 0 WHERE run_id = ?",
                (report.run_id,),
            )

        # Verify BEFORE DELETE trigger on eval_runs_ledger
        with self.assertRaises(AppendOnlyLedgerError):
            self.store.execute_raw_mutation_for_test(
                "DELETE FROM eval_runs_ledger WHERE run_id = ?",
                (report.run_id,),
            )

        # Record must remain intact
        eval_reports_after = self.store.list_eval_reports()
        self.assertEqual(len(eval_reports_after), 1)
        self.assertEqual(eval_reports_after[0]["passed_cases"], report.passed_cases)


if __name__ == "__main__":
    unittest.main()
