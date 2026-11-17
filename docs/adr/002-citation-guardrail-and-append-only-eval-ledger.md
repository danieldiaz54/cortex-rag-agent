# ADR-002: Citation Guardrail Verification & Database-Enforced Append-Only Audit/Evaluation Ledgers

- **Status**: Accepted
- **Date**: 2026-11-17
- **Deciders**: `domain-architect`, `data-architect`, `security-auditor`

---

## 1. Context & Problem Statement

Production RAG systems face two critical reliability risks:
1. **Hallucinated & Ungrounded Claims**: Synthesized responses may state facts not supported by retrieved evidence or fabricate non-existent citation identifiers.
2. **Silent Quality Regression & Audit Tampering**: Prompt, chunking, or weighting changes can degrade retrieval recall or grounding faithfulness over time if evaluation runs and query audits can be overwritten or silently ignored.

## 2. Decision

1. **Mandatory Inline Provenance (`[cite:<chunk_id>]`)**:
   - `SynthesizerAgent` attaches explicit `[cite:<chunk_id>]` markers to every synthesized technical claim.
   - `CitationGuardrailAgent` parses all `[cite:<chunk_id>]` markers and validates:
     - **Non-Empty Citations**: Raises `UngroundedClaimError` (`HTTP 422`) if zero citations are present.
     - **Candidate Membership**: Raises `InvalidCitationError` (`HTTP 422`) if any cited `chunk_id` does not exist in the top-$k$ retrieved candidate set.
     - **Lexical/Stem Grounding Faithfulness**: Computes a blended token and segment grounding score $\in [0.0, 1.0]$ against the cited chunks and raises `UngroundedClaimError` (`HTTP 422`) if `faithfulness_score < min_faithfulness` (`0.65`).
2. **Append-Only Audit & Evaluation Ledgers (`query_audit_log`, `eval_runs_ledger`)**:
   - Every verified `RagResponse` and every `EvalSuiteReport` is persisted in append-only tables.
   - Both SQLite (`src/infrastructure/persistence/sqlite_store.py`) and PostgreSQL (`migrations/001_init_pgvector_hybrid.sql`) attach `BEFORE UPDATE` and `BEFORE DELETE` database triggers that abort any mutation attempt with `APPEND_ONLY_LEDGER_VIOLATION` (`AppendOnlyLedgerError`).

## 3. Consequences

- **Positive**:
  - Hallucinated citations or ungrounded answers are blocked before reaching API consumers.
  - Full cryptographic/temporal provenance of every RAG answer and evaluation benchmark is preserved at the storage engine layer.
- **Negative / Trade-offs**:
  - Ledger tables grow monotonically and require periodic cold-storage archival partitioning in high-throughput production deployments.
