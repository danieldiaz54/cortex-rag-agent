# Agentic Architecture Changelog — cortex-rag-agent

## [2026-11-17] - Release v0.1.0: Multi-Agent Hybrid RAG, Append-Only Eval Ledger & SDD Certification

### Added
- **SDD Specifications (`.specs/`)**:
  - `.specs/01-domain/hybrid-rag-multiagent.spec.md` (`domain-architect`)
  - `.specs/02-data/hybrid-store-and-append-only-ledger.spec.md` (`data-architect`)
  - `.specs/03-integration/http-sse-api-contracts.spec.md` (`integration-architect`)
  - `.specs/04-security/security-audit-report.spec.md` (`security-auditor`)
- **Multi-Agent & Hybrid RAG Engine (`src/domain/`)**:
  - `RouterAgent`, `HybridRetrieverAgent` (Okapi BM25 + 64-D L2 Vector Cosine + Weighted RRF), `SynthesizerAgent`, `CitationGuardrailAgent`, `MultiAgentRagPipeline` y `RagEvalHarness`.
- **Append-Only Persistence (`src/infrastructure/persistence/` & `migrations/`)**:
  - `SQLiteHybridStore` con triggers de inmutabilidad `BEFORE UPDATE` / `BEFORE DELETE` en `query_audit_log` y `eval_runs_ledger`, `seed_corpus.py` (8 documentos técnicos y 5 casos dorados) y `001_init_pgvector_hybrid.sql`.
- **HTTP + SSE API & Minimalist UI (`src/application/`, `src/interfaces/http/`, `ui/`)**:
  - `CortexApplicationService`, `server.py` (`ThreadingHTTPServer` con SSE), `fastapi_app.py` y consola editorial minimalista en `ui/index.html`.
- **QA & DevSecOps Certification (`tests/`)**:
  - 15/15 pruebas unitarias, AST domain purity, triggers append-only e integración HTTP/SSE certificadas por `qa-verifier` y `security-auditor`.

## [2026-11-17] - Adopción de Estructura Agéntica (Estándar GNV Manager)

### Added
- **`AGENTS.md`**: Gobernanza del *Lead Orchestrator*, Ley de Ramas (`dev` / `main` / `sdd/agentic-optimization`) y catálogo de 7 subagentes.
- **`.agents/`**: Personas de los 7 subagentes, reglas contextuales (`domain-purity.md`, `ui-strict-types.md`, `gitflow-and-env-isolation.md`), Safety Gate (`hooks.json`, `safety-gate.py`, `safety-gate.js`) y catálogo de skills UI/UX.
- **`.specs/`**: Especificaciones fundacionales en `.specs/00-infrastructure/` y `.specs/01-domain/architecture-baseline.spec.md`.
- **Aislamiento de Entornos**: `.env.development.example`, `.env.production.example` y `docker-compose.dev.yml`.
