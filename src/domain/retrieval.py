"""Deterministic Hybrid Retrieval Engine: Technical Tokenizer, BM25, Hash Embeddings, and RRF."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import hashlib
import math
import re

from src.domain.entities import DocumentChunk, RetrievalCandidate
from src.domain.exceptions import EmptyRetrievalError, InvalidQueryError

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

# Domain semantic concept clusters for strong conceptual cosine similarity without external APIs
_CONCEPT_CLUSTERS: dict[str, tuple[str, ...]] = {
    "concept:hybrid_retrieval": (
        "hybrid",
        "bm25",
        "lexical",
        "semantic",
        "dense",
        "vector",
        "rrf",
        "reciprocal",
        "rank",
        "fusion",
        "retrieval",
        "retriever",
        "search",
        "scoring",
        "cosine",
        "embedding",
        "embeddings",
        "pgvector",
    ),
    "concept:guardrails_grounding": (
        "citation",
        "citations",
        "cite",
        "guardrail",
        "guardrails",
        "grounding",
        "grounded",
        "faithfulness",
        "hallucination",
        "hallucinations",
        "ungrounded",
        "verification",
        "verify",
        "provenance",
        "evidence",
    ),
    "concept:multi_agent_routing": (
        "router",
        "routing",
        "intent",
        "agent",
        "agents",
        "multi-agent",
        "multiagent",
        "pipeline",
        "orchestrator",
        "orchestration",
        "synthesizer",
        "synthesis",
        "trace",
        "traces",
    ),
    "concept:evaluation_quality": (
        "eval",
        "evaluation",
        "harness",
        "recall",
        "precision",
        "golden",
        "regression",
        "benchmark",
        "metric",
        "metrics",
        "gate",
        "latency",
        "threshold",
    ),
    "concept:distributed_consensus": (
        "raft",
        "consensus",
        "quorum",
        "leader",
        "follower",
        "replication",
        "split-brain",
        "fencing",
        "lease",
        "epoch",
        "partition",
        "failover",
        "ha",
    ),
    "concept:ledger_idempotency": (
        "ledger",
        "outbox",
        "idempotency",
        "idempotent",
        "append-only",
        "audit",
        "hash",
        "sha-256",
        "sha256",
        "immutability",
        "immutable",
        "transaction",
        "concurrency",
        "err-409",
        "conflict",
    ),
    "concept:telemetry_custody": (
        "aga-8",
        "api-mpms",
        "telemetry",
        "scada",
        "modbus",
        "custody",
        "compressibility",
        "flow",
        "meter",
        "pressure",
        "temperature",
        "pipeline-integrity",
    ),
}


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercased alphanumeric and hyphen/underscore technical tokens."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _stem_token(token: str) -> str:
    """Lightweight deterministic suffix stemmer for English technical terms."""
    if len(token) <= 4 or "-" in token or "_" in token or any(ch.isdigit() for ch in token):
        return token
    for suffix in ("ations", "ation", "ingly", "ments", "ment", "ities", "ity", "ings", "ing", "edly", "ness", "ives", "ive", "ions", "ion", "ies", "es", "ed", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if suffix == "ies":
                return stem + "y"
            return stem
    return token


def _expand_tokens_for_indexing(tokens: Sequence[str]) -> list[str]:
    """Expand compound tokens (e.g., err-409 -> err, 409) and stems for robust matching."""
    expanded: list[str] = []
    for tok in tokens:
        expanded.append(tok)
        stem = _stem_token(tok)
        if stem != tok:
            expanded.append(stem)
        if "-" in tok or "_" in tok:
            parts = [p for p in re.split(r"[-_]+", tok) if p]
            expanded.extend(parts)
    return expanded


def _hash_feature_to_bucket(feature: str, dim: int, salt: str = "") -> tuple[int, float]:
    """Map a string feature deterministically to (dimension_index, sign)."""
    digest = hashlib.blake2s(f"{salt}:{feature}".encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, byteorder="big", signed=False)
    bucket = val % dim
    sign = 1.0 if ((val >> 32) & 1) == 0 else -1.0
    return bucket, sign


def compute_deterministic_embedding(text: str, dim: int = 64) -> tuple[float, ...]:
    """Compute a deterministic L2-normalized dense vector of dimension `dim`."""
    if dim <= 0:
        raise ValueError("Embedding dimension `dim` must be a positive integer.")

    raw_tokens = tokenize(text)
    if not raw_tokens:
        return tuple(0.0 for _ in range(dim))

    content_tokens = [t for t in raw_tokens if t not in _STOPWORDS]
    if not content_tokens:
        content_tokens = raw_tokens

    vec = [0.0] * dim
    features: list[tuple[str, float]] = []

    token_set = set(content_tokens)
    stemmed_set = {_stem_token(t) for t in content_tokens}

    for tok in content_tokens:
        features.append((f"tok:{tok}", 2.2))
        stem = _stem_token(tok)
        features.append((f"stem:{stem}", 1.8))
        if len(stem) >= 4:
            features.append((f"pref4:{stem[:4]}", 0.9))
        if "-" in tok or "_" in tok:
            for part in re.split(r"[-_]+", tok):
                if part:
                    features.append((f"sub:{part}", 1.4))
        # Character n-grams for morphological resilience
        padded = f"<{stem}>"
        for n in (3, 4):
            if len(padded) >= n:
                for i in range(len(padded) - n + 1):
                    features.append((f"ng{n}:{padded[i : i + n]}", 0.35))

    # Bigram features for phrase sensitivity
    for i in range(len(content_tokens) - 1):
        s1 = _stem_token(content_tokens[i])
        s2 = _stem_token(content_tokens[i + 1])
        features.append((f"bi:{s1}_{s2}", 1.1))

    # Concept cluster projection for semantic alignment
    for cluster_name, keywords in _CONCEPT_CLUSTERS.items():
        hits = sum(1 for kw in keywords if kw in token_set or _stem_token(kw) in stemmed_set)
        if hits > 0:
            weight = 1.75 * min(2.5, 1.0 + 0.4 * (hits - 1))
            features.append((cluster_name, weight))

    for feat, weight in features:
        b1, s1 = _hash_feature_to_bucket(feat, dim, salt="p1")
        b2, s2 = _hash_feature_to_bucket(feat, dim, salt="p2")
        vec[b1] += s1 * weight
        vec[b2] += s2 * (weight * 0.5)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        unit = [0.0] * dim
        unit[0] = 1.0
        return tuple(unit)

    return tuple(round(v / norm, 8) for v in vec)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors in [-1.0, 1.0]."""
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch in cosine_similarity: {len(vec_a)} != {len(vec_b)}"
        )
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, round(sim, 8)))


class BM25Scorer:
    """Okapi BM25 lexical scorer with k1=1.5 and b=0.75."""

    def __init__(
        self,
        corpus: Sequence[DocumentChunk] | float | None = None,
        b_positional: float | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if isinstance(corpus, (int, float)):
            self.k1 = float(corpus)
            self.b = float(b_positional) if b_positional is not None else b
            self._corpus: tuple[DocumentChunk, ...] = ()
        else:
            self.k1 = float(k1)
            self.b = float(b)
            self._corpus = tuple(corpus) if corpus is not None else ()

    @staticmethod
    def _chunk_tokens(chunk: DocumentChunk) -> list[str]:
        # Title, section, and tags are included with slight field boost
        header = f"{chunk.title} {chunk.title} {chunk.section} {' '.join(chunk.tags)} {' '.join(chunk.tags)}"
        body = f"{header} {chunk.content}"
        return _expand_tokens_for_indexing(tokenize(body))

    def score_chunks(
        self,
        query: str,
        chunks: Sequence[DocumentChunk] | None = None,
    ) -> list[float]:
        """Compute BM25 scores for each chunk in `chunks` against `query`."""
        target_chunks = tuple(chunks) if chunks is not None else self._corpus
        if not target_chunks:
            return []

        raw_query_tokens = tokenize(query)
        query_tokens = [t for t in _expand_tokens_for_indexing(raw_query_tokens) if t not in _STOPWORDS]
        if not query_tokens:
            query_tokens = _expand_tokens_for_indexing(raw_query_tokens)
        if not query_tokens:
            return [0.0 for _ in target_chunks]

        doc_tokens_list = [self._chunk_tokens(chunk) for chunk in target_chunks]
        doc_lengths = [len(tokens) for tokens in doc_tokens_list]
        num_docs = len(target_chunks)
        avgdl = sum(doc_lengths) / num_docs if num_docs > 0 else 1.0
        if avgdl <= 0:
            avgdl = 1.0

        doc_Counters = [Counter(tokens) for tokens in doc_tokens_list]
        unique_query_terms = list(dict.fromkeys(query_tokens))

        # Document frequency per query term
        df_map: dict[str, int] = {}
        for term in unique_query_terms:
            df_map[term] = sum(1 for counts in doc_Counters if counts.get(term, 0) > 0)

        scores: list[float] = []
        for idx, counts in enumerate(doc_Counters):
            dl = doc_lengths[idx]
            score = 0.0
            norm_factor = self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
            for term in query_tokens:
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                df = df_map.get(term, 0)
                idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
                denom = tf + norm_factor
                score += idf * ((tf * (self.k1 + 1.0)) / denom)
            scores.append(round(score, 8))

        return scores

    def score_all(
        self,
        query: str,
        chunks: Sequence[DocumentChunk] | None = None,
    ) -> list[float]:
        """Alias for score_chunks."""
        return self.score_chunks(query, chunks)

    def score(
        self,
        query: str,
        chunks: Sequence[DocumentChunk] | None = None,
    ) -> list[float]:
        """Alias for score_chunks."""
        return self.score_chunks(query, chunks)


class HybridSearchEngine:
    """Combines BM25 lexical scoring and Dense Vector Cosine Similarity via Weighted RRF."""

    MAX_QUERY_LENGTH = 2000

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        k1: float = 1.5,
        b: float = 0.75,
        embedding_dim: int = 64,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer.")
        self.rrf_k = rrf_k
        self.embedding_dim = embedding_dim
        self.bm25_scorer = BM25Scorer(k1=k1, b=b)

    def search(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
        *,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Rank chunks using BM25 + Vector Similarity + Weighted Reciprocal Rank Fusion."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("Query must be a non-empty string.")
        if len(query.strip()) > self.MAX_QUERY_LENGTH:
            raise InvalidQueryError(
                f"Query length {len(query.strip())} exceeds maximum of {self.MAX_QUERY_LENGTH} characters."
            )
        if top_k <= 0:
            raise InvalidQueryError("top_k must be a positive integer.")
        if not chunks:
            raise EmptyRetrievalError("Cannot execute hybrid search against an empty document corpus.")

        k_const = rrf_k if rrf_k is not None else self.rrf_k
        bm25_scores = self.bm25_scorer.score_chunks(query, chunks)
        query_vec = compute_deterministic_embedding(query, dim=self.embedding_dim)

        vector_scores: list[float] = []
        for chunk in chunks:
            if chunk.embedding and len(chunk.embedding) == self.embedding_dim:
                chunk_vec = chunk.embedding
            else:
                rich_text = f"{chunk.title} {chunk.section} {' '.join(chunk.tags)} {chunk.content}"
                chunk_vec = compute_deterministic_embedding(rich_text, dim=self.embedding_dim)
            vector_scores.append(cosine_similarity(query_vec, chunk_vec))

        indices = list(range(len(chunks)))
        bm25_order = sorted(
            indices,
            key=lambda i: (-bm25_scores[i], -vector_scores[i], chunks[i].chunk_id),
        )
        vector_order = sorted(
            indices,
            key=lambda i: (-vector_scores[i], -bm25_scores[i], chunks[i].chunk_id),
        )

        bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_order)}
        vector_ranks = {idx: rank + 1 for rank, idx in enumerate(vector_order)}

        candidates: list[RetrievalCandidate] = []
        for idx, chunk in enumerate(chunks):
            b_rank = bm25_ranks[idx]
            v_rank = vector_ranks[idx]
            rrf = (bm25_weight / (k_const + b_rank)) + (vector_weight / (k_const + v_rank))
            candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    bm25_score=round(bm25_scores[idx], 6),
                    vector_score=round(vector_scores[idx], 6),
                    bm25_rank=b_rank,
                    vector_rank=v_rank,
                    rrf_score=round(rrf, 8),
                )
            )

        candidates.sort(
            key=lambda c: (
                -c.rrf_score,
                -(c.bm25_score * bm25_weight + c.vector_score * vector_weight),
                c.chunk.chunk_id,
            )
        )
        return tuple(candidates[:top_k])

    def retrieve(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
        *,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Alias for search()."""
        return self.search(
            query,
            chunks,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rrf_k=rrf_k,
        )
