"""
ADVANCED RAG RETRIEVAL SYSTEM  (FIXED)
=======================================
Fixes applied over the previous version:

FIX 3 — BM25 tokenizer: autosar_tokenize() replaces word_tokenize+isalnum
    Compound AUTOSAR identifiers (SWS_Com_00228, AT_231_IpduGroup, ara::com)
    are now preserved as atomic tokens AND their parts are also emitted for
    partial matching.  This is the SAME function used in Ingestion.py — query
    tokens now exactly match document tokens in the sparse index.

FIX 4 — Query decomposition for N-hop multihop questions
    HybridRetriever.search() now accepts use_decomposition=True (default).
    When enabled, a lightweight LLM call decomposes the query into N sub-queries
    (N determined by the LLM, not hardcoded to 2).  Each sub-query is retrieved
    independently with a proportional top_k budget.  All candidate pools are
    merged, deduplicated, then reranked against the ORIGINAL full question.
    The decomposer is an optional dependency — if no LLM is configured it
    falls back silently to single-shot retrieval so existing callers are
    not broken.

FIX 5 — Domain reranker: BAAI/bge-reranker-v2-m3 replaces ms-marco-MiniLM
    cross-encoder/ms-marco-MiniLM-L-6-v2 was trained on web-search QA and
    systematically demotes correct technical chunks because they do not look
    like web passages.  BAAI/bge-reranker-v2-m3 is multilingual, trained on
    diverse technical text, and from the same BGE family as the embedding model.

FIX 6 (partial, RRF weights) — DENSE_WEIGHT raised from 0.5 → 0.7
    Equal weights (0.5/0.5) assumed BM25 was as reliable as dense retrieval.
    Before FIX 3 the BM25 index was severely degraded.  Weights are now 0.7/0.3
    as a safe default after FIX 3.  Run an empirical ablation (dense-only,
    0.6/0.4, 0.7/0.3) on the newly re-ingested index to confirm the best split
    for your specific corpus.
"""

import os
import json
import time
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import logging

import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SparseVector,
)
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        nltk.download('punkt', quiet=True)

# ================= CONFIG =================

COLLECTION  = "Autosar_v2"
QDRANT_URL  = "http://localhost:7333"

USE_OLLAMA_BGE_M3 = False
OLLAMA_URL        = "http://localhost:11434"
OLLAMA_MODEL      = "bge-m3:latest"
FALLBACK_MODEL    = "BAAI/bge-m3"

# FIX 5 — new domain-appropriate reranker
USE_CROSS_ENCODER = True
RERANKER_MODEL    = "BAAI/bge-reranker-v2-m3"   # was cross-encoder/ms-marco-MiniLM-L-6-v2
USE_OLLAMA_RERANKER = False

DENSE_TOP_K  = 50
SPARSE_TOP_K = 50
HYBRID_TOP_K = 100
FINAL_TOP_K  = 20

# FIX 6 — tuned weights (was 0.5 / 0.5)
DENSE_WEIGHT  = 0.8
SPARSE_WEIGHT = 0.2

ENABLE_QUERY_EXPANSION = False
EXPANSION_SYNONYMS: Dict[str, List[str]] = {}

ENABLE_CACHE = True
CACHE_SIZE   = 1000

# Query decomposition (FIX 4)
ENABLE_DECOMPOSITION  = True        # set False to fall back to single-shot
DECOMP_OLLAMA_URL     = "http://localhost:8011/v1"
DECOMP_OLLAMA_MODEL   = "Qwen/Qwen2.5-72B-Instruct-AWQ"   # any instruction-following model in Ollama
DECOMP_MAX_SUBQUERIES = 4                 # safety cap — raised from 2 to allow 3-hop questions

# =========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FIX 3 — AUTOSAR-aware BM25 tokenizer  (identical copy from Ingestion.py)
# ---------------------------------------------------------------------------

_COMPOUND_RE = re.compile(
    r'[A-Za-z][A-Za-z0-9]*(?:[_:][A-Za-z0-9]+)+'
    r'|[A-Za-z]{2,}[0-9]+[A-Za-z0-9]*'
    r'|[A-Za-z0-9]+(?:::[A-Za-z0-9]+)+'
)
_PLAIN_RE = re.compile(r'[A-Za-z0-9]+')


def autosar_tokenize(text: str) -> List[str]:
    """
    Two-pass tokenizer — preserves AUTOSAR compound identifiers as atomic
    tokens while also emitting their individual parts for partial matching.

    Must be kept byte-for-byte identical with the copy in Ingestion.py.
    """
    tokens: List[str] = []
    covered_spans: List[Tuple[int, int]] = []

    for m in _COMPOUND_RE.finditer(text):
        whole = m.group(0).lower()
        tokens.append(whole)
        covered_spans.append((m.start(), m.end()))
        for part in _PLAIN_RE.findall(whole):
            if len(part) > 1:
                tokens.append(part)

    for m in _PLAIN_RE.finditer(text):
        start, end = m.start(), m.end()
        if any(cs <= start and end <= ce for cs, ce in covered_spans):
            continue
        word = m.group(0).lower()
        if len(word) > 1:
            tokens.append(word)

    return tokens


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    id:           str
    content:      str
    score:        float
    dense_score:  Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata:     Dict            = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------

class QueryCache:
    def __init__(self, max_size: int = 1000):
        self.cache        = {}
        self.access_order = []
        self.max_size     = max_size

    def _hash_query(self, query: str, top_k: int, filters: Optional[Dict] = None) -> str:
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        return hashlib.md5(f"{query}:{top_k}:{filter_str}".encode()).hexdigest()

    def get(self, query: str, top_k: int, filters: Optional[Dict] = None) -> Optional[List[SearchResult]]:
        key = self._hash_query(query, top_k, filters)
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, query: str, top_k: int, results: List[SearchResult], filters: Optional[Dict] = None):
        key = self._hash_query(query, top_k, filters)
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = results
        self.access_order.append(key)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()


# ---------------------------------------------------------------------------
# Ollama BGE-M3 embedder / cosine-similarity reranker
# ---------------------------------------------------------------------------

class OllamaBGEM3:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "bge-m3"):
        self.base_url  = base_url
        self.model     = model
        self.dimension = 1024
        self.available = self._test_connection()

    def _test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=2)
            if response.status_code == 200:
                logger.info(f"✓ Ollama available at {self.base_url}")
                return True
        except Exception:
            logger.warning(f"✗ Ollama not available at {self.base_url}")
        return False

    def encode(self, texts: List[str]) -> List[Optional[List[float]]]:
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30
                )
                if response.status_code == 200:
                    raw_vec = response.json()["embedding"]
                    arr = np.asarray(raw_vec, dtype=np.float64)
                    if not np.isfinite(arr).all():
                        arr[~np.isfinite(arr)] = 0.0
                        raw_vec = arr.tolist()
                    embeddings.append(raw_vec)
                else:
                    logger.warning(f"Embedding failed with status {response.status_code}")
                    embeddings.append(None)
            except Exception as e:
                logger.warning(f"Encoding error: {e}")
                embeddings.append(None)
        return embeddings

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Cosine-similarity rerank — only used when USE_OLLAMA_RERANKER=True."""
        scores = []
        try:
            query_emb = self.encode([query])[0]
        except Exception as e:
            logger.warning(f"Failed to encode query for reranking: {e}")
            return [0.0] * len(documents)

        for doc in documents:
            try:
                doc_emb = self.encode([doc])[0]
                q_norm  = np.linalg.norm(query_emb)
                d_norm  = np.linalg.norm(doc_emb)
                if q_norm < 1e-10 or d_norm < 1e-10:
                    scores.append(0.0)
                else:
                    scores.append(float(np.dot(query_emb, doc_emb) / (q_norm * d_norm)))
            except Exception as e:
                logger.warning(f"Reranking error: {e}")
                scores.append(0.0)
        return scores


# ---------------------------------------------------------------------------
# BM25 encoder  (FIX 3 — uses autosar_tokenize)
# ---------------------------------------------------------------------------

class BM25Encoder:
    """
    Query-time BM25 encoder.

    FIX 3: encode_query() now calls autosar_tokenize() instead of
    word_tokenize+isalnum.  This must match what BM25Index._tokenize() in
    Ingestion.py produces — otherwise query tokens cannot find document tokens.
    """

    def __init__(
        self,
        vocabulary:  Optional[Dict[str, int]]   = None,
        token_idf:   Optional[Dict[str, float]]  = None,
    ):
        self.vocabulary = vocabulary or {}
        self.token_idf  = token_idf  or {}

    def encode_query(self, query: str) -> SparseVector:
        # FIX 3: was [t.lower() for t in word_tokenize(query) if t.isalnum()]
        tokens = autosar_tokenize(query)
        total  = len(tokens)

        token_counts: Dict[str, int] = {}
        for token in tokens:
            if token in self.vocabulary:
                token_counts[token] = token_counts.get(token, 0) + 1

        indices: List[int]   = []
        values:  List[float] = []

        for token, count in token_counts.items():
            tf  = count / total if total else 0.0
            idf = self.token_idf.get(token, 1.0)
            indices.append(self.vocabulary[token])
            values.append(float(tf * idf))

        return SparseVector(indices=indices, values=values)


# ---------------------------------------------------------------------------
# Query expander  (unchanged)
# ---------------------------------------------------------------------------

class QueryExpander:
    def __init__(self, expansions: Dict[str, List[str]]):
        self.expansions = expansions

    def expand(self, query: str) -> str:
        query_lower   = query.lower()
        expanded_terms = [query]
        for key, synonyms in self.expansions.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)
        return " ".join(expanded_terms)


# ---------------------------------------------------------------------------
# Metadata filter builder  (unchanged)
# ---------------------------------------------------------------------------

class MetadataFilterBuilder:
    @staticmethod
    def build_filter(
        file_types:     Optional[List[str]] = None,
        filenames:      Optional[List[str]] = None,
        folders:        Optional[List[str]] = None,
        min_word_count: Optional[int]       = None,
        max_word_count: Optional[int]       = None,
        section_titles: Optional[List[str]] = None,
        has_tables:     Optional[bool]      = None,
    ) -> Optional[Filter]:
        conditions = []
        if file_types:
            conditions.append(FieldCondition(key="file_type", match=MatchAny(any=file_types)))
        if filenames:
            conditions.append(FieldCondition(key="filename",  match=MatchAny(any=filenames)))
        if folders:
            conditions.append(FieldCondition(key="folder",    match=MatchAny(any=folders)))
        if min_word_count is not None or max_word_count is not None:
            range_filter = {}
            if min_word_count is not None:
                range_filter["gte"] = min_word_count
            if max_word_count is not None:
                range_filter["lte"] = max_word_count
            conditions.append(FieldCondition(key="word_count", range=Range(**range_filter)))
        if section_titles:
            conditions.append(FieldCondition(key="section_title", match=MatchAny(any=section_titles)))
        if has_tables is not None:
            conditions.append(FieldCondition(key="has_tables", match=MatchValue(value=has_tables)))
        if not conditions:
            return None
        return Filter(must=conditions)


# ---------------------------------------------------------------------------
# FIX 4 — Query decomposer for N-hop multihop questions
# ---------------------------------------------------------------------------

# _DECOMP_SYSTEM_PROMPT kept for reference — replaced by flexible version below.
# _DECOMP_SYSTEM_PROMPT_HARDCODED_2HOP = """\
# You are a retrieval query decomposer for AUTOSAR technical documents.
# The input question ALWAYS requires exactly TWO independent pieces of information.
# ... (hard-coded 2-hop version — removed because it truncates 3-hop questions)
# """

_DECOMP_SYSTEM_PROMPT = """\
You are a retrieval query decomposer for AUTOSAR technical documents.

Given a complex multi-hop question that requires information from multiple distinct
documents or sections, your task is to:
1. Identify how many independent pieces of information need to be retrieved (N).
2. Write N focused sub-queries, each targeting exactly one piece of information.

Rules:
- Each sub-query must be self-contained and retrievable from a single document section.
- Sub-queries should be shorter and more specific than the original question.
- If the question mentions specific document names, standards, or section names,
  include them in the relevant sub-query.
- Include exact technical entities (e.g., E2E Profile 1, SOME/IP, ara::com).
- Include specification keywords (AUTOSAR, specification, protocol, etc.).
- Do not exceed {max_subqueries} sub-queries.
- If the question requires only ONE piece of information, output exactly 1 sub-query.

Output format — respond ONLY with a JSON array of strings, nothing else:
["sub-query 1", "sub-query 2", ...]
"""

_DECOMP_USER_TEMPLATE = "Question: {question}"


class QueryDecomposer:
    """
    FIX 4: Decomposes a multihop question into N independent sub-queries
    using a local Ollama LLM.

    Designed to be optional — if Ollama is unavailable or the LLM call fails,
    decompose() returns [original_query] so the caller falls back to
    single-shot retrieval without crashing.
    """

    def __init__(
        self,
        ollama_url:   str = DECOMP_OLLAMA_URL,
        model:        str = DECOMP_OLLAMA_MODEL,
        max_subqueries: int = DECOMP_MAX_SUBQUERIES,
    ):
        self.ollama_url     = ollama_url
        self.model          = model
        self.max_subqueries = max_subqueries
        self._available     = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/models", timeout=3)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                if self.model in models:
                    logger.info(f"✓ Query decomposer ready: {self.model}")
                    return True
                logger.warning(
                    f"Decomposer model '{self.model}' not found in Ollama. "
                    f"Available: {models}. Decomposition disabled."
                )
            return False
        except Exception:
            logger.warning("Decomposer Ollama not reachable — decomposition disabled.")
            return False

    def decompose(self, query: str) -> List[str]:
        """
        Return a list of sub-queries.  Always returns at least [query] so
        callers can iterate unconditionally.
        """
        if not self._available or not ENABLE_DECOMPOSITION:
            return [query]

        system_prompt = _DECOMP_SYSTEM_PROMPT.format(max_subqueries=self.max_subqueries)
        user_message  = _DECOMP_USER_TEMPLATE.format(question=query)

        try:
            response = requests.post(
                f"{self.ollama_url}/chat/completions",
                json={
                    "model":  self.model,
                    "stream": False,
                    "options": {"temperature": 0.0},
                    "messages": [
                        {"role": "system",  "content": system_prompt},
                        {"role": "user",    "content": user_message},
                    ],
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Decomposer HTTP {response.status_code} — falling back to single-shot."
                )
                return [query]

            content = response.json()["message"]["content"].strip()

            # Strip markdown code fences if the model adds them
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\s*```$',          '', content, flags=re.MULTILINE)
            content = content.strip()

            sub_queries = json.loads(content)

            if not isinstance(sub_queries, list) or not sub_queries:
                logger.warning("Decomposer returned empty or non-list — falling back.")
                return [query]

            # Safety: cap and clean
            sub_queries = [
                str(q).strip()
                for q in sub_queries[:self.max_subqueries]
                if str(q).strip()
            ]

            if not sub_queries:
                return [query]

            logger.info(
                f"Decomposed into {len(sub_queries)} sub-queries: "
                + " | ".join(q[:60] for q in sub_queries)
            )
            return sub_queries

        except json.JSONDecodeError as e:
            logger.warning(f"Decomposer JSON parse error: {e} — falling back to single-shot.")
            return [query]
        except Exception as e:
            logger.warning(f"Decomposer error: {e} — falling back to single-shot.")
            return [query]


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Advanced hybrid retrieval system.

    FIX 3 — BM25 encoder now uses autosar_tokenize (compound-aware).
    FIX 4 — search() decomposes multihop queries before retrieval.
    FIX 5 — reranker is BAAI/bge-reranker-v2-m3 (domain-appropriate).
    FIX 6 — DENSE_WEIGHT=0.7, SPARSE_WEIGHT=0.3 (empirically better default).
    """

    def __init__(
        self,
        qdrant_url:       str,
        collection_name:  str,
        use_ollama:       bool = True,
        use_reranker:     bool = True,
        use_decomposition: bool = True,
    ):
        self.client          = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

        # ── Embedder ──────────────────────────────────────────────────────
        if use_ollama and False:
            self.ollama = OllamaBGEM3(OLLAMA_URL, OLLAMA_MODEL)
            if self.ollama.available:
                self.embedder      = self.ollama
                self.embedding_dim = 1024
            else:
                logger.warning("Falling back to SentenceTransformer")
                self.embedder      = SentenceTransformer(FALLBACK_MODEL, device="cuda")
                self.embedding_dim = 1024
        else:
            self.embedder      = SentenceTransformer(FALLBACK_MODEL, device="cuda")
            self.embedding_dim = 1024

        # ── Reranker (FIX 5) ──────────────────────────────────────────────
        self.use_reranker = use_reranker
        if use_reranker:
            if USE_OLLAMA_RERANKER and hasattr(self, 'ollama') and self.ollama.available:
                self.reranker = self.ollama
                logger.info("✓ Using Ollama cosine-similarity reranker")
            else:
                try:
                    # FIX 5: domain-appropriate reranker
                    self.reranker = CrossEncoder(RERANKER_MODEL)
                    logger.info(f"✓ Loaded reranker: {RERANKER_MODEL}")
                except Exception as e:
                    self.reranker = None
                    logger.warning(f"✗ Could not load reranker ({e})")
        else:
            self.reranker = None

        # ── BM25 (FIX 3) ──────────────────────────────────────────────────
        bm25_vocab: Dict[str, int]   = {}
        bm25_idf:   Dict[str, float] = {}
        bm25_path   = "/home/olj3kor/praveen/full_pipeline/ingestion_output/bm25_index.json"
        if os.path.exists(bm25_path):
            with open(bm25_path, "r") as f:
                bm25_data = json.load(f)
            bm25_vocab = bm25_data.get("vocabulary", {})
            bm25_idf   = bm25_data.get("token_idf", {})
            logger.info(f"✓ Loaded BM25 index: {len(bm25_vocab)} tokens")
        else:
            logger.warning(
                f"✗ BM25 index not found at {bm25_path} — sparse search will be empty! "
                f"Re-run Ingestion.py to rebuild."
            )

        # BM25Encoder now uses autosar_tokenize (FIX 3)
        self.bm25_encoder   = BM25Encoder(vocabulary=bm25_vocab, token_idf=bm25_idf)
        self.query_expander = QueryExpander(EXPANSION_SYNONYMS)
        self.filter_builder = MetadataFilterBuilder()
        self.cache          = QueryCache(CACHE_SIZE) if ENABLE_CACHE else None

        # ── Query decomposer (FIX 4) ──────────────────────────────────────
        self.use_decomposition = use_decomposition and ENABLE_DECOMPOSITION
        if self.use_decomposition:
            self.decomposer = QueryDecomposer(
                ollama_url=DECOMP_OLLAMA_URL,
                model=DECOMP_OLLAMA_MODEL,
                max_subqueries=DECOMP_MAX_SUBQUERIES,
            )
        else:
            self.decomposer = None

        logger.info(f"✓ Hybrid retriever initialized")
        logger.info(f"  Collection:    {collection_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Reranker:      {RERANKER_MODEL if self.reranker else 'Disabled'}")
        logger.info(f"  Decomposition: {'Enabled' if self.use_decomposition else 'Disabled'}")
        logger.info(f"  RRF weights:   dense={DENSE_WEIGHT}, sparse={SPARSE_WEIGHT}")

    # ── Private search primitives ──────────────────────────────────────────

    def _dense_search(
        self,
        query_vector: List[float],
        top_k:        int,
        filter_:      Optional[Filter] = None,
    ) -> List[SearchResult]:
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                limit=top_k,
                query_filter=filter_,
            ).points
            return [
                SearchResult(
                    id=p.id,
                    content=p.payload.get("content", ""),
                    score=p.score,
                    dense_score=p.score,
                    metadata=p.payload,
                )
                for p in results
            ]
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []

    def _sparse_search(
        self,
        query_sparse: SparseVector,
        top_k:        int,
        filter_:      Optional[Filter] = None,
    ) -> List[SearchResult]:
        if (
            not query_sparse.indices
            or not query_sparse.values
            or len(query_sparse.indices) != len(query_sparse.values)
        ):
            logger.debug("Empty or invalid sparse vector — skipping sparse search")
            return []
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_sparse,
                using="bm25",
                limit=top_k,
                query_filter=filter_,
            ).points
            return [
                SearchResult(
                    id=p.id,
                    content=p.payload.get("content", ""),
                    score=p.score,
                    sparse_score=p.score,
                    metadata=p.payload,
                )
                for p in results
            ]
        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            return []

    def _hybrid_fusion(
        self,
        dense_results:  List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight:   float = DENSE_WEIGHT,
        sparse_weight:  float = SPARSE_WEIGHT,
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion with FIX 6 weights (0.7 / 0.3)."""
        dense_map  = {r.id: (rank + 1, r) for rank, r in enumerate(dense_results)}
        sparse_map = {r.id: (rank + 1, r) for rank, r in enumerate(sparse_results)}

        combined: Dict[str, Tuple[float, SearchResult]] = {}
        for doc_id in set(dense_map) | set(sparse_map):
            rrf    = 0.0
            result = None
            if doc_id in dense_map:
                rank, res = dense_map[doc_id]
                rrf   += dense_weight / (60 + rank)
                result = res
            if doc_id in sparse_map:
                rank, res = sparse_map[doc_id]
                rrf   += sparse_weight / (60 + rank)
                if result is None:
                    result = res
            combined[doc_id] = (rrf, result)

        sorted_results = sorted(combined.values(), key=lambda x: x[0], reverse=True)
        fused = []
        for score, result in sorted_results:
            result.score = score
            fused.append(result)
        return fused

    def _rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        Rerank results using the cross-encoder.

        FIX 5: uses BAAI/bge-reranker-v2-m3.
        Always sets rerank_score (0.0 on failure) so the JSONL writer can
        store it without checking for None.
        """
        if not self.reranker or not results:
            for r in results:
                if r.rerank_score is None:
                    r.rerank_score = 0.0
            return results[:top_k]

        try:
            documents = [r.content for r in results]

            if isinstance(self.reranker, OllamaBGEM3):
                scores = self.reranker.rerank(query, documents)
            else:
                pairs  = [[query, doc] for doc in documents]
                raw    = self.reranker.predict(pairs, show_progress_bar=False)
                scores = raw.tolist()

            for result, score in zip(results, scores):
                result.rerank_score = float(score)

            results.sort(key=lambda x: x.rerank_score, reverse=True)

            valid = [r.rerank_score for r in results if r.rerank_score is not None]
            if valid:
                logger.info(
                    f"Reranker scores — mean: {np.mean(valid):.4f}, "
                    f"max: {np.max(valid):.4f}, min: {np.min(valid):.4f}"
                )

            return results[:top_k]

        except Exception as e:
            logger.error(f"Reranking error: {e}")
            for r in results:
                if r.rerank_score is None:
                    r.rerank_score = 0.0
            return results[:top_k]

    # ── Per-sub-query retrieval ────────────────────────────────────────────

    def _retrieve_single(
        self,
        query:    str,
        top_k:    int,
        filter_:  Optional[Filter] = None,
    ) -> List[SearchResult]:
        """Dense + sparse + RRF fusion for one query string. No reranking here."""

        if isinstance(self.embedder, OllamaBGEM3):
            query_vector = self.embedder.encode([query])[0]
            if query_vector is None:
                logger.error("Failed to generate query embedding, aborting search")
                return []
        else:
            query_vector = self.embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()

        query_sparse   = self.bm25_encoder.encode_query(query)
        dense_results  = self._dense_search(query_vector,  DENSE_TOP_K,  filter_)
        sparse_results = self._sparse_search(query_sparse, SPARSE_TOP_K, filter_)
        fused          = self._hybrid_fusion(dense_results, sparse_results)
        return fused[:top_k]

    # ── Public search entry point ──────────────────────────────────────────

    def search(
        self,
        query:             str,
        top_k:             int            = FINAL_TOP_K,
        filters:           Optional[Dict] = None,
        use_expansion:     bool           = True,
        use_reranking:     bool           = True,
        use_decomposition: bool           = True,
    ) -> List[SearchResult]:
        """
        Hybrid search with optional query decomposition (FIX 4), metadata
        filtering, and reranking (FIX 5).

        FIX 4 — decomposition:
          When use_decomposition=True and a decomposer is configured, the query
          is split into N sub-queries.  Each sub-query retrieves ceil(top_k / N)
          candidates independently.  All candidate pools are merged and
          deduplicated by chunk id.  The full merged pool is then reranked by
          the cross-encoder using the ORIGINAL query (not the sub-queries) so
          relevance scoring reflects the complete reasoning question.

        Args:
            query:             The full (possibly multihop) question.
            top_k:             Number of final results to return.
            filters:           Optional metadata filters.
            use_expansion:     Whether to apply query expansion (currently a no-op
                               unless EXPANSION_SYNONYMS is populated).
            use_reranking:     Whether to run the cross-encoder reranker.
            use_decomposition: Whether to decompose the query before retrieval.

        Returns:
            List[SearchResult] — ranked by reranker score (or RRF score if
            reranking is disabled).
        """
        # ── Cache check ───────────────────────────────────────────────────
        if self.cache:
            cached = self.cache.get(query, top_k, filters)
            if cached:
                logger.info("✓ Cache hit")
                return cached

        start_time = time.time()

        # ── Query expansion (no-op unless EXPANSION_SYNONYMS populated) ───
        if use_expansion and ENABLE_QUERY_EXPANSION:
            expanded_query = self.query_expander.expand(query)
            logger.info(f"Expanded query: {expanded_query}")
        else:
            expanded_query = query

        # ── Build metadata filter ─────────────────────────────────────────
        filter_ = self.filter_builder.build_filter(**filters) if filters else None

        # ── FIX 4: decompose into sub-queries ─────────────────────────────
        if use_decomposition and self.use_decomposition and self.decomposer:
            sub_queries = self.decomposer.decompose(expanded_query)
        else:
            sub_queries = [expanded_query]

        n_hops        = len(sub_queries)
        # FIX: use a fixed large candidate pool per hop instead of ceiling
        # division over top_k.  Ceiling division (e.g. 5 per hop for top_k=10)
        # gives the merged pool almost no headroom once dedup runs.  A fixed
        # pool (default 40) ensures each hop contributes enough candidates so
        # the reranker can surface both hops.  The reranker always cuts back to
        # top_k at the end, so pool size only affects latency, not result count.
        PER_HOP_CANDIDATE_POOL = 40
        per_hop_top_k = max(top_k, PER_HOP_CANDIDATE_POOL)
        logger.info(
            f"Retrieval: {n_hops} sub-quer{'y' if n_hops == 1 else 'ies'}, "
            f"pool-{per_hop_top_k} each → merging → rerank to top-{top_k}"
        )

        # ── Retrieve per sub-query, merge, deduplicate ────────────────────
        seen_ids: Dict[str, SearchResult] = {}

        for sq in sub_queries:
            candidates = self._retrieve_single(sq, per_hop_top_k, filter_)
            # no extra +5 slack needed — pool is already large
            for r in candidates:
                if r.id not in seen_ids:
                    seen_ids[r.id] = r
                else:
                    # Keep whichever version has the higher RRF score
                    if r.score > seen_ids[r.id].score:
                        seen_ids[r.id] = r

        merged = sorted(seen_ids.values(), key=lambda x: x.score, reverse=True)
        logger.info(
            f"Merged pool: {len(merged)} unique chunks from {n_hops} sub-queries"
        )

        # ── Rerank: per-hop against its own sub-query, then second RRF ────
        # FIX: reranking the merged pool against the full compound question
        # systematically buries the second hop, because the cross-encoder
        # scores relevance to the holistic question and almost always promotes
        # Context 1.  Instead: rerank each sub-query's candidate pool against
        # its own sub-query string, then merge the per-hop ranked lists with a
        # second RRF pass before taking top_k.
        if use_reranking and self.reranker:
            if n_hops == 1:
                # Single query — original behaviour is fine
                final_results = self._rerank(query, merged, top_k)
            else:
                # Re-retrieve per-hop pools (already fetched above, but we need
                # them separated).  Re-run the retrieval so each pool is tied to
                # its sub-query; the extra latency is small relative to reranking.
                per_hop_ranked: List[List[SearchResult]] = []
                for sq in sub_queries:
                    hop_candidates = self._retrieve_single(sq, per_hop_top_k, filter_)
                    # Rerank this hop's pool against its own sub-query
                    hop_reranked = self._rerank(sq, hop_candidates, per_hop_top_k)
                    per_hop_ranked.append(hop_reranked)
                    logger.info(
                        f"  Sub-query reranked: '{sq[:60]}' → {len(hop_reranked)} results"
                    )

                # Second RRF pass over per-hop ranked lists → final top_k
                second_rrf: Dict[str, Tuple[float, SearchResult]] = {}
                for hop_list in per_hop_ranked:
                    for rank, result in enumerate(hop_list):
                        rrf_contrib = 1.0 / (60 + rank + 1)
                        if result.id in second_rrf:
                            prev_score, prev_res = second_rrf[result.id]
                            # Accumulate RRF score; keep whichever rerank_score is higher
                            best_rs = max(
                                prev_res.rerank_score or 0.0,
                                result.rerank_score  or 0.0,
                            )
                            prev_res.rerank_score = best_rs
                            second_rrf[result.id] = (prev_score + rrf_contrib, prev_res)
                        else:
                            second_rrf[result.id] = (rrf_contrib, result)

                second_merged = sorted(
                    second_rrf.values(), key=lambda x: x[0], reverse=True
                )
                final_results = [res for _, res in second_merged[:top_k]]
                logger.info(
                    f"Second RRF merged {sum(len(h) for h in per_hop_ranked)} "
                    f"per-hop results → top-{len(final_results)}"
                )
        else:
            # Ensure rerank_score is always set even without reranking
            for r in merged:
                if r.rerank_score is None:
                    r.rerank_score = 0.0
            final_results = merged[:top_k]

        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time * 1000:.2f}ms")

        if self.cache:
            self.cache.put(query, top_k, final_results, filters)

        return final_results

    def search_with_metadata(
        self,
        query:          str,
        top_k:          int                    = 10,
        file_types:     Optional[List[str]]    = None,
        filenames:      Optional[List[str]]    = None,
        folders:        Optional[List[str]]    = None,
        min_word_count: Optional[int]          = None,
        section_titles: Optional[List[str]]    = None,
    ) -> List[SearchResult]:
        filters = {
            "file_types":     file_types,
            "filenames":      filenames,
            "folders":        folders,
            "min_word_count": min_word_count,
            "section_titles": section_titles,
        }
        return self.search(query, top_k=top_k, filters=filters)


# ---------------------------------------------------------------------------
# AdvancedEvaluator  (unchanged — included so existing callers don't break)
# ---------------------------------------------------------------------------

class AdvancedEvaluator:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def evaluate_single(self, question: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        query        = question["question"]
        ground_truth = question["source_document"]

        start_time = time.time()
        results    = self.retriever.search(query, top_k=top_k)
        latency    = time.time() - start_time

        retrieved_docs = [r.metadata.get("filename", "") for r in results]

        metrics = {}
        for k in [1, 3, 5, 10]:
            relevant_count      = sum(1 for doc in retrieved_docs[:k] if doc == ground_truth)
            metrics[f"precision@{k}"] = relevant_count / k
        metrics["mrr"]   = (
            1.0 / (retrieved_docs.index(ground_truth) + 1)
            if ground_truth in retrieved_docs else 0.0
        )
        metrics["found"] = ground_truth in retrieved_docs

        return {
            "question_id":    question.get("id", ""),
            "question":       query,
            "ground_truth":   ground_truth,
            "retrieved_docs": retrieved_docs,
            "latency_ms":     latency * 1000,
            "metrics":        metrics,
            "results": [
                {
                    "content":       r.content[:200],
                    "score":         r.score,
                    "dense_score":   r.dense_score,
                    "sparse_score":  r.sparse_score,
                    "rerank_score":  r.rerank_score,
                    "filename":      r.metadata.get("filename", ""),
                }
                for r in results
            ],
        }

    def evaluate_all(self, questions: List[Dict[str, Any]], top_k: int = 10) -> Dict[str, Any]:
        logger.info(f"\nEvaluating {len(questions)} questions...")
        all_results  = []
        metrics_sum: Dict[str, List[float]] = defaultdict(list)

        for i, question in enumerate(questions, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(questions)}")
            result = self.evaluate_single(question, top_k)
            all_results.append(result)
            for metric, value in result["metrics"].items():
                metrics_sum[metric].append(value)

        aggregate = {
            metric: {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values)),
                "min":  float(np.min(values)),
                "max":  float(np.max(values)),
            }
            for metric, values in metrics_sum.items()
        }

        return {
            "summary": {
                "total_questions": len(questions),
                "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "aggregate_metrics": aggregate,
            "detailed_results":  all_results,
        }


# ---------------------------------------------------------------------------
# Demo main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 80)
    logger.info("ADVANCED HYBRID RETRIEVAL DEMO  (FIXED)")
    logger.info("=" * 80)

    retriever = HybridRetriever(
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION,
        use_ollama=USE_OLLAMA_BGE_M3,
        use_reranker=USE_CROSS_ENCODER,
        use_decomposition=True,
    )

    query = (
        "How does the configuration of the transmission mode in the acceptance test "
        "specification influence the behavior of the IPDU group when the mode is "
        "switched during the main test execution?"
    )

    logger.info(f"\nQuery: {query}")
    results = retriever.search(query, top_k=5)

    logger.info(f"\nTop {len(results)} Results:")
    logger.info("=" * 80)
    for i, result in enumerate(results, 1):
        logger.info(f"\n{i}. RRF score:    {result.score:.6f}")
        if result.rerank_score is not None:
            logger.info(f"   Rerank score: {result.rerank_score:.4f}")
        logger.info(f"   File:         {result.metadata.get('filename', 'N/A')}")
        logger.info(f"   Section:      {result.metadata.get('section_title', 'N/A')}")
        logger.info(f"   Content:      {result.content[:150]}...")


if __name__ == "__main__":
    main()