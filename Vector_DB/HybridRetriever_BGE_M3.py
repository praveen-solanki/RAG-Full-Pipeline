"""
HybridRetriever  —  Standard QD+RR hybrid retrieval
=====================================================

Architecture
------------
Single-hop query
    dense + sparse  →  RRF fusion  →  cross-encoder rerank (vs original query)  →  top-k

Multi-hop query  (QD+RR, ACL 2025 pattern)
    Step 1 – LLM decomposes query into N sub-queries
    Step 2 – For each sub-query (iteratively):
               a. Retrieve a candidate pool (dense + sparse + RRF)
               b. Feed top retrieved chunks as context to LLM for next sub-query
                  (IRCoT-style context carryover)
    Step 3 – Merge ALL per-hop candidate pools into one deduplicated set
    Step 4 – Single cross-encoder rerank of merged pool against the ORIGINAL query
              (not against sub-queries — this is the key fix)
    Step 5 – Enforce min-coverage: at least one chunk per hop survives in top-k

Key differences from the previous version
------------------------------------------
OLD: one-shot decomposition (blind to evidence), per-hop rerank vs sub-query,
     second RRF over per-hop ranked lists, double-retrieval in rerank path.
NEW: iterative decomposition (each hop sees prior evidence), single rerank vs
     original query on the merged pool, no second RRF, no double-retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

# Must be set before any CUDA context is created (before torch/sentence-transformers
# are imported). expandable_segments reduces fragmentation and avoids OOM on large
# rerank pools by allowing the allocator to grow/return segments on demand.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from nltk.tokenize import word_tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
    NamedVector,
    NamedSparseVector,
)
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)

def _setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure root logger with console + optional file handler.
    Call once from main(); importing modules get the root config automatically.
    """
    fmt     = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    root    = logging.getLogger()
    root.setLevel(logging.INFO)

    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info(f"Logging to file: {log_file}")

    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

COLLECTION     = "Dear_autosar"   # Changed: was "Autosar_v2"
QDRANT_URL     = "http://localhost:7333"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"

# Rerank mode — controls which scorer is used after the multi-hop merge step.
#
#   "cross_encoder"  (default)
#       Loads bge-reranker-v2-m3 as a separate CrossEncoder.  Higher quality
#       but costs an extra ~1.6 GB VRAM and is vulnerable to OOM on very large
#       merged pools.
#
#   "colbert"
#       Reuses the already-loaded BGEM3FlagModel (self.embedder) and scores via
#       colbert_score(query_vecs, doc_vecs).  No second model load; peak VRAM
#       stays flat.  Recommended when the GPU is under memory pressure.
#
# Set via HybridRetriever(rerank_mode="colbert") or --rerank-mode colbert CLI flag.
RERANK_MODE = "cross_encoder"

# Batch size used only when rerank_mode="colbert".  Lower to 16 on busier GPUs.
COLBERT_RERANK_BATCH = 32

# BM25_INDEX_PATH removed — BGE-M3 native sparse vectors are used instead.
# No external BM25 JSON file is needed or loaded.

# RRF constant (standard k=60)
RRF_K = 60

# Asymmetric pool sizes per hop (PAR²-RAG Coverage Anchor pattern)
# hop-1 uses broad pool for recall; hop-2+ use smaller targeted pool for precision
CANDIDATE_POOL_SINGLE_HOP = 100   # single-hop: balanced 100
CANDIDATE_POOL_HOP1       = 120   # hop-1: broad, maximise recall
CANDIDATE_POOL_HOP_N      = 80    # hop-2+: targeted, bridge entities give precision

# Kept for backward-compat references; maps to hop-1 default
CANDIDATE_POOL_PER_HOP = CANDIDATE_POOL_HOP1

# Final top-k returned to caller (overridden by search(top_k=…))
DEFAULT_TOP_K = 10

# Decomposition
DECOMP_BASE_URL    = "http://localhost:8011/v1"
DECOMP_MODEL       = "mistralai/Mistral-7B-Instruct-v0.3"
# DECOMP_MODEL       = "Qwen/Qwen2.5-32B-Instruct-AWQ"
DECOMP_MAX_HOPS    = 4
DECOMP_TEMPERATURE = 0.0

# Cache
ENABLE_CACHE = True
CACHE_SIZE   = 1000

# Min-coverage enforcement: for multi-hop, guarantee this many chunks
# per hop survive in the final top-k (if pool had results for that hop)
MIN_CHUNKS_PER_HOP = 1

# RRF fusion weights (used only in fallback path; primary retrieval uses Qdrant prefetch)
DENSE_WEIGHT  = 0.5
SPARSE_WEIGHT = 0.5

# HyDE pre-retrieval for analytical/conceptual single-hop queries
# When enabled, an LLM generates a hypothetical AUTOSAR paragraph and its
# dense embedding is averaged with the original query embedding before retrieval.
# Only dense embedding benefits from HyDE; sparse vector uses original query.
ENABLE_HYDE      = True
HYDE_BASE_URL    = DECOMP_BASE_URL
HYDE_MODEL       = DECOMP_MODEL
HYDE_TEMPERATURE = 0.3
HYDE_KEYWORDS    = frozenset([
    "how", "why", "influence", "relationship", "effect", "impact",
    "affect", "difference", "between", "compare", "explain", "describe",
    "mechanism", "process", "role", "purpose", "interaction",
])


# ──────────────────────────────────────────────────────────────────────────────
# AUTOSAR-aware BM25 tokenizer  (must match Ingestion.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

_COMPOUND_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[_:][A-Za-z0-9]+)+"
    r"|[A-Za-z]{2,}[0-9]+[A-Za-z0-9]*"
    r"|[A-Za-z0-9]+(?:::[A-Za-z0-9]+)+"
)
_PLAIN_RE = re.compile(r"[A-Za-z0-9]+")


def autosar_tokenize(text: str) -> List[str]:
    """
    Two-pass tokenizer.  Preserves compound AUTOSAR identifiers as atomic
    tokens (e.g. SWS_Com_00228, ara::com) and also emits individual parts
    for partial matching.  Must be kept identical with the copy in Ingestion.py.
    """
    tokens: List[str] = []
    covered: List[Tuple[int, int]] = []

    for m in _COMPOUND_RE.finditer(text):
        whole = m.group(0).lower()
        tokens.append(whole)
        covered.append((m.start(), m.end()))
        for part in _PLAIN_RE.findall(whole):
            if len(part) > 1:
                tokens.append(part)

    for m in _PLAIN_RE.finditer(text):
        s, e = m.start(), m.end()
        if any(cs <= s and e <= ce for cs, ce in covered):
            continue
        word = m.group(0).lower()
        if len(word) > 1:
            tokens.append(word)

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    id:           str
    content:      str
    score:        float                    # final score (rerank or RRF)
    parent_text:  Optional[str]  = None   # full section text (1600-token parent); use as LLM context
    dense_score:  Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rrf_score:    Optional[float] = None
    hop_index:    Optional[int]   = None   # which sub-query produced this chunk
    metadata:     Dict            = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Query cache
# ──────────────────────────────────────────────────────────────────────────────

class QueryCache:
    def __init__(self, max_size: int = CACHE_SIZE):
        self._store: Dict[str, List[SearchResult]] = {}
        self._order: List[str] = []
        self._max   = max_size

    def _key(self, query: str, top_k: int, filters: Optional[Dict]) -> str:
        s = json.dumps(filters, sort_keys=True) if filters else ""
        return hashlib.md5(f"{query}:{top_k}:{s}".encode()).hexdigest()

    def get(self, query: str, top_k: int, filters: Optional[Dict] = None) -> Optional[List[SearchResult]]:
        k = self._key(query, top_k, filters)
        if k in self._store:
            self._order.remove(k)
            self._order.append(k)
            return self._store[k]
        return None

    def put(self, query: str, top_k: int, results: List[SearchResult], filters: Optional[Dict] = None):
        k = self._key(query, top_k, filters)
        if len(self._store) >= self._max:
            oldest = self._order.pop(0)
            del self._store[oldest]
        self._store[k] = results
        self._order.append(k)

    def clear(self):
        self._store.clear()
        self._order.clear()


# ──────────────────────────────────────────────────────────────────────────────
# NOTE: BM25Encoder class REMOVED.
# Sparse vectors are now produced natively by BGEM3FlagModel.encode() via the
# lexical_weights output — same model used for dense and ColBERT vectors.
# No external BM25 JSON file is loaded or needed.
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Iterative query decomposer  (IRCoT-style context carryover)
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_FIRST_HOP = """\
You are a retrieval reasoning assistant for AUTOSAR technical documentation.

Given a complex question, write ONE focused reasoning sentence that targets the \
FIRST piece of information needed to answer it.

Rules:
- The sentence must be self-contained and embeddable as a retrieval query.
- Include exact technical identifiers (e.g. E2E Profile 1, SOME/IP, ara::com).
- Write as a declarative sentence or specific technical phrase — NOT a question.
  Good: "E2E Profile 1 uses a 16-bit CRC computed over the data bytes."
  Bad:  "What CRC does E2E Profile 1 use?"
- Keep it concise (one sentence, no more than 30 words).
- Output ONLY a JSON object: {{"sub_query": "...", "is_final": false}}
  Set "is_final": true ONLY if this is unmistakably a single-fact lookup \
(e.g. "What is the value of constant X?").  For analytical / multi-part \
questions, always set is_final=false — hop-1 retrieval is mandatory.
"""

_SYSTEM_NEXT_HOP = """\
You are a retrieval reasoning assistant for AUTOSAR technical documentation.

You have already retrieved evidence for earlier reasoning steps (shown below). \
Based on that evidence, write ONE reasoning sentence that bridges from what you \
know to the NEXT piece of information still needed.

Rules:
- Your sentence MUST incorporate specific entities, values, or identifiers found \
in the retrieved evidence (bridge-entity carryover).
  Example: if hop-1 found "E2E Profile 1 uses 16-bit CRC", your next sentence \
might be "The 16-bit CRC computation covers the entire data area including the \
E2E header byte at offset 0."
- Write as a declarative sentence or targeted technical claim — NOT a question.
- Keep it concise (one sentence, no more than 40 words).
- Output ONLY a JSON object: {{"sub_query": "...", "is_final": false}}
  Set "is_final": true if the evidence already gathered is sufficient to fully \
answer the original question (no more retrieval needed).
"""

_USER_FIRST_HOP = "Original question: {question}"

_USER_NEXT_HOP  = """\
Original question: {question}

Retrieved evidence so far:
{evidence}

Generate the next sub-query."""


class IterativeDecomposer:
    """
    Generates sub-queries one at a time, feeding prior retrieved evidence
    back to the LLM before generating the next sub-query.  This is the
    IRCoT / FAIR-RAG context-carryover pattern.

    If the LLM is unavailable the decomposer returns [original_query] so
    callers degrade gracefully to single-shot retrieval.
    """

    def __init__(
        self,
        base_url:    str = DECOMP_BASE_URL,
        model:       str = DECOMP_MODEL,
        max_hops:    int = DECOMP_MAX_HOPS,
        temperature: float = DECOMP_TEMPERATURE,
    ):
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.max_hops    = max_hops
        self.temperature = temperature
        self._available  = self._check()

    # ── availability ──────────────────────────────────────────────────────

    def _check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=3)
            if r.status_code == 200:
                names = [m.get("id", m.get("name", "")) for m in r.json().get("data", r.json().get("models", []))]
                if any(self.model in n for n in names):
                    logger.info(f"✓ Decomposer ready: {self.model}")
                    return True
                logger.warning(f"Decomposer model '{self.model}' not found. Available: {names}")
        except Exception as e:
            logger.warning(f"Decomposer not reachable ({e}) — will use single-shot fallback.")
        return False

    # ── single LLM call ───────────────────────────────────────────────────

    def _call(self, system: str, user: str) -> Optional[Dict]:
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model":  self.model,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                },
                timeout=60,
            )
            if resp.status_code != 200:
                logger.warning(f"Decomposer HTTP {resp.status_code}")
                return None

            raw = resp.json()
            # Support both OpenAI-compatible and Ollama response shapes
            content = (
                raw.get("choices", [{}])[0].get("message", {}).get("content")
                or raw.get("message", {}).get("content", "")
            ).strip()

            # Strip markdown fences
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
            content = re.sub(r"\s*```$",           "", content, flags=re.MULTILINE).strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Decomposer JSON parse error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Decomposer call failed: {e}")
            return None

    # ── public interface ───────────────────────────────────────────────────

    def first_sub_query(self, question: str) -> Tuple[str, bool]:
        """
        Returns (sub_query, is_final).
        is_final=True means single-hop — no further decomposition needed.
        """
        if not self._available:
            return question, True

        result = self._call(
            system=_SYSTEM_FIRST_HOP,
            user=_USER_FIRST_HOP.format(question=question),
        )
        if not result or "sub_query" not in result:
            logger.warning("Decomposer returned bad first hop — falling back.")
            return question, True

        sq       = str(result["sub_query"]).strip() or question
        is_final = bool(result.get("is_final", False))
        logger.info(f"  Hop 1 sub-query: «{sq[:80]}»  is_final={is_final}")
        return sq, is_final

    def next_sub_query(self, question: str, evidence_chunks: List[str], hop_number: int = 1) -> Tuple[str, bool]:
        """
        Returns (sub_query, is_final).
        is_final=True means enough evidence has been gathered.
        evidence_chunks: list of content strings from all prior hops' top results.
        hop_number: 1-indexed hop being generated (used for CoT step labels).

        IRCoT-style structured CoT carryover: formats evidence as numbered steps
        so bridge entities survive truncation and seed the next query precisely.
        """
        if not self._available:
            return "", True

        # Structured CoT chain format (BridgeRAG / IRCoT pattern):
        # 5 chunks × 900 chars = up to 4500 chars of evidence.
        # Label each step to make bridge entities explicit.
        evidence_parts = []
        for i, c in enumerate(evidence_chunks[:5]):
            chunk_preview = c[:900]
            evidence_parts.append(f"Step {i+1} evidence:\n{chunk_preview}")
        evidence_text = "\n\n---\n\n".join(evidence_parts)

        result = self._call(
            system=_SYSTEM_NEXT_HOP,
            user=_USER_NEXT_HOP.format(question=question, evidence=evidence_text),
        )
        if not result or "sub_query" not in result:
            logger.warning("Decomposer returned bad next hop — stopping.")
            return "", True

        sq       = str(result["sub_query"]).strip()
        is_final = bool(result.get("is_final", False))
        logger.info(f"  Next sub-query (hop {hop_number+1}): «{sq[:80]}»  is_final={is_final}")
        return sq, is_final


# ──────────────────────────────────────────────────────────────────────────────
# Metadata filter builder  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

class MetadataFilterBuilder:
    @staticmethod
    def build(
        file_types:     Optional[List[str]] = None,
        filenames:      Optional[List[str]] = None,
        folders:        Optional[List[str]] = None,
        min_word_count: Optional[int]       = None,
        max_word_count: Optional[int]       = None,
        section_titles: Optional[List[str]] = None,
        has_tables:     Optional[bool]      = None,
    ) -> Optional[Filter]:
        conds = []
        if file_types:
            conds.append(FieldCondition(key="file_type",  match=MatchAny(any=file_types)))
        if filenames:
            conds.append(FieldCondition(key="filename",   match=MatchAny(any=filenames)))
        if folders:
            conds.append(FieldCondition(key="folder",     match=MatchAny(any=folders)))
        if section_titles:
            conds.append(FieldCondition(key="section_title", match=MatchAny(any=section_titles)))
        if has_tables is not None:
            conds.append(FieldCondition(key="has_tables", match=MatchValue(value=has_tables)))
        rng: Dict = {}
        if min_word_count is not None:
            rng["gte"] = min_word_count
        if max_word_count is not None:
            rng["lte"] = max_word_count
        if rng:
            conds.append(FieldCondition(key="word_count", range=Range(**rng)))
        return Filter(must=conds) if conds else None


# ──────────────────────────────────────────────────────────────────────────────
# Main retriever
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Standard hybrid retriever implementing the QD+RR pattern.

    Single-hop:
        RRF(dense, sparse)  →  cross-encoder rerank vs original query  →  top-k

    Multi-hop (iterative QD+RR):
        for each hop:
            sub_query = LLM(original_question, prior_evidence)   # context carryover
            pool_i    = RRF(dense(sub_query), sparse(sub_query))
        merged_pool = deduplicate(all pools)
        final       = cross_encoder_rerank(original_query, merged_pool)[:top_k]
        enforce min-coverage per hop
    """

    def __init__(
        self,
        qdrant_url:        str  = QDRANT_URL,
        collection_name:   str  = COLLECTION,
        use_reranker:      bool = True,
        use_decomposition: bool = True,
        device:            str  = "cuda",
        rerank_mode:       str  = RERANK_MODE,
    ):
        self.collection  = collection_name
        self.client      = QdrantClient(url=qdrant_url)

        # ── BGE-M3 Embedder (dense + sparse + ColBERT in one model) ──────
        # Single BGEM3FlagModel.encode() call produces all three signal types.
        # This replaces the old SentenceTransformer (dense) + BM25Encoder (sparse)
        # two-model setup and ensures query/document vector spaces are aligned.
        logger.info(f"Loading BGE-M3 model: {EMBEDDING_MODEL}")
        self.embedder = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True, device=device)
        logger.info("✓ BGE-M3 embedder ready (dense + sparse + ColBERT)")

        # ── Optional: verify _model_fingerprint in collection payload ─────
        # Ingestion.py writes a _model_fingerprint field; warn if mismatch.
        self._verify_model_fingerprint(collection_name)

        # ── Reranker ──────────────────────────────────────────────────────
        # rerank_mode="cross_encoder" (default): load bge-reranker-v2-m3 as a
        #   separate CrossEncoder — best quality, uses ~1.6 GB extra VRAM.
        # rerank_mode="colbert": reuse self.embedder's ColBERT weights via
        #   colbert_score() — no second model load, lower peak VRAM.
        self.use_reranker = use_reranker
        self.rerank_mode  = rerank_mode.lower()
        self.reranker: Optional[CrossEncoder] = None

        if use_reranker:
            if self.rerank_mode == "cross_encoder":
                try:
                    logger.info(f"Loading reranker: {RERANKER_MODEL}")
                    self.reranker = CrossEncoder(RERANKER_MODEL)
                    logger.info("✓ Reranker ready (cross-encoder)")
                except Exception as e:
                    logger.warning(
                        f"Could not load cross-encoder reranker ({e}) — "
                        "falling back to colbert rerank mode."
                    )
                    self.rerank_mode = "colbert"
                    logger.info("✓ Reranker ready (BGE-M3 ColBERT — reuses embedder)")
            else:
                # colbert mode: embedder already holds the weights, nothing extra to load
                logger.info("✓ Reranker ready (BGE-M3 ColBERT — reuses embedder, no second model)")

        # ── Decomposer ────────────────────────────────────────────────────
        self.use_decomposition = use_decomposition
        self.decomposer: Optional[IterativeDecomposer] = None
        if use_decomposition:
            self.decomposer = IterativeDecomposer()

        # ── Misc ──────────────────────────────────────────────────────────
        self.filter_builder = MetadataFilterBuilder()
        self.cache          = QueryCache() if ENABLE_CACHE else None

        logger.info(
            f"HybridRetriever ready  |  collection={collection_name}  "
            f"reranker={'off' if not self.use_reranker else self.rerank_mode}  "
            f"decomposition={'on' if self.use_decomposition else 'off'}  "
            f"hyde={'on' if ENABLE_HYDE else 'off'}"
        )

    def _verify_model_fingerprint(self, collection_name: str) -> None:
        """Nice-to-have: check that the collection was indexed with the same model."""
        try:
            pts = self.client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True,
            )[0]
            if pts:
                fp = pts[0].payload.get("_model_fingerprint")
                if fp and fp != EMBEDDING_MODEL:
                    logger.warning(
                        f"Model fingerprint mismatch: collection was indexed with "
                        f"'{fp}' but retriever is using '{EMBEDDING_MODEL}'. "
                        "Re-run Ingestion.py to rebuild the index."
                    )
                elif fp == EMBEDDING_MODEL:
                    logger.info(f"✓ Model fingerprint match: {EMBEDDING_MODEL}")
        except Exception:
            pass  # Collection may not exist yet or have no payload field

    # ── Low-level search primitives ───────────────────────────────────────

    def _encode_query(self, query: str) -> Dict:
        """
        Single BGE-M3 encode call → dense + sparse + ColBERT vectors.
        Returns dict with keys: 'dense', 'lexical_weights', 'colbert_vecs'.
        """
        output = self.embedder.encode(
            query,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        return output

    def _hyde_dense_vec(self, query: str) -> Optional[List[float]]:
        """
        HyDE pre-retrieval: generate a hypothetical AUTOSAR paragraph, embed it
        with BGE-M3 dense, and return the average of original + hypothetical
        dense vectors.  Returns None if HyDE is disabled or LLM call fails.
        """
        if not ENABLE_HYDE:
            return None

        q_lower = query.lower()
        if not any(kw in q_lower for kw in HYDE_KEYWORDS):
            return None

        hyde_prompt = (
            "Write a concise paragraph from an AUTOSAR technical specification "
            f"that directly answers the following question:\n\n{query}\n\n"
            "Write only the specification paragraph, no preamble or explanation."
        )
        try:
            resp = requests.post(
                f"{HYDE_BASE_URL}/chat/completions",
                json={
                    "model":  HYDE_MODEL,
                    "stream": False,
                    "options": {"temperature": HYDE_TEMPERATURE},
                    "messages": [{"role": "user", "content": hyde_prompt}],
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.debug(f"HyDE LLM returned HTTP {resp.status_code}")
                return None

            raw = resp.json()
            hypo_text = (
                raw.get("choices", [{}])[0].get("message", {}).get("content")
                or raw.get("message", {}).get("content", "")
            ).strip()
            if not hypo_text:
                return None

            hypo_enc  = self.embedder.encode(hypo_text, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            orig_enc  = self.embedder.encode(query,     return_dense=True, return_sparse=False, return_colbert_vecs=False)
            avg_dense = (
                np.array(orig_enc["dense_vecs"]) + np.array(hypo_enc["dense_vecs"])
            ) / 2.0
            # Normalise
            norm = np.linalg.norm(avg_dense)
            if norm > 0:
                avg_dense = avg_dense / norm
            logger.info(f"HyDE activated — hypothetical paragraph generated ({len(hypo_text)} chars)")
            return avg_dense.tolist()

        except Exception as e:
            logger.debug(f"HyDE failed ({e}) — using original dense vector.")
            return None

    def _retrieve_pool(
        self,
        query:    str,
        top_k:    int,
        filter_:  Optional[Filter] = None,
        hop_index: int = 0,
    ) -> List[SearchResult]:
        """
        Single Qdrant Universal Query API call:
          prefetch dense(120) + sparse(120) → ColBERT rerank server-side → top_k

        For single-hop analytical queries, HyDE replaces the dense vector.
        Named vectors match what Ingestion.py writes: 'dense', 'sparse', 'colbert'.
        """
        enc = self._encode_query(query)

        # Dense vector — optionally replaced by HyDE average for single-hop
        dense_vec: List[float] = enc["dense_vecs"].tolist() if hasattr(enc["dense_vecs"], "tolist") else list(enc["dense_vecs"])

        # HyDE only for first hop (hop_index == 0) — analytical queries
        if hop_index == 0:
            hyde_vec = self._hyde_dense_vec(query)
            if hyde_vec is not None:
                dense_vec = hyde_vec

        # Sparse vector from BGE-M3 lexical_weights
        lw = enc["lexical_weights"]
        # lexical_weights is a dict {token_id_str: weight} or similar
        if isinstance(lw, dict):
            indices = [int(k) for k in lw.keys()]
            values  = [float(v) for v in lw.values()]
        else:
            # Already a list of (index, value) or similar; handle gracefully
            try:
                indices = [int(x[0]) for x in lw]
                values  = [float(x[1]) for x in lw]
            except Exception:
                indices, values = [], []

        sparse_vec = SparseVector(indices=indices, values=values)

        # ColBERT multi-vector for reranking
        colbert_vecs = enc.get("colbert_vecs", None)
        if colbert_vecs is not None:
            if hasattr(colbert_vecs, "tolist"):
                colbert_list = colbert_vecs.tolist()
            else:
                colbert_list = list(colbert_vecs)
        else:
            colbert_list = None

        prefetch_limit = max(top_k * 2, 120)

        try:
            if colbert_list is not None:
                pts = self.client.query_points(
                    collection_name=self.collection,
                    prefetch=[
                        Prefetch(query=dense_vec,  using="dense",  limit=prefetch_limit),
                        Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
                    ],
                    query=colbert_list,
                    using="colbert",
                    limit=top_k,
                    with_payload=True,
                    query_filter=filter_,
                ).points
            else:
                # Fallback: no ColBERT — just prefetch dense+sparse, fuse server-side
                pts = self.client.query_points(
                    collection_name=self.collection,
                    prefetch=[
                        Prefetch(query=dense_vec,  using="dense",  limit=prefetch_limit),
                        Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
                    ],
                    query=dense_vec,
                    using="dense",
                    limit=top_k,
                    with_payload=True,
                    query_filter=filter_,
                ).points
        except Exception as e:
            logger.error(f"Qdrant prefetch query failed ({e})")
            return []

        results = []
        for p in pts:
            payload = p.payload or {}
            results.append(SearchResult(
                id=str(p.id),
                content=payload.get("content", ""),
                score=float(p.score),
                parent_text=payload.get("parent_text") or payload.get("parent_content"),
                dense_score=None,
                sparse_score=None,
                rerank_score=None,
                rrf_score=float(p.score),   # use ColBERT/fused score as rrf_score for dedup
                metadata=payload,
            ))
        return results

    def _rerank(
        self,
        query:   str,
        pool:    List[SearchResult],
        top_k:   int,
        hop1_evidence: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Post-retrieval reranking.  Behaviour is controlled by self.rerank_mode:

        "cross_encoder"  (default)
            Uses bge-reranker-v2-m3 (CrossEncoder.predict).  Highest quality;
            requires ~1.6 GB extra VRAM.  Batched in chunks of COLBERT_RERANK_BATCH
            with empty_cache() between batches to reduce OOM risk on large pools.

        "colbert"
            Reuses self.embedder (BGEM3FlagModel).  Scores via late-interaction
            MaxSim (colbert_score).  No second model on GPU; peak VRAM stays flat.
            Recommended when the GPU is under memory pressure.

        Both modes support BridgeRAG M7 (hop-1 bridge conditioning for hop-2+ chunks)
        and fall back to RRF order on any exception.
        """
        if not self.use_reranker or not pool:
            for r in pool:
                if r.rerank_score is None:
                    r.rerank_score = 0.0
            return pool[:top_k]

        try:
            if self.rerank_mode == "cross_encoder":
                scores = self._rerank_cross_encoder(query, pool, hop1_evidence)
            else:
                scores = self._rerank_colbert(query, pool, hop1_evidence)

            for r, s in zip(pool, scores):
                r.rerank_score = float(s)
                r.score        = float(s)
            pool.sort(key=lambda x: x.rerank_score, reverse=True)  # type: ignore[arg-type]
            logger.info(
                f"Reranked {len(pool)} chunks ({self.rerank_mode}) — "
                f"top={pool[0].rerank_score:.4f}  "
                f"mean={np.mean([x.rerank_score for x in pool]):.4f}"
            )

        except Exception as e:
            logger.error(f"Reranking failed ({e}) — using RRF order.")
            for r in pool:
                if r.rerank_score is None:
                    r.rerank_score = 0.0

        return pool[:top_k]

    def _rerank_cross_encoder(
        self,
        query:         str,
        pool:          List[SearchResult],
        hop1_evidence: Optional[str],
    ) -> List[float]:
        """
        Score pool against query using the CrossEncoder (bge-reranker-v2-m3).
        Batched to avoid a single large CUDA allocation on deep merged pools.
        """
        import torch

        pairs: List[List[str]] = []
        for r in pool:
            if hop1_evidence and r.hop_index is not None and r.hop_index >= 1:
                conditioned_query = (
                    f"{query}\n\nBridge evidence from prior step:\n{hop1_evidence}"
                )
                pairs.append([conditioned_query, r.content])
            else:
                pairs.append([query, r.content])

        scores: List[float] = []
        for batch_start in range(0, len(pairs), COLBERT_RERANK_BATCH):
            batch_pairs  = pairs[batch_start: batch_start + COLBERT_RERANK_BATCH]
            batch_scores = self.reranker.predict(batch_pairs, show_progress_bar=False)
            scores.extend(batch_scores.tolist())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return scores

    def _rerank_colbert(
        self,
        query:         str,
        pool:          List[SearchResult],
        hop1_evidence: Optional[str],
    ) -> List[float]:
        """
        Score pool against query using BGE-M3 ColBERT late-interaction MaxSim.
        Reuses self.embedder — no second model load, lower peak VRAM.
        """
        import torch

        # Encode query ColBERT vecs once
        q_enc  = self.embedder.encode(
            query,
            return_dense=False, return_sparse=False, return_colbert_vecs=True,
        )
        q_vecs = q_enc["colbert_vecs"]          # (nq_tokens, dim)

        # Bridge-conditioned query vecs for hop-2+ chunks (BridgeRAG M7)
        bridge_q_vecs = None
        if hop1_evidence and any(
            r.hop_index is not None and r.hop_index >= 1 for r in pool
        ):
            conditioned_query = (
                f"{query}\n\nBridge evidence from prior step:\n{hop1_evidence}"
            )
            bq_enc        = self.embedder.encode(
                conditioned_query,
                return_dense=False, return_sparse=False, return_colbert_vecs=True,
            )
            bridge_q_vecs = bq_enc["colbert_vecs"]

        scores: List[float] = []
        for batch_start in range(0, len(pool), COLBERT_RERANK_BATCH):
            batch     = pool[batch_start: batch_start + COLBERT_RERANK_BATCH]
            doc_texts = [r.content for r in batch]

            d_enc       = self.embedder.encode(
                doc_texts,
                return_dense=False, return_sparse=False, return_colbert_vecs=True,
            )
            d_vecs_list = d_enc["colbert_vecs"]   # list of (nd_tokens, dim) per doc

            for r, d_vecs in zip(batch, d_vecs_list):
                qv = (
                    bridge_q_vecs
                    if (bridge_q_vecs is not None
                        and r.hop_index is not None
                        and r.hop_index >= 1)
                    else q_vecs
                )
                # colbert_score expects tensors of shape (1, n_tokens, dim)
                qv_t = torch.tensor(qv).unsqueeze(0)
                dv_t = torch.tensor(d_vecs).unsqueeze(0)
                scores.append(float(self.embedder.colbert_score(qv_t, dv_t).item()))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return scores

    # ── Min-coverage enforcement ──────────────────────────────────────────

    @staticmethod
    def _enforce_coverage(
        ranked:     List[SearchResult],
        hop_pools:  Dict[int, List[SearchResult]],
        top_k:      int,
    ) -> List[SearchResult]:
        """
        Guarantees that at least MIN_CHUNKS_PER_HOP chunks from each hop
        appear in the returned list.  If a hop is absent, the top-ranked
        result from that hop's pool is inserted and the lowest overall
        result is dropped (keeping list length == top_k).

        M5 fix: When deduplicating seen_ids, the hop_index of the FIRST
        finder is preserved — not overwritten by the higher-scoring copy.
        This ensures _enforce_coverage correctly identifies which hops
        are already covered in the ranked list.

        M6 fix: Force-inserted chunks are re-scored as min(existing rerank
        scores) so they don't corrupt the sorted order with stale RRF scores.
        """
        if not hop_pools:
            return ranked

        seen_ids = {r.id for r in ranked}
        result   = list(ranked)

        # Compute the minimum rerank score among already-ranked results
        # (used to assign a neutral score to force-inserted chunks)
        rerank_scores = [r.rerank_score for r in result if r.rerank_score is not None]
        min_rerank    = min(rerank_scores) if rerank_scores else 0.0

        for hop_idx, pool in sorted(hop_pools.items()):
            represented = any(r.hop_index == hop_idx for r in result)
            if represented:
                continue
            # Find the highest-ranked chunk from this hop not already present
            for candidate in pool:
                if candidate.id not in seen_ids:
                    # M6: assign min score so insertion doesn't corrupt rank order
                    candidate.hop_index    = hop_idx
                    candidate.rerank_score = min_rerank
                    candidate.score        = min_rerank
                    if len(result) >= top_k:
                        result.pop()           # drop lowest
                    result.append(candidate)
                    seen_ids.add(candidate.id)
                    logger.info(
                        f"  Coverage: forced hop-{hop_idx} chunk "
                        f"(id={candidate.id[:12]}…) into top-{top_k} "
                        f"with score={min_rerank:.4f}"
                    )
                    break

        # Re-sort by score after possible insertions
        result.sort(key=lambda x: x.score, reverse=True)
        return result[:top_k]

    # ── Public search entry point ─────────────────────────────────────────

    def search(
        self,
        query:             str,
        top_k:             int            = DEFAULT_TOP_K,
        filters:           Optional[Dict] = None,
        use_reranking:     bool           = True,
        use_decomposition: bool           = True,
    ) -> List[SearchResult]:
        """
        Main search method.  Automatically selects single-hop or multi-hop
        path depending on what the decomposer returns.

        Parameters
        ----------
        query             : The full (possibly multi-hop) question.
        top_k             : Number of results to return.
        filters           : Optional metadata filter dict (see MetadataFilterBuilder).
        use_reranking     : Whether to apply cross-encoder reranking.
        use_decomposition : Whether to attempt query decomposition.

        Returns
        -------
        List[SearchResult] sorted by score descending.
        """
        # ── Cache check ───────────────────────────────────────────────────
        if self.cache:
            hit = self.cache.get(query, top_k, filters)
            if hit:
                logger.info("✓ Cache hit")
                return hit

        t0      = time.time()
        filter_ = self.filter_builder.build(**(filters or {})) if filters else None

        # ── Decide: single-hop or iterative multi-hop ─────────────────────
        can_decompose = (
            use_decomposition
            and self.use_decomposition
            and self.decomposer is not None
            and self.decomposer._available
        )

        if can_decompose:
            results = self._search_multihop(query, top_k, filter_, use_reranking)
        else:
            results = self._search_singlehop(query, top_k, filter_, use_reranking)

        logger.info(f"Search done in {(time.time()-t0)*1000:.1f}ms — {len(results)} results")

        if self.cache:
            self.cache.put(query, top_k, results, filters)

        return results

    # ── Single-hop path ───────────────────────────────────────────────────

    def _search_singlehop(
        self,
        query:     str,
        top_k:     int,
        filter_:   Optional[Filter],
        do_rerank: bool,
    ) -> List[SearchResult]:
        logger.info(f"Single-hop retrieval: «{query[:80]}»")
        pool = self._retrieve_pool(query, max(top_k * 5, CANDIDATE_POOL_PER_HOP), filter_)
        logger.info(f"  Pool size: {len(pool)}")

        if do_rerank:
            # Rerank against the original query
            return self._rerank(query, pool, top_k)
        else:
            for r in pool:
                if r.rerank_score is None:
                    r.rerank_score = 0.0
            return pool[:top_k]

    # ── Multi-hop path  (iterative QD+RR) ────────────────────────────────

    def _search_multihop(
        self,
        query:     str,
        top_k:     int,
        filter_:   Optional[Filter],
        do_rerank: bool,
    ) -> List[SearchResult]:
        """
        Iterative retrieval loop:

        hop=1  sub_query = decomposer.first_sub_query(query)
               pool_1   = retrieve(sub_query)
               evidence = pool_1[:3].content         ← context carryover

        hop=2  sub_query = decomposer.next_sub_query(query, evidence)
               pool_2   = retrieve(sub_query)
               evidence += pool_2[:3].content

        ...repeat until is_final=True or max_hops reached...

        merged_pool = deduplicate(pool_1 ∪ pool_2 ∪ …)
        final       = cross_encoder_rerank(ORIGINAL_QUERY, merged_pool)[:top_k]
        """
        assert self.decomposer is not None

        hop_pools:    Dict[int, List[SearchResult]] = {}   # hop_idx → sorted pool
        seen_ids:     Dict[str, SearchResult]        = {}   # id → best result
        all_evidence: List[str]                      = []   # content strings for carryover

        # ── Hop 1 ─────────────────────────────────────────────────────────
        # Change #3 (IRCoT / PAR²-RAG): is_final from first_sub_query() is
        # IGNORED. Hop-1 retrieval ALWAYS executes — is_final is a stop
        # condition checked only AFTER evidence is gathered.  Single-hop
        # questions stop naturally after hop-1 when next_sub_query() returns
        # is_final=True once evidence is sufficient.
        sub_query, _is_final_from_first = self.decomposer.first_sub_query(query)
        is_final = False   # always execute hop-1 regardless of LLM hint
        logger.info(f"Multi-hop retrieval  |  hop=1  (is_final hint ignored — hop-1 always runs)")

        # Change #7 (PAR²-RAG Coverage Anchor): hop-1 uses broad pool for recall
        hop1_pool_size = max(top_k * 5, CANDIDATE_POOL_HOP1)
        pool = self._retrieve_pool(sub_query, hop1_pool_size, filter_, hop_index=0)
        for r in pool:
            r.hop_index = 0
        hop_pools[0] = pool

        # Change #8 (M5 fix): When updating seen_ids, preserve hop_index from
        # the FIRST finder — only overwrite the score, not the hop_index.
        for r in pool:
            if r.id not in seen_ids or (r.rrf_score or 0) > (seen_ids[r.id].rrf_score or 0):
                if r.id in seen_ids:
                    # Keep the hop_index of whichever pool first found this chunk
                    r.hop_index = seen_ids[r.id].hop_index
                seen_ids[r.id] = r

        # Change #6 (BridgeRAG / IRCoT): 5 chunks × 900 chars for carryover
        all_evidence.extend(r.content for r in pool[:5])

        # ── Subsequent hops ───────────────────────────────────────────────
        hop_idx = 1
        while not is_final and hop_idx < DECOMP_MAX_HOPS:
            sub_query, is_final = self.decomposer.next_sub_query(
                query, all_evidence, hop_number=hop_idx
            )
            if not sub_query:
                break

            logger.info(f"Multi-hop retrieval  |  hop={hop_idx + 1}  is_final={is_final}")

            # Change #7: hop-2+ use smaller targeted pool (bridge entities give precision)
            hopn_pool_size = max(top_k * 5, CANDIDATE_POOL_HOP_N)
            pool = self._retrieve_pool(sub_query, hopn_pool_size, filter_, hop_index=hop_idx)
            for r in pool:
                r.hop_index = hop_idx
            hop_pools[hop_idx] = pool

            # Change #8 (M5 fix): preserve first-seen hop_index on dedup
            for r in pool:
                if r.id not in seen_ids or (r.rrf_score or 0) > (seen_ids[r.id].rrf_score or 0):
                    if r.id in seen_ids:
                        r.hop_index = seen_ids[r.id].hop_index  # preserve first-finder hop
                    seen_ids[r.id] = r

            # Change #6: 5 chunks × 900 chars for carryover
            all_evidence.extend(r.content for r in pool[:5])
            hop_idx += 1

        # ── Merge and deduplicate ─────────────────────────────────────────
        merged = sorted(seen_ids.values(), key=lambda x: x.rrf_score or 0.0, reverse=True)
        logger.info(
            f"Merged pool: {len(merged)} unique chunks from {len(hop_pools)} hop(s)"
        )

        # ── Single rerank against the ORIGINAL query ──────────────────────
        # Change #10 (BridgeRAG M7): extract hop-1 bridge summary to condition
        # reranking of hop-2+ chunks (tripartite scorer pattern).
        hop1_evidence_text: Optional[str] = None
        if hop_pools.get(0):
            # Top-2 sentences from hop-1 evidence as bridge summary
            top_hop1 = hop_pools[0][:2]
            hop1_evidence_text = "".join(r.content[:450] for r in top_hop1)

        if do_rerank:
            ranked = self._rerank(query, merged, top_k, hop1_evidence=hop1_evidence_text)
        else:
            for r in merged:
                if r.rerank_score is None:
                    r.rerank_score = 0.0
            ranked = merged[:top_k]

        # ── Enforce per-hop coverage ──────────────────────────────────────
        if len(hop_pools) > 1:
            ranked = self._enforce_coverage(ranked, hop_pools, top_k)

        return ranked

    # ── Convenience wrapper ───────────────────────────────────────────────

    def search_with_filters(
        self,
        query:          str,
        top_k:          int                  = DEFAULT_TOP_K,
        file_types:     Optional[List[str]]  = None,
        filenames:      Optional[List[str]]  = None,
        folders:        Optional[List[str]]  = None,
        min_word_count: Optional[int]        = None,
        section_titles: Optional[List[str]]  = None,
    ) -> List[SearchResult]:
        filters = {k: v for k, v in {
            "file_types":     file_types,
            "filenames":      filenames,
            "folders":        folders,
            "min_word_count": min_word_count,
            "section_titles": section_titles,
        }.items() if v is not None}
        return self.search(query, top_k=top_k, filters=filters or None)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(description="HybridRetriever demo")
    ap.add_argument("--qdrant-url",    default=QDRANT_URL)
    ap.add_argument("--collection",    default=COLLECTION)
    ap.add_argument("--query",         default=(
        "How does the configuration of the transmission mode in the acceptance test "
        "specification influence the behavior of the IPDU group when the mode is "
        "switched during the main test execution?"
    ))
    ap.add_argument("--top-k",         type=int, default=5)
    ap.add_argument("--no-reranker",   action="store_true")
    ap.add_argument("--no-decompose",  action="store_true")
    ap.add_argument("--rerank-mode",   default=RERANK_MODE,
                    choices=["cross_encoder", "colbert"],
                    help="Reranker backend: 'cross_encoder' (default, bge-reranker-v2-m3) "
                         "or 'colbert' (BGE-M3 ColBERT, reuses embedder, lower VRAM).")
    ap.add_argument("--log-file",      default=None,
                    help="Path to write log file (optional, e.g. logs/retriever.log)")
    args = ap.parse_args()

    global logger
    logger = _setup_logging(args.log_file)

    retriever = HybridRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        use_reranker=not args.no_reranker,
        use_decomposition=not args.no_decompose,
        rerank_mode=args.rerank_mode,
    )

    logger.info(f"\nQuery: {args.query}\n")
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"\n{'='*80}")
    print(f"Top {len(results)} results")
    print("=" * 80)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. score={r.score:.6f}  rrf={r.rrf_score:.6f}  "
              f"rerank={r.rerank_score:.4f}  hop={r.hop_index}")
        print(f"   file:    {r.metadata.get('filename','N/A')}")
        print(f"   section: {r.metadata.get('section_title','N/A')}")
        print(f"   content: {r.content[:200]}…")


if __name__ == "__main__":
    main()