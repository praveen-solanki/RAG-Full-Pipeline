"""
Retrieval Evaluation Pipeline
==============================
Evaluates HybridRetriever against Ragas-style ground truth (chunk-level).

Metrics computed per question
------------------------------
  recall@k        — fraction of reference contexts found in top-k
  precision@k     — fraction of top-k chunks that are relevant
  ndcg@k          — normalised discounted cumulative gain
  all_found@k     — 1.0 if every reference context appears in top-k (strict multihop)
  mrr             — 1 / rank of the first relevant retrieved chunk (standard MRR)
  found           — 1.0 if all reference contexts were found (any k)

Failure taxonomy  (per missed context)
---------------------------------------
  wrong_doc       — correct document never appeared in retrieved pool
  wrong_chunk     — correct document appeared but not this specific chunk
  below_cutoff    — chunk exists in extended pool but ranked > top-k
  unknown         — cannot determine

Usage
-----
    python retrieval_eval.py --questions evaluation_questions.json

    python retrieval_eval.py \\
        --questions evaluation_questions.json \\
        --resume    progress_75.json \\
        --top-k     10 \\
        --store-full-pool
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from HybridRetriever_BGE_M3 import HybridRetriever, SearchResult

def _setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Attach console + file handlers to the root logger.
    Both HybridRetriever and this module share the root logger, so
    retriever internals (sub-query generation, reranker scores, etc.)
    also land in the log file.
    """
    fmt  = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    root = logging.getLogger()
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
        root.info(f"Log file: {log_file}")

    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

# When True, retrieve top_k * POOL_MULTIPLIER to detect below_cutoff failures.
# Adds latency.  Set via --store-full-pool flag.
STORE_FULL_POOL = False
POOL_MULTIPLIER = 3

# Context-matching mode used to decide whether a retrieved chunk satisfies a
# reference context.  Applies everywhere evaluate_single checks for a match.
#
#   "fuzzy"     (default) — fast text-based matching, no GPU needed.
#   "embedding" — BGE-M3 dense cosine similarity. Semantically robust;
#                 handles OCR artefacts, paraphrasing, and the common case
#                 where a GT context is longer than a single chunk.
#                 Reuses the retriever's already-loaded embedder (no extra VRAM).
#
# CLI: --match-mode fuzzy | embedding
MATCH_MODE                = "fuzzy"
EMBEDDING_MATCH_THRESHOLD = 0.70


# ──────────────────────────────────────────────────────────────────────────────
# Fuzzy matching helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return (
        text.replace("\n", " ")
            .replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
    )


def _despace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _strip_pipes(text: str) -> str:
    return re.sub(r"\s*\|\s*", " ", text).strip()


def _kv_to_text(snippet: str) -> str:
    parts  = re.split(r"[,;]\s*", snippet)
    values = []
    for part in parts:
        if "=" in part:
            _, _, val = part.partition("=")
            values.append(val.strip())
        else:
            values.append(part.strip())
    return " ".join(v for v in values if v)


def _sliding_match(snippet: str, text: str, threshold: float) -> bool:
    ws = len(snippet)
    if ws == 0:
        return False
    if ws > len(text):
        return SequenceMatcher(None, snippet, text).ratio() >= threshold
    step = max(1, ws // 4)
    for i in range(0, len(text) - ws + 1, step):
        window = text[i: min(i + ws + step, len(text))]
        if SequenceMatcher(None, snippet, window).ratio() >= threshold:
            return True
    return False


def fuzzy_match(snippet: str, text: str, threshold: float = 0.80) -> bool:
    """
    Bidirectional fuzzy match.

    Direction 1 (snippet in text): original behaviour.
    Direction 2 (text in snippet): handles the common case where the retrieved
      chunk is a sub-window of a longer GT reference context.  When snippet is
      much longer than text the sliding-window step becomes snippet_len//4
      which is too coarse to catch real matches — reversing the roles fixes it.
    """
    sc = _normalize(snippet.lower())
    tc = _normalize(text.lower())

    # Direction 1
    if sc in tc:
        return True
    if _sliding_match(sc, tc, threshold):
        return True
    sds = _despace(sc)
    tds = _despace(tc)
    if sds:
        if sds in tds or _sliding_match(sds, tds, threshold):
            return True
    tc_np = _strip_pipes(tc)
    if sc in tc_np or _sliding_match(sc, tc_np, threshold):
        return True
    tds_np = _despace(tc_np)
    if sds and (sds in tds_np):
        return True
    if "=" in sc:
        kv    = _kv_to_text(sc)
        kv_ds = _despace(kv)
        MIN   = 8
        if kv and len(kv) > MIN:
            if kv in tc or kv in tc_np:
                return True
            if _sliding_match(kv, tc, threshold) or _sliding_match(kv, tc_np, threshold):
                return True
        if kv_ds and len(kv_ds) > MIN:
            if kv_ds in tds or kv_ds in tds_np:
                return True

    # Direction 2: text (chunk) as snippet, snippet (GT context) as haystack.
    # Guard: chunk must be >= 80 chars to avoid trivial false positives.
    if len(tc) >= 80 and len(tc) < len(sc):
        if tc in sc:
            return True
        if _sliding_match(tc, sc, threshold):
            return True
        sc_np = _strip_pipes(sc)
        if tc in sc_np or _sliding_match(tc, sc_np, threshold):
            return True
        if _despace(tc) in _despace(sc):
            return True

    return False


def embedding_match(
    snippet:   str,
    text:      str,
    embedder,
    threshold: float = EMBEDDING_MATCH_THRESHOLD,
) -> bool:
    """
    Semantic match via BGE-M3 dense cosine similarity.
    Reuses the already-loaded BGEM3FlagModel — no extra model needed.
    Falls back to fuzzy_match if embedder is None.
    """
    if embedder is None:
        return fuzzy_match(snippet, text)
    import numpy as np
    encs = embedder.encode(
        [snippet, text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        batch_size=2,
    )
    vecs = encs["dense_vecs"]        # shape (2, dim)
    a, b = vecs[0], vecs[1]
    cos  = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return cos >= threshold


# ──────────────────────────────────────────────────────────────────────────────
# Context-label decomposition helper
# ──────────────────────────────────────────────────────────────────────────────

def _split_by_context_labels(query: str) -> List[str]:
    """
    Split a query that contains explicit (Context N) labels into sub-queries.

    Example input:
        "How does X (Context 1) influence Y when Z (Context 2)?"

    Returns:
        ["How does X", "influence Y when Z?"]

    The trailing punctuation of the original query is appended to the last
    sub-query so it remains a well-formed question.

    Falls back to [query] (single-element list) when no context labels are
    found, so callers can always iterate over the result safely.
    """
    # Match segments that end just before a (Context N) marker or end of string
    parts = re.split(r"\s*\(Context\s+\d+\)\s*", query, flags=re.IGNORECASE)
    # Filter empty strings that can arise from leading/trailing markers
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        # No context labels found — return the original query unchanged
        return [query]

    # Preserve the trailing punctuation of the full query on the last sub-query
    last = parts[-1]
    if query.rstrip() and query.rstrip()[-1] in ".?!" and not last.endswith((".", "?", "!")):
        parts[-1] = last + query.rstrip()[-1]

    return parts


# ──────────────────────────────────────────────────────────────────────────────
# Failure taxonomy helpers
# ──────────────────────────────────────────────────────────────────────────────

def _doc_key(source_id: str) -> str:
    if "_chunk_" in source_id:
        return source_id.rsplit("_chunk_", 1)[0]
    return source_id


def _classify_miss(
    ctx_text:  str,
    retrieved: List[Dict],
    full_pool: Optional[List[Dict]] = None,
) -> str:
    """
    Classify why a reference context was not matched in the top-k results.

    Uses only fuzzy_match (bidirectional, no embedder needed) plus filename-stem
    matching — no GT source_documents, no regex inference.

    Order:
      1. fuzzy_match against retrieved chunks            -> wrong_chunk
      2. filename-stem token match against retrieved     -> wrong_chunk
      3. fuzzy_match against extended pool               -> below_cutoff
      4. otherwise                                       -> wrong_doc
    """
    # 1. Direct fuzzy match against top-k chunks (bidirectional)
    for c in retrieved:
        if (fuzzy_match(ctx_text, c.get("parent_text") or c.get("content", ""))
                or fuzzy_match(ctx_text, c.get("content", ""))):
            return "wrong_chunk"

    # 2. Filename-stem token match: correct doc retrieved but wrong chunk
    header_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", ctx_text[:200]))
    header_tokens -= {"the","and","for","via","with","that","from","this",
                      "not","are","shall","autosar","release","document",
                      "specification","requirements","note","used"}
    if header_tokens:
        for c in retrieved:
            fn = Path(c.get("filename", "")).stem.lower()
            hits = sum(1 for t in header_tokens if t.lower() in fn)
            if hits >= max(1, len(header_tokens) // 3):
                return "wrong_chunk"

    # 3. Extended pool check
    if full_pool is not None:
        for c in full_pool:
            if (fuzzy_match(ctx_text, c.get("parent_text") or c.get("content", ""))
                    or fuzzy_match(ctx_text, c.get("content", ""))):
                # Found in pool — was it from a doc already in top-k?
                pool_fn = Path(c.get("filename", "")).stem
                retrieved_fns = {Path(r.get("filename","")).stem for r in retrieved}
                return "wrong_chunk" if pool_fn in retrieved_fns else "below_cutoff"

    return "wrong_doc"


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

def _make_chunk_dict(result: SearchResult, top_k: int) -> Dict:
    stem = Path(result.metadata.get("filename", "unknown")).stem
    return {
        "filename":     result.metadata.get("filename", ""),
        "content":      result.content,
        "parent_text":  result.parent_text,   # full section text for fuzzy matching
        "score":        result.score,
        "rerank_score": result.rerank_score,
        "rrf_score":    result.rrf_score,
        "dense_score":  result.dense_score,
        "sparse_score": result.sparse_score,
        "hop_index":    result.hop_index,
        "section":      result.metadata.get("section_title", ""),
        "source_id":    f"doc_{stem}_chunk_{result.metadata.get('chunk_id', 0)}",
    }


class ComprehensiveEvaluator:
    """Chunk-level retrieval evaluator for Ragas-style ground truth."""

    def __init__(
        self,
        retriever:                HybridRetriever,
        match_mode:               str = MATCH_MODE,
        query_decomposition_mode: str = "llm",
        fallback_mode:            str = "full_query",
    ):
        self.retriever                = retriever
        self.match_mode               = match_mode.lower()
        # "llm"           — pass the full query to the retriever and let it
        #                   decompose via LLM internally (original behaviour).
        # "context_split" — parse (Context N) labels from the query and issue
        #                   one retriever call per sub-query, then merge results.
        self.query_decomposition_mode = query_decomposition_mode.lower()
        # fallback_mode controls what happens in "context_split" mode when a
        # query has no (Context N) labels:
        #   "full_query" — send the full query as-is, no LLM decomposition.
        #   "llm"        — let the retriever use its internal LLM decomposition.
        # Ignored entirely when query_decomposition_mode="llm".
        self.fallback_mode            = fallback_mode.lower()
        # Reuse the retriever's already-loaded BGE-M3 embedder for semantic matching.
        # None when match_mode="fuzzy" so no GPU memory is touched for matching.
        self.embedder   = retriever.embedder if self.match_mode == "embedding" else None

    def _ctx_match(self, ctx: str, chunk_text: str) -> bool:
        """Single dispatch: fuzzy or embedding depending on match_mode."""
        if self.match_mode == "embedding":
            return embedding_match(ctx, chunk_text, self.embedder)
        return fuzzy_match(ctx, chunk_text)

    # ── Retrieval dispatch ────────────────────────────────────────────────

    def _retrieve(self, query: str, top_k: int) -> List:
        """
        Dispatch retrieval based on self.query_decomposition_mode.

        "llm"
            Pass the full query to the retriever unchanged.  The retriever
            may call its internal LLM decomposition if it was initialised
            with use_decomposition=True.

        "context_split"
            Split the query on (Context N) labels into sub-queries and
            issue one retriever call per sub-query with LLM decomposition
            disabled (the sub-queries are already atomic).  Results are
            merged by deduplication — highest score per unique source_id
            is kept — then re-sorted and trimmed to top_k.

            If the query contains no (Context N) labels the full query is
            used as-is (safe fallback for mixed datasets).
        """
        if self.query_decomposition_mode == "context_split":
            sub_queries = _split_by_context_labels(query)

            if len(sub_queries) == 1:
                # No (Context N) labels found — apply fallback_mode
                if self.fallback_mode == "llm":
                    logger.debug("context_split: no labels found, fallback → LLM decomposition")
                    return self.retriever.search(query, top_k=top_k)
                else:  # "full_query"
                    logger.debug("context_split: no labels found, fallback → full query (no decomposition)")
                    return self.retriever.search(query, top_k=top_k, use_decomposition=False)

            logger.debug(
                f"context_split: {len(sub_queries)} sub-queries — "
                + " | ".join(f'"{sq}"' for sq in sub_queries)
            )

            # Collect results from each sub-query; disable LLM decomposition
            # inside the retriever since the sub-queries are already atomic.
            seen:    Dict[str, Any] = {}   # source_id -> SearchResult
            for sq in sub_queries:
                sub_results = self.retriever.search(
                    sq,
                    top_k=top_k,
                    use_decomposition=False,
                )
                for r in sub_results:
                    # Use content as dedup key (source_id may not exist on all
                    # retriever versions; content is always present)
                    key = r.metadata.get("chunk_id", r.content[:120])
                    if key not in seen or r.score > seen[key].score:
                        seen[key] = r

            # Re-sort merged pool by score descending, trim to top_k
            merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
            return merged[:top_k]

        # Default: "llm" mode — full query, retriever handles decomposition
        return self.retriever.search(query, top_k=top_k)

    # ── Single question ───────────────────────────────────────────────────

    def evaluate_single(
        self,
        question_data:  Dict[str, Any],
        question_index: int,
        top_k:          int  = 10,
        verbose:        bool = False,
    ) -> Dict[str, Any]:
        query              = question_data["user_input"]
        reference          = question_data.get("reference", "")
        reference_contexts = question_data.get("reference_contexts", []) or []

        if verbose:
            logger.info(f"\n[{question_index}] {query}")

        t0 = time.time()
        try:
            # ── Primary retrieval ─────────────────────────────────────────
            results  = self._retrieve(query, top_k=top_k)
            latency  = time.time() - t0
            chunks   = [_make_chunk_dict(r, top_k) for r in results]

            # ── Extended pool for below_cutoff detection ──────────────────
            full_pool_chunks: Optional[List[Dict]] = None
            if STORE_FULL_POOL:
                extended = self.retriever.search(
                    query,
                    top_k=top_k * POOL_MULTIPLIER,
                    use_reranking=False,
                    use_decomposition=False,      # raw pool, no extra LLM calls
                )
                full_pool_chunks = [_make_chunk_dict(r, top_k * POOL_MULTIPLIER) for r in extended]

            # ══════════════════════════════════════════════════════════════
            # UNIFIED MATCHING: single pass, content-only, two signals
            # ══════════════════════════════════════════════════════════════
            #
            # One pass builds two independent tracking structures:
            #
            #  ctx_first_rank[i]  — 1-indexed rank of the first chunk whose
            #    own content matches GT context i.  Drives Recall, MRR,
            #    found, and all_found.  Records the rank even when another
            #    context was already satisfied at the same rank.
            #
            #  chunk_relevance[rank-1]  — binary: True if the chunk at that
            #    rank position matched ANY GT context (regardless of how many
            #    contexts it satisfied).  Drives Precision and NDCG.
            #
            # WHY two signals instead of one:
            #   ctx_first_rank correctly counts how many information needs
            #   were satisfied and at what rank, but using it directly for
            #   Precision/NDCG causes > 1.0 values when one chunk satisfies
            #   multiple contexts simultaneously (e.g., a dense chunk covers
            #   two related GT passages, both get first_rank = 1 ->
            #   n_found_in_k = 2 at k=1 -> P@1 = 2.0, NDCG@1 = 2.0).
            #
            #   chunk_relevance treats each retrieved rank position as a
            #   single binary judgment (relevant = 1, not relevant = 0),
            #   matching the standard IR definition used by BEIR, TREC, and
            #   Ragas. A chunk satisfying two contexts still counts as one
            #   relevant chunk, keeping P@k and NDCG@k in [0, 1].
            #
            # WHY content-only (not parent_text):
            #   parent_text is the full section containing the chunk.
            #   In embedding mode its cosine with the GT context is ~0.99,
            #   so every chunk from the right section matches at rank 1,
            #   collapsing Recall@k to a flat line. In fuzzy mode the same
            #   effect arises via direction-2 substring matching.
            #   Content-only gives a genuine, differentiated signal.
            # ══════════════════════════════════════════════════════════════

            total_ctx = len(reference_contexts)

            # ctx_first_rank[i]  = 1-indexed rank of first content match for
            #                      GT context i.  None = never found.
            # chunk_relevance    = per-position binary relevance list;
            #                      chunk_relevance[rank-1] is True if the
            #                      chunk at that rank matched any GT context.
            ctx_first_rank:  List[Optional[int]] = [None] * total_ctx
            chunk_relevance: List[bool]           = []

            for rank, chunk in enumerate(chunks, start=1):
                chunk_rel = False
                for i, ctx in enumerate(reference_contexts):
                    if self._ctx_match(ctx, chunk["content"]):
                        chunk_rel = True
                        if ctx_first_rank[i] is None:
                            ctx_first_rank[i] = rank
                chunk_relevance.append(chunk_rel)

            # ── Derived scalars ───────────────────────────────────────────
            found = bool(reference_contexts) and all(
                r is not None for r in ctx_first_rank
            )

            # Rank of the last (worst) satisfied context; set only when all
            # contexts are found.
            bottleneck_rank = (
                max(r for r in ctx_first_rank if r is not None)
                if found else None
            )

            # MRR = 1 / rank of first context satisfied (standard definition).
            first_rel_rank = min(
                (r for r in ctx_first_rank if r is not None), default=None
            )
            mrr = 1.0 / first_rel_rank if first_rel_rank else 0.0

            # ── Per-k metrics ─────────────────────────────────────────────
            k_values = sorted({1, 3, 5, 10, top_k})
            metrics: Dict[str, Any] = {
                "found": found,
                "rank":  bottleneck_rank,
                "mrr":   mrr,
            }

            for k in k_values:
                # ── Recall@k / all_found@k  (per-context signal) ──────────
                # How many of the total_ctx GT contexts are satisfied by at
                # least one chunk within the top-k results?
                n_ctx_in_k = sum(
                    1 for r in ctx_first_rank if r is not None and r <= k
                )
                metrics[f"recall@{k}"]    = n_ctx_in_k / total_ctx if total_ctx else 0.0
                metrics[f"all_found@{k}"] = (
                    1.0 if (total_ctx and n_ctx_in_k == total_ctx) else 0.0
                )

                # ── Precision@k  (per-chunk signal) ───────────────────────
                # Of the k retrieved chunks, how many are relevant (match any
                # GT context)? One chunk satisfying multiple contexts still
                # counts as 1 relevant chunk -> P@k in [0, 1].
                n_rel_chunks_in_k = sum(chunk_relevance[:k])
                metrics[f"precision@{k}"] = n_rel_chunks_in_k / k if k else 0.0

            # ── NDCG@k  (per-chunk signal) ────────────────────────────────
            #
            # Binary chunk_relevance means each rank position contributes at
            # most one discounted gain term — matching the standard BEIR/TREC
            # formulation and guaranteeing NDCG in [0, 1].
            #
            # DCG@k  = sum( 1/log2(r+1)  for relevant positions r in 1..k )
            # IDCG@k = sum( 1/log2(j+1)  for j = 1..min(total_ctx, k) )
            #
            # IDCG uses total_ctx as the oracle count of relevant items in
            # the corpus (one per GT context) — a fixed denominator that does
            # not shrink when the retriever misses contexts.
            for k in k_values:
                dcg = sum(
                    1.0 / np.log2(r + 1)
                    for r, rel in enumerate(chunk_relevance[:k], start=1)
                    if rel
                )
                idcg = sum(
                    1.0 / np.log2(j + 1)
                    for j in range(1, min(total_ctx, k) + 1)
                )
                metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

            # ── Per-context match record + failure taxonomy ───────────────
            # _classify_miss always uses fuzzy_match for failure diagnosis
            # regardless of --match-mode.  This is intentional: fuzzy is
            # more permissive and can detect near-misses that the embedding
            # threshold might reject.  The taxonomy is diagnostic only —
            # it does not feed into any metric value.
            context_match = []
            for i, ctx in enumerate(reference_contexts):
                ok      = ctx_first_rank[i] is not None
                at_rank = ctx_first_rank[i]
                at_file = chunks[at_rank - 1]["filename"] if at_rank else None
                fail_type = None if ok else _classify_miss(ctx, chunks, full_pool_chunks)
                context_match.append({
                    "reference_context":  ctx,
                    "found":              ok,
                    "found_in_rank":      at_rank,
                    "found_in_filename":  at_file,
                    "failure_type":       fail_type,
                })

            n_found_total = sum(1 for r in ctx_first_rank if r is not None)
            if verbose:
                logger.info(
                    f"  found={found}  rank={bottleneck_rank}  mrr={mrr:.4f}  "
                    f"coverage={n_found_total}/{total_ctx}"
                )

            return {
                "question_index":     question_index,
                "question":           query,
                "reference":          reference,
                "reference_contexts": reference_contexts,
                "context_match":      context_match,
                "retrieved_chunks":   chunks,
                "latency_ms":         latency * 1000,
                "metrics":            metrics,
            }

        except Exception as e:
            logger.error(f"[{question_index}] Error: {e}")
            k_values = sorted({1, 3, 5, 10, top_k})
            err_metrics: Dict[str, Any] = {"found": False, "rank": None, "mrr": 0.0}
            for k in k_values:
                for m in ["precision", "recall", "ndcg", "all_found"]:
                    err_metrics[f"{m}@{k}"] = 0.0
            return {
                "question_index":     question_index,
                "question":           query,
                "reference":          reference,
                "reference_contexts": reference_contexts,
                "context_match":      [],
                "retrieved_chunks":   [],
                "latency_ms":         0.0,
                "error":              str(e),
                "metrics":            err_metrics,
            }

    # ── Full dataset ──────────────────────────────────────────────────────

    def evaluate_all(
        self,
        questions:           List[Dict[str, Any]],
        top_k:               int           = 10,
        verbose:             bool          = False,
        save_progress_every: int           = 25,
        resume_file:         Optional[str] = None,
        jsonl_writer:        Any           = None,
    ) -> Dict[str, Any]:

        logger.info("=" * 80)
        logger.info(f"EVALUATING {len(questions)} QUESTIONS")
        logger.info("=" * 80)

        all_results:       List[Dict] = []
        completed_indices: set        = set()

        if resume_file and Path(resume_file).exists():
            logger.info(f"Resuming from {resume_file}")
            with open(resume_file) as f:
                all_results = json.load(f)
            completed_indices = {r["question_index"] for r in all_results}
            logger.info(f"  already done: {len(completed_indices)}  remaining: {len(questions)-len(completed_indices)}")
            if jsonl_writer:
                jsonl_writer.rewrite_from_results(all_results)

        t0           = time.time()
        n_this_run   = 0

        for idx, q in enumerate(questions):
            if idx in completed_indices:
                continue

            n_this_run += 1
            if n_this_run % 10 == 0 or verbose:
                elapsed   = time.time() - t0
                remaining = len(questions) - len(completed_indices) - n_this_run
                eta       = (elapsed / n_this_run) * remaining if n_this_run else 0
                logger.info(f"  progress: {n_this_run}/{len(questions)-len(completed_indices)}  ETA: {eta/60:.1f}min")

            result = self.evaluate_single(q, idx, top_k, verbose)
            all_results.append(result)

            if jsonl_writer:
                jsonl_writer.append(result)

            if save_progress_every and n_this_run % save_progress_every == 0:
                self._save_progress(all_results, f"./progress/progress_{len(all_results)}.json")

        total_time = time.time() - t0
        logger.info(f"✓ {n_this_run} questions in {total_time/60:.2f}min (total {len(all_results)})")

        agg = self._aggregate(all_results)
        return {
            "evaluation_info": {
                "total_questions":       len(all_results),
                "evaluation_mode":       "chunk",
                "total_time_seconds":    total_time,
                "avg_time_per_question": total_time / n_this_run if n_this_run else 0.0,
                "timestamp":             time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "aggregate_metrics": agg["aggregate_metrics"],
            "latency_stats":     agg["latency_stats"],
            "failure_analysis":  agg["failure_analysis"],
            "detailed_results":  all_results,
        }

    def _aggregate(self, results: List[Dict]) -> Dict:
        metric_vals: Dict[str, List[float]] = defaultdict(list)
        latencies:   List[float]            = []
        taxonomy:    Dict[str, int]         = {
            "wrong_doc": 0, "wrong_chunk": 0, "below_cutoff": 0, "unknown": 0,
        }

        for r in results:
            if "error" in r:
                continue
            for m, v in r["metrics"].items():
                if v is not None:
                    metric_vals[m].append(float(v))
            latencies.append(r["latency_ms"])
            for cm in r.get("context_match", []):
                ft = cm.get("failure_type")
                if ft in taxonomy:
                    taxonomy[ft] += 1

        agg_metrics: Dict[str, Dict] = {}
        for m, vals in metric_vals.items():
            arr = np.array(vals, dtype=float)
            agg_metrics[m] = {
                "mean":   float(np.mean(arr)),
                "std":    float(np.std(arr)),
                "min":    float(np.min(arr)),
                "max":    float(np.max(arr)),
                "median": float(np.median(arr)),
            }

        if latencies:
            lat_arr      = np.array(latencies, dtype=float)
            latency_stats = {
                "mean_ms":   float(np.mean(lat_arr)),
                "median_ms": float(np.median(lat_arr)),
                "std_ms":    float(np.std(lat_arr)),
                "min_ms":    float(np.min(lat_arr)),
                "max_ms":    float(np.max(lat_arr)),
                "p95_ms":    float(np.percentile(lat_arr, 95)),
                "p99_ms":    float(np.percentile(lat_arr, 99)),
            }
        else:
            latency_stats = {k: 0.0 for k in ["mean_ms","median_ms","std_ms","min_ms","max_ms","p95_ms","p99_ms"]}

        failures = [r for r in results if not r["metrics"].get("found", False)]
        total_missed = sum(taxonomy.values())

        sample_failures = []
        for f in failures[:10]:
            sample_failures.append({
                "question_index":       f["question_index"],
                "question":             f["question"],
                "reference":            f["reference"],
                "reference_contexts":   f.get("reference_contexts", []),
                "got_top3_filenames":   [c.get("filename","") for c in f.get("retrieved_chunks",[])[:3]],
                "missed_context_types": [
                    {
                        "context_preview": cm.get("reference_context","")[:80],
                        "failure_type":    cm.get("failure_type","unknown"),
                    }
                    for cm in f.get("context_match", []) if not cm.get("found", False)
                ],
            })

        return {
            "aggregate_metrics": agg_metrics,
            "latency_stats":     latency_stats,
            "failure_analysis":  {
                "total_failures":  len(failures),
                "failure_rate":    len(failures) / len(results) if results else 0.0,
                "failure_taxonomy": {
                    **taxonomy,
                    "_total_missed_contexts": total_missed,
                },
                "sample_failures": sample_failures,
            },
        }

    def _save_progress(self, results: List[Dict], path: str):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(results, f, indent=2, default=_json_serial)
            logger.info(f"  💾 saved {path}")
        except Exception as e:
            logger.error(f"Could not save progress: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# JSONL writer
# ──────────────────────────────────────────────────────────────────────────────

def _result_to_jsonl_row(result: Dict[str, Any], retriever_name: str) -> Dict:
    context_chunks = []
    for c in result.get("retrieved_chunks", []):
        context_chunks.append({
            "text":         c.get("content", ""),
            "source_id":    c.get("source_id", ""),
            "score":        float(c.get("score") or 0.0),
            "rerank_score": c.get("rerank_score"),
            "rrf_score":    c.get("rrf_score"),
            "dense_score":  c.get("dense_score"),
            "sparse_score": c.get("sparse_score"),
            "hop_index":    c.get("hop_index"),
        })
    return {
        "query":           result["question"],
        "ground_truth":    result.get("reference", ""),
        "expected_chunks": result.get("reference_contexts", []),
        "retriever":       retriever_name,
        "context_chunks":  context_chunks,
    }


class JsonlWriter:
    def __init__(self, path: str, retriever_name: str = "hybrid_qdrr"):
        self.path           = path
        self.retriever_name = retriever_name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(path).exists():
            open(path, "w").close()

    def append(self, result: Dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(_result_to_jsonl_row(result, self.retriever_name), default=_json_serial) + "\n")

    def rewrite_from_results(self, results: List[Dict]):
        with open(self.path, "w") as f:
            for r in results:
                f.write(json.dumps(_result_to_jsonl_row(r, self.retriever_name), default=_json_serial) + "\n")
        logger.info(f"✓ rebuilt {self.path} from {len(results)} results")


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: Dict):
    sep = "=" * 100
    thin = "-" * 100
    print(f"\n{sep}")
    print("RETRIEVAL EVALUATION RESULTS  (chunk-level, QD+RR)")
    print(sep)

    info = results["evaluation_info"]
    print(f"\n{'OVERVIEW':}")
    print(f"  Mode:               {info['evaluation_mode'].upper()}")
    print(f"  Questions:          {info['total_questions']}")
    print(f"  Total time:         {info['total_time_seconds']/60:.2f} min")
    print(f"  Avg per question:   {info['avg_time_per_question']:.2f} s")
    print(f"  Timestamp:          {info['timestamp']}")

    print(f"\n{thin}\nPERFORMANCE\n{thin}")
    agg = results["aggregate_metrics"]

    if "found" in agg:
        sr = agg["found"]["mean"]
        n  = info["total_questions"]
        print(f"\n  Strict success rate (all contexts found):  {sr:.2%}  ({int(sr*n)}/{n})")

    if "mrr" in agg:
        print(f"  MRR (first relevant):                      {agg['mrr']['mean']:.4f}")

    print(f"\n  Precision @ k:")
    for k in [1, 3, 5, 10]:
        if f"precision@{k}" in agg:
            p = agg[f"precision@{k}"]
            print(f"    P@{k:2d}:  {p['mean']:.4f} ± {p['std']:.4f}")

    print(f"\n  Recall @ k:")
    for k in [1, 3, 5, 10]:
        if f"recall@{k}" in agg:
            print(f"    R@{k:2d}:  {agg[f'recall@{k}']['mean']:.4f}")

    print(f"\n  All contexts found @ k  (strict multi-hop):")
    for k in [1, 3, 5, 10]:
        if f"all_found@{k}" in agg:
            a = agg[f"all_found@{k}"]
            print(f"    All@{k:2d}: {a['mean']:.4f}  ({a['mean']:.2%})")

    print(f"\n  NDCG @ k:")
    for k in [1, 3, 5, 10]:
        if f"ndcg@{k}" in agg:
            print(f"    NDCG@{k:2d}: {agg[f'ndcg@{k}']['mean']:.4f}")

    print(f"\n{thin}\nLATENCY\n{thin}")
    lat = results["latency_stats"]
    print(f"  Mean:   {lat['mean_ms']:.1f} ms")
    print(f"  Median: {lat['median_ms']:.1f} ms")
    print(f"  P95:    {lat['p95_ms']:.1f} ms")
    print(f"  P99:    {lat['p99_ms']:.1f} ms")

    print(f"\n{thin}\nFAILURE ANALYSIS\n{thin}")
    fail = results["failure_analysis"]
    print(f"  Total failures:  {fail['total_failures']} ({fail['failure_rate']:.2%})")

    tx = fail.get("failure_taxonomy", {})
    total_missed = tx.get("_total_missed_contexts", sum(v for k,v in tx.items() if not k.startswith("_")))
    if total_missed:
        print(f"\n  Missed-context taxonomy ({total_missed} total):")
        labels = {
            "wrong_doc":    "Wrong document (never retrieved)",
            "wrong_chunk":  "Right doc, wrong chunk",
            "below_cutoff": "Below cutoff (in pool, ranked > top-k)",
            "unknown":      "Unknown",
        }
        for key, label in labels.items():
            cnt = tx.get(key, 0)
            pct = 100 * cnt / total_missed if total_missed else 0
            print(f"    {label:45s}: {cnt:4d}  ({pct:.1f}%)")

    if fail.get("sample_failures"):
        print(f"\n  Sample failures:")
        for i, sf in enumerate(fail["sample_failures"][:5], 1):
            q   = sf["question"]
            ref = sf["reference"][:100] + ("…" if len(sf["reference"]) > 100 else "")
            print(f"\n  {i}. [{sf['question_index']}] {q}")
            print(f"      ref: {ref}")
            print(f"      top-3 files: {', '.join(sf['got_top3_filenames'])}")
            for ct in sf.get("missed_context_types", []):
                print(f"      → [{ct['failure_type']:15s}] {ct['context_preview'][:70]}")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _json_serial(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj).__name__}")


def save_results(results: Dict, path: str):
    def _convert(obj):
        if isinstance(obj, (np.integer, np.int64)):    return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray):                return obj.tolist()
        if isinstance(obj, dict):  return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_convert(v) for v in obj]
        return obj
    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    logger.info(f"✓ results saved to {path}")


def load_questions(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError(f"Cannot find questions list in {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Chunk-level retrieval evaluation (QD+RR)")
    ap.add_argument("--questions",        required=True)
    ap.add_argument("--resume",           default=None)
    ap.add_argument("--collection",       default="Dear_autosar")
    ap.add_argument("--qdrant-url",       default="http://localhost:7333")
    ap.add_argument("--top-k",            type=int,  default=10)
    ap.add_argument("--output",           default="evaluation_results.json")
    ap.add_argument("--jsonl-output",     default="retrieval_results.jsonl")
    ap.add_argument("--retriever-name",   default="hybrid_qdrr")
    ap.add_argument("--log-file",         default=None,
                    help="Path for run log file (e.g. logs/eval_run.log). "
                         "Captures all logger output from both this module and HybridRetriever.")
    ap.add_argument("--verbose",          action="store_true")
    ap.add_argument("--no-reranker",      action="store_true")
    ap.add_argument("--no-decomposition", action="store_true")
    ap.add_argument("--rerank-mode",      default="cross_encoder",
                    choices=["cross_encoder", "colbert"],
                    help="Reranker backend.")
    ap.add_argument("--match-mode",       default=MATCH_MODE,
                    choices=["fuzzy", "embedding"],
                    help="Context-match mode: 'fuzzy' (default) or 'embedding' (BGE-M3 cosine).")
    ap.add_argument("--save-progress",    type=int, default=25)
    ap.add_argument("--store-full-pool",  action="store_true",
                    help="Retrieve extended pool to classify below_cutoff failures (adds latency).")
    ap.add_argument("--decomposition-mode", default="llm",
                    choices=["llm", "context_split"],
                    help="Query decomposition strategy: "
                         "'llm' (default) — retriever uses its internal LLM decomposition; "
                         "'context_split' — split query on (Context N) labels directly, "
                         "no LLM call needed (fast, deterministic, for structured queries).")
    ap.add_argument("--fallback-mode",      default="full_query",
                    choices=["full_query", "llm"],
                    help="Fallback when --decomposition-mode=context_split but a query has no "
                         "(Context N) labels. 'full_query' (default) — send the full query as-is; "
                         "'llm' — let the retriever decompose via LLM. Ignored in llm mode.")
    args = ap.parse_args()

    # ── Logging — must be set up before any logger.info calls ────────────
    # Derive a default log path alongside the output file if not specified.
    log_file = args.log_file or args.output.replace(".json", ".log")
    global logger
    logger = _setup_logging(log_file)

    global STORE_FULL_POOL
    if args.store_full_pool:
        STORE_FULL_POOL = True

    logger.info(f"Loading questions from {args.questions}")
    questions = load_questions(args.questions)
    logger.info(f"✓ {len(questions)} questions loaded")

    logger.info("Initialising retriever…")
    retriever = HybridRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        use_reranker=not args.no_reranker,
        use_decomposition=not args.no_decomposition,
        rerank_mode=args.rerank_mode,
    )

    evaluator    = ComprehensiveEvaluator(
        retriever,
        match_mode=args.match_mode,
        query_decomposition_mode=args.decomposition_mode,
        fallback_mode=args.fallback_mode,
    )
    logger.info(f"Query decomposition mode: {args.decomposition_mode.upper()}")
    if args.decomposition_mode == "context_split":
        logger.info(f"Fallback mode (no labels): {args.fallback_mode.upper()}")
    jsonl_writer = JsonlWriter(args.jsonl_output, args.retriever_name)

    results = evaluator.evaluate_all(
        questions,
        top_k=args.top_k,
        verbose=args.verbose,
        save_progress_every=args.save_progress,
        resume_file=args.resume,
        jsonl_writer=jsonl_writer,
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    save_results(results, args.output)

    # Text summary — write directly to file (no sys.stdout redirect needed)
    summary_path = args.output.replace(".json", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        # Temporarily redirect print() to the file for print_summary
        old_stdout  = sys.stdout
        sys.stdout  = f
        print_summary(results)
        sys.stdout  = old_stdout
    logger.info(f"✓ summary saved to {summary_path}")

    # Also print summary to console
    print_summary(results)

    print(f"\nOutputs saved:")
    print(f"  JSON results:  {args.output}")
    print(f"  Text summary:  {summary_path}")
    print(f"  JSONL rows:    {args.jsonl_output}")
    print(f"  Run log:       {log_file}")


if __name__ == "__main__":
    main()