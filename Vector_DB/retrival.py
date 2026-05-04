# #!/usr/bin/env python3
# """
# RETRIEVAL EVALUATION PIPELINE (chunk-level only)
# =================================================
# Evaluates a hybrid retriever against a Ragas-style ground-truth JSON:

#     [
#       {
#         "user_input": "...",
#         "reference": "...",
#         "reference_contexts": ["chunk text ...", ...],
#         "synthesizer_name": "...",
#         "metadata": {...}
#       },
#       ...
#     ]

# A retrieved chunk counts as a HIT for a question when fuzzy_match(...) finds
# any of that question's reference_contexts inside the chunk's content.

# Two files are produced:
#   1. <o>.json               — detailed metrics + aggregates + failure analysis
#   2. bgem3_retrieval.jsonl  — one JSON object per question, for downstream
#                               retriever-comparison tooling.
# """

# """
# python Evaluate_Retrieval_Takes_Json_Questions.py \
#   --questions evaluation_questions.json \
#   --resume progress_75.json
# OR
# python Evaluate_Retrieval_Takes_Json_Questions.py --questions evaluation_questions.json
# """
# import sys
# import re
# from io import StringIO
# from difflib import SequenceMatcher
# import unicodedata
# import json
# import time
# import argparse
# from pathlib import Path
# from typing import Dict, List, Any
# from collections import defaultdict
# import logging

# import numpy as np
# from Evaluate_Retrieval_With_Reranker_Template import HybridRetriever

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # =============================================================================
# # Fuzzy-match machinery (unchanged — handles PDF-extraction quirks:
# # hyphenated line breaks, space-concatenation, pipe-tables, Key=Value snippets)
# # =============================================================================

# def normalize(text: str) -> str:
#     """Normalize text for comparison: NFKC + line-break hyphens + dash variants."""
#     text = unicodedata.normalize("NFKC", text)
#     # Remove hyphenated line breaks: "word-\nword" → "wordword" (PDF line-wrap artifact)
#     text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
#     return (
#         text
#         .replace("\n", " ")
#         .replace("\u2010", "-")
#         .replace("\u2011", "-")
#         .replace("\u2012", "-")
#         .replace("\u2013", "-")
#     )


# def _despace(text: str) -> str:
#     """Remove all whitespace — used to match space-concatenated PDF text."""
#     return re.sub(r'\s+', '', text)


# def _strip_pipes(text: str) -> str:
#     """Replace pipe-table separators with spaces for natural-language comparison."""
#     return re.sub(r'\s*\|\s*', ' ', text).strip()


# def _kv_to_text(snippet: str) -> str:
#     """
#     Strip 'Key=' prefixes from LLM-synthesized Key=Value snippets.
#     e.g. 'Cause=Bus failure, Remedy=Check cables' → 'Bus failure Check cables'
#     """
#     parts = re.split(r'[,;]\s*', snippet)
#     values = []
#     for part in parts:
#         if '=' in part:
#             _, _, val = part.partition('=')
#             values.append(val.strip())
#         else:
#             values.append(part.strip())
#     return ' '.join(v for v in values if v)


# def _sliding_match(snippet: str, text: str, threshold: float) -> bool:
#     """Sliding-window fuzzy match with correct step-aware padding."""
#     ws = len(snippet)
#     if ws == 0:
#         return False
#     if ws > len(text):
#         return SequenceMatcher(None, snippet, text).ratio() >= threshold
#     step = max(1, ws // 4)
#     for i in range(0, len(text) - ws + 1, step):
#         # Pad by step size so that a snippet starting mid-step is still fully covered.
#         window = text[i: min(i + ws + step, len(text))]
#         if SequenceMatcher(None, snippet, window).ratio() >= threshold:
#             return True
#     return False


# def fuzzy_match(snippet: str, text: str, threshold: float = 0.8) -> bool:
#     """
#     Multi-strategy fuzzy match of *snippet* against *text*.

#     Strategies applied in order (short-circuits on first success):
#     1. Exact substring match (after normalize).
#     2. Sliding-window fuzzy match with fixed-step padding (normalize).
#     3. De-spaced exact match — handles PDF space-concatenation artifacts
#        e.g. snippet 'It is at a low level' vs chunk 'Itisatalowlevel'.
#     4. De-spaced sliding-window fuzzy match.
#     5. Pipe-stripped match — handles pdfplumber pipe-table format.
#     6. Key=Value extraction — handles LLM-synthesized 'Cause=X, Remedy=Y' snippets.
#     """
#     sc = normalize(snippet.lower())
#     tc = normalize(text.lower())

#     # 1. Exact
#     if sc in tc:
#         return True

#     # 2. Sliding window (normalized)
#     if _sliding_match(sc, tc, threshold):
#         return True

#     # 3 & 4. De-spaced (handles concatenated words in snippet OR chunk)
#     sds = _despace(sc)
#     tds = _despace(tc)
#     if sds:
#         if sds in tds:
#             return True
#         if _sliding_match(sds, tds, threshold):
#             return True

#     # 5. Pipe-stripped (handles pdfplumber table chunks: 'A | B | C')
#     tc_no_pipe = _strip_pipes(tc)
#     if sc in tc_no_pipe:
#         return True
#     if _sliding_match(sc, tc_no_pipe, threshold):
#         return True
#     # Also try de-spaced against pipe-stripped
#     tds_no_pipe = _despace(tc_no_pipe)
#     if sds and sds in tds_no_pipe:
#         return True

#     # 6. Key=Value extraction (handles 'Cause=X, Remedy=Y' synthesized snippets)
#     # MIN_KV_TEXT_LENGTH ensures we don't try to match trivially short extracted values
#     MIN_KV_TEXT_LENGTH = 8
#     if '=' in sc:
#         kv = _kv_to_text(sc)
#         kv_ds = _despace(kv)
#         if kv and len(kv) > MIN_KV_TEXT_LENGTH:
#             if kv in tc or kv in tc_no_pipe:
#                 return True
#             if _sliding_match(kv, tc, threshold) or _sliding_match(kv, tc_no_pipe, threshold):
#                 return True
#         if kv_ds and len(kv_ds) > MIN_KV_TEXT_LENGTH:
#             if kv_ds in tds or kv_ds in tds_no_pipe:
#                 return True

#     return False


# # =============================================================================
# # Evaluator
# # =============================================================================

# class ComprehensiveEvaluator:
#     """Chunk-level retrieval evaluator for Ragas-style ground truth."""

#     def __init__(self, retriever: HybridRetriever):
#         self.retriever = retriever

#     # def evaluate_single_question(
#     #         self,
#     #         question_data: Dict[str, Any],
#     #         question_index: int,
#     #         top_k: int = 10,
#     #         verbose: bool = False,
#     #     ) -> Dict[str, Any]:
#     #     """Evaluate a single question with chunk-level metrics."""

#     #     query = question_data["user_input"]
#     #     reference = question_data.get("reference", "")
#     #     reference_contexts = question_data.get("reference_contexts", []) or []

#     #     if verbose:
#     #         logger.info(f"\nQuestion [{question_index}]: {query}")
#     #         logger.info(f"Reference: {reference[:120]}...")

#     #     start_time = time.time()
#     #     try:
#     #         search_results = self.retriever.search(query, top_k=top_k)
#     #         latency = time.time() - start_time

#     #         # Extract retrieved chunks
#     #         retrieved_chunks = []
#     #         for result in search_results:
#     #             retrieved_chunks.append({
#     #                 'filename': result.metadata.get('filename', ''),
#     #                 'content': result.content,
#     #                 'score': result.score,
#     #                 'section': result.metadata.get('section_title', ''),
#     #                 'dense_score': result.dense_score,
#     #                 'sparse_score': result.sparse_score,
#     #                 'rerank_score': result.rerank_score,
#     #             })

#     #         # Chunk-level hit list: a chunk is a hit if ANY reference_context
#     #         # fuzzy-matches inside it.
#     #         HIT = "__hit__"
#     #         MISS = "__no_match__"
#     #         hit_list: List[str] = []
#     #         if not reference_contexts:
#     #             logger.warning(
#     #                 f"No reference_contexts for question [{question_index}] — "
#     #                 f"all chunks will be marked as misses."
#     #             )
#     #             hit_list = [MISS] * len(retrieved_chunks)
#     #         else:
#     #             for chunk in retrieved_chunks:
#     #                 matched = any(
#     #                     fuzzy_match(ctx, chunk["content"])
#     #                     for ctx in reference_contexts
#     #                 )
#     #                 hit_list.append(HIT if matched else MISS)

#     #         found = HIT in hit_list

#     #         # Rank of first hit (1-based)
#     #         try:
#     #             rank = hit_list.index(HIT) + 1
#     #             reciprocal_rank = 1.0 / rank
#     #         except ValueError:
#     #             rank = None
#     #             reciprocal_rank = 0.0

#     #         # Precision@K / Recall@K / NDCG@K
#     #         metrics: Dict[str, Any] = {
#     #             'found': found,
#     #             'rank': rank,
#     #             'mrr': reciprocal_rank,
#     #         }
#     #         k_values = sorted(set([1, 3, 5, 10] + ([top_k] if top_k not in [1, 3, 5, 10] else [])))
#     #         for k in k_values:
#     #             relevant_count = sum(1 for x in hit_list[:k] if x == HIT)
#     #             metrics[f'precision@{k}'] = relevant_count / k
#     #             metrics[f'recall@{k}'] = 1.0 if HIT in hit_list[:k] else 0.0

#     #         for k in k_values:
#     #             dcg = 0.0
#     #             num_relevant_in_list = sum(1 for x in hit_list[:k] if x == HIT)
#     #             for i, x in enumerate(hit_list[:k], start=1):
#     #                 if x == HIT:
#     #                     dcg += 1.0 / np.log2(i + 1)
#     #             idcg = sum(1.0 / np.log2(j + 1) for j in range(1, min(num_relevant_in_list, k) + 1))
#     #             metrics[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0.0

#     #         if verbose:
#     #             logger.info(f"Found: {found}, Rank: {rank}, MRR: {reciprocal_rank:.4f}")

#     #         # Per-context match info for post-hoc inspection
#     #         context_match = []
#     #         for ctx in reference_contexts:
#     #             match = {"reference_context": ctx, "found_in_rank": None, "found_in_filename": None}
#     #             for chunk_rank, chunk in enumerate(retrieved_chunks, start=1):
#     #                 if fuzzy_match(ctx, chunk["content"]):
#     #                     match["found_in_rank"] = chunk_rank
#     #                     match["found_in_filename"] = chunk["filename"]
#     #                     break
#     #             context_match.append(match)

#     #         return {
#     #             'question_index': question_index,
#     #             'question': query,
#     #             'reference': reference,
#     #             'reference_contexts': reference_contexts,
#     #             'context_match': context_match,
#     #             'retrieved_chunks': retrieved_chunks[:5],
#     #             'latency_ms': latency * 1000,
#     #             'metrics': metrics,
#     #         }

#     #     except Exception as e:
#     #         logger.error(f"Error processing question [{question_index}]: {e}")
#     #         k_values = sorted(set([1, 3, 5, 10] + ([top_k] if top_k not in [1, 3, 5, 10] else [])))
#     #         error_metrics: Dict[str, Any] = {'found': 0.0, 'rank': None, 'mrr': 0.0}
#     #         for k in k_values:
#     #             error_metrics[f'precision@{k}'] = 0.0
#     #             error_metrics[f'recall@{k}'] = 0.0
#     #             error_metrics[f'ndcg@{k}'] = 0.0
#     #         return {
#     #             'question_index': question_index,
#     #             'question': query,
#     #             'reference': reference,
#     #             'reference_contexts': reference_contexts,
#     #             'context_match': [],
#     #             'retrieved_chunks': [],
#     #             'latency_ms': 0.0,
#     #             'error': str(e),
#     #             'metrics': error_metrics,
#     #         }

#     def evaluate_single_question(
#         self,
#         question_data: Dict[str, Any],
#         question_index: int,
#         top_k: int = 10,
#         verbose: bool = False,
#     ) -> Dict[str, Any]:
#         """Evaluate a single question with chunk-level metrics."""

#         query = question_data["user_input"]
#         reference = question_data.get("reference", "")
#         reference_contexts = question_data.get("reference_contexts", []) or []

#         if verbose:
#             logger.info(f"\nQuestion [{question_index}]: {query}")
#             logger.info(f"Reference: {reference[:120]}...")

#         start_time = time.time()
#         try:
#             search_results = self.retriever.search(query, top_k=top_k)
#             latency = time.time() - start_time

#             # Extract retrieved chunks
#             retrieved_chunks = []
#             for result in search_results:
#                 retrieved_chunks.append({
#                     'filename': result.metadata.get('filename', ''),
#                     'content': result.content,
#                     'score': result.score,
#                     'section': result.metadata.get('section_title', ''),
#                     'dense_score': result.dense_score,
#                     'sparse_score': result.sparse_score,
#                     'rerank_score': result.rerank_score,
#                 })

#             # Per-context tracking: for each reference context, find the earliest
#             # rank at which it was matched.
#             if not reference_contexts:
#                 logger.warning(
#                     f"No reference_contexts for question [{question_index}] — "
#                     f"all contexts will be marked as not found."
#                 )
#                 context_found = []
#                 context_found_at_rank = []
#             else:
#                 context_found = [False] * len(reference_contexts)
#                 context_found_at_rank = [None] * len(reference_contexts)

#                 for chunk_rank, chunk in enumerate(retrieved_chunks, start=1):
#                     for ctx_idx, ctx in enumerate(reference_contexts):
#                         if not context_found[ctx_idx] and fuzzy_match(ctx, chunk["content"]):
#                             context_found[ctx_idx] = True
#                             context_found_at_rank[ctx_idx] = chunk_rank

#             total_contexts = len(reference_contexts)

#             # found = ALL reference contexts were retrieved somewhere in top_k
#             found = bool(context_found) and all(context_found)

#             # Rank = rank of the last (bottleneck) context found
#             if found:
#                 rank = max(context_found_at_rank)
#                 reciprocal_rank = 1.0 / rank
#             else:
#                 rank = None
#                 reciprocal_rank = 0.0

#             metrics: Dict[str, Any] = {
#                 'found': found,
#                 'rank': rank,
#                 'mrr': reciprocal_rank,
#             }

#             k_values = sorted(set([1, 3, 5, 10] + ([top_k] if top_k not in [1, 3, 5, 10] else [])))

#             for k in k_values:
#                 # How many distinct reference contexts were found within top-k?
#                 contexts_found_in_k = sum(
#                     1 for r in context_found_at_rank if r is not None and r <= k
#                 )

#                 # Recall: fraction of required contexts retrieved in top-k
#                 metrics[f'recall@{k}'] = contexts_found_in_k / total_contexts if total_contexts > 0 else 0.0

#                 # Strict multihop: ALL contexts found within top-k
#                 metrics[f'all_found@{k}'] = 1.0 if (total_contexts > 0 and contexts_found_in_k == total_contexts) else 0.0

#                 # Precision: fraction of top-k chunks that matched any reference context
#                 relevant_chunks_in_k = sum(
#                     1 for chunk in retrieved_chunks[:k]
#                     if any(fuzzy_match(ctx, chunk["content"]) for ctx in reference_contexts)
#                 )
#                 metrics[f'precision@{k}'] = relevant_chunks_in_k / k

#             for k in k_values:
#                 dcg = 0.0
#                 for i, chunk in enumerate(retrieved_chunks[:k], start=1):
#                     if any(fuzzy_match(ctx, chunk["content"]) for ctx in reference_contexts):
#                         dcg += 1.0 / np.log2(i + 1)
#                 num_relevant = sum(1 for r in context_found_at_rank if r is not None and r <= k)
#                 idcg = sum(1.0 / np.log2(j + 1) for j in range(1, min(num_relevant, k) + 1))
#                 metrics[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0.0

#             if verbose:
#                 logger.info(f"Found all: {found}, Bottleneck rank: {rank}, MRR: {reciprocal_rank:.4f}")
#                 logger.info(f"Context coverage: {sum(context_found)}/{total_contexts}")

#             # Per-context match info for post-hoc inspection
#             context_match = []
#             for ctx_idx, ctx in enumerate(reference_contexts):
#                 context_match.append({
#                     "reference_context": ctx,
#                     "found": context_found[ctx_idx] if context_found else False,
#                     "found_in_rank": context_found_at_rank[ctx_idx] if context_found_at_rank else None,
#                     "found_in_filename": (
#                         retrieved_chunks[context_found_at_rank[ctx_idx] - 1]["filename"]
#                         if context_found_at_rank and context_found_at_rank[ctx_idx] is not None
#                         else None
#                     ),
#                 })

#             return {
#                 'question_index': question_index,
#                 'question': query,
#                 'reference': reference,
#                 'reference_contexts': reference_contexts,
#                 'context_match': context_match,
#                 'retrieved_chunks': retrieved_chunks[:5],
#                 'latency_ms': latency * 1000,
#                 'metrics': metrics,
#             }

#         except Exception as e:
#             logger.error(f"Error processing question [{question_index}]: {e}")
#             k_values = sorted(set([1, 3, 5, 10] + ([top_k] if top_k not in [1, 3, 5, 10] else [])))
#             error_metrics: Dict[str, Any] = {'found': 0.0, 'rank': None, 'mrr': 0.0}
#             for k in k_values:
#                 error_metrics[f'precision@{k}'] = 0.0
#                 error_metrics[f'recall@{k}'] = 0.0
#                 error_metrics[f'ndcg@{k}'] = 0.0
#                 error_metrics[f'all_found@{k}'] = 0.0
#             return {
#                 'question_index': question_index,
#                 'question': query,
#                 'reference': reference,
#                 'reference_contexts': reference_contexts,
#                 'context_match': [],
#                 'retrieved_chunks': [],
#                 'latency_ms': 0.0,
#                 'error': str(e),
#                 'metrics': error_metrics,
#             }

#     def evaluate_all(
#         self,
#         questions: List[Dict[str, Any]],
#         top_k: int = 10,
#         verbose: bool = False,
#         save_progress_every: int = 10,
#         resume_file: str = None,
#         jsonl_writer: "JsonlWriter" = None,
#     ) -> Dict[str, Any]:
#         """Evaluate all questions with progress tracking."""

#         logger.info("=" * 80)
#         logger.info(f"EVALUATING {len(questions)} QUESTIONS (chunk-level)")
#         logger.info("=" * 80)

#         all_results: List[Dict[str, Any]] = []
#         completed_indices: set = set()

#         # ------------------ RESUME LOGIC ------------------
#         if resume_file and Path(resume_file).exists():
#             logger.info(f"Resuming from {resume_file}...")
#             with open(resume_file, 'r') as f:
#                 all_results = json.load(f)
#             completed_indices = {r['question_index'] for r in all_results}
#             logger.info(f"✓ Already completed: {len(completed_indices)}")
#             logger.info(f"→ Remaining: {len(questions) - len(completed_indices)} / {len(questions)}")

#             # Rebuild the JSONL file from completed results so we don't duplicate
#             # appended rows on resume.
#             if jsonl_writer is not None:
#                 jsonl_writer.rewrite_from_results(all_results)
#         # ---------------------------------------------------

#         start_time = time.time()
#         processed_this_run = 0

#         for idx, question in enumerate(questions):
#             if idx in completed_indices:
#                 continue

#             processed_this_run += 1
#             if processed_this_run % 10 == 0 or verbose:
#                 elapsed = time.time() - start_time
#                 avg_time = elapsed / processed_this_run
#                 remaining = len(questions) - len(completed_indices) - processed_this_run
#                 eta = avg_time * remaining
#                 logger.info(
#                     f"\nProgress: {processed_this_run}/"
#                     f"{len(questions) - len(completed_indices)} "
#                     f"- ETA: {eta/60:.1f} min"
#                 )

#             result = self.evaluate_single_question(question, idx, top_k, verbose)
#             all_results.append(result)

#             # Append one line to the JSONL file in lock-step.
#             if jsonl_writer is not None:
#                 jsonl_writer.append(result)

#             if save_progress_every and processed_this_run % save_progress_every == 0:
#                 self._save_progress(all_results, f"progress_{len(all_results)}.json")

#         total_time = time.time() - start_time
#         logger.info(
#             f"\n✓ Completed {processed_this_run} questions in {total_time/60:.2f} minutes "
#             f"(total evaluated: {len(all_results)})"
#         )

#         aggregate_results = self._calculate_aggregates(all_results)

#         return {
#             'evaluation_info': {
#                 'total_questions': len(all_results),
#                 'evaluation_mode': 'chunk',
#                 'total_time_seconds': total_time,
#                 'avg_time_per_question': (
#                     total_time / processed_this_run if processed_this_run > 0 else 0.0
#                 ),
#                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
#             },
#             'aggregate_metrics': aggregate_results['aggregate_metrics'],
#             'latency_stats': aggregate_results['latency_stats'],
#             'failure_analysis': aggregate_results['failure_analysis'],
#             'detailed_results': all_results,
#         }

#     def _calculate_aggregates(self, results: List[Dict]) -> Dict:
#         """Calculate overall metrics, latency stats, and failure analysis."""

#         metric_values: Dict[str, List[float]] = defaultdict(list)
#         latencies: List[float] = []

#         for result in results:
#             if 'error' not in result:
#                 for metric_name, value in result['metrics'].items():
#                     if value is not None:
#                         metric_values[metric_name].append(value)
#                 latencies.append(result['latency_ms'])

#         aggregate_metrics: Dict[str, Dict[str, float]] = {}
#         for metric_name, values in metric_values.items():
#             if not values:
#                 continue
#             arr = np.array(values, dtype=float)
#             aggregate_metrics[metric_name] = {
#                 'mean': float(np.mean(arr)),
#                 'std': float(np.std(arr)),
#                 'min': float(np.min(arr)),
#                 'max': float(np.max(arr)),
#                 'median': float(np.median(arr)),
#             }

#         if latencies:
#             latency_stats = {
#                 'mean_ms': float(np.mean(latencies)),
#                 'median_ms': float(np.median(latencies)),
#                 'std_ms': float(np.std(latencies)),
#                 'min_ms': float(np.min(latencies)),
#                 'max_ms': float(np.max(latencies)),
#                 'p95_ms': float(np.percentile(latencies, 95)),
#                 'p99_ms': float(np.percentile(latencies, 99)),
#             }
#         else:
#             latency_stats = {
#                 'mean_ms': 0.0, 'median_ms': 0.0, 'std_ms': 0.0,
#                 'min_ms': 0.0, 'max_ms': 0.0, 'p95_ms': 0.0, 'p99_ms': 0.0,
#             }

#         # Failure analysis — questions where no retrieved chunk was a hit.
#         failures = [r for r in results if not r['metrics'].get('found', False)]
#         failure_analysis: Dict[str, Any] = {
#             'total_failures': len(failures),
#             'failure_rate': len(failures) / len(results) if results else 0.0,
#             'sample_failures': [],
#         }
#         for failure in failures[:10]:
#             top3_filenames = [c.get('filename', '') for c in failure.get('retrieved_chunks', [])[:3]]
#             failure_analysis['sample_failures'].append({
#                 'question_index': failure['question_index'],
#                 'question': failure['question'],
#                 'reference': failure['reference'],
#                 'reference_contexts': failure.get('reference_contexts', []),
#                 'got_top3_filenames': top3_filenames,
#             })

#         return {
#             'aggregate_metrics': aggregate_metrics,
#             'latency_stats': latency_stats,
#             'failure_analysis': failure_analysis,
#         }

#     def _save_progress(self, results: List[Dict], filename: str):
#         try:
#             with open(filename, 'w') as f:
#                 json.dump(results, f, indent=2, default=_json_default)
#             logger.info(f"  💾 Progress saved to {filename}")
#         except Exception as e:
#             logger.error(f"  ✗ Could not save progress: {e}")


# # =============================================================================
# # JSONL writer — produces bgem3_retrieval.jsonl, one row per question
# # =============================================================================

# def _result_to_jsonl_row(result: Dict[str, Any], retriever_name: str) -> Dict[str, Any]:
#     """Convert an internal per-question result into the bgem3_retrieval.jsonl row."""
#     context_chunks = []
#     for i, chunk in enumerate(result.get('retrieved_chunks', [])):
#         filename = chunk.get('filename') or 'unknown'
#         # Synthesize a chunk ID in the style of the target schema:
#         #   doc_<filename-without-ext>_chunk_<rank-index>
#         stem = Path(filename).stem if filename != 'unknown' else 'unknown'
#         context_chunks.append({
#             'text': chunk.get('content', ''),
#             'source_id': f'doc_{stem}_chunk_{i}',
#             'score': float(chunk.get('score') or 0.0),
#         })
#     return {
#         'query': result['question'],
#         'ground_truth': result.get('reference', ''),
#         'expected_chunks': result.get('reference_contexts', []),
#         'retriever': retriever_name,
#         'context_chunks': context_chunks,
#     }


# class JsonlWriter:
#     """Append-only writer for bgem3_retrieval.jsonl.  Supports rebuild-on-resume."""

#     def __init__(self, path: str, retriever_name: str):
#         self.path = path
#         self.retriever_name = retriever_name
#         # Ensure the file exists (truncated to empty) at start so fresh runs don't
#         # accidentally append to stale data.
#         Path(self.path).parent.mkdir(parents=True, exist_ok=True)
#         if not Path(self.path).exists():
#             open(self.path, 'w').close()

#     def append(self, result: Dict[str, Any]):
#         row = _result_to_jsonl_row(result, self.retriever_name)
#         with open(self.path, 'a') as f:
#             f.write(json.dumps(row, default=_json_default) + '\n')

#     def rewrite_from_results(self, results: List[Dict[str, Any]]):
#         """Called on resume — truncate and re-emit from all completed results."""
#         with open(self.path, 'w') as f:
#             for r in results:
#                 row = _result_to_jsonl_row(r, self.retriever_name)
#                 f.write(json.dumps(row, default=_json_default) + '\n')
#         logger.info(f"✓ Rebuilt {self.path} from {len(results)} completed results")


# # =============================================================================
# # Reporting
# # =============================================================================

# # def print_detailed_summary(results: Dict):
# #     """Print evaluation summary (overall metrics + latency + failures)."""

# #     print("\n" + "=" * 100)
# #     print("RETRIEVAL EVALUATION RESULTS  (chunk-level)")
# #     print("=" * 100)

# #     info = results['evaluation_info']
# #     print(f"\n📊 OVERVIEW:")
# #     print(f"   Evaluation Mode: {info['evaluation_mode'].upper()}")
# #     print(f"   Total Questions: {info['total_questions']}")
# #     print(f"   Total Time: {info['total_time_seconds']/60:.2f} minutes")
# #     print(f"   Avg Time/Question: {info['avg_time_per_question']:.2f} seconds")
# #     print(f"   Timestamp: {info['timestamp']}")

# #     print("\n" + "-" * 100)
# #     print("OVERALL PERFORMANCE")
# #     print("-" * 100)

# #     agg = results['aggregate_metrics']

# #     if 'found' in agg:
# #         success = agg['found']['mean']
# #         print(
# #             f"\n✅ SUCCESS RATE: {success:.2%} "
# #             f"({int(success * info['total_questions'])}/{info['total_questions']})"
# #         )

# #     if 'mrr' in agg:
# #         print(f"\n🎯 MEAN RECIPROCAL RANK: {agg['mrr']['mean']:.4f}")

# #     print(f"\n📍 PRECISION @ K:")
# #     for k in [1, 3, 5, 10]:
# #         if f'precision@{k}' in agg:
# #             p = agg[f'precision@{k}']
# #             print(f"   P@{k:2d}: {p['mean']:.4f} ± {p['std']:.4f} (min: {p['min']:.4f}, max: {p['max']:.4f})")

# #     print(f"\n📊 RECALL @ K:")
# #     for k in [1, 3, 5, 10]:
# #         if f'recall@{k}' in agg:
# #             r = agg[f'recall@{k}']
# #             print(f"   R@{k:2d}: {r['mean']:.4f}")

# #     print(f"\n📊 NDCG @ K:")
# #     for k in [1, 3, 5, 10]:
# #         if f'ndcg@{k}' in agg:
# #             ndcg = agg[f'ndcg@{k}']
# #             print(f"   NDCG@{k:2d}: {ndcg['mean']:.4f}")

# #     # Latency
# #     print("\n" + "-" * 100)
# #     print("LATENCY STATISTICS")
# #     print("-" * 100)

# #     lat = results['latency_stats']
# #     print(f"\n   Mean:   {lat['mean_ms']:.2f} ms")
# #     print(f"   Median: {lat['median_ms']:.2f} ms")
# #     print(f"   Std:    {lat['std_ms']:.2f} ms")
# #     print(f"   Min:    {lat['min_ms']:.2f} ms")
# #     print(f"   Max:    {lat['max_ms']:.2f} ms")
# #     print(f"   P95:    {lat['p95_ms']:.2f} ms")
# #     print(f"   P99:    {lat['p99_ms']:.2f} ms")

# #     # Failure analysis
# #     print("\n" + "-" * 100)
# #     print("FAILURE ANALYSIS")
# #     print("-" * 100)

# #     fail = results['failure_analysis']
# #     print(f"\n   Total Failures: {fail['total_failures']} ({fail['failure_rate']:.2%})")

# #     if fail['sample_failures']:
# #         print(f"\n   Sample Failed Queries:")
# #         for i, failure in enumerate(fail['sample_failures'][:5], 1):
# #             print(f"\n   {i}. [idx {failure['question_index']}] {failure['question']}")
# #             ref_preview = failure['reference'][:100] + ('...' if len(failure['reference']) > 100 else '')
# #             print(f"      Reference: {ref_preview}")
# #             if failure['got_top3_filenames']:
# #                 print(f"      Top-3 filenames: {', '.join(failure['got_top3_filenames'])}")

# #     print("\n" + "=" * 100)

# def print_detailed_summary(results: Dict):
#     """Print evaluation summary (overall metrics + latency + failures)."""

#     print("\n" + "=" * 100)
#     print("RETRIEVAL EVALUATION RESULTS  (chunk-level)")
#     print("=" * 100)

#     info = results['evaluation_info']
#     print(f"\n📊 OVERVIEW:")
#     print(f"   Evaluation Mode: {info['evaluation_mode'].upper()}")
#     print(f"   Total Questions: {info['total_questions']}")
#     print(f"   Total Time: {info['total_time_seconds']/60:.2f} minutes")
#     print(f"   Avg Time/Question: {info['avg_time_per_question']:.2f} seconds")
#     print(f"   Timestamp: {info['timestamp']}")

#     print("\n" + "-" * 100)
#     print("OVERALL PERFORMANCE")
#     print("-" * 100)

#     agg = results['aggregate_metrics']

#     if 'found' in agg:
#         success = agg['found']['mean']
#         print(
#             f"\n✅ STRICT MULTIHOP SUCCESS RATE (all contexts found): {success:.2%} "
#             f"({int(success * info['total_questions'])}/{info['total_questions']})"
#         )

#     if 'mrr' in agg:
#         print(f"\n🎯 MEAN RECIPROCAL RANK (bottleneck hop): {agg['mrr']['mean']:.4f}")

#     print(f"\n📍 PRECISION @ K (fraction of retrieved chunks that matched any context):")
#     for k in [1, 3, 5, 10]:
#         if f'precision@{k}' in agg:
#             p = agg[f'precision@{k}']
#             print(f"   P@{k:2d}: {p['mean']:.4f} ± {p['std']:.4f} (min: {p['min']:.4f}, max: {p['max']:.4f})")

#     print(f"\n📊 RECALL @ K (avg fraction of required contexts retrieved):")
#     for k in [1, 3, 5, 10]:
#         if f'recall@{k}' in agg:
#             r = agg[f'recall@{k}']
#             print(f"   R@{k:2d}: {r['mean']:.4f}")

#     print(f"\n🎯 STRICT MULTIHOP SUCCESS @ K (ALL contexts found within top-k):")
#     for k in [1, 3, 5, 10]:
#         if f'all_found@{k}' in agg:
#             a = agg[f'all_found@{k}']
#             print(f"   All@{k:2d}: {a['mean']:.4f} ({a['mean']:.2%})")

#     print(f"\n📊 NDCG @ K:")
#     for k in [1, 3, 5, 10]:
#         if f'ndcg@{k}' in agg:
#             ndcg = agg[f'ndcg@{k}']
#             print(f"   NDCG@{k:2d}: {ndcg['mean']:.4f}")

#     print("\n" + "-" * 100)
#     print("LATENCY STATISTICS")
#     print("-" * 100)

#     lat = results['latency_stats']
#     print(f"\n   Mean:   {lat['mean_ms']:.2f} ms")
#     print(f"   Median: {lat['median_ms']:.2f} ms")
#     print(f"   Std:    {lat['std_ms']:.2f} ms")
#     print(f"   Min:    {lat['min_ms']:.2f} ms")
#     print(f"   Max:    {lat['max_ms']:.2f} ms")
#     print(f"   P95:    {lat['p95_ms']:.2f} ms")
#     print(f"   P99:    {lat['p99_ms']:.2f} ms")

#     print("\n" + "-" * 100)
#     print("FAILURE ANALYSIS")
#     print("-" * 100)

#     fail = results['failure_analysis']
#     print(f"\n   Total Failures: {fail['total_failures']} ({fail['failure_rate']:.2%})")

#     if fail['sample_failures']:
#         print(f"\n   Sample Failed Queries:")
#         for i, failure in enumerate(fail['sample_failures'][:5], 1):
#             print(f"\n   {i}. [idx {failure['question_index']}] {failure['question']}")
#             ref_preview = failure['reference'][:100] + ('...' if len(failure['reference']) > 100 else '')
#             print(f"      Reference: {ref_preview}")
#             if failure.get('got_top3_filenames'):
#                 print(f"      Top-3 filenames: {', '.join(failure['got_top3_filenames'])}")
#             total = len(failure.get('reference_contexts', []))
#             # Count how many contexts were partially found via context_match if available
#             print(f"      Required contexts: {total}")

#     print("\n" + "=" * 100)

# # =============================================================================
# # JSON serialization helpers
# # =============================================================================

# def _json_default(obj):
#     """Convert numpy scalars/arrays for JSON serialization."""
#     if isinstance(obj, (np.integer,)):
#         return int(obj)
#     if isinstance(obj, (np.floating,)):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# def save_results(results: Dict, output_path: str):
#     """Save detailed results as pretty-printed JSON."""

#     def convert_types(obj):
#         if isinstance(obj, (np.integer, np.int64)):
#             return int(obj)
#         elif isinstance(obj, (np.floating, np.float64)):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, dict):
#             return {k: convert_types(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [convert_types(item) for item in obj]
#         return obj

#     results = convert_types(results)

#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=2)

#     logger.info(f"\n✓ Results saved to: {output_path}")


# # =============================================================================
# # Main
# # =============================================================================

# def _load_questions(path: str) -> List[Dict[str, Any]]:
#     """
#     Load GT questions.  Accepts either:
#       - a plain list of question objects: [ {...}, {...} ]
#       - an object wrapping them:          { "questions": [...] }
#     """
#     with open(path, 'r') as f:
#         data = json.load(f)
#     if isinstance(data, list):
#         return data
#     if isinstance(data, dict) and 'questions' in data:
#         return data['questions']
#     raise ValueError(
#         f"Could not find questions in {path}. "
#         f"Expected a list or an object with a 'questions' key."
#     )


# def main():
#     parser = argparse.ArgumentParser(
#         description="Chunk-level retrieval evaluation against Ragas-style GT"
#     )
#     parser.add_argument('--questions', type=str, required=True,
#                         help='Path to GT JSON (list of {user_input, reference, reference_contexts, ...})')
#     parser.add_argument('--resume', type=str, default=None,
#                         help='Path to progress JSON file to resume from')
#     parser.add_argument('--collection', type=str, default='Autosar_chunks_ragas',
#                         help='Qdrant collection name')
#     parser.add_argument('--qdrant-url', type=str, default='http://localhost:7333',
#                         help='Qdrant URL')
#     parser.add_argument('--top-k', type=int, default=10,
#                         help='Number of results to retrieve per question')
#     parser.add_argument('--output', type=str, default='complete_evaluation_results.json',
#                         help='Output JSON file for detailed metrics')
#     parser.add_argument('--jsonl-output', type=str, default='bgem3_retrieval.jsonl',
#                         help='Output JSONL file, one row per question, for retriever comparison')
#     parser.add_argument('--retriever-name', type=str, default='bge_m3',
#                         help='Name recorded in each JSONL row\'s "retriever" field')
#     parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
#     parser.add_argument('--no-ollama', action='store_true', help='Disable Ollama BGE-M3')
#     parser.add_argument('--no-reranker', action='store_true', help='Disable cross-encoder reranking')
#     parser.add_argument('--save-progress', type=int, default=25,
#                         help='Save progress every N questions (0 to disable)')

#     args = parser.parse_args()

#     # Load questions
#     logger.info(f"Loading questions from {args.questions}...")
#     questions = _load_questions(args.questions)
#     logger.info(f"✓ Loaded {len(questions)} questions")

#     # Initialize retriever
#     logger.info("\nInitializing hybrid retriever...")
#     retriever = HybridRetriever(
#         qdrant_url=args.qdrant_url,
#         collection_name=args.collection,
#         use_ollama=not args.no_ollama,
#         use_reranker=not args.no_reranker,
#     )
#     logger.info("✓ Retriever initialized")

#     # Initialize evaluator + JSONL writer
#     evaluator = ComprehensiveEvaluator(retriever)
#     jsonl_writer = JsonlWriter(args.jsonl_output, args.retriever_name)

#     # Run evaluation
#     logger.info("\nStarting evaluation...\n")
#     results = evaluator.evaluate_all(
#         questions,
#         top_k=args.top_k,
#         verbose=args.verbose,
#         save_progress_every=args.save_progress,
#         resume_file=args.resume,
#         jsonl_writer=jsonl_writer,
#     )

#     # Print + save
#     print_detailed_summary(results)
#     save_results(results, args.output)

#     # Text summary next to the JSON
#     summary_path = args.output.replace('.json', '_summary.txt')
#     with open(summary_path, 'w') as f:
#         old_stdout = sys.stdout
#         try:
#             sys.stdout = StringIO()
#             print_detailed_summary(results)
#             summary_text = sys.stdout.getvalue()
#         finally:
#             sys.stdout = old_stdout
#         f.write(summary_text)
#     logger.info(f"✓ Summary saved to: {summary_path}")

#     print("\n" + "=" * 100)
#     print("EVALUATION COMPLETE!")
#     print("=" * 100)
#     print(f"\nResults saved to:")
#     print(f"  • Detailed JSON: {args.output}")
#     print(f"  • Text Summary:  {summary_path}")
#     print(f"  • JSONL rows:    {args.jsonl_output}")
#     print("=" * 100 + "\n")


# if __name__ == "__main__":
#     main()

# # #!/usr/bin/env python3
# # """
# # RETRIEVAL EVALUATION PIPELINE (chunk-level only)
# # =================================================
# # Evaluates a hybrid retriever against a Ragas-style ground-truth JSON:

# #     [
# #       {
# #         "user_input": "...",
# #         "reference": "...",
# #         "reference_contexts": ["chunk text ...", ...],
# #         "synthesizer_name": "...",
# #         "metadata": {...}
# #       },
# #       ...
# #     ]

# # A retrieved chunk counts as a HIT for a question when fuzzy_match(...) finds
# # any of that question's reference_contexts inside the chunk's content.

# # Two files are produced:
# #   1. <o>.json               — detailed metrics + aggregates + failure analysis
# #   2. bgem3_retrieval.jsonl  — one JSON object per question, for downstream
# #                               retriever-comparison tooling.
# # """

# # """
# # python Evaluate_Retrieval_Takes_Json_Questions.py \
# #   --questions evaluation_questions.json \
# #   --resume progress_75.json
# # OR
# # python Evaluate_Retrieval_Takes_Json_Questions.py --questions evaluation_questions.json
# # """
# # import sys
# # import re
# # from io import StringIO
# # from difflib import SequenceMatcher
# # import unicodedata
# # import json
# # import time
# # import argparse
# # from pathlib import Path
# # from typing import Dict, List, Any
# # from collections import defaultdict
# # import logging

# # import numpy as np
# # from sentence_transformers import SentenceTransformer
# # from Evaluate_Retrieval_With_Reranker_Template import HybridRetriever


# # # =============================================================================
# # # Sentence-Transformers embedding wrapper
# # # =============================================================================

# # class SentenceTransformerEmbedder:
# #     """Dense embedding model using sentence-transformers with BAAI/bge-m3."""

# #     MODEL_NAME = "BAAI/bge-m3"

# #     def __init__(self):
# #         import torch
# #         device = "cuda" if torch.cuda.is_available() else "cpu"
# #         logger.info(f"Loading SentenceTransformer model '{self.MODEL_NAME}' on {device} ...")
# #         self.model = SentenceTransformer(self.MODEL_NAME, device=device)
# #         self.device = device
# #         logger.info("✓ SentenceTransformer model loaded")

# #     def encode(self, texts, batch_size: int = 32, **kwargs) -> np.ndarray:
# #         """
# #         Primary entry point — matches the interface HybridRetriever calls.
# #         Accepts a single string or a list of strings.
# #         Always returns L2-normalised float32 embeddings.
# #         """
# #         if isinstance(texts, str):
# #             texts = [texts]
# #         embeddings = self.model.encode(
# #             texts,
# #             batch_size=batch_size,
# #             normalize_embeddings=True,
# #             show_progress_bar=False,
# #             **kwargs,
# #         )
# #         return np.array(embeddings, dtype=np.float32)

# #     def embed(self, texts, batch_size: int = 32) -> np.ndarray:
# #         """Alias kept for compatibility."""
# #         return self.encode(texts, batch_size=batch_size)

# #     def embed_query(self, text: str) -> np.ndarray:
# #         """Embed a single query string."""
# #         return self.encode([text])[0]

# # # Setup logging
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(levelname)s - %(message)s'
# # )
# # logger = logging.getLogger(__name__)


# # # =============================================================================
# # # Fuzzy-match machinery (unchanged — handles PDF-extraction quirks:
# # # hyphenated line breaks, space-concatenation, pipe-tables, Key=Value snippets)
# # # =============================================================================

# # def normalize(text: str) -> str:
# #     """Normalize text for comparison: NFKC + line-break hyphens + dash variants."""
# #     text = unicodedata.normalize("NFKC", text)
# #     # Remove hyphenated line breaks: "word-\nword" → "wordword" (PDF line-wrap artifact)
# #     text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
# #     return (
# #         text
# #         .replace("\n", " ")
# #         .replace("\u2010", "-")
# #         .replace("\u2011", "-")
# #         .replace("\u2012", "-")
# #         .replace("\u2013", "-")
# #     )


# # def _despace(text: str) -> str:
# #     """Remove all whitespace — used to match space-concatenated PDF text."""
# #     return re.sub(r'\s+', '', text)


# # def _strip_pipes(text: str) -> str:
# #     """Replace pipe-table separators with spaces for natural-language comparison."""
# #     return re.sub(r'\s*\|\s*', ' ', text).strip()


# # def _kv_to_text(snippet: str) -> str:
# #     """
# #     Strip 'Key=' prefixes from LLM-synthesized Key=Value snippets.
# #     e.g. 'Cause=Bus failure, Remedy=Check cables' → 'Bus failure Check cables'
# #     """
# #     parts = re.split(r'[,;]\s*', snippet)
# #     values = []
# #     for part in parts:
# #         if '=' in part:
# #             _, _, val = part.partition('=')
# #             values.append(val.strip())
# #         else:
# #             values.append(part.strip())
# #     return ' '.join(v for v in values if v)


# # def _sliding_match(snippet: str, text: str, threshold: float) -> bool:
# #     """Sliding-window fuzzy match with correct step-aware padding."""
# #     ws = len(snippet)
# #     if ws == 0:
# #         return False
# #     if ws > len(text):
# #         return SequenceMatcher(None, snippet, text).ratio() >= threshold
# #     step = max(1, ws // 4)
# #     for i in range(0, len(text) - ws + 1, step):
# #         # Pad by step size so that a snippet starting mid-step is still fully covered.
# #         window = text[i: min(i + ws + step, len(text))]
# #         if SequenceMatcher(None, snippet, window).ratio() >= threshold:
# #             return True
# #     return False


# # def fuzzy_match(snippet: str, text: str, threshold: float = 0.8) -> bool:
# #     """
# #     Multi-strategy fuzzy match of *snippet* against *text*.

# #     Strategies applied in order (short-circuits on first success):
# #     1. Exact substring match (after normalize).
# #     2. Sliding-window fuzzy match with fixed-step padding (normalize).
# #     3. De-spaced exact match — handles PDF space-concatenation artifacts
# #        e.g. snippet 'It is at a low level' vs chunk 'Itisatalowlevel'.
# #     4. De-spaced sliding-window fuzzy match.
# #     5. Pipe-stripped match — handles pdfplumber pipe-table format.
# #     6. Key=Value extraction — handles LLM-synthesized 'Cause=X, Remedy=Y' snippets.
# #     """
# #     sc = normalize(snippet.lower())
# #     tc = normalize(text.lower())

# #     # 1. Exact
# #     if sc in tc:
# #         return True

# #     # 2. Sliding window (normalized)
# #     if _sliding_match(sc, tc, threshold):
# #         return True

# #     # 3 & 4. De-spaced (handles concatenated words in snippet OR chunk)
# #     sds = _despace(sc)
# #     tds = _despace(tc)
# #     if sds:
# #         if sds in tds:
# #             return True
# #         if _sliding_match(sds, tds, threshold):
# #             return True

# #     # 5. Pipe-stripped (handles pdfplumber table chunks: 'A | B | C')
# #     tc_no_pipe = _strip_pipes(tc)
# #     if sc in tc_no_pipe:
# #         return True
# #     if _sliding_match(sc, tc_no_pipe, threshold):
# #         return True
# #     # Also try de-spaced against pipe-stripped
# #     tds_no_pipe = _despace(tc_no_pipe)
# #     if sds and sds in tds_no_pipe:
# #         return True

# #     # 6. Key=Value extraction (handles 'Cause=X, Remedy=Y' synthesized snippets)
# #     # MIN_KV_TEXT_LENGTH ensures we don't try to match trivially short extracted values
# #     MIN_KV_TEXT_LENGTH = 8
# #     if '=' in sc:
# #         kv = _kv_to_text(sc)
# #         kv_ds = _despace(kv)
# #         if kv and len(kv) > MIN_KV_TEXT_LENGTH:
# #             if kv in tc or kv in tc_no_pipe:
# #                 return True
# #             if _sliding_match(kv, tc, threshold) or _sliding_match(kv, tc_no_pipe, threshold):
# #                 return True
# #         if kv_ds and len(kv_ds) > MIN_KV_TEXT_LENGTH:
# #             if kv_ds in tds or kv_ds in tds_no_pipe:
# #                 return True

# #     return False


# # # =============================================================================
# # # Evaluator
# # # =============================================================================

# # class ComprehensiveEvaluator:
# #     """Chunk-level retrieval evaluator for Ragas-style ground truth."""

# #     def __init__(self, retriever: HybridRetriever):
# #         self.retriever = retriever

# #     def evaluate_single_question(
# #         self,
# #         question_data: Dict[str, Any],
# #         question_index: int,
# #         top_k: int = 10,
# #         verbose: bool = False,
# #     ) -> Dict[str, Any]:
# #         """Evaluate a single question with chunk-level metrics."""

# #         query = question_data["user_input"]
# #         reference = question_data.get("reference", "")
# #         reference_contexts = question_data.get("reference_contexts", []) or []

# #         if verbose:
# #             logger.info(f"\nQuestion [{question_index}]: {query}")
# #             logger.info(f"Reference: {reference[:120]}...")

# #         start_time = time.time()
# #         try:
# #             search_results = self.retriever.search(query, top_k=top_k)
# #             latency = time.time() - start_time

# #             # Extract retrieved chunks
# #             retrieved_chunks = []
# #             for result in search_results:
# #                 retrieved_chunks.append({
# #                     'filename': result.metadata.get('filename', ''),
# #                     'content': result.content,
# #                     'score': result.score,
# #                     'section': result.metadata.get('section_title', ''),
# #                     'dense_score': result.dense_score,
# #                     'sparse_score': result.sparse_score,
# #                     'rerank_score': result.rerank_score,
# #                 })

# #             # Chunk-level hit list: a chunk is a hit if ANY reference_context
# #             # fuzzy-matches inside it.
# #             HIT = "__hit__"
# #             MISS = "__no_match__"
# #             hit_list: List[str] = []
# #             if not reference_contexts:
# #                 logger.warning(
# #                     f"No reference_contexts for question [{question_index}] — "
# #                     f"all chunks will be marked as misses."
# #                 )
# #                 hit_list = [MISS] * len(retrieved_chunks)
# #             else:
# #                 for chunk in retrieved_chunks:
# #                     matched = any(
# #                         fuzzy_match(ctx, chunk["content"])
# #                         for ctx in reference_contexts
# #                     )
# #                     hit_list.append(HIT if matched else MISS)

# #             found = HIT in hit_list

# #             # Rank of first hit (1-based)
# #             try:
# #                 rank = hit_list.index(HIT) + 1
# #                 reciprocal_rank = 1.0 / rank
# #             except ValueError:
# #                 rank = None
# #                 reciprocal_rank = 0.0

# #             # Precision@K / Recall@K / NDCG@K
# #             metrics: Dict[str, Any] = {
# #                 'found': found,
# #                 'rank': rank,
# #                 'mrr': reciprocal_rank,
# #             }
# #             k_values = sorted(set([1, 3, 5, 10] + ([top_k] if top_k not in [1, 3, 5, 10] else [])))
# #             for k in k_values:
# #                 relevant_count = sum(1 for x in hit_list[:k] if x == HIT)
# #                 metrics[f'precision@{k}'] = relevant_count / k
# #                 metrics[f'recall@{k}'] = 1.0 if HIT in hit_list[:k] else 0.0

# #             for k in k_values:
# #                 dcg = 0.0
# #                 num_relevant_in_list = sum(1 for x in hit_list[:k] if x == HIT)
# #                 for i, x in enumerate(hit_list[:k], start=1):
# #                     if x == HIT:
# #                         dcg += 1.0 / np.log2(i + 1)
# #                 idcg = sum(1.0 / np.log2(j + 1) for j in range(1, min(num_relevant_in_list, k) + 1))
# #                 metrics[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0.0

# #             if verbose:
# #                 logger.info(f"Found: {found}, Rank: {rank}, MRR: {reciprocal_rank:.4f}")

# #             # Per-context match info for post-hoc inspection
# #             context_match = []
# #             for ctx in reference_contexts:
# #                 match = {"reference_context": ctx, "found_in_rank": None, "found_in_filename": None}
# #                 for chunk_rank, chunk in enumerate(retrieved_chunks, start=1):
# #                     if fuzzy_match(ctx, chunk["content"]):
# #                         match["found_in_rank"] = chunk_rank
# #                         match["found_in_filename"] = chunk["filename"]
# #                         break
# #                 context_match.append(match)

# #             return {
# #                 'question_index': question_index,
# #                 'question': query,
# #                 'reference': reference,
# #                 'reference_contexts': reference_contexts,
# #                 'context_match': context_match,
# #                 'retrieved_chunks': retrieved_chunks[:5],
# #                 'latency_ms': latency * 1000,
# #                 'metrics': metrics,
# #             }

# #         except Exception as e:
# #             logger.error(f"Error processing question [{question_index}]: {e}")
# #             k_values = sorted(set([1, 3, 5, 10] + ([top_k] if top_k not in [1, 3, 5, 10] else [])))
# #             error_metrics: Dict[str, Any] = {'found': 0.0, 'rank': None, 'mrr': 0.0}
# #             for k in k_values:
# #                 error_metrics[f'precision@{k}'] = 0.0
# #                 error_metrics[f'recall@{k}'] = 0.0
# #                 error_metrics[f'ndcg@{k}'] = 0.0
# #             return {
# #                 'question_index': question_index,
# #                 'question': query,
# #                 'reference': reference,
# #                 'reference_contexts': reference_contexts,
# #                 'context_match': [],
# #                 'retrieved_chunks': [],
# #                 'latency_ms': 0.0,
# #                 'error': str(e),
# #                 'metrics': error_metrics,
# #             }

# #     def evaluate_all(
# #         self,
# #         questions: List[Dict[str, Any]],
# #         top_k: int = 10,
# #         verbose: bool = False,
# #         save_progress_every: int = 10,
# #         resume_file: str = None,
# #         jsonl_writer: "JsonlWriter" = None,
# #     ) -> Dict[str, Any]:
# #         """Evaluate all questions with progress tracking."""

# #         logger.info("=" * 80)
# #         logger.info(f"EVALUATING {len(questions)} QUESTIONS (chunk-level)")
# #         logger.info("=" * 80)

# #         all_results: List[Dict[str, Any]] = []
# #         completed_indices: set = set()

# #         # ------------------ RESUME LOGIC ------------------
# #         if resume_file and Path(resume_file).exists():
# #             logger.info(f"Resuming from {resume_file}...")
# #             with open(resume_file, 'r') as f:
# #                 all_results = json.load(f)
# #             completed_indices = {r['question_index'] for r in all_results}
# #             logger.info(f"✓ Already completed: {len(completed_indices)}")
# #             logger.info(f"→ Remaining: {len(questions) - len(completed_indices)} / {len(questions)}")

# #             # Rebuild the JSONL file from completed results so we don't duplicate
# #             # appended rows on resume.
# #             if jsonl_writer is not None:
# #                 jsonl_writer.rewrite_from_results(all_results)
# #         # ---------------------------------------------------

# #         start_time = time.time()
# #         processed_this_run = 0

# #         for idx, question in enumerate(questions):
# #             if idx in completed_indices:
# #                 continue

# #             processed_this_run += 1
# #             if processed_this_run % 10 == 0 or verbose:
# #                 elapsed = time.time() - start_time
# #                 avg_time = elapsed / processed_this_run
# #                 remaining = len(questions) - len(completed_indices) - processed_this_run
# #                 eta = avg_time * remaining
# #                 logger.info(
# #                     f"\nProgress: {processed_this_run}/"
# #                     f"{len(questions) - len(completed_indices)} "
# #                     f"- ETA: {eta/60:.1f} min"
# #                 )

# #             result = self.evaluate_single_question(question, idx, top_k, verbose)
# #             all_results.append(result)

# #             # Append one line to the JSONL file in lock-step.
# #             if jsonl_writer is not None:
# #                 jsonl_writer.append(result)

# #             if save_progress_every and processed_this_run % save_progress_every == 0:
# #                 self._save_progress(all_results, f"progress_{len(all_results)}.json")

# #         total_time = time.time() - start_time
# #         logger.info(
# #             f"\n✓ Completed {processed_this_run} questions in {total_time/60:.2f} minutes "
# #             f"(total evaluated: {len(all_results)})"
# #         )

# #         aggregate_results = self._calculate_aggregates(all_results)

# #         return {
# #             'evaluation_info': {
# #                 'total_questions': len(all_results),
# #                 'evaluation_mode': 'chunk',
# #                 'total_time_seconds': total_time,
# #                 'avg_time_per_question': (
# #                     total_time / processed_this_run if processed_this_run > 0 else 0.0
# #                 ),
# #                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
# #             },
# #             'aggregate_metrics': aggregate_results['aggregate_metrics'],
# #             'latency_stats': aggregate_results['latency_stats'],
# #             'failure_analysis': aggregate_results['failure_analysis'],
# #             'detailed_results': all_results,
# #         }

# #     def _calculate_aggregates(self, results: List[Dict]) -> Dict:
# #         """Calculate overall metrics, latency stats, and failure analysis."""

# #         metric_values: Dict[str, List[float]] = defaultdict(list)
# #         latencies: List[float] = []

# #         for result in results:
# #             if 'error' not in result:
# #                 for metric_name, value in result['metrics'].items():
# #                     if value is not None:
# #                         metric_values[metric_name].append(value)
# #                 latencies.append(result['latency_ms'])

# #         aggregate_metrics: Dict[str, Dict[str, float]] = {}
# #         for metric_name, values in metric_values.items():
# #             if not values:
# #                 continue
# #             arr = np.array(values, dtype=float)
# #             aggregate_metrics[metric_name] = {
# #                 'mean': float(np.mean(arr)),
# #                 'std': float(np.std(arr)),
# #                 'min': float(np.min(arr)),
# #                 'max': float(np.max(arr)),
# #                 'median': float(np.median(arr)),
# #             }

# #         if latencies:
# #             latency_stats = {
# #                 'mean_ms': float(np.mean(latencies)),
# #                 'median_ms': float(np.median(latencies)),
# #                 'std_ms': float(np.std(latencies)),
# #                 'min_ms': float(np.min(latencies)),
# #                 'max_ms': float(np.max(latencies)),
# #                 'p95_ms': float(np.percentile(latencies, 95)),
# #                 'p99_ms': float(np.percentile(latencies, 99)),
# #             }
# #         else:
# #             latency_stats = {
# #                 'mean_ms': 0.0, 'median_ms': 0.0, 'std_ms': 0.0,
# #                 'min_ms': 0.0, 'max_ms': 0.0, 'p95_ms': 0.0, 'p99_ms': 0.0,
# #             }

# #         # Failure analysis — questions where no retrieved chunk was a hit.
# #         failures = [r for r in results if not r['metrics'].get('found', False)]
# #         failure_analysis: Dict[str, Any] = {
# #             'total_failures': len(failures),
# #             'failure_rate': len(failures) / len(results) if results else 0.0,
# #             'sample_failures': [],
# #         }
# #         for failure in failures[:10]:
# #             top3_filenames = [c.get('filename', '') for c in failure.get('retrieved_chunks', [])[:3]]
# #             failure_analysis['sample_failures'].append({
# #                 'question_index': failure['question_index'],
# #                 'question': failure['question'],
# #                 'reference': failure['reference'],
# #                 'reference_contexts': failure.get('reference_contexts', []),
# #                 'got_top3_filenames': top3_filenames,
# #             })

# #         return {
# #             'aggregate_metrics': aggregate_metrics,
# #             'latency_stats': latency_stats,
# #             'failure_analysis': failure_analysis,
# #         }

# #     def _save_progress(self, results: List[Dict], filename: str):
# #         try:
# #             with open(filename, 'w') as f:
# #                 json.dump(results, f, indent=2, default=_json_default)
# #             logger.info(f"  💾 Progress saved to {filename}")
# #         except Exception as e:
# #             logger.error(f"  ✗ Could not save progress: {e}")


# # # =============================================================================
# # # JSONL writer — produces bgem3_retrieval.jsonl, one row per question
# # # =============================================================================

# # def _result_to_jsonl_row(result: Dict[str, Any], retriever_name: str) -> Dict[str, Any]:
# #     """Convert an internal per-question result into the bgem3_retrieval.jsonl row."""
# #     context_chunks = []
# #     for i, chunk in enumerate(result.get('retrieved_chunks', [])):
# #         filename = chunk.get('filename') or 'unknown'
# #         # Synthesize a chunk ID in the style of the target schema:
# #         #   doc_<filename-without-ext>_chunk_<rank-index>
# #         stem = Path(filename).stem if filename != 'unknown' else 'unknown'
# #         context_chunks.append({
# #             'text': chunk.get('content', ''),
# #             'source_id': f'doc_{stem}_chunk_{i}',
# #             'score': float(chunk.get('score') or 0.0),
# #         })
# #     return {
# #         'query': result['question'],
# #         'ground_truth': result.get('reference', ''),
# #         'expected_chunks': result.get('reference_contexts', []),
# #         'retriever': retriever_name,
# #         'context_chunks': context_chunks,
# #     }


# # class JsonlWriter:
# #     """Append-only writer for bgem3_retrieval.jsonl.  Supports rebuild-on-resume."""

# #     def __init__(self, path: str, retriever_name: str):
# #         self.path = path
# #         self.retriever_name = retriever_name
# #         # Ensure the file exists (truncated to empty) at start so fresh runs don't
# #         # accidentally append to stale data.
# #         Path(self.path).parent.mkdir(parents=True, exist_ok=True)
# #         if not Path(self.path).exists():
# #             open(self.path, 'w').close()

# #     def append(self, result: Dict[str, Any]):
# #         row = _result_to_jsonl_row(result, self.retriever_name)
# #         with open(self.path, 'a') as f:
# #             f.write(json.dumps(row, default=_json_default) + '\n')

# #     def rewrite_from_results(self, results: List[Dict[str, Any]]):
# #         """Called on resume — truncate and re-emit from all completed results."""
# #         with open(self.path, 'w') as f:
# #             for r in results:
# #                 row = _result_to_jsonl_row(r, self.retriever_name)
# #                 f.write(json.dumps(row, default=_json_default) + '\n')
# #         logger.info(f"✓ Rebuilt {self.path} from {len(results)} completed results")


# # # =============================================================================
# # # Reporting
# # # =============================================================================

# # def print_detailed_summary(results: Dict):
# #     """Print evaluation summary (overall metrics + latency + failures)."""

# #     print("\n" + "=" * 100)
# #     print("RETRIEVAL EVALUATION RESULTS  (chunk-level)")
# #     print("=" * 100)

# #     info = results['evaluation_info']
# #     print(f"\n📊 OVERVIEW:")
# #     print(f"   Evaluation Mode: {info['evaluation_mode'].upper()}")
# #     print(f"   Total Questions: {info['total_questions']}")
# #     print(f"   Total Time: {info['total_time_seconds']/60:.2f} minutes")
# #     print(f"   Avg Time/Question: {info['avg_time_per_question']:.2f} seconds")
# #     print(f"   Timestamp: {info['timestamp']}")

# #     print("\n" + "-" * 100)
# #     print("OVERALL PERFORMANCE")
# #     print("-" * 100)

# #     agg = results['aggregate_metrics']

# #     if 'found' in agg:
# #         success = agg['found']['mean']
# #         print(
# #             f"\n✅ SUCCESS RATE: {success:.2%} "
# #             f"({int(success * info['total_questions'])}/{info['total_questions']})"
# #         )

# #     if 'mrr' in agg:
# #         print(f"\n🎯 MEAN RECIPROCAL RANK: {agg['mrr']['mean']:.4f}")

# #     print(f"\n📍 PRECISION @ K:")
# #     for k in [1, 3, 5, 10]:
# #         if f'precision@{k}' in agg:
# #             p = agg[f'precision@{k}']
# #             print(f"   P@{k:2d}: {p['mean']:.4f} ± {p['std']:.4f} (min: {p['min']:.4f}, max: {p['max']:.4f})")

# #     print(f"\n📊 RECALL @ K:")
# #     for k in [1, 3, 5, 10]:
# #         if f'recall@{k}' in agg:
# #             r = agg[f'recall@{k}']
# #             print(f"   R@{k:2d}: {r['mean']:.4f}")

# #     print(f"\n📊 NDCG @ K:")
# #     for k in [1, 3, 5, 10]:
# #         if f'ndcg@{k}' in agg:
# #             ndcg = agg[f'ndcg@{k}']
# #             print(f"   NDCG@{k:2d}: {ndcg['mean']:.4f}")

# #     # Latency
# #     print("\n" + "-" * 100)
# #     print("LATENCY STATISTICS")
# #     print("-" * 100)

# #     lat = results['latency_stats']
# #     print(f"\n   Mean:   {lat['mean_ms']:.2f} ms")
# #     print(f"   Median: {lat['median_ms']:.2f} ms")
# #     print(f"   Std:    {lat['std_ms']:.2f} ms")
# #     print(f"   Min:    {lat['min_ms']:.2f} ms")
# #     print(f"   Max:    {lat['max_ms']:.2f} ms")
# #     print(f"   P95:    {lat['p95_ms']:.2f} ms")
# #     print(f"   P99:    {lat['p99_ms']:.2f} ms")

# #     # Failure analysis
# #     print("\n" + "-" * 100)
# #     print("FAILURE ANALYSIS")
# #     print("-" * 100)

# #     fail = results['failure_analysis']
# #     print(f"\n   Total Failures: {fail['total_failures']} ({fail['failure_rate']:.2%})")

# #     if fail['sample_failures']:
# #         print(f"\n   Sample Failed Queries:")
# #         for i, failure in enumerate(fail['sample_failures'][:5], 1):
# #             print(f"\n   {i}. [idx {failure['question_index']}] {failure['question']}")
# #             ref_preview = failure['reference'][:100] + ('...' if len(failure['reference']) > 100 else '')
# #             print(f"      Reference: {ref_preview}")
# #             if failure['got_top3_filenames']:
# #                 print(f"      Top-3 filenames: {', '.join(failure['got_top3_filenames'])}")

# #     print("\n" + "=" * 100)


# # # =============================================================================
# # # JSON serialization helpers
# # # =============================================================================

# # def _json_default(obj):
# #     """Convert numpy scalars/arrays for JSON serialization."""
# #     if isinstance(obj, (np.integer,)):
# #         return int(obj)
# #     if isinstance(obj, (np.floating,)):
# #         return float(obj)
# #     if isinstance(obj, np.ndarray):
# #         return obj.tolist()
# #     raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# # def save_results(results: Dict, output_path: str):
# #     """Save detailed results as pretty-printed JSON."""

# #     def convert_types(obj):
# #         if isinstance(obj, (np.integer, np.int64)):
# #             return int(obj)
# #         elif isinstance(obj, (np.floating, np.float64)):
# #             return float(obj)
# #         elif isinstance(obj, np.ndarray):
# #             return obj.tolist()
# #         elif isinstance(obj, dict):
# #             return {k: convert_types(v) for k, v in obj.items()}
# #         elif isinstance(obj, list):
# #             return [convert_types(item) for item in obj]
# #         return obj

# #     results = convert_types(results)

# #     with open(output_path, 'w') as f:
# #         json.dump(results, f, indent=2)

# #     logger.info(f"\n✓ Results saved to: {output_path}")


# # # =============================================================================
# # # Main
# # # =============================================================================

# # def _load_questions(path: str) -> List[Dict[str, Any]]:
# #     """
# #     Load GT questions.  Accepts either:
# #       - a plain list of question objects: [ {...}, {...} ]
# #       - an object wrapping them:          { "questions": [...] }
# #     """
# #     with open(path, 'r') as f:
# #         data = json.load(f)
# #     if isinstance(data, list):
# #         return data
# #     if isinstance(data, dict) and 'questions' in data:
# #         return data['questions']
# #     raise ValueError(
# #         f"Could not find questions in {path}. "
# #         f"Expected a list or an object with a 'questions' key."
# #     )


# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="Chunk-level retrieval evaluation against Ragas-style GT"
# #     )
# #     parser.add_argument('--questions', type=str, required=True,
# #                         help='Path to GT JSON (list of {user_input, reference, reference_contexts, ...})')
# #     parser.add_argument('--resume', type=str, default=None,
# #                         help='Path to progress JSON file to resume from')
# #     parser.add_argument('--collection', type=str, default='Autosar_v1',
# #                         help='Qdrant collection name')
# #     parser.add_argument('--qdrant-url', type=str, default='http://localhost:7333',
# #                         help='Qdrant URL')
# #     parser.add_argument('--top-k', type=int, default=10,
# #                         help='Number of results to retrieve per question')
# #     parser.add_argument('--output', type=str, default='complete_evaluation_results.json',
# #                         help='Output JSON file for detailed metrics')
# #     parser.add_argument('--jsonl-output', type=str, default='bgem3_retrieval.jsonl',
# #                         help='Output JSONL file, one row per question, for retriever comparison')
# #     parser.add_argument('--retriever-name', type=str, default='bge_m3',
# #                         help='Name recorded in each JSONL row\'s "retriever" field')
# #     parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
# #     parser.add_argument('--no-reranker', action='store_true', help='Disable cross-encoder reranking')
# #     parser.add_argument('--save-progress', type=int, default=25,
# #                         help='Save progress every N questions (0 to disable)')

# #     args = parser.parse_args()

# #     # Load questions
# #     logger.info(f"Loading questions from {args.questions}...")
# #     questions = _load_questions(args.questions)
# #     logger.info(f"✓ Loaded {len(questions)} questions")

# #     # Initialize BGE-M3 via sentence-transformers
# #     logger.info("\nInitializing sentence-transformers embedding model (BAAI/bge-m3)...")
# #     embedder = SentenceTransformerEmbedder()

# #     # Initialize retriever, then inject the ST embedder in place of Ollama.
# #     logger.info("Initializing hybrid retriever...")
# #     retriever = HybridRetriever(
# #         qdrant_url=args.qdrant_url,
# #         collection_name=args.collection,
# #         use_ollama=False,
# #         use_reranker=not args.no_reranker,
# #     )

# #     # Patch the dense-encoder attribute with our local embedder.
# #     # If the attribute name below doesn't match, check vars(retriever) and update it.
# #     _DENSE_ATTR = next(
# #         (a for a in ("embedding_model", "dense_model", "embedder", "encoder")
# #          if hasattr(retriever, a)),
# #         None,
# #     )
# #     if _DENSE_ATTR:
# #         setattr(retriever, _DENSE_ATTR, embedder)
# #         logger.info(f"✓ Injected BAAI/bge-m3 into retriever.{_DENSE_ATTR}")
# #     else:
# #         logger.warning(
# #             "Could not find dense-encoder attribute on HybridRetriever. "
# #             f"Instance attributes: {list(vars(retriever).keys())}"
# #         )

# #     logger.info("✓ Retriever initialized")

# #     # Initialize evaluator + JSONL writer
# #     evaluator = ComprehensiveEvaluator(retriever)
# #     jsonl_writer = JsonlWriter(args.jsonl_output, args.retriever_name)

# #     # Run evaluation
# #     logger.info("\nStarting evaluation...\n")
# #     results = evaluator.evaluate_all(
# #         questions,
# #         top_k=args.top_k,
# #         verbose=args.verbose,
# #         save_progress_every=args.save_progress,
# #         resume_file=args.resume,
# #         jsonl_writer=jsonl_writer,
# #     )

# #     # Print + save
# #     print_detailed_summary(results)
# #     save_results(results, args.output)

# #     # Text summary next to the JSON
# #     summary_path = args.output.replace('.json', '_summary.txt')
# #     with open(summary_path, 'w') as f:
# #         old_stdout = sys.stdout
# #         try:
# #             sys.stdout = StringIO()
# #             print_detailed_summary(results)
# #             summary_text = sys.stdout.getvalue()
# #         finally:
# #             sys.stdout = old_stdout
# #         f.write(summary_text)
# #     logger.info(f"✓ Summary saved to: {summary_path}")

# #     print("\n" + "=" * 100)
# #     print("EVALUATION COMPLETE!")
# #     print("=" * 100)
# #     print(f"\nResults saved to:")
# #     print(f"  • Detailed JSON: {args.output}")
# #     print(f"  • Text Summary:  {summary_path}")
# #     print(f"  • JSONL rows:    {args.jsonl_output}")
# #     print("=" * 100 + "\n")


# # if __name__ == "__main__":
# #     main()

#!/usr/bin/env python3
"""
RETRIEVAL EVALUATION PIPELINE  (FIXED)
=======================================
Fixes applied over the previous version:

FIX 6 — JSONL stores all four scores + full top-k
    _result_to_jsonl_row() previously stored only the RRF fusion score and
    truncated context_chunks to top-5.  Now stores:
      • score         — RRF fusion score (as before)
      • rerank_score  — cross-encoder score (NEW)
      • dense_score   — dense cosine score (NEW)
      • sparse_score  — BM25 score (NEW)
    And retrieved_chunks[:top_k] instead of [:5] so no retrieval evidence
    is silently discarded from the output.

FIX 7 — Three-level failure taxonomy
    Every missed context is now classified into one of three buckets:
      1. wrong_doc      — the correct document was never in the retrieved pool
      2. wrong_chunk    — the correct document appeared but the specific chunk
                          did not (right doc, wrong section boundary)
      3. below_cutoff   — the correct chunk existed in the pool (fuzzy match
                          in any position) but was ranked below top-k after
                          reranking  [requires STORE_FULL_POOL=True, see below]
    This taxonomy is printed in the failure analysis section and saved in the
    JSON output, giving you a precise signal for which fix addresses which
    failure mode.

    STORE_FULL_POOL (default False): when True, the retriever is queried with
    top_k * POOL_MULTIPLIER and only the top_k results are used for metrics,
    but the extended pool is stored in the result for below_cutoff detection.
    Set True to distinguish "wrong_chunk" from "below_cutoff".  Adds latency.

Usage (unchanged):
    python retrival.py --questions evaluation_questions.json
    python retrival.py --questions evaluation_questions.json --resume progress_75.json
"""

import sys
import re
from io import StringIO
from difflib import SequenceMatcher
import unicodedata
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

import numpy as np
from Evaluate_Retrieval_With_Reranker_Template import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FIX 7 — extended pool for below_cutoff detection
# Set True to distinguish "wrong_chunk" from "below_cutoff".
# Adds extra retrieval latency per question.
STORE_FULL_POOL   = False
POOL_MULTIPLIER   = 3      # retrieve top_k * POOL_MULTIPLIER when STORE_FULL_POOL=True


# =============================================================================
# Fuzzy-match machinery  (unchanged)
# =============================================================================

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    return (
        text
        .replace("\n", " ")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
    )


def _despace(text: str) -> str:
    return re.sub(r'\s+', '', text)


def _strip_pipes(text: str) -> str:
    return re.sub(r'\s*\|\s*', ' ', text).strip()


def _kv_to_text(snippet: str) -> str:
    parts  = re.split(r'[,;]\s*', snippet)
    values = []
    for part in parts:
        if '=' in part:
            _, _, val = part.partition('=')
            values.append(val.strip())
        else:
            values.append(part.strip())
    return ' '.join(v for v in values if v)


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


def fuzzy_match(snippet: str, text: str, threshold: float = 0.8) -> bool:
    sc = normalize(snippet.lower())
    tc = normalize(text.lower())

    if sc in tc:
        return True
    if _sliding_match(sc, tc, threshold):
        return True

    sds = _despace(sc)
    tds = _despace(tc)
    if sds:
        if sds in tds:
            return True
        if _sliding_match(sds, tds, threshold):
            return True

    tc_no_pipe = _strip_pipes(tc)
    if sc in tc_no_pipe:
        return True
    if _sliding_match(sc, tc_no_pipe, threshold):
        return True
    tds_no_pipe = _despace(tc_no_pipe)
    if sds and sds in tds_no_pipe:
        return True

    MIN_KV_TEXT_LENGTH = 8
    if '=' in sc:
        kv    = _kv_to_text(sc)
        kv_ds = _despace(kv)
        if kv and len(kv) > MIN_KV_TEXT_LENGTH:
            if kv in tc or kv in tc_no_pipe:
                return True
            if _sliding_match(kv, tc, threshold) or _sliding_match(kv, tc_no_pipe, threshold):
                return True
        if kv_ds and len(kv_ds) > MIN_KV_TEXT_LENGTH:
            if kv_ds in tds or kv_ds in tds_no_pipe:
                return True

    return False


# =============================================================================
# FIX 7 — failure taxonomy helpers
# =============================================================================

def _doc_key_from_source_id(source_id: str) -> str:
    """
    Extract a stable document key from the source_id stored in a chunk dict.
    Strips the '_chunk_N' suffix produced by _result_to_jsonl_row().
    """
    if "_chunk_" in source_id:
        return source_id.rsplit("_chunk_", 1)[0]
    return source_id


def _infer_doc_key_from_context(ctx_text: str) -> Optional[str]:
    """
    Best-effort: extract a document identifier from the first 200 chars of an
    expected context.  AUTOSAR chunks typically start with the document title.
    Returns None if nothing recognisable is found.
    """
    header = ctx_text[:200].lower()
    # Look for AUTOSAR PDF stem patterns: AUTOSAR_XXX_YYY
    m = re.search(r'autosar[_\s][a-z0-9_]+', header)
    if m:
        return re.sub(r'\s+', '_', m.group(0))
    return None


def classify_missed_context(
    ctx_text:        str,
    retrieved_chunks: List[Dict],
    full_pool:       Optional[List[Dict]] = None,
) -> str:
    """
    FIX 7: classify why a specific reference context was not found.

    Returns one of:
        "wrong_doc"     — the correct document never appeared in the retrieved pool
        "wrong_chunk"   — the correct document appeared but this chunk did not
        "below_cutoff"  — chunk exists in the extended pool but ranked > top_k
                          (only possible when full_pool is provided)
        "unknown"       — cannot determine (fallback)
    """
    # ── Step 1: check whether any retrieved chunk comes from the right document
    ctx_doc_key = _infer_doc_key_from_context(ctx_text)

    # Check top-k pool for the right document
    top_k_has_doc = False
    if ctx_doc_key:
        for chunk in retrieved_chunks:
            chunk_doc = _doc_key_from_source_id(chunk.get("source_id", ""))
            if ctx_doc_key and ctx_doc_key in chunk_doc:
                top_k_has_doc = True
                break
    else:
        # Fall back: look for any chunk whose content shares significant text
        # with the start of the expected context
        probe = ctx_text[:80].lower().strip()
        if probe:
            for chunk in retrieved_chunks:
                if probe[:40] in chunk.get("content", "").lower():
                    top_k_has_doc = True
                    break

    if not top_k_has_doc:
        # ── Step 2: if full_pool provided, check if the chunk exists there
        if full_pool is not None:
            found_in_pool = any(
                fuzzy_match(ctx_text, c.get("content", ""))
                for c in full_pool
            )
            if found_in_pool:
                return "below_cutoff"
        return "wrong_doc"

    # Right document is in the pool but this specific chunk wasn't matched
    return "wrong_chunk"


# =============================================================================
# Evaluator
# =============================================================================

class ComprehensiveEvaluator:
    """Chunk-level retrieval evaluator for Ragas-style ground truth."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def evaluate_single_question(
        self,
        question_data:  Dict[str, Any],
        question_index: int,
        top_k:          int  = 10,
        verbose:        bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question with chunk-level metrics.

        FIX 6: stores all four scores (rrf, rerank, dense, sparse) per chunk.
        FIX 6: stores retrieved_chunks[:top_k] instead of [:5].
        FIX 7: classifies each missed context into wrong_doc / wrong_chunk /
               below_cutoff.
        """
        query             = question_data["user_input"]
        reference         = question_data.get("reference", "")
        reference_contexts = question_data.get("reference_contexts", []) or []

        if verbose:
            logger.info(f"\nQuestion [{question_index}]: {query}")
            logger.info(f"Reference: {reference[:120]}...")

        start_time = time.time()
        try:
            # ── Retrieve top_k results ─────────────────────────────────────
            search_results = self.retriever.search(query, top_k=top_k)
            latency = time.time() - start_time

            # ── Optionally retrieve extended pool for below_cutoff detection
            full_pool_chunks: Optional[List[Dict]] = None
            if STORE_FULL_POOL:
                extended = self.retriever.search(
                    query,
                    top_k=top_k * POOL_MULTIPLIER,
                    use_reranking=False,    # don't double-rerank; use RRF order
                )
                full_pool_chunks = [
                    {"content": r.content, "source_id": r.metadata.get("filename", "")}
                    for r in extended
                ]

            # ── Build retrieved_chunks list with ALL scores (FIX 6) ────────
            retrieved_chunks = []
            for result in search_results:
                retrieved_chunks.append({
                    "filename":     result.metadata.get("filename", ""),
                    "content":      result.content,
                    "score":        result.score,          # RRF score
                    "rerank_score": result.rerank_score,   # FIX 6 — was discarded
                    "dense_score":  result.dense_score,    # FIX 6 — was discarded
                    "sparse_score": result.sparse_score,   # FIX 6 — was discarded
                    "section":      result.metadata.get("section_title", ""),
                    "source_id":    (
                        "doc_"
                        + Path(result.metadata.get("filename", "unknown")).stem
                        + f"_chunk_{result.metadata.get('chunk_id', 0)}"
                    ),
                })

            # ── Per-context tracking ───────────────────────────────────────
            if not reference_contexts:
                logger.warning(
                    f"No reference_contexts for question [{question_index}] — "
                    f"all contexts will be marked as not found."
                )
                context_found         = []
                context_found_at_rank = []
            else:
                context_found         = [False] * len(reference_contexts)
                context_found_at_rank = [None]  * len(reference_contexts)

                for chunk_rank, chunk in enumerate(retrieved_chunks, start=1):
                    for ctx_idx, ctx in enumerate(reference_contexts):
                        if not context_found[ctx_idx] and fuzzy_match(ctx, chunk["content"]):
                            context_found[ctx_idx]         = True
                            context_found_at_rank[ctx_idx] = chunk_rank

            total_contexts = len(reference_contexts)
            found          = bool(context_found) and all(context_found)

            if found:
                rank             = max(context_found_at_rank)
                reciprocal_rank  = 1.0 / rank
            else:
                rank            = None
                reciprocal_rank = 0.0

            metrics: Dict[str, Any] = {
                "found": found,
                "rank":  rank,
                "mrr":   reciprocal_rank,
            }

            k_values = sorted(set([1, 3, 5, 10] + ([top_k] if top_k not in [1, 3, 5, 10] else [])))

            for k in k_values:
                contexts_found_in_k = sum(
                    1 for r in context_found_at_rank if r is not None and r <= k
                )
                metrics[f"recall@{k}"]    = contexts_found_in_k / total_contexts if total_contexts > 0 else 0.0
                metrics[f"all_found@{k}"] = 1.0 if (total_contexts > 0 and contexts_found_in_k == total_contexts) else 0.0
                relevant_chunks_in_k = sum(
                    1 for chunk in retrieved_chunks[:k]
                    if any(fuzzy_match(ctx, chunk["content"]) for ctx in reference_contexts)
                )
                metrics[f"precision@{k}"] = relevant_chunks_in_k / k

            for k in k_values:
                dcg = 0.0
                for i, chunk in enumerate(retrieved_chunks[:k], start=1):
                    if any(fuzzy_match(ctx, chunk["content"]) for ctx in reference_contexts):
                        dcg += 1.0 / np.log2(i + 1)
                num_relevant = sum(1 for r in context_found_at_rank if r is not None and r <= k)
                idcg         = sum(1.0 / np.log2(j + 1) for j in range(1, min(num_relevant, k) + 1))
                metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

            if verbose:
                logger.info(f"Found all: {found}, Bottleneck rank: {rank}, MRR: {reciprocal_rank:.4f}")
                logger.info(f"Context coverage: {sum(context_found)}/{total_contexts}")

            # ── Per-context match info + FIX 7 failure classification ──────
            context_match = []
            for ctx_idx, ctx in enumerate(reference_contexts):
                found_flag = context_found[ctx_idx] if context_found else False
                found_rank = context_found_at_rank[ctx_idx] if context_found_at_rank else None
                found_file = (
                    retrieved_chunks[found_rank - 1]["filename"]
                    if found_rank is not None
                    else None
                )

                # FIX 7: classify missed contexts
                if not found_flag:
                    failure_type = classify_missed_context(
                        ctx_text=ctx,
                        retrieved_chunks=retrieved_chunks,
                        full_pool=full_pool_chunks,
                    )
                else:
                    failure_type = None   # not a failure

                context_match.append({
                    "reference_context": ctx,
                    "found":             found_flag,
                    "found_in_rank":     found_rank,
                    "found_in_filename": found_file,
                    "failure_type":      failure_type,   # FIX 7
                })

            return {
                "question_index":     question_index,
                "question":           query,
                "reference":          reference,
                "reference_contexts": reference_contexts,
                "context_match":      context_match,
                # FIX 6: store full top_k, not just top-5
                "retrieved_chunks":   retrieved_chunks[:top_k],
                "latency_ms":         latency * 1000,
                "metrics":            metrics,
            }

        except Exception as e:
            logger.error(f"Error processing question [{question_index}]: {e}")
            k_values = sorted(set([1, 3, 5, 10] + ([top_k] if top_k not in [1, 3, 5, 10] else [])))
            error_metrics: Dict[str, Any] = {"found": 0.0, "rank": None, "mrr": 0.0}
            for k in k_values:
                error_metrics[f"precision@{k}"] = 0.0
                error_metrics[f"recall@{k}"]    = 0.0
                error_metrics[f"ndcg@{k}"]      = 0.0
                error_metrics[f"all_found@{k}"] = 0.0
            return {
                "question_index":     question_index,
                "question":           query,
                "reference":          reference,
                "reference_contexts": reference_contexts,
                "context_match":      [],
                "retrieved_chunks":   [],
                "latency_ms":         0.0,
                "error":              str(e),
                "metrics":            error_metrics,
            }

    def evaluate_all(
        self,
        questions:           List[Dict[str, Any]],
        top_k:               int            = 10,
        verbose:             bool           = False,
        save_progress_every: int            = 10,
        resume_file:         Optional[str]  = None,
        jsonl_writer:        Any            = None,
    ) -> Dict[str, Any]:
        """Evaluate all questions with progress tracking."""

        logger.info("=" * 80)
        logger.info(f"EVALUATING {len(questions)} QUESTIONS (chunk-level)")
        logger.info("=" * 80)

        all_results:       List[Dict[str, Any]] = []
        completed_indices: set                  = set()

        if resume_file and Path(resume_file).exists():
            logger.info(f"Resuming from {resume_file}...")
            with open(resume_file, "r") as f:
                all_results = json.load(f)
            completed_indices = {r["question_index"] for r in all_results}
            logger.info(f"✓ Already completed: {len(completed_indices)}")
            logger.info(f"→ Remaining: {len(questions) - len(completed_indices)} / {len(questions)}")
            if jsonl_writer is not None:
                jsonl_writer.rewrite_from_results(all_results)

        start_time         = time.time()
        processed_this_run = 0

        for idx, question in enumerate(questions):
            if idx in completed_indices:
                continue

            processed_this_run += 1
            if processed_this_run % 10 == 0 or verbose:
                elapsed   = time.time() - start_time
                avg_time  = elapsed / processed_this_run
                remaining = len(questions) - len(completed_indices) - processed_this_run
                eta       = avg_time * remaining
                logger.info(
                    f"\nProgress: {processed_this_run}/"
                    f"{len(questions) - len(completed_indices)} "
                    f"- ETA: {eta / 60:.1f} min"
                )

            result = self.evaluate_single_question(question, idx, top_k, verbose)
            all_results.append(result)

            if jsonl_writer is not None:
                jsonl_writer.append(result)

            if save_progress_every and processed_this_run % save_progress_every == 0:
                self._save_progress(all_results, f"./progress/progress_{len(all_results)}.json")

        total_time = time.time() - start_time
        logger.info(
            f"\n✓ Completed {processed_this_run} questions in {total_time / 60:.2f} minutes "
            f"(total evaluated: {len(all_results)})"
        )

        aggregate_results = self._calculate_aggregates(all_results)

        return {
            "evaluation_info": {
                "total_questions":      len(all_results),
                "evaluation_mode":      "chunk",
                "total_time_seconds":   total_time,
                "avg_time_per_question": (
                    total_time / processed_this_run if processed_this_run > 0 else 0.0
                ),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "aggregate_metrics": aggregate_results["aggregate_metrics"],
            "latency_stats":     aggregate_results["latency_stats"],
            "failure_analysis":  aggregate_results["failure_analysis"],
            "detailed_results":  all_results,
        }

    def _calculate_aggregates(self, results: List[Dict]) -> Dict:
        """Calculate overall metrics, latency stats, and FIX 7 failure taxonomy."""

        metric_values: Dict[str, List[float]] = defaultdict(list)
        latencies: List[float] = []

        # FIX 7: failure taxonomy counters
        taxonomy_counts: Dict[str, int] = {
            "wrong_doc":    0,
            "wrong_chunk":  0,
            "below_cutoff": 0,
            "unknown":      0,
        }

        for result in results:
            if "error" not in result:
                for metric_name, value in result["metrics"].items():
                    if value is not None:
                        metric_values[metric_name].append(value)
                latencies.append(result["latency_ms"])

                # Accumulate failure taxonomy from context_match
                for cm in result.get("context_match", []):
                    ft = cm.get("failure_type")
                    if ft in taxonomy_counts:
                        taxonomy_counts[ft] += 1

        aggregate_metrics: Dict[str, Dict[str, float]] = {}
        for metric_name, values in metric_values.items():
            if not values:
                continue
            arr = np.array(values, dtype=float)
            aggregate_metrics[metric_name] = {
                "mean":   float(np.mean(arr)),
                "std":    float(np.std(arr)),
                "min":    float(np.min(arr)),
                "max":    float(np.max(arr)),
                "median": float(np.median(arr)),
            }

        if latencies:
            latency_stats = {
                "mean_ms":   float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "std_ms":    float(np.std(latencies)),
                "min_ms":    float(np.min(latencies)),
                "max_ms":    float(np.max(latencies)),
                "p95_ms":    float(np.percentile(latencies, 95)),
                "p99_ms":    float(np.percentile(latencies, 99)),
            }
        else:
            latency_stats = {k: 0.0 for k in
                             ["mean_ms","median_ms","std_ms","min_ms","max_ms","p95_ms","p99_ms"]}

        failures = [r for r in results if not r["metrics"].get("found", False)]
        failure_analysis: Dict[str, Any] = {
            "total_failures": len(failures),
            "failure_rate":   len(failures) / len(results) if results else 0.0,
            # FIX 7: taxonomy breakdown
            "failure_taxonomy": {
                "wrong_doc":    taxonomy_counts["wrong_doc"],
                "wrong_chunk":  taxonomy_counts["wrong_chunk"],
                "below_cutoff": taxonomy_counts["below_cutoff"],
                "unknown":      taxonomy_counts["unknown"],
            },
            "sample_failures": [],
        }

        for failure in failures[:10]:
            top3_filenames = [
                c.get("filename", "") for c in failure.get("retrieved_chunks", [])[:3]
            ]
            # FIX 7: per-context failure types in the sample
            ctx_types = [
                {
                    "context_preview": cm.get("reference_context", "")[:80],
                    "failure_type":    cm.get("failure_type", "unknown"),
                }
                for cm in failure.get("context_match", [])
                if not cm.get("found", False)
            ]
            failure_analysis["sample_failures"].append({
                "question_index":      failure["question_index"],
                "question":            failure["question"],
                "reference":           failure["reference"],
                "reference_contexts":  failure.get("reference_contexts", []),
                "got_top3_filenames":  top3_filenames,
                "missed_context_types": ctx_types,   # FIX 7
            })

        return {
            "aggregate_metrics": aggregate_metrics,
            "latency_stats":     latency_stats,
            "failure_analysis":  failure_analysis,
        }

    def _save_progress(self, results: List[Dict], filename: str):
        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=_json_default)
            logger.info(f"  💾 Progress saved to {filename}")
        except Exception as e:
            logger.error(f"  ✗ Could not save progress: {e}")


# =============================================================================
# JSONL writer  (FIX 6 — stores all four scores, full top-k)
# =============================================================================

def _result_to_jsonl_row(result: Dict[str, Any], retriever_name: str) -> Dict[str, Any]:
    """
    FIX 6: stores rerank_score, dense_score, sparse_score alongside the RRF
    score, and writes all top-k chunks instead of only the first 5.

    The source_id is synthesised from the stored source_id field in the chunk
    dict (which already has the doc_<stem>_chunk_<id> format set in
    evaluate_single_question).
    """
    context_chunks = []
    for chunk in result.get("retrieved_chunks", []):
        # Use the pre-built source_id if available, fall back to filename
        source_id = chunk.get("source_id") or (
            "doc_"
            + Path(chunk.get("filename") or "unknown").stem
            + "_chunk_0"
        )
        context_chunks.append({
            "text":         chunk.get("content", ""),
            "source_id":    source_id,
            "score":        float(chunk.get("score")        or 0.0),   # RRF
            "rerank_score": chunk.get("rerank_score"),                  # FIX 6
            "dense_score":  chunk.get("dense_score"),                   # FIX 6
            "sparse_score": chunk.get("sparse_score"),                  # FIX 6
        })

    return {
        "query":          result["question"],
        "ground_truth":   result.get("reference", ""),
        "expected_chunks": result.get("reference_contexts", []),
        "retriever":      retriever_name,
        "context_chunks": context_chunks,
    }


class JsonlWriter:
    """Append-only writer for bgem3_retrieval.jsonl.  Supports rebuild-on-resume."""

    def __init__(self, path: str, retriever_name: str):
        self.path           = path
        self.retriever_name = retriever_name
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(self.path).exists():
            open(self.path, "w").close()

    def append(self, result: Dict[str, Any]):
        row = _result_to_jsonl_row(result, self.retriever_name)
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")

    def rewrite_from_results(self, results: List[Dict[str, Any]]):
        with open(self.path, "w") as f:
            for r in results:
                row = _result_to_jsonl_row(r, self.retriever_name)
                f.write(json.dumps(row, default=_json_default) + "\n")
        logger.info(f"✓ Rebuilt {self.path} from {len(results)} completed results")


# =============================================================================
# Reporting  (extended to print FIX 7 taxonomy)
# =============================================================================

def print_detailed_summary(results: Dict):
    print("\n" + "=" * 100)
    print("RETRIEVAL EVALUATION RESULTS  (chunk-level)")
    print("=" * 100)

    info = results["evaluation_info"]
    print(f"\n📊 OVERVIEW:")
    print(f"   Evaluation Mode: {info['evaluation_mode'].upper()}")
    print(f"   Total Questions: {info['total_questions']}")
    print(f"   Total Time: {info['total_time_seconds'] / 60:.2f} minutes")
    print(f"   Avg Time/Question: {info['avg_time_per_question']:.2f} seconds")
    print(f"   Timestamp: {info['timestamp']}")

    print("\n" + "-" * 100)
    print("OVERALL PERFORMANCE")
    print("-" * 100)

    agg = results["aggregate_metrics"]

    if "found" in agg:
        success = agg["found"]["mean"]
        print(
            f"\n✅ STRICT MULTIHOP SUCCESS RATE (all contexts found): {success:.2%} "
            f"({int(success * info['total_questions'])}/{info['total_questions']})"
        )

    if "mrr" in agg:
        print(f"\n🎯 MEAN RECIPROCAL RANK (bottleneck hop): {agg['mrr']['mean']:.4f}")

    print(f"\n📍 PRECISION @ K:")
    for k in [1, 3, 5, 10]:
        if f"precision@{k}" in agg:
            p = agg[f"precision@{k}"]
            print(f"   P@{k:2d}: {p['mean']:.4f} ± {p['std']:.4f} (min: {p['min']:.4f}, max: {p['max']:.4f})")

    print(f"\n📊 RECALL @ K:")
    for k in [1, 3, 5, 10]:
        if f"recall@{k}" in agg:
            r = agg[f"recall@{k}"]
            print(f"   R@{k:2d}: {r['mean']:.4f}")

    print(f"\n🎯 STRICT MULTIHOP SUCCESS @ K (ALL contexts found within top-k):")
    for k in [1, 3, 5, 10]:
        if f"all_found@{k}" in agg:
            a = agg[f"all_found@{k}"]
            print(f"   All@{k:2d}: {a['mean']:.4f} ({a['mean']:.2%})")

    print(f"\n📊 NDCG @ K:")
    for k in [1, 3, 5, 10]:
        if f"ndcg@{k}" in agg:
            ndcg = agg[f"ndcg@{k}"]
            print(f"   NDCG@{k:2d}: {ndcg['mean']:.4f}")

    print("\n" + "-" * 100)
    print("LATENCY STATISTICS")
    print("-" * 100)
    lat = results["latency_stats"]
    print(f"\n   Mean:   {lat['mean_ms']:.2f} ms")
    print(f"   Median: {lat['median_ms']:.2f} ms")
    print(f"   Std:    {lat['std_ms']:.2f} ms")
    print(f"   Min:    {lat['min_ms']:.2f} ms")
    print(f"   Max:    {lat['max_ms']:.2f} ms")
    print(f"   P95:    {lat['p95_ms']:.2f} ms")
    print(f"   P99:    {lat['p99_ms']:.2f} ms")

    print("\n" + "-" * 100)
    print("FAILURE ANALYSIS")
    print("-" * 100)
    fail = results["failure_analysis"]
    print(f"\n   Total Failures: {fail['total_failures']} ({fail['failure_rate']:.2%})")

    # FIX 7: print taxonomy
    taxonomy = fail.get("failure_taxonomy", {})
    if taxonomy:
        total_missed_contexts = sum(taxonomy.values())
        print(f"\n   Missed-context failure taxonomy ({total_missed_contexts} missed contexts total):")
        labels = {
            "wrong_doc":    "Wrong document retrieved  — document never appeared in top-k",
            "wrong_chunk":  "Right doc, wrong chunk   — correct doc retrieved but not this section",
            "below_cutoff": "Below cutoff             — chunk exists in pool but ranked > top-k",
            "unknown":      "Unknown                  — could not determine reason",
        }
        for key, label in labels.items():
            count = taxonomy.get(key, 0)
            pct   = 100 * count / total_missed_contexts if total_missed_contexts else 0
            print(f"     {label}: {count} ({pct:.1f}%)")

    if fail["sample_failures"]:
        print(f"\n   Sample Failed Queries:")
        for i, failure in enumerate(fail["sample_failures"][:5], 1):
            print(f"\n   {i}. [idx {failure['question_index']}] {failure['question']}")
            ref_preview = failure["reference"][:100] + (
                "..." if len(failure["reference"]) > 100 else ""
            )
            print(f"      Reference: {ref_preview}")
            if failure.get("got_top3_filenames"):
                print(f"      Top-3 filenames: {', '.join(failure['got_top3_filenames'])}")
            print(f"      Required contexts: {len(failure.get('reference_contexts', []))}")
            # FIX 7: show per-context failure type
            for ct in failure.get("missed_context_types", []):
                print(
                    f"      → [{ct['failure_type']:15s}] {ct['context_preview'][:70]}"
                )

    print("\n" + "=" * 100)


# =============================================================================
# JSON serialization helpers
# =============================================================================

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_results(results: Dict, output_path: str):
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    results = convert_types(results)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def _load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError(
        f"Could not find questions in {path}. "
        f"Expected a list or an object with a 'questions' key."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Chunk-level retrieval evaluation against Ragas-style GT  (FIXED)"
    )
    parser.add_argument("--questions",       type=str, required=True)
    parser.add_argument("--resume",          type=str, default=None)
    parser.add_argument("--collection",      type=str, default="Autosar_v2")
    parser.add_argument("--qdrant-url",      type=str, default="http://localhost:7333")
    parser.add_argument("--top-k",           type=int, default=10)
    parser.add_argument("--output",          type=str, default="complete_evaluation_results.json")
    parser.add_argument("--jsonl-output",    type=str, default="bgem3_retrieval.jsonl")
    parser.add_argument("--retriever-name",  type=str, default="bge_m3")
    parser.add_argument("--verbose",         action="store_true")
    parser.add_argument("--no-ollama",       action="store_true")
    parser.add_argument("--no-reranker",     action="store_true")
    parser.add_argument("--no-decomposition",action="store_true",
                        help="Disable query decomposition (single-shot retrieval)")
    parser.add_argument("--save-progress",   type=int, default=25)
    parser.add_argument("--store-full-pool", action="store_true",
                        help="Retrieve top_k * POOL_MULTIPLIER to enable below_cutoff detection "
                             "in the failure taxonomy.  Adds latency.")

    args = parser.parse_args()

    # Allow --store-full-pool to override the module-level default
    global STORE_FULL_POOL
    if args.store_full_pool:
        STORE_FULL_POOL = True

    logger.info(f"Loading questions from {args.questions}...")
    questions = _load_questions(args.questions)
    logger.info(f"✓ Loaded {len(questions)} questions")

    logger.info("\nInitializing hybrid retriever...")
    retriever = HybridRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        use_ollama=not args.no_ollama,
        use_reranker=not args.no_reranker,
        use_decomposition=not args.no_decomposition,
    )
    logger.info("✓ Retriever initialized")

    evaluator    = ComprehensiveEvaluator(retriever)
    jsonl_writer = JsonlWriter(args.jsonl_output, args.retriever_name)

    logger.info("\nStarting evaluation...\n")
    results = evaluator.evaluate_all(
        questions,
        top_k=args.top_k,
        verbose=args.verbose,
        save_progress_every=args.save_progress,
        resume_file=args.resume,
        jsonl_writer=jsonl_writer,
    )

    print_detailed_summary(results)
    save_results(results, args.output)

    summary_path = args.output.replace(".json", "_summary.txt")
    with open(summary_path, "w") as f:
        old_stdout = sys.stdout
        try:
            sys.stdout = StringIO()
            print_detailed_summary(results)
            summary_text = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        f.write(summary_text)
    logger.info(f"✓ Summary saved to: {summary_path}")

    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE!")
    print("=" * 100)
    print(f"\nResults saved to:")
    print(f"  • Detailed JSON: {args.output}")
    print(f"  • Text Summary:  {summary_path}")
    print(f"  • JSONL rows:    {args.jsonl_output}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()