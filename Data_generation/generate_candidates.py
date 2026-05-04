
# """
# Stage A: Generate candidate QA pairs from the knowledge graph.

# What makes this different from RAGAS's TestsetGenerator:
#   - We define our own scenarios. No auto-generated "casual typer" personas
#     that produce `wat iz AUTOSAR` questions.
#   - We split question generation and answer generation into TWO separate
#     LLM calls (RAGalyst Nov 2025 showed this beats the combined prompt).
#   - The answer prompt is strict-grounding: returns NOT_ANSWERABLE when the
#     context doesn't support an answer, and those are dropped.
#   - multi_hop scenarios are GUARANTEED to have >=2 contexts by construction.
#   - We over-generate 2x the target size so Stage B filtering has room.

# Model: Qwen/Qwen2.5-72B-Instruct-AWQ (AWQ 4-bit, TP=2 on your 2x48GB).
# Throughput: ~30-60 candidates/min with batched vLLM server inference.

# Resumability: candidates.jsonl is append-only. If the script is killed
# mid-run, rerunning picks up from the last saved candidate.

# Usage:
#     # Terminal 1: start vLLM server
#     vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 2 \
#         --max-model-len 8192 --quantization awq --port 8011

#     # Terminal 2:
#     python generate_candidates.py \
#         --kg-file ./output/kg/knowledge_graph.json \
#         --output-dir ./output \
#         --target 500
# """

# from __future__ import annotations

# import argparse
# import json
# import random
# import sys
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Any

# from shared.io_utils import atomic_write_json, append_jsonl, count_jsonl
# from shared.personas import AUTOSAR_PERSONAS, Persona
# from shared.prompts import (
#     QUESTION_GEN_SYSTEM, QUESTION_GEN_USER_SINGLEHOP,
#     ANSWER_GEN_SYSTEM, ANSWER_GEN_USER, STYLE_HINTS,
# )
# from shared.schemas import (
#     DEFAULT_SYNTH_DISTRIBUTION,
#     SYNTH_SINGLE_HOP_SPECIFIC, SYNTH_SINGLE_HOP_ABSTRACT,
#     SYNTH_MULTI_HOP_SPECIFIC,  SYNTH_MULTI_HOP_ABSTRACT,
#     new_candidate,
# )
# from shared.llm_batch import messages


# # ══════════════════════════════════════════════════════════════════════════════
# # PROMPT OVERRIDES — multi-hop only
# #
# # QUESTION_GEN_USER_MULTIHOP: defined here (not imported from shared/prompts.py).
# # The shared version takes {contexts_block} — one merged string — so the model
# # latches onto the first relevant chunk and ignores the other. This version
# # takes {context_1} and {context_2} as separate named slots with an explicit
# # BRIDGING REQUIREMENT.
# #
# # ANSWER_GEN_USER_MULTIHOP: new constant (not in shared/prompts.py at all).
# # ANSWER_GEN_USER is still used unchanged for single-hop.
# # ══════════════════════════════════════════════════════════════════════════════

# QUESTION_GEN_USER_MULTIHOP = """\
# You are a question-writing expert for a technical RAG evaluation dataset.
# You produce ONE well-formed question per call, in clean, formal technical English.

# Strict rules:
# 1. The question must be answerable using ONLY the provided contexts.
# 2. The question must be self-contained: a reader without the contexts should
#    still understand what is being asked.
# 3. Do NOT reference figure numbers, document IDs, page numbers, or section
#    numbers unless those artifacts are themselves the topic.
# 4. Do NOT ask "what does the document say about X" — ask directly about X.
# 5. Do not write casual or informal English. Use full technical terms.
# 6. The question must match the requested TYPE and PERSONA.

# You have been given TWO separate source contexts. Your task is to generate a
# single question that REQUIRES INFORMATION FROM BOTH contexts to answer.

# --- CONTEXT 1 ---
# {context_1}

# --- CONTEXT 2 ---
# {context_2}
# --- END CONTEXTS ---

# Persona: {persona_role}

# Question type: {question_type}

# BRIDGING REQUIREMENT (mandatory):
#   - The question must be impossible to answer using CONTEXT 1 alone.
#   - The question must be impossible to answer using CONTEXT 2 alone.
#   - The question must require combining or contrasting information from both.
#   - Do NOT ask about something that appears in only one context.

# Good bridging patterns:
#   - "How does [concept from ctx1] interact with / constrain [concept from ctx2]?"
#   - "What are the implications of [rule from ctx1] for the design in ctx2?"
#   - "Compare the approach in [ctx1 topic] with the requirement in [ctx2 topic]."
#   - "Given [constraint in ctx1], what must change in the design in ctx2?"

# REJECT: if no genuine informational bridge exists between the two contexts,
# output exactly: {{"question": null}}

# Otherwise output exactly this JSON (no markdown, no extra keys):
# {{"question": "<your bridging question here>"}}

# Write one question per the rules. {style_hint}"""


# ANSWER_GEN_USER_MULTIHOP = """\
# You are an expert answer-writer for a technical RAG evaluation dataset.
# You produce ONE ground-truth answer grounded strictly in the provided contexts.

# Strict rules:
# 1. Your answer MUST draw on information from BOTH Context 1 and Context 2.
#    An answer that uses only one context is NOT acceptable here.
# 2. If the question can be fully answered from one context alone, output
#    exactly: {{"answer": "NOT_ANSWERABLE"}}
# 3. If the contexts do not contain enough information to answer, output
#    exactly: {{"answer": "NOT_ANSWERABLE"}}
# 4. Use ONLY the information in the provided contexts. Do NOT add outside knowledge.
# 5. The answer should be complete but concise. Explain the connection between
#    the two contexts as it relates to the question.
# 6. Do NOT say "according to context 1" etc. Just give the factual answer.
# 7. Write in clean, formal technical English.

# --- CONTEXT 1 ---
# {context_1}

# --- CONTEXT 2 ---
# {context_2}
# --- END CONTEXTS ---

# Question: {question}

# Respond ONLY with valid JSON in this exact format:
# {{"answer": "<the answer>"}}
# No preamble, no markdown, no code fences."""


# # ══════════════════════════════════════════════════════════════════════════════
# # CLI
# # ══════════════════════════════════════════════════════════════════════════════

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     p.add_argument("--kg-file", required=True)
#     p.add_argument("--output-dir", required=True)
#     p.add_argument("--target", type=int, default=500,
#                    help="Final dataset size target (Stage C output)")
#     p.add_argument("--overgen-ratio", type=float, default=2.0,
#                    help="How many candidates to generate per target (default: 2.0)")
#     # Model
#     p.add_argument("--generator-model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
#     p.add_argument("--vllm-url",        default="http://localhost:8011/v1",
#                    help="URL of the running vLLM server (same as build_kg.py)")
#     # Generation
#     p.add_argument("--batch-size",      type=int,   default=32)
#     p.add_argument("--q-temperature",   type=float, default=0.3,
#                    help="Temperature for question generation (variety)")
#     p.add_argument("--a-temperature",   type=float, default=0.0,
#                    help="Temperature for answer generation (determinism)")
#     p.add_argument("--seed",            type=int,   default=42)
#     # FIX 3 — context quality gate
#     p.add_argument("--min-context-chars", type=int, default=150,
#                    help="Min chars a context chunk must have to be used (default: 150)")
#     # Stratified per-PDF sampling
#     p.add_argument("--min-per-pdf", type=int, default=20,
#                    help="Guaranteed minimum scenarios sampled from each PDF (default: 20). "
#                         "If total guaranteed budget exceeds n_candidates, reduced automatically.")
#     return p.parse_args()


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 1 — LOAD KG AND BUILD SCENARIOS
# # ══════════════════════════════════════════════════════════════════════════════

# def load_kg(path: Path):
#     from ragas.testset.graph import KnowledgeGraph
#     return KnowledgeGraph.load(str(path))


# def extract_content_nodes(kg) -> list:
#     """
#     Return content-bearing nodes (chunks) from the KG. These are the leaf
#     nodes produced by HeadlineSplitter — NOT the original DOCUMENT nodes,
#     which are full pages and too large.
#     """
#     from ragas.testset.graph import NodeType

#     # Prefer CHUNK nodes if the splitter ran; otherwise fall back to DOCUMENT
#     chunks = [n for n in kg.nodes if n.type == NodeType.CHUNK]
#     if chunks:
#         return chunks
#     return [n for n in kg.nodes if n.type == NodeType.DOCUMENT]


# def build_chunk_to_pdf_map(kg) -> dict[str, str]:
#     """
#     Walk KG relationships to map each CHUNK node id → source PDF filename.

#     CHUNK nodes produced by HeadlineSplitter carry only 'themes' and
#     'entities' — they have no 'filename' / 'source' in their own properties.
#     The PDF metadata lives on the parent DOCUMENT nodes.  This function
#     traverses every relationship once and builds the lookup so that
#     build_scenarios() can correctly assign CHUNK nodes to their PDF bucket.

#     Both edge directions are handled:
#       CHUNK ──rel──> DOCUMENT   (most common after HeadlineSplitter)
#       DOCUMENT ──rel──> CHUNK   (seen in some KG builds)

#     ISSUE-7 FIX: DOCUMENT-DOCUMENT relationships are explicitly skipped so
#     that document IDs never pollute the chunk→pdf map.
#     """
#     from ragas.testset.graph import NodeType

#     doc_by_id: dict[str, Any] = {
#         str(n.id): n
#         for n in kg.nodes
#         if n.type == NodeType.DOCUMENT
#     }

#     chunk_to_pdf: dict[str, str] = {}
#     for rel in kg.relationships:
#         src_id = str(rel.source.id)
#         tgt_id = str(rel.target.id)

#         if tgt_id in doc_by_id:
#             chunk_id = src_id
#             doc = doc_by_id[tgt_id]
#         elif src_id in doc_by_id:
#             chunk_id = tgt_id
#             doc = doc_by_id[src_id]
#         else:
#             continue

#         # Skip DOC-DOC relationships: chunk_id would be another doc ID
#         if chunk_id in doc_by_id:
#             continue

#         if chunk_id in chunk_to_pdf:
#             continue  # already resolved; first-found wins

#         meta = doc.properties.get("document_metadata", {})
#         pdf = meta.get("filename") or meta.get("source")
#         if pdf:
#             chunk_to_pdf[chunk_id] = pdf

#     return chunk_to_pdf


# def get_multihop_pairs(kg, min_overlap: float = 0.5) -> list[tuple]:
#     """
#     Return pairs of CHUNK nodes connected by a meaningful relationship, for
#     multi-hop scenarios. We use keyphrase-overlap and cosine-similarity
#     relationships if present.

#     Only CHUNK-CHUNK pairs are returned — DOCUMENT nodes are full pages and
#     are intentionally excluded from multi-hop contexts (too large, wrong level).

#     BUG-FIX: score extraction uses explicit None-check instead of `or` chaining
#     so that a legitimate score of 0.0 is not treated as missing and replaced
#     with 1.0 (Python falsy-zero bug).
#     """
#     from ragas.testset.graph import NodeType

#     # Pre-index chunk IDs so we can filter pairs to CHUNK-CHUNK only
#     chunk_ids: set[str] = {
#         str(n.id) for n in kg.nodes if n.type == NodeType.CHUNK
#     }

#     pairs: list[tuple] = []
#     seen: set[tuple[str, str]] = set()
#     for rel in kg.relationships:
#         src_id = str(rel.source.id)
#         tgt_id = str(rel.target.id)
#         if src_id == tgt_id:
#             continue
#         # Only keep CHUNK-CHUNK pairs (exclude DOCUMENT nodes)
#         if src_id not in chunk_ids or tgt_id not in chunk_ids:
#             continue
#         key = tuple(sorted([src_id, tgt_id]))
#         if key in seen:
#             continue
#         # FIX: use explicit None-check so score=0.0 is not falsily skipped
#         score = None
#         for prop_key in ("entity_jaccard_similarity", "cosine_similarity",
#                          "summary_similarity"):
#             val = rel.properties.get(prop_key)
#             if val is not None:
#                 score = val
#                 break
#         if score is None:
#             score = 1.0  # no score property → treat as fully connected
#         if not isinstance(score, (int, float)) or score < min_overlap:
#             continue
#         pairs.append((rel.source, rel.target))
#         seen.add(key)
#     return pairs


# def build_scenarios(
#     kg,
#     n_candidates: int,
#     distribution: dict[str, float],
#     rng: random.Random,
#     min_context_chars: int,
#     min_per_pdf: int = 20,
# ) -> list[dict[str, Any]]:
#     """
#     Build n scenarios, one per desired candidate. Each scenario describes:
#       - synthesizer name
#       - persona
#       - list of source nodes (1 for single-hop, 2+ for multi-hop)

#     FIX 3: nodes whose text is shorter than min_context_chars are excluded
#     from sampling, so degenerate KG chunks (header-only pages, reference-ID
#     rows) never reach the LLM.

#     STRATIFIED SAMPLING: the total budget is split into two buckets.
#       Bucket 1 (Guaranteed): each PDF with ≥1 valid node gets exactly
#         min_per_pdf single-hop scenarios drawn from its own nodes only.
#         If the guaranteed budget exceeds n_candidates, min_per_pdf is
#         reduced proportionally.
#       Bucket 2 (Random): remaining scenarios are drawn from the global
#         pool using the original distribution (including multi-hop), exactly
#         as before.
#     """
#     content_nodes = extract_content_nodes(kg)
#     if not content_nodes:
#         sys.exit("No content nodes in KG — rebuild with --force")

#     # FIX 3 — filter degenerate nodes from the sampling pool
#     content_nodes_filtered = [
#         n for n in content_nodes
#         if len(_node_text(n).strip()) >= min_context_chars
#     ]
#     dropped_nodes = len(content_nodes) - len(content_nodes_filtered)
#     if dropped_nodes:
#         print(f"  FIX-3: dropped {dropped_nodes} degenerate nodes "
#               f"(< {min_context_chars} chars) from sampling pool")
#     if not content_nodes_filtered:
#         sys.exit("All content nodes are below min_context_chars — "
#                  "lower --min-context-chars or rebuild KG")
#     content_nodes = content_nodes_filtered

#     # FIX 3 — also filter multihop pairs where either node is degenerate
#     multihop_pairs_raw = get_multihop_pairs(kg)
#     multihop_pairs = [
#         (src, tgt) for src, tgt in multihop_pairs_raw
#         if len(_node_text(src).strip()) >= min_context_chars
#         and len(_node_text(tgt).strip()) >= min_context_chars
#     ]
#     dropped_pairs = len(multihop_pairs_raw) - len(multihop_pairs)
#     if dropped_pairs:
#         print(f"  FIX-3: dropped {dropped_pairs} multi-hop pairs "
#               f"containing a degenerate node")

#     print(f"  Found {len(content_nodes)} content nodes, "
#           f"{len(multihop_pairs)} multi-hop pairs")

#     if not multihop_pairs:
#         # Force all to single-hop if there are no pairs
#         print("  WARNING: no multi-hop pairs; degrading to single-hop only")
#         distribution = {
#             SYNTH_SINGLE_HOP_SPECIFIC: 0.7,
#             SYNTH_SINGLE_HOP_ABSTRACT: 0.3,
#         }

#     # ── STRATIFIED SAMPLING ──────────────────────────────────────────────────
#     # Step 1: Group filtered nodes by source PDF.
#     # CHUNK nodes don't carry filename/source in their own properties —
#     # that metadata lives on the parent DOCUMENT nodes.  We build a
#     # CHUNK-id → PDF lookup by traversing KG relationships first, then
#     # fall back to _source_docs() for any node that wasn't resolved.
#     # Nodes still unresolved after both attempts go into "__unknown__" and
#     # are excluded from the guaranteed bucket (Bucket 2 only).
#     chunk_to_pdf = build_chunk_to_pdf_map(kg)
#     resolved = sum(1 for n in content_nodes if str(n.id) in chunk_to_pdf)
#     print(f"  PDF map: resolved {resolved}/{len(content_nodes)} chunk nodes "
#           f"via KG relationships")

#     pdf_to_nodes: dict[str, list] = {}
#     for node in content_nodes:
#         key = chunk_to_pdf.get(str(node.id))
#         if not key:
#             docs = _source_docs(node)
#             key = docs[0] if docs else "__unknown__"
#         pdf_to_nodes.setdefault(key, []).append(node)

#     # PDFs that have at least one valid node (excluding anonymous nodes)
#     eligible_pdfs = sorted(k for k in pdf_to_nodes if k != "__unknown__")
#     n_eligible = len(eligible_pdfs)

#     # Step 2: Calculate the two-bucket budgets.
#     # Cap min_per_pdf so the guaranteed bucket never exceeds n_candidates.
#     effective_min_per_pdf = min_per_pdf
#     if n_eligible > 0 and n_eligible * min_per_pdf > n_candidates:
#         effective_min_per_pdf = n_candidates // n_eligible
#         print(f"  STRATIFIED: min_per_pdf reduced from {min_per_pdf} to "
#               f"{effective_min_per_pdf} to fit within n_candidates={n_candidates}")

#     guaranteed_budget = n_eligible * effective_min_per_pdf
#     random_budget     = n_candidates - guaranteed_budget

#     print(f"\n  PDF Coverage Report (Stratified Sampling):")
#     print(f"    PDFs with valid nodes      : {n_eligible} / "
#           f"{n_eligible + (1 if '__unknown__' in pdf_to_nodes else 0)} groups")
#     if "__unknown__" in pdf_to_nodes:
#         print(f"    Nodes with no PDF metadata : {len(pdf_to_nodes['__unknown__'])} "
#               f"(random bucket only)")
#     print(f"    effective min_per_pdf      : {effective_min_per_pdf}")
#     print(f"    Guaranteed budget (Bucket 1): {guaranteed_budget}")
#     print(f"    Random budget     (Bucket 2): {random_budget}")
#     print(f"    Total                       : {n_candidates}")

#     # Ratio for splitting single-hop types within the guaranteed bucket.
#     # Uses the same proportion as the original distribution (spec:abstract).
#     sh_spec_w = distribution.get(SYNTH_SINGLE_HOP_SPECIFIC, 0.35)
#     sh_abst_w = distribution.get(SYNTH_SINGLE_HOP_ABSTRACT, 0.15)
#     sh_total_w = sh_spec_w + sh_abst_w if (sh_spec_w + sh_abst_w) > 0 else 1.0
#     single_hop_types = [SYNTH_SINGLE_HOP_SPECIFIC, SYNTH_SINGLE_HOP_ABSTRACT]
#     single_hop_weights = [sh_spec_w / sh_total_w, sh_abst_w / sh_total_w]

#     # ── BUCKET 1: Guaranteed per-PDF single-hop scenarios ───────────────────
#     guaranteed_scenarios: list[dict[str, Any]] = []

#     for pdf_key in eligible_pdfs:
#         pdf_nodes = pdf_to_nodes[pdf_key]
#         sampled = rng.choices(pdf_nodes, k=effective_min_per_pdf)
#         for node in sampled:
#             synth = rng.choices(single_hop_types, weights=single_hop_weights, k=1)[0]
#             persona = rng.choice(AUTOSAR_PERSONAS)
#             guaranteed_scenarios.append({
#                 "synthesizer_name": synth,
#                 "persona": persona,
#                 "nodes": [node],
#             })

#     # ── BUCKET 2: Random scenarios from the global pool ─────────────────────
#     # Apply the full original distribution to the random_budget.
#     # Single-hop draws from the full content_nodes pool (all PDFs);
#     # multi-hop draws from multihop_pairs as before.
#     random_scenarios: list[dict[str, Any]] = []

#     if random_budget > 0:
#         counts = {k: int(round(random_budget * v)) for k, v in distribution.items()}
#         # Round-off fix
#         diff = random_budget - sum(counts.values())
#         if diff:
#             k_max = max(counts, key=counts.get)
#             counts[k_max] += diff

#         print(f"\n  Bucket 2 scenario counts:")
#         for k, v in counts.items():
#             print(f"     {k:<28} {v}")

#         # Single-hop from global pool
#         for synth in (SYNTH_SINGLE_HOP_SPECIFIC, SYNTH_SINGLE_HOP_ABSTRACT):
#             n = counts.get(synth, 0)
#             if not n:
#                 continue
#             nodes = rng.choices(content_nodes, k=n)
#             for node in nodes:
#                 persona = rng.choice(AUTOSAR_PERSONAS)
#                 random_scenarios.append({
#                     "synthesizer_name": synth,
#                     "persona": persona,
#                     "nodes": [node],
#                 })

#         # Multi-hop from global pairs pool
#         for synth in (SYNTH_MULTI_HOP_SPECIFIC, SYNTH_MULTI_HOP_ABSTRACT):
#             n = counts.get(synth, 0)
#             if not n or not multihop_pairs:
#                 continue
#             pairs = rng.choices(multihop_pairs, k=n)
#             for pair in pairs:
#                 persona = rng.choice(AUTOSAR_PERSONAS)
#                 random_scenarios.append({
#                     "synthesizer_name": synth,
#                     "persona": persona,
#                     "nodes": list(pair),
#                 })

#     # ── Merge both buckets and shuffle ───────────────────────────────────────
#     scenarios = guaranteed_scenarios + random_scenarios
#     rng.shuffle(scenarios)

#     print(f"\n  Final scenario totals:")
#     print(f"    Bucket 1 (guaranteed, single-hop) : {len(guaranteed_scenarios)}")
#     print(f"    Bucket 2 (random, mixed)          : {len(random_scenarios)}")
#     print(f"    Combined                          : {len(scenarios)}")

#     return scenarios


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 2 — CONVERT A SCENARIO TO QUESTION-GEN CHAT MESSAGES
# # ══════════════════════════════════════════════════════════════════════════════

# def _node_text(node) -> str:
#     return node.properties.get("page_content") or node.properties.get("text") or ""


# def _source_docs(node) -> list[str]:
#     meta = node.properties.get("document_metadata") or {}
#     src = meta.get("filename") or meta.get("source")
#     return [src] if src else []


# def scenario_to_question_messages(scenario: dict[str, Any]) -> list[dict]:
#     synth = scenario["synthesizer_name"]
#     persona: Persona = scenario["persona"]
#     nodes = scenario["nodes"]
#     style = "abstract" if synth.endswith("abstract") else "specific"
#     style_hint = STYLE_HINTS[style]

#     if synth.startswith("single_hop"):
#         # Single-hop: uses imported prompt with {context}
#         return messages(
#             QUESTION_GEN_SYSTEM,
#             QUESTION_GEN_USER_SINGLEHOP.format(
#                 context=_node_text(nodes[0]),
#                 persona_role=persona.role_description,
#                 question_type=style,
#                 style_hint=style_hint,
#             ),
#         )
#     else:
#         # Multi-hop: uses QUESTION_GEN_USER_MULTIHOP (defined above) which
#         # takes {context_1} and {context_2} separately and enforces bridging.
#         # Guard: if somehow nodes has < 2 entries, fall back to single-hop
#         # rather than raising an IndexError.
#         if len(nodes) < 2:
#             return messages(
#                 QUESTION_GEN_SYSTEM,
#                 QUESTION_GEN_USER_SINGLEHOP.format(
#                     context=_node_text(nodes[0]),
#                     persona_role=persona.role_description,
#                     question_type=style,
#                     style_hint=style_hint,
#                 ),
#             )
#         return messages(
#             QUESTION_GEN_SYSTEM,
#             QUESTION_GEN_USER_MULTIHOP.format(
#                 context_1=_node_text(nodes[0]),
#                 context_2=_node_text(nodes[1]),
#                 persona_role=persona.role_description,
#                 question_type=style,
#                 style_hint=style_hint,
#             ),
#         )


# def scenario_with_question_to_answer_messages(
#     scenario: dict[str, Any],
#     question: str,
# ) -> list[dict]:
#     nodes = scenario["nodes"]

#     if len(nodes) == 1:
#         # Single-hop: unchanged — uses imported ANSWER_GEN_USER with {context}
#         return messages(
#             ANSWER_GEN_SYSTEM,
#             ANSWER_GEN_USER.format(
#                 context=_node_text(nodes[0]),
#                 question=question,
#             ),
#         )
#     else:
#         # FIX 1 — uses ANSWER_GEN_USER_MULTIHOP (defined above) which takes
#         # {context_1} and {context_2} separately and requires both to be used.
#         return messages(
#             ANSWER_GEN_SYSTEM,
#             ANSWER_GEN_USER_MULTIHOP.format(
#                 context_1=_node_text(nodes[0]),
#                 context_2=_node_text(nodes[1]),
#                 question=question,
#             ),
#         )


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 2b — HTTP CLIENT (vLLM server, same approach as build_kg.py)
# # ══════════════════════════════════════════════════════════════════════════════

# def build_client(args: argparse.Namespace):
#     from openai import OpenAI
#     client = OpenAI(
#         api_key="dummy",
#         base_url=args.vllm_url,
#     )
#     return client


# def check_vllm_connectivity(client, model: str) -> None:
#     """
#     QUALITY-12 FIX: Verify the vLLM server is alive and serving the expected
#     model before generating thousands of scenarios. Exits with a clear message
#     on failure instead of silently producing 0 candidates across all batches.
#     """
#     try:
#         resp = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "user", "content": "Reply with the single word: OK"}],
#             max_tokens=5,
#             temperature=0.0,
#         )
#         reply = resp.choices[0].message.content.strip()
#         print(f"  Server check: OK (reply='{reply}')")
#     except Exception as e:
#         print(f"\n  [FATAL] vLLM server is not responding: {type(e).__name__}: {e}")
#         print(f"  Make sure the server is running and the model is loaded.")
#         sys.exit(1)


# def _extract_json_from_text(text: str) -> dict | None:
#     """
#     QUALITY-13 FIX: Robustly extract a JSON object from model output.

#     Handles:
#       - Clean JSON:            {"question": "..."}
#       - Fenced with ```json:   ```json\n{"question": "..."}\n```
#       - Fenced with ```:       ```\n{"question": "..."}\n```
#       - Preamble text:         "Sure!\n```json\n{"question": "..."}\n```"
#       - Inline JSON in prose:  finds the first {...} or [...] span

#     Returns the parsed dict/list or None on failure.
#     """
#     import re
#     text = text.strip()

#     # Try direct parse first (most common for well-behaved models)
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         pass

#     # Strip code fences and retry
#     # Match ```json ... ``` or ``` ... ``` with optional whitespace
#     fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
#     if fence_match:
#         try:
#             return json.loads(fence_match.group(1))
#         except json.JSONDecodeError:
#             pass

#     # Last resort: find the first JSON object {...} in the text
#     obj_match = re.search(r"\{[\s\S]+\}", text)
#     if obj_match:
#         try:
#             return json.loads(obj_match.group(0))
#         except json.JSONDecodeError:
#             pass

#     return None


# def _single_chat_json(
#     client,
#     conv: list[dict],
#     temperature: float,
#     top_p: float,
#     max_tokens: int,
#     seed: int,
#     model: str,
#     call_idx: int,
# ) -> tuple[int, dict | None, str | None]:
#     """
#     Single HTTP call to vLLM. Returns (call_idx, parsed_dict_or_None, error_str_or_None).
#     Used by _batch_chat_json for concurrent execution.
#     """
#     try:
#         resp = client.chat.completions.create(
#             model=model,
#             messages=conv,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#             seed=seed + call_idx,   # ISSUE-11 FIX: unique seed per call
#         )
#         text = resp.choices[0].message.content.strip()
#         parsed = _extract_json_from_text(text)
#         if parsed is None:
#             return call_idx, None, f"JSON parse failed; raw='{text[:120]}'"
#         return call_idx, parsed, None
#     except Exception as e:
#         return call_idx, None, f"{type(e).__name__}: {e}"


# def _batch_chat_json(
#     client,
#     conversations: list[list[dict]],
#     temperature: float,
#     top_p: float,
#     max_tokens: int,
#     seed: int,
#     model: str,
#     max_workers: int = 8,
# ) -> list[dict | None]:
#     """
#     Send all conversations to the vLLM server concurrently and collect JSON
#     results.  Returns a list of parsed dicts in the same order as
#     conversations; None for any call that failed.

#     BUG-1 FIX: Exceptions are now logged (first error per batch) instead of
#     being swallowed silently.
#     BUG-2 FIX: Uses ThreadPoolExecutor for concurrent HTTP calls instead of
#     sequential one-by-one execution. max_workers=8 is safe for vLLM since it
#     has its own internal queue; tune up if your server has spare capacity.
#     QUALITY-13 FIX: JSON extraction uses _extract_json_from_text() which
#     handles fenced, preambled, and inline JSON robustly.
#     ISSUE-11 FIX: Each call gets seed + call_idx so intra-batch calls are
#     not identically seeded.
#     """
#     results: list[dict | None] = [None] * len(conversations)
#     first_error: str | None = None
#     error_count = 0

#     with ThreadPoolExecutor(max_workers=max_workers) as pool:
#         futures = {
#             pool.submit(
#                 _single_chat_json,
#                 client, conv, temperature, top_p, max_tokens, seed, model, idx,
#             ): idx
#             for idx, conv in enumerate(conversations)
#         }
#         for future in as_completed(futures):
#             idx, parsed, err = future.result()
#             results[idx] = parsed
#             if err:
#                 error_count += 1
#                 if first_error is None:
#                     first_error = err

#     if first_error:
#         print(f"  [WARN] {error_count}/{len(conversations)} calls failed "
#               f"in this batch. First error: {first_error}")

#     return results


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 2c — FIX 2 + FIX 1 answer-guard: POST-GENERATION VALIDATORS
# # ══════════════════════════════════════════════════════════════════════════════

# def _is_restatement(question: str, answer: str) -> bool:
#     """
#     Detect answers that merely restate the question.

#     Two signals:
#       (a) Structural echo: the answer opens by mirroring the question's
#           subject phrase (model just flips the interrogative into declarative).
#           e.g. Q: "What is the purpose of X?" -> A: "The purpose of X is..."
#       (b) Trivial pointer: answer is short (<120 chars) with no technical
#           elaboration — just a pure lookup with no reasoning content.

#     ISSUE-8 FIX: Expanded prefix list covers "how does/is/are", "why is/does",
#     "which", "when is" — not just "what is/are/does".
#     ISSUE-9 FIX: Expanded elaboration keywords cover common technical verbs
#     so valid short answers are not incorrectly dropped.

#     Returns True if the candidate should be dropped.
#     """
#     q = question.strip()
#     a = answer.strip()

#     # Signal (a): strip wh-word/phrase, check if first 40 chars of question
#     # body appear verbatim in the first 160 chars of the answer.
#     q_body = q.lower()
#     # Ordered longest-first to avoid partial matches (e.g. "what is" vs "what is the")
#     _PREFIXES = (
#         "what is the ", "what are the ", "what does the ", "what is a ",
#         "what are a ", "where can the ", "where is the ", "where does the ",
#         "how does the ", "how is the ", "how are the ", "how do the ",
#         "why is the ", "why does the ", "why are the ",
#         "which ", "when is the ", "when does the ",
#         "what role does ", "what is ", "what are ", "how does ", "how is ",
#     )
#     for prefix in _PREFIXES:
#         if q_body.startswith(prefix):
#             q_body = q_body[len(prefix):]
#             break
#     q_body = q_body.rstrip("?").strip()
#     if q_body[:40] and q_body[:40] in a.lower()[:160]:
#         return True

#     # Signal (b): very short answer with no elaboration markers
#     if len(a) < 120:
#         has_elaboration = any(c.isdigit() for c in a) or \
#                           any(kw in a.lower() for kw in (
#                               # reasoning connectors
#                               "because", "therefore", "thus", "hence",
#                               "in order", "so that", "as a result",
#                               # technical verbs common in AUTOSAR answers
#                               "ensures", "requires", "defined", "specified",
#                               "allows", "prevents", "means", "refers",
#                               "used to", "contains", "implements", "provides",
#                               "supports", "enables", "manages", "constrains",
#                               "consists", "comprises", "represents", "performs",
#                               "handles", "defines", "separates", "abstracts",
#                               "encapsulates", "exposes", "describes", "maps",
#                           ))
#         if not has_elaboration:
#             return True

#     return False


# def _multihop_uses_both_contexts(answer: str, ctx1: str, ctx2: str,
#                                   min_coverage: float = 0.10) -> bool:
#     """
#     Verify the answer draws from both contexts via word-overlap.
#     min_coverage=0.10 is permissive: only rejects answers that completely
#     ignore one context, not ones that merely lean on one more.

#     ISSUE-10 FIX: Common stop words are excluded from the overlap calculation.
#     Without this, any answer trivially passes the 10% threshold via words like
#     "the", "a", "is", "of" that appear in every answer and every context,
#     letting single-context answers slip through.
#     """
#     _STOP_WORDS = frozenset({
#         "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
#         "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
#         "with", "by", "from", "as", "it", "its", "this", "that", "these",
#         "those", "which", "who", "what", "how", "when", "where", "why",
#         "can", "may", "must", "shall", "will", "should", "would", "could",
#         "have", "has", "had", "do", "does", "did", "if", "then", "than",
#         "so", "also", "both", "each", "their", "they", "them", "there",
#     })

#     def _content_words(text: str) -> set[str]:
#         return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 2}

#     ans_words = _content_words(answer)
#     if not ans_words:
#         return False

#     ctx1_words = _content_words(ctx1)
#     ctx2_words = _content_words(ctx2)

#     hit1 = len(ans_words & ctx1_words) / max(len(ctx1_words), 1)
#     hit2 = len(ans_words & ctx2_words) / max(len(ctx2_words), 1)
#     return hit1 >= min_coverage and hit2 >= min_coverage


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 3 — GENERATION LOOP (batched, checkpointed)
# # ══════════════════════════════════════════════════════════════════════════════

# def run_generation(
#     scenarios: list[dict[str, Any]],
#     client,
#     args: argparse.Namespace,
#     output_path: Path,
#     already_done: int,
#     chunk_to_pdf: dict[str, str],
# ) -> None:
#     """
#     For each batch of scenarios:
#       1. Run question-gen -> N questions
#       2. For the subset that produced valid questions, run answer-gen -> N answers
#       3. Drop NOT_ANSWERABLE, JSON-parse failures, restatements,
#          multi-hop answers that ignore one context
#       4. Append the survivors to candidates.jsonl

#     BUG-4 FIX: chunk_to_pdf is passed in so source_documents is populated
#     correctly for CHUNK-based candidates. Previously _source_docs() always
#     returned [] for CHUNK nodes (they carry no document_metadata).
#     """
#     scenarios = scenarios[already_done:]
#     total = already_done + len(scenarios)
#     print(f"\n  Generating {len(scenarios)} candidates "
#           f"(resuming from {already_done}/{total}) ...")
#     print(f"  Batch size: {args.batch_size}")

#     current = already_done
#     batch_num = 0

#     for i in range(0, len(scenarios), args.batch_size):
#         batch_num += 1
#         batch = scenarios[i:i + args.batch_size]
#         t0 = time.time()

#         # --- Phase 1: generate questions for the batch ---
#         q_convs = [scenario_to_question_messages(s) for s in batch]
#         q_results = _batch_chat_json(
#             client, q_convs,
#             temperature=args.q_temperature,
#             top_p=0.9,
#             max_tokens=300,
#             seed=args.seed + batch_num,
#             model=args.generator_model,
#         )

#         # Pair scenarios with generated questions (drop parse failures)
#         with_q: list[tuple[dict[str, Any], str]] = []
#         parse_fail = 0
#         for s, q in zip(batch, q_results):
#             if not q or "question" not in q:
#                 parse_fail += 1
#                 continue
#             # FIX 1 — {"question": null} means no genuine bridge existed;
#             # drop cleanly instead of letting a fake multi-hop through.
#             if q["question"] is None:
#                 parse_fail += 1
#                 continue
#             qtxt = str(q["question"]).strip()
#             if not qtxt or len(qtxt) < 10:
#                 parse_fail += 1
#                 continue
#             with_q.append((s, qtxt))

#         if not with_q:
#             print(f"     Batch {batch_num}: 0/{len(batch)} questions parseable — skipping")
#             continue

#         # --- Phase 2: generate answers for those questions ---
#         a_convs = [scenario_with_question_to_answer_messages(s, q) for s, q in with_q]
#         a_results = _batch_chat_json(
#             client, a_convs,
#             temperature=args.a_temperature,
#             top_p=1.0,
#             max_tokens=512,
#             seed=args.seed + batch_num,
#             model=args.generator_model,
#         )

#         # --- Phase 3: build candidates, apply all quality filters ---
#         new_candidates: list[dict[str, Any]] = []
#         not_answerable = 0
#         a_parse_fail = 0
#         restatement_drop = 0   # FIX 2
#         single_ctx_drop = 0    # FIX 1 answer guard

#         for (s, q), a in zip(with_q, a_results):
#             if not a or "answer" not in a:
#                 a_parse_fail += 1
#                 continue
#             atxt = str(a["answer"]).strip()
#             if not atxt or atxt.upper().startswith("NOT_ANSWERABLE"):
#                 not_answerable += 1
#                 continue
#             if len(atxt) < 20:
#                 a_parse_fail += 1
#                 continue

#             # FIX 2 — drop restatements
#             if _is_restatement(q, atxt):
#                 restatement_drop += 1
#                 continue

#             # FIX 1 (answer guard) — drop multi-hop answers that ignore one context
#             if "multi_hop" in s["synthesizer_name"] and len(s["nodes"]) == 2:
#                 ctx1 = _node_text(s["nodes"][0])
#                 ctx2 = _node_text(s["nodes"][1])
#                 if not _multihop_uses_both_contexts(atxt, ctx1, ctx2):
#                     single_ctx_drop += 1
#                     continue

#             contexts = [_node_text(n) for n in s["nodes"]]
#             source_node_ids = [str(n.id) for n in s["nodes"]]
#             # BUG-4 FIX: CHUNK nodes have no document_metadata, so _source_docs()
#             # always returns []. Use chunk_to_pdf map (traversed from KG
#             # relationships) to correctly resolve source PDF paths.
#             source_docs: list[str] = []
#             for n in s["nodes"]:
#                 pdf = chunk_to_pdf.get(str(n.id))
#                 if pdf:
#                     source_docs.append(pdf)
#                 else:
#                     source_docs.extend(_source_docs(n))  # fallback for DOCUMENT nodes
#             source_docs = sorted(set(source_docs))

#             cand = new_candidate(
#                 candidate_id=f"c_{current + len(new_candidates):06d}",
#                 user_input=q,
#                 reference=atxt,
#                 reference_contexts=contexts,
#                 synthesizer_name=s["synthesizer_name"],
#                 persona_name=s["persona"].name,
#                 source_node_ids=source_node_ids,
#                 source_documents=source_docs,
#                 generator_model=args.generator_model,
#                 generator_config={
#                     "q_temperature": args.q_temperature,
#                     "a_temperature": args.a_temperature,
#                     "seed": args.seed + batch_num,
#                 },
#             )
#             new_candidates.append(cand)

#         # --- Persist ---
#         append_jsonl(new_candidates, output_path)
#         current += len(new_candidates)

#         dt = time.time() - t0
#         print(
#             f"     Batch {batch_num:>3}: "
#             f"+{len(new_candidates)}/{len(batch)} kept "
#             f"(q_fail={parse_fail}, a_fail={a_parse_fail}, "
#             f"not_answerable={not_answerable}, "
#             f"restatement={restatement_drop}, "
#             f"single_ctx={single_ctx_drop})  "
#             f"[{dt:.1f}s]  total={current}"
#         )

#     print(f"\n  Generation complete. Total candidates persisted: {current}")


# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def main() -> None:
#     args = parse_args()
#     output_dir = Path(args.output_dir)
#     stage_dir = output_dir / "stage_a_generation"
#     stage_dir.mkdir(parents=True, exist_ok=True)
#     candidates_path = stage_dir / "candidates.jsonl"
#     config_path = stage_dir / "generation_config.json"

#     n_candidates = int(args.target * args.overgen_ratio)

#     print("=" * 70)
#     print(" Stage A :: Generate Candidates")
#     print("=" * 70)
#     print(f" Target final size : {args.target}")
#     print(f" Over-gen ratio    : {args.overgen_ratio}")
#     print(f" Candidates to gen : {n_candidates}")
#     print(f" Output            : {candidates_path}")
#     print(f" Generator         : {args.generator_model}")
#     print(f" vLLM URL          : {args.vllm_url}")
#     print(f" Min context chars : {args.min_context_chars}")
#     print("=" * 70)

#     # --- Resume check ---
#     already_done = count_jsonl(candidates_path)
#     if already_done >= n_candidates:
#         print(f"\n  Already have {already_done} candidates (target {n_candidates}). Done.")
#         return
#     if already_done > 0:
#         print(f"\n  Resuming: {already_done} candidates already persisted")

#     # --- Load KG ---
#     print("\n[1/4] Loading KG ...")
#     kg = load_kg(Path(args.kg_file))
#     print(f"  Loaded: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")

#     # --- Build scenarios ---
#     print("\n[2/4] Building scenarios ...")
#     rng = random.Random(args.seed)
#     scenarios = build_scenarios(
#         kg,
#         n_candidates=n_candidates,
#         distribution=DEFAULT_SYNTH_DISTRIBUTION,
#         rng=rng,
#         min_context_chars=args.min_context_chars,
#         min_per_pdf=args.min_per_pdf,
#     )
#     print(f"  Built {len(scenarios)} scenarios")

#     # --- Connect to vLLM server ---
#     print("\n[3/4] Connecting to vLLM server ...")
#     print(f"  URL   : {args.vllm_url}")
#     print(f"  Model : {args.generator_model}")
#     client = build_client(args)
#     check_vllm_connectivity(client, args.generator_model)  # QUALITY-12 FIX

#     # --- Save config (for reproducibility) ---
#     atomic_write_json(
#         {
#             "target":             args.target,
#             "overgen_ratio":      args.overgen_ratio,
#             "n_scenarios":        len(scenarios),
#             "distribution":       DEFAULT_SYNTH_DISTRIBUTION,
#             "personas":           [p.name for p in AUTOSAR_PERSONAS],
#             "generator_model":    args.generator_model,
#             "vllm_url":           args.vllm_url,
#             "q_temperature":      args.q_temperature,
#             "a_temperature":      args.a_temperature,
#             "seed":               args.seed,
#             "batch_size":         args.batch_size,
#             "min_context_chars":  args.min_context_chars,
#             "min_per_pdf":        args.min_per_pdf,
#         },
#         config_path,
#     )

#     # --- Generate ---
#     print("\n[4/4] Generating ...")
#     # chunk_to_pdf is rebuilt here from the already-loaded kg so run_generation
#     # can correctly populate source_documents (BUG-4 FIX).
#     chunk_to_pdf = build_chunk_to_pdf_map(kg)
#     run_generation(scenarios, client, args, candidates_path, already_done, chunk_to_pdf)

#     print("\n" + "=" * 70)
#     print(" Stage A complete.")
#     print(f" Next: python validate_candidates.py --output-dir {args.output_dir}")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()




"""
Stage A: Generate candidate QA pairs from the knowledge graph.

What makes this different from RAGAS's TestsetGenerator:
  - We define our own scenarios. No auto-generated "casual typer" personas
    that produce `wat iz AUTOSAR` questions.
  - We split question generation and answer generation into TWO separate
    LLM calls (RAGalyst Nov 2025 showed this beats the combined prompt).
  - The answer prompt is strict-grounding: returns NOT_ANSWERABLE when the
    context doesn't support an answer, and those are dropped.
  - multi_hop scenarios are GUARANTEED to have >=2 contexts by construction.
  - We over-generate 2x the target size so Stage B filtering has room.

Model: Qwen/Qwen2.5-72B-Instruct-AWQ (AWQ 4-bit, TP=2 on your 2x48GB).
Throughput: ~30-60 candidates/min with batched vLLM server inference.

Resumability: candidates.jsonl is append-only. If the script is killed
mid-run, rerunning picks up from the last saved candidate.

Usage:
    # Terminal 1: start vLLM server
    vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 2 \
        --max-model-len 8192 --quantization awq --port 8011

    # Terminal 2:
    python generate_candidates.py \
        --kg-file ./output/kg/knowledge_graph.json \
        --output-dir ./output \
        --target 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from collections import defaultdict

from shared.io_utils import atomic_write_json, append_jsonl, count_jsonl
from shared.personas import AUTOSAR_PERSONAS, Persona
from shared.prompts import (
    QUESTION_GEN_SYSTEM, QUESTION_GEN_USER_SINGLEHOP,
    ANSWER_GEN_SYSTEM, ANSWER_GEN_USER, STYLE_HINTS,
)
from shared.schemas import (
    DEFAULT_SYNTH_DISTRIBUTION,
    SYNTH_SINGLE_HOP_SPECIFIC, SYNTH_SINGLE_HOP_ABSTRACT,
    SYNTH_MULTI_HOP_SPECIFIC,  SYNTH_MULTI_HOP_ABSTRACT,
    new_candidate,
)
from shared.llm_batch import messages


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT OVERRIDES — multi-hop only
#
# QUESTION_GEN_USER_MULTIHOP: defined here (not imported from shared/prompts.py).
# The shared version takes {contexts_block} — one merged string — so the model
# latches onto the first relevant chunk and ignores the other. This version
# takes {context_1} and {context_2} as separate named slots with an explicit
# BRIDGING REQUIREMENT.
#
# ANSWER_GEN_USER_MULTIHOP: new constant (not in shared/prompts.py at all).
# ANSWER_GEN_USER is still used unchanged for single-hop.
# ══════════════════════════════════════════════════════════════════════════════

QUESTION_GEN_USER_MULTIHOP = """\
You are a question-writing expert for a technical RAG evaluation dataset.
You produce ONE well-formed question per call, in clean, formal technical English.

Strict rules:
1. The question must be answerable using ONLY the provided contexts.
2. The question must be self-contained: a reader without the contexts should
   still understand what is being asked.
3. Do NOT reference figure numbers, document IDs, page numbers, or section
   numbers unless those artifacts are themselves the topic.
4. Do NOT ask "what does the document say about X" — ask directly about X.
5. Do not write casual or informal English. Use full technical terms.
6. The question must match the requested TYPE and PERSONA.

You have been given TWO separate source contexts. Your task is to generate a
single question that REQUIRES INFORMATION FROM BOTH contexts to answer.

--- CONTEXT 1 ---
{context_1}

--- CONTEXT 2 ---
{context_2}
--- END CONTEXTS ---

Persona: {persona_role}

Question type: {question_type}

BRIDGING REQUIREMENT (mandatory):
  - The question must be impossible to answer using CONTEXT 1 alone.
  - The question must be impossible to answer using CONTEXT 2 alone.
  - The question must require combining or contrasting information from both.
  - Do NOT ask about something that appears in only one context.

Good bridging patterns:
  - "How does [concept from ctx1] interact with / constrain [concept from ctx2]?"
  - "What are the implications of [rule from ctx1] for the design in ctx2?"
  - "Compare the approach in [ctx1 topic] with the requirement in [ctx2 topic]."
  - "Given [constraint in ctx1], what must change in the design in ctx2?"

REJECT: if no genuine informational bridge exists between the two contexts,
output exactly: {{"question": null}}

Otherwise output exactly this JSON (no markdown, no extra keys):
{{"question": "<your bridging question here>"}}

Write one question per the rules. {style_hint}"""


ANSWER_GEN_USER_MULTIHOP = """\
You are an expert answer-writer for a technical RAG evaluation dataset.
You produce ONE ground-truth answer grounded strictly in the provided contexts.

Strict rules:
1. Your answer MUST draw on information from BOTH Context 1 and Context 2.
   An answer that uses only one context is NOT acceptable here.
2. If the question can be fully answered from one context alone, output
   exactly: {{"answer": "NOT_ANSWERABLE"}}
3. If the contexts do not contain enough information to answer, output
   exactly: {{"answer": "NOT_ANSWERABLE"}}
4. Use ONLY the information in the provided contexts. Do NOT add outside knowledge.
5. The answer should be complete but concise. Explain the connection between
   the two contexts as it relates to the question.
6. Do NOT say "according to context 1" etc. Just give the factual answer.
7. Write in clean, formal technical English.

--- CONTEXT 1 ---
{context_1}

--- CONTEXT 2 ---
{context_2}
--- END CONTEXTS ---

Question: {question}

Respond ONLY with valid JSON in this exact format:
{{"answer": "<the answer>"}}
No preamble, no markdown, no code fences."""


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--kg-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", type=int, default=500,
                   help="Final dataset size target (Stage C output)")
    p.add_argument("--overgen-ratio", type=float, default=2.0,
                   help="Default over-generation ratio for single-hop types (default: 2.0)")
    p.add_argument("--overgen-ratio-multihop", type=float, default=3.5,
                   help="Over-generation ratio for multi-hop types — higher because "
                        "null-bridge and single-ctx drops are more frequent (default: 3.5)")
    # Model
    p.add_argument("--generator-model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p.add_argument("--vllm-url",        default="http://localhost:8011/v1",
                   help="URL of the running vLLM server")
    # Generation
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--q-temperature",   type=float, default=0.3)
    p.add_argument("--a-temperature",   type=float, default=0.0)
    p.add_argument("--seed",            type=int,   default=42)
    # Context quality gate
    p.add_argument("--min-context-chars", type=int, default=150,
                   help="Min chars a context chunk must have to be used (default: 150)")
    # Stratified per-PDF sampling
    p.add_argument("--min-per-pdf", type=int, default=20,
                   help="Guaranteed minimum scenarios sampled from each PDF (default: 20).")
    # Per-type distribution (must sum to 1.0)
    p.add_argument("--dist-single-hop-specific", type=float, default=0.35,
                   help="Fraction of final dataset that is single_hop_specific (default: 0.35)")
    p.add_argument("--dist-single-hop-abstract", type=float, default=0.15,
                   help="Fraction of final dataset that is single_hop_abstract (default: 0.15)")
    p.add_argument("--dist-multi-hop-specific", type=float, default=0.35,
                   help="Fraction of final dataset that is multi_hop_specific (default: 0.35)")
    p.add_argument("--dist-multi-hop-abstract", type=float, default=0.15,
                   help="Fraction of final dataset that is multi_hop_abstract (default: 0.15)")
    # Retry cap
    p.add_argument("--max-retries-per-type", type=int, default=5,
                   help="Max extra generation passes per type when deficit remains (default: 5)")

    args = p.parse_args()

    # Validate distribution sums to 1.0
    total = (args.dist_single_hop_specific + args.dist_single_hop_abstract +
             args.dist_multi_hop_specific + args.dist_multi_hop_abstract)
    if abs(total - 1.0) > 0.01:
        p.error(f"Distribution fractions must sum to 1.0, got {total:.4f}. "
                f"Check --dist-* arguments.")

    return args

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD KG AND BUILD SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

def load_kg(path: Path):
    from ragas.testset.graph import KnowledgeGraph
    return KnowledgeGraph.load(str(path))


def extract_content_nodes(kg) -> list:
    """
    Return content-bearing nodes (chunks) from the KG. These are the leaf
    nodes produced by HeadlineSplitter — NOT the original DOCUMENT nodes,
    which are full pages and too large.
    """
    from ragas.testset.graph import NodeType

    # Prefer CHUNK nodes if the splitter ran; otherwise fall back to DOCUMENT
    chunks = [n for n in kg.nodes if n.type == NodeType.CHUNK]
    if chunks:
        return chunks
    return [n for n in kg.nodes if n.type == NodeType.DOCUMENT]


def build_chunk_to_pdf_map(kg) -> dict[str, str]:
    """
    Walk KG relationships to map each CHUNK node id → source PDF filename.

    CHUNK nodes produced by HeadlineSplitter carry only 'themes' and
    'entities' — they have no 'filename' / 'source' in their own properties.
    The PDF metadata lives on the parent DOCUMENT nodes.  This function
    traverses every relationship once and builds the lookup so that
    build_scenarios() can correctly assign CHUNK nodes to their PDF bucket.

    Both edge directions are handled:
      CHUNK ──rel──> DOCUMENT   (most common after HeadlineSplitter)
      DOCUMENT ──rel──> CHUNK   (seen in some KG builds)

    ISSUE-7 FIX: DOCUMENT-DOCUMENT relationships are explicitly skipped so
    that document IDs never pollute the chunk→pdf map.
    """
    from ragas.testset.graph import NodeType

    doc_by_id: dict[str, Any] = {
        str(n.id): n
        for n in kg.nodes
        if n.type == NodeType.DOCUMENT
    }

    chunk_to_pdf: dict[str, str] = {}
    for rel in kg.relationships:
        src_id = str(rel.source.id)
        tgt_id = str(rel.target.id)

        if tgt_id in doc_by_id:
            chunk_id = src_id
            doc = doc_by_id[tgt_id]
        elif src_id in doc_by_id:
            chunk_id = tgt_id
            doc = doc_by_id[src_id]
        else:
            continue

        # Skip DOC-DOC relationships: chunk_id would be another doc ID
        if chunk_id in doc_by_id:
            continue

        if chunk_id in chunk_to_pdf:
            continue  # already resolved; first-found wins

        meta = doc.properties.get("document_metadata", {})
        pdf = meta.get("filename") or meta.get("source")
        if pdf:
            chunk_to_pdf[chunk_id] = pdf

    return chunk_to_pdf


def get_multihop_pairs(kg, min_overlap: float = 0.5) -> list[tuple]:
    """
    Return pairs of CHUNK nodes connected by a meaningful relationship, for
    multi-hop scenarios. We use keyphrase-overlap and cosine-similarity
    relationships if present.

    Only CHUNK-CHUNK pairs are returned — DOCUMENT nodes are full pages and
    are intentionally excluded from multi-hop contexts (too large, wrong level).

    BUG-FIX: score extraction uses explicit None-check instead of `or` chaining
    so that a legitimate score of 0.0 is not treated as missing and replaced
    with 1.0 (Python falsy-zero bug).
    """
    from ragas.testset.graph import NodeType

    # Pre-index chunk IDs so we can filter pairs to CHUNK-CHUNK only
    chunk_ids: set[str] = {
        str(n.id) for n in kg.nodes if n.type == NodeType.CHUNK
    }

    pairs: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    for rel in kg.relationships:
        src_id = str(rel.source.id)
        tgt_id = str(rel.target.id)
        if src_id == tgt_id:
            continue
        # Only keep CHUNK-CHUNK pairs (exclude DOCUMENT nodes)
        if src_id not in chunk_ids or tgt_id not in chunk_ids:
            continue
        key = tuple(sorted([src_id, tgt_id]))
        if key in seen:
            continue
        # FIX: use explicit None-check so score=0.0 is not falsily skipped
        score = None
        for prop_key in ("entity_jaccard_similarity", "cosine_similarity",
                         "summary_similarity"):
            val = rel.properties.get(prop_key)
            if val is not None:
                score = val
                break
        if score is None:
            score = 1.0  # no score property → treat as fully connected
        if not isinstance(score, (int, float)) or score < min_overlap:
            continue
        pairs.append((rel.source, rel.target))
        seen.add(key)
    return pairs


def build_scenarios(
    kg,
    target: int,
    distribution: dict[str, float],
    overgen_ratio_singlehop: float,
    overgen_ratio_multihop: float,
    rng: random.Random,
    min_context_chars: int,
    min_per_pdf: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """
    Build scenarios grouped by synthesizer type, with per-type overgen ratios
    so that after expected drop rates, each type hits its target count.

    Returns a dict: { synthesizer_name -> [scenario, ...] }
    so run_generation() can track and refill per type independently.
    """
    content_nodes = extract_content_nodes(kg)
    if not content_nodes:
        sys.exit("No content nodes in KG — rebuild with --force")

    # Filter degenerate nodes
    content_nodes_filtered = [
        n for n in content_nodes
        if len(_node_text(n).strip()) >= min_context_chars
    ]
    dropped_nodes = len(content_nodes) - len(content_nodes_filtered)
    if dropped_nodes:
        print(f"  FIX-3: dropped {dropped_nodes} degenerate nodes "
              f"(< {min_context_chars} chars) from sampling pool")
    if not content_nodes_filtered:
        sys.exit("All content nodes are below min_context_chars — "
                 "lower --min-context-chars or rebuild KG")
    content_nodes = content_nodes_filtered

    # Filter degenerate multihop pairs
    multihop_pairs_raw = get_multihop_pairs(kg)
    multihop_pairs = [
        (src, tgt) for src, tgt in multihop_pairs_raw
        if len(_node_text(src).strip()) >= min_context_chars
        and len(_node_text(tgt).strip()) >= min_context_chars
    ]
    dropped_pairs = len(multihop_pairs_raw) - len(multihop_pairs)
    if dropped_pairs:
        print(f"  FIX-3: dropped {dropped_pairs} multi-hop pairs "
              f"containing a degenerate node")

    print(f"  Found {len(content_nodes)} content nodes, "
          f"{len(multihop_pairs)} multi-hop pairs")

    if not multihop_pairs:
        print("  WARNING: no multi-hop pairs; degrading to single-hop only")
        total_sh = distribution.get(SYNTH_SINGLE_HOP_SPECIFIC, 0.5) + \
                   distribution.get(SYNTH_SINGLE_HOP_ABSTRACT, 0.5)
        distribution = {
            SYNTH_SINGLE_HOP_SPECIFIC: distribution.get(SYNTH_SINGLE_HOP_SPECIFIC, 0.5) / total_sh,
            SYNTH_SINGLE_HOP_ABSTRACT: distribution.get(SYNTH_SINGLE_HOP_ABSTRACT, 0.5) / total_sh,
        }

    # Per-type target counts (what we want to survive after all drops)
    type_targets: dict[str, int] = {}
    for synth, frac in distribution.items():
        type_targets[synth] = int(round(target * frac))
    # Fix rounding so targets sum exactly to target
    diff = target - sum(type_targets.values())
    if diff:
        k_max = max(type_targets, key=type_targets.get)
        type_targets[k_max] += diff

    print(f"\n  Per-type survival targets:")
    for k, v in type_targets.items():
        print(f"    {k:<28} {v}")

    # Per-type scenario counts to generate (apply overgen ratio)
    type_scenario_counts: dict[str, int] = {}
    for synth, tgt in type_targets.items():
        ratio = overgen_ratio_multihop if "multi_hop" in synth else overgen_ratio_singlehop
        type_scenario_counts[synth] = int(round(tgt * ratio))

    print(f"\n  Per-type scenario counts (after overgen ratio):")
    for k, v in type_scenario_counts.items():
        ratio = overgen_ratio_multihop if "multi_hop" in k else overgen_ratio_singlehop
        print(f"    {k:<28} {v}  (ratio={ratio})")

    # Build chunk → PDF map for stratified sampling
    chunk_to_pdf = build_chunk_to_pdf_map(kg)
    resolved = sum(1 for n in content_nodes if str(n.id) in chunk_to_pdf)
    print(f"\n  PDF map: resolved {resolved}/{len(content_nodes)} chunk nodes")

    pdf_to_nodes: dict[str, list] = {}
    for node in content_nodes:
        key = chunk_to_pdf.get(str(node.id))
        if not key:
            docs = _source_docs(node)
            key = docs[0] if docs else "__unknown__"
        pdf_to_nodes.setdefault(key, []).append(node)

    eligible_pdfs = sorted(k for k in pdf_to_nodes if k != "__unknown__")
    n_eligible = len(eligible_pdfs)

    # Cap min_per_pdf to avoid exceeding total single-hop scenario budget
    sh_total_scenarios = (type_scenario_counts.get(SYNTH_SINGLE_HOP_SPECIFIC, 0) +
                          type_scenario_counts.get(SYNTH_SINGLE_HOP_ABSTRACT, 0))
    effective_min_per_pdf = min_per_pdf
    if n_eligible > 0 and n_eligible * min_per_pdf > sh_total_scenarios:
        effective_min_per_pdf = sh_total_scenarios // n_eligible
        print(f"  STRATIFIED: min_per_pdf reduced from {min_per_pdf} to "
              f"{effective_min_per_pdf} to fit within single-hop budget")

    sh_spec_w = distribution.get(SYNTH_SINGLE_HOP_SPECIFIC, 0.5)
    sh_abst_w = distribution.get(SYNTH_SINGLE_HOP_ABSTRACT, 0.5)
    sh_total_w = sh_spec_w + sh_abst_w if (sh_spec_w + sh_abst_w) > 0 else 1.0
    single_hop_types = [SYNTH_SINGLE_HOP_SPECIFIC, SYNTH_SINGLE_HOP_ABSTRACT]
    single_hop_weights = [sh_spec_w / sh_total_w, sh_abst_w / sh_total_w]

    # Scenarios grouped by type
    scenarios_by_type: dict[str, list[dict[str, Any]]] = {s: [] for s in distribution}

    # Bucket 1: guaranteed per-PDF single-hop scenarios
    guaranteed_per_type: dict[str, int] = {s: 0 for s in distribution}
    for pdf_key in eligible_pdfs:
        pdf_nodes = pdf_to_nodes[pdf_key]
        sampled = rng.choices(pdf_nodes, k=effective_min_per_pdf)
        for node in sampled:
            synth = rng.choices(single_hop_types, weights=single_hop_weights, k=1)[0]
            persona = rng.choice(AUTOSAR_PERSONAS)
            scenarios_by_type[synth].append({
                "synthesizer_name": synth,
                "persona": persona,
                "nodes": [node],
            })
            guaranteed_per_type[synth] += 1

    print(f"\n  Bucket 1 (guaranteed per-PDF) counts:")
    for k, v in guaranteed_per_type.items():
        print(f"    {k:<28} {v}")

    # Bucket 2: remaining scenarios up to type_scenario_counts
    for synth in distribution:
        already = len(scenarios_by_type[synth])
        remaining = max(0, type_scenario_counts[synth] - already)
        if remaining == 0:
            continue
        if synth in (SYNTH_SINGLE_HOP_SPECIFIC, SYNTH_SINGLE_HOP_ABSTRACT):
            nodes = rng.choices(content_nodes, k=remaining)
            for node in nodes:
                persona = rng.choice(AUTOSAR_PERSONAS)
                scenarios_by_type[synth].append({
                    "synthesizer_name": synth,
                    "persona": persona,
                    "nodes": [node],
                })
        else:
            if not multihop_pairs:
                continue
            pairs = rng.choices(multihop_pairs, k=remaining)
            for pair in pairs:
                persona = rng.choice(AUTOSAR_PERSONAS)
                scenarios_by_type[synth].append({
                    "synthesizer_name": synth,
                    "persona": persona,
                    "nodes": list(pair),
                })

    print(f"\n  Final scenario counts per type (Bucket 1 + Bucket 2):")
    for k, v in scenarios_by_type.items():
        print(f"    {k:<28} {len(v)}")

    # Shuffle within each type
    for synth in scenarios_by_type:
        rng.shuffle(scenarios_by_type[synth])

    return scenarios_by_type, type_targets, multihop_pairs, content_nodes

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CONVERT A SCENARIO TO QUESTION-GEN CHAT MESSAGES
# ══════════════════════════════════════════════════════════════════════════════

def _node_text(node) -> str:
    return node.properties.get("page_content") or node.properties.get("text") or ""


def _source_docs(node) -> list[str]:
    meta = node.properties.get("document_metadata") or {}
    src = meta.get("filename") or meta.get("source")
    return [src] if src else []


def scenario_to_question_messages(scenario: dict[str, Any]) -> list[dict]:
    synth = scenario["synthesizer_name"]
    persona: Persona = scenario["persona"]
    nodes = scenario["nodes"]
    style = "abstract" if synth.endswith("abstract") else "specific"
    style_hint = STYLE_HINTS[style]

    if synth.startswith("single_hop"):
        # Single-hop: uses imported prompt with {context}
        return messages(
            QUESTION_GEN_SYSTEM,
            QUESTION_GEN_USER_SINGLEHOP.format(
                context=_node_text(nodes[0]),
                persona_role=persona.role_description,
                question_type=style,
                style_hint=style_hint,
            ),
        )
    else:
        # Multi-hop: uses QUESTION_GEN_USER_MULTIHOP (defined above) which
        # takes {context_1} and {context_2} separately and enforces bridging.
        # Guard: if somehow nodes has < 2 entries, fall back to single-hop
        # rather than raising an IndexError.
        if len(nodes) < 2:
            return messages(
                QUESTION_GEN_SYSTEM,
                QUESTION_GEN_USER_SINGLEHOP.format(
                    context=_node_text(nodes[0]),
                    persona_role=persona.role_description,
                    question_type=style,
                    style_hint=style_hint,
                ),
            )
        return messages(
            QUESTION_GEN_SYSTEM,
            QUESTION_GEN_USER_MULTIHOP.format(
                context_1=_node_text(nodes[0]),
                context_2=_node_text(nodes[1]),
                persona_role=persona.role_description,
                question_type=style,
                style_hint=style_hint,
            ),
        )


def scenario_with_question_to_answer_messages(
    scenario: dict[str, Any],
    question: str,
) -> list[dict]:
    nodes = scenario["nodes"]

    if len(nodes) == 1:
        # Single-hop: unchanged — uses imported ANSWER_GEN_USER with {context}
        return messages(
            ANSWER_GEN_SYSTEM,
            ANSWER_GEN_USER.format(
                context=_node_text(nodes[0]),
                question=question,
            ),
        )
    else:
        # FIX 1 — uses ANSWER_GEN_USER_MULTIHOP (defined above) which takes
        # {context_1} and {context_2} separately and requires both to be used.
        return messages(
            ANSWER_GEN_SYSTEM,
            ANSWER_GEN_USER_MULTIHOP.format(
                context_1=_node_text(nodes[0]),
                context_2=_node_text(nodes[1]),
                question=question,
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2b — HTTP CLIENT (vLLM server, same approach as build_kg.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_client(args: argparse.Namespace):
    from openai import OpenAI
    client = OpenAI(
        api_key="dummy",
        base_url=args.vllm_url,
    )
    return client


def check_vllm_connectivity(client, model: str) -> None:
    """
    QUALITY-12 FIX: Verify the vLLM server is alive and serving the expected
    model before generating thousands of scenarios. Exits with a clear message
    on failure instead of silently producing 0 candidates across all batches.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
            temperature=0.0,
        )
        reply = resp.choices[0].message.content.strip()
        print(f"  Server check: OK (reply='{reply}')")
    except Exception as e:
        print(f"\n  [FATAL] vLLM server is not responding: {type(e).__name__}: {e}")
        print(f"  Make sure the server is running and the model is loaded.")
        sys.exit(1)


def _extract_json_from_text(text: str) -> dict | None:
    """
    QUALITY-13 FIX: Robustly extract a JSON object from model output.

    Handles:
      - Clean JSON:            {"question": "..."}
      - Fenced with ```json:   ```json\n{"question": "..."}\n```
      - Fenced with ```:       ```\n{"question": "..."}\n```
      - Preamble text:         "Sure!\n```json\n{"question": "..."}\n```"
      - Inline JSON in prose:  finds the first {...} or [...] span

    Returns the parsed dict/list or None on failure.
    """
    import re
    text = text.strip()

    # Try direct parse first (most common for well-behaved models)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip code fences and retry
    # Match ```json ... ``` or ``` ... ``` with optional whitespace
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find the first JSON object {...} in the text
    obj_match = re.search(r"\{[\s\S]+\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _single_chat_json(
    client,
    conv: list[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    model: str,
    call_idx: int,
) -> tuple[int, dict | None, str | None]:
    """
    Single HTTP call to vLLM. Returns (call_idx, parsed_dict_or_None, error_str_or_None).
    Used by _batch_chat_json for concurrent execution.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=conv,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed + call_idx,   # ISSUE-11 FIX: unique seed per call
        )
        text = resp.choices[0].message.content.strip()
        parsed = _extract_json_from_text(text)
        if parsed is None:
            return call_idx, None, f"JSON parse failed; raw='{text[:120]}'"
        return call_idx, parsed, None
    except Exception as e:
        return call_idx, None, f"{type(e).__name__}: {e}"


def _batch_chat_json(
    client,
    conversations: list[list[dict]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    model: str,
    max_workers: int = 8,
) -> list[dict | None]:
    """
    Send all conversations to the vLLM server concurrently and collect JSON
    results.  Returns a list of parsed dicts in the same order as
    conversations; None for any call that failed.

    BUG-1 FIX: Exceptions are now logged (first error per batch) instead of
    being swallowed silently.
    BUG-2 FIX: Uses ThreadPoolExecutor for concurrent HTTP calls instead of
    sequential one-by-one execution. max_workers=8 is safe for vLLM since it
    has its own internal queue; tune up if your server has spare capacity.
    QUALITY-13 FIX: JSON extraction uses _extract_json_from_text() which
    handles fenced, preambled, and inline JSON robustly.
    ISSUE-11 FIX: Each call gets seed + call_idx so intra-batch calls are
    not identically seeded.
    """
    results: list[dict | None] = [None] * len(conversations)
    first_error: str | None = None
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _single_chat_json,
                client, conv, temperature, top_p, max_tokens, seed, model, idx,
            ): idx
            for idx, conv in enumerate(conversations)
        }
        for future in as_completed(futures):
            idx, parsed, err = future.result()
            results[idx] = parsed
            if err:
                error_count += 1
                if first_error is None:
                    first_error = err

    if first_error:
        print(f"  [WARN] {error_count}/{len(conversations)} calls failed "
              f"in this batch. First error: {first_error}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2c — FIX 2 + FIX 1 answer-guard: POST-GENERATION VALIDATORS
# ══════════════════════════════════════════════════════════════════════════════

def _is_restatement(question: str, answer: str) -> bool:
    """
    Detect answers that merely restate the question.

    Two signals:
      (a) Structural echo: the answer opens by mirroring the question's
          subject phrase (model just flips the interrogative into declarative).
          e.g. Q: "What is the purpose of X?" -> A: "The purpose of X is..."
      (b) Trivial pointer: answer is short (<120 chars) with no technical
          elaboration — just a pure lookup with no reasoning content.

    ISSUE-8 FIX: Expanded prefix list covers "how does/is/are", "why is/does",
    "which", "when is" — not just "what is/are/does".
    ISSUE-9 FIX: Expanded elaboration keywords cover common technical verbs
    so valid short answers are not incorrectly dropped.

    Returns True if the candidate should be dropped.
    """
    q = question.strip()
    a = answer.strip()

    # Signal (a): strip wh-word/phrase, check if first 40 chars of question
    # body appear verbatim in the first 160 chars of the answer.
    q_body = q.lower()
    # Ordered longest-first to avoid partial matches (e.g. "what is" vs "what is the")
    _PREFIXES = (
        "what is the ", "what are the ", "what does the ", "what is a ",
        "what are a ", "where can the ", "where is the ", "where does the ",
        "how does the ", "how is the ", "how are the ", "how do the ",
        "why is the ", "why does the ", "why are the ",
        "which ", "when is the ", "when does the ",
        "what role does ", "what is ", "what are ", "how does ", "how is ",
    )
    for prefix in _PREFIXES:
        if q_body.startswith(prefix):
            q_body = q_body[len(prefix):]
            break
    q_body = q_body.rstrip("?").strip()
    if q_body[:40] and q_body[:40] in a.lower()[:160]:
        return True

    # Signal (b): very short answer with no elaboration markers
    if len(a) < 120:
        has_elaboration = any(c.isdigit() for c in a) or \
                          any(kw in a.lower() for kw in (
                              # reasoning connectors
                              "because", "therefore", "thus", "hence",
                              "in order", "so that", "as a result",
                              # technical verbs common in AUTOSAR answers
                              "ensures", "requires", "defined", "specified",
                              "allows", "prevents", "means", "refers",
                              "used to", "contains", "implements", "provides",
                              "supports", "enables", "manages", "constrains",
                              "consists", "comprises", "represents", "performs",
                              "handles", "defines", "separates", "abstracts",
                              "encapsulates", "exposes", "describes", "maps",
                          ))
        if not has_elaboration:
            return True

    return False


def _multihop_uses_both_contexts(answer: str, ctx1: str, ctx2: str,
                                  min_coverage: float = 0.10) -> bool:
    """
    Verify the answer draws from both contexts via word-overlap.
    min_coverage=0.10 is permissive: only rejects answers that completely
    ignore one context, not ones that merely lean on one more.

    ISSUE-10 FIX: Common stop words are excluded from the overlap calculation.
    Without this, any answer trivially passes the 10% threshold via words like
    "the", "a", "is", "of" that appear in every answer and every context,
    letting single-context answers slip through.
    """
    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
        "with", "by", "from", "as", "it", "its", "this", "that", "these",
        "those", "which", "who", "what", "how", "when", "where", "why",
        "can", "may", "must", "shall", "will", "should", "would", "could",
        "have", "has", "had", "do", "does", "did", "if", "then", "than",
        "so", "also", "both", "each", "their", "they", "them", "there",
    })

    def _content_words(text: str) -> set[str]:
        return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 2}

    ans_words = _content_words(answer)
    if not ans_words:
        return False

    ctx1_words = _content_words(ctx1)
    ctx2_words = _content_words(ctx2)

    hit1 = len(ans_words & ctx1_words) / max(len(ctx1_words), 1)
    hit2 = len(ans_words & ctx2_words) / max(len(ctx2_words), 1)
    return hit1 >= min_coverage and hit2 >= min_coverage


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GENERATION LOOP (batched, checkpointed)
# ══════════════════════════════════════════════════════════════════════════════

def run_generation(
    scenarios_by_type: dict[str, list[dict[str, Any]]],
    type_targets: dict[str, int],
    multihop_pairs: list[tuple],
    content_nodes: list,
    client,
    args: argparse.Namespace,
    output_path: Path,
    already_done_by_type: dict[str, int],
    chunk_to_pdf: dict[str, str],
    rng: random.Random,
) -> None:
    """
    Generate candidates per type, with per-type counters and a retry/refill
    loop so that each type strictly hits its target count (subject to
    --max-retries-per-type hard cap).
    """
    print(f"\n  Per-type targets: {type_targets}")
    print(f"  Already done:     {already_done_by_type}")
    print(f"  Batch size: {args.batch_size}")

    # Per-type survivor counters (resume-aware)
    generated: dict[str, int] = {s: already_done_by_type.get(s, 0) for s in type_targets}
    retry_counts: dict[str, int] = {s: 0 for s in type_targets}

    single_hop_types = [SYNTH_SINGLE_HOP_SPECIFIC, SYNTH_SINGLE_HOP_ABSTRACT]
    multi_hop_types  = [SYNTH_MULTI_HOP_SPECIFIC,  SYNTH_MULTI_HOP_ABSTRACT]

    # Working scenario queues — start from where build_scenarios left off
    queues: dict[str, list] = {s: list(scenarios) for s, scenarios in scenarios_by_type.items()}

    global_counter = sum(generated.values())

    def _types_with_deficit() -> list[str]:
        return [s for s in type_targets if generated[s] < type_targets[s]]

    def _make_refill_scenarios(synth: str, n: int) -> list[dict[str, Any]]:
        """Generate fresh scenarios for a deficit type."""
        new_scenarios = []
        if synth in single_hop_types:
            nodes = rng.choices(content_nodes, k=n)
            for node in nodes:
                persona = rng.choice(AUTOSAR_PERSONAS)
                new_scenarios.append({
                    "synthesizer_name": synth,
                    "persona": persona,
                    "nodes": [node],
                })
        else:
            if not multihop_pairs:
                return []
            pairs = rng.choices(multihop_pairs, k=n)
            for pair in pairs:
                persona = rng.choice(AUTOSAR_PERSONAS)
                new_scenarios.append({
                    "synthesizer_name": synth,
                    "persona": persona,
                    "nodes": list(pair),
                })
        return new_scenarios

    def _run_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run one batch through question + answer generation. Returns survivors."""
        nonlocal global_counter

        # Phase 1: question generation
        q_convs = [scenario_to_question_messages(s) for s in batch]
        q_results = _batch_chat_json(
            client, q_convs,
            temperature=args.q_temperature,
            top_p=0.9,
            max_tokens=300,
            seed=args.seed + global_counter,
            model=args.generator_model,
        )

        with_q: list[tuple[dict[str, Any], str]] = []
        for s, q in zip(batch, q_results):
            if not q or "question" not in q or q["question"] is None:
                continue
            qtxt = str(q["question"]).strip()
            if not qtxt or len(qtxt) < 10:
                continue
            with_q.append((s, qtxt))

        if not with_q:
            return []

        # Phase 2: answer generation
        a_convs = [scenario_with_question_to_answer_messages(s, q) for s, q in with_q]
        a_results = _batch_chat_json(
            client, a_convs,
            temperature=args.a_temperature,
            top_p=1.0,
            max_tokens=512,
            seed=args.seed + global_counter,
            model=args.generator_model,
        )

        # Phase 3: quality filters + build candidates
        survivors: list[dict[str, Any]] = []
        for (s, q), a in zip(with_q, a_results):
            if not a or "answer" not in a:
                continue
            atxt = str(a["answer"]).strip()
            if not atxt or atxt.upper().startswith("NOT_ANSWERABLE") or len(atxt) < 20:
                continue
            if _is_restatement(q, atxt):
                continue
            if "multi_hop" in s["synthesizer_name"] and len(s["nodes"]) == 2:
                ctx1 = _node_text(s["nodes"][0])
                ctx2 = _node_text(s["nodes"][1])
                if not _multihop_uses_both_contexts(atxt, ctx1, ctx2):
                    continue

            contexts = [_node_text(n) for n in s["nodes"]]
            source_node_ids = [str(n.id) for n in s["nodes"]]
            source_docs: list[str] = []
            for n in s["nodes"]:
                pdf = chunk_to_pdf.get(str(n.id))
                if pdf:
                    source_docs.append(pdf)
                else:
                    source_docs.extend(_source_docs(n))
            source_docs = sorted(set(source_docs))

            cand = new_candidate(
                candidate_id=f"c_{global_counter + len(survivors):06d}",
                user_input=q,
                reference=atxt,
                reference_contexts=contexts,
                synthesizer_name=s["synthesizer_name"],
                persona_name=s["persona"].name,
                source_node_ids=source_node_ids,
                source_documents=source_docs,
                generator_model=args.generator_model,
                generator_config={
                    "q_temperature": args.q_temperature,
                    "a_temperature": args.a_temperature,
                    "seed": args.seed + global_counter,
                },
            )
            survivors.append(cand)

        return survivors

    # ── Main generation loop ─────────────────────────────────────────────────
    while _types_with_deficit():
        deficit_types = _types_with_deficit()

        # Build a mixed batch from all deficit types proportionally
        batch: list[dict[str, Any]] = []
        for synth in deficit_types:
            needed = type_targets[synth] - generated[synth]
            # Take up to batch_size // len(deficit_types) from each type's queue
            take = min(needed, args.batch_size // len(deficit_types), len(queues[synth]))
            batch.extend(queues[synth][:take])
            queues[synth] = queues[synth][take:]

        if not batch:
            # Queues exhausted — trigger refill for all deficit types
            for synth in deficit_types:
                if retry_counts[synth] >= args.max_retries_per_type:
                    print(f"  [WARN] {synth}: hit max retries ({args.max_retries_per_type}), "
                          f"deficit={type_targets[synth] - generated[synth]} — stopping retries.")
                    # Remove from future deficit checks to avoid infinite loop
                    type_targets[synth] = generated[synth]
                    continue
                needed = type_targets[synth] - generated[synth]
                refill_n = int(needed * (args.overgen_ratio_multihop
                                         if "multi_hop" in synth
                                         else args.overgen_ratio))
                new_scenarios = _make_refill_scenarios(synth, refill_n)
                queues[synth].extend(new_scenarios)
                retry_counts[synth] += 1
                print(f"  [REFILL] {synth}: retry {retry_counts[synth]}/{args.max_retries_per_type}, "
                      f"added {len(new_scenarios)} new scenarios, "
                      f"deficit={needed}")
            continue

        survivors = _run_batch(batch)

        # Credit survivors to their type
        new_by_type: dict[str, list] = defaultdict(list)
        for cand in survivors:
            new_by_type[cand["synthesizer_name"]].append(cand)

        to_persist: list[dict[str, Any]] = []
        for synth, cands in new_by_type.items():
            # Only keep up to the remaining target for this type
            remaining_needed = type_targets[synth] - generated[synth]
            accepted = cands[:remaining_needed]
            generated[synth] += len(accepted)
            global_counter += len(accepted)
            to_persist.extend(accepted)

        if to_persist:
            append_jsonl(to_persist, output_path)

        print(
            f"  Progress: "
            + ", ".join(f"{s.split('_hop_')[1][:4]}={generated[s]}/{type_targets[s]}"
                        for s in type_targets)
            + f"  |  total={global_counter}"
        )

    print(f"\n  Generation complete.")
    print(f"  Final counts per type:")
    for synth, count in generated.items():
        print(f"    {synth:<28} {count}/{type_targets[synth]}")
    print(f"  Total persisted: {global_counter}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    stage_dir = output_dir / "stage_a_generation"
    stage_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = stage_dir / "candidates.jsonl"
    config_path = stage_dir / "generation_config.json"

    # Build distribution dict from CLI args
    distribution = {
        SYNTH_SINGLE_HOP_SPECIFIC: args.dist_single_hop_specific,
        SYNTH_SINGLE_HOP_ABSTRACT: args.dist_single_hop_abstract,
        SYNTH_MULTI_HOP_SPECIFIC:  args.dist_multi_hop_specific,
        SYNTH_MULTI_HOP_ABSTRACT:  args.dist_multi_hop_abstract,
    }

    print("=" * 70)
    print(" Stage A :: Generate Candidates")
    print("=" * 70)
    print(f" Target final size        : {args.target}")
    print(f" Overgen ratio (single)   : {args.overgen_ratio}")
    print(f" Overgen ratio (multi)    : {args.overgen_ratio_multihop}")
    print(f" Max retries per type     : {args.max_retries_per_type}")
    print(f" Distribution             :")
    for k, v in distribution.items():
        print(f"   {k:<28} {v:.0%}")
    print(f" Output                   : {candidates_path}")
    print(f" Generator                : {args.generator_model}")
    print(f" vLLM URL                 : {args.vllm_url}")
    print(f" Min context chars        : {args.min_context_chars}")
    print("=" * 70)

    # Resume: count already-done per type from existing candidates.jsonl
    already_done_by_type: dict[str, int] = {s: 0 for s in distribution}
    total_already_done = 0
    if candidates_path.exists():
        with open(candidates_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    synth = row.get("synthesizer_name", "")
                    if synth in already_done_by_type:
                        already_done_by_type[synth] += 1
                        total_already_done += 1
                except json.JSONDecodeError:
                    continue
        if total_already_done > 0:
            print(f"\n  Resuming: {total_already_done} candidates already persisted")
            print(f"  Per-type: {already_done_by_type}")

    # Check if already complete
    type_targets = {s: int(round(args.target * frac)) for s, frac in distribution.items()}
    if all(already_done_by_type[s] >= type_targets[s] for s in type_targets):
        print(f"\n  All type targets already met. Done.")
        return

    # Load KG
    print("\n[1/4] Loading KG ...")
    kg = load_kg(Path(args.kg_file))
    print(f"  Loaded: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")

    # Build scenarios
    print("\n[2/4] Building scenarios ...")
    rng = random.Random(args.seed)
    scenarios_by_type, type_targets, multihop_pairs, content_nodes = build_scenarios(
        kg,
        target=args.target,
        distribution=distribution,
        overgen_ratio_singlehop=args.overgen_ratio,
        overgen_ratio_multihop=args.overgen_ratio_multihop,
        rng=rng,
        min_context_chars=args.min_context_chars,
        min_per_pdf=args.min_per_pdf,
    )

    # Connect to vLLM
    print("\n[3/4] Connecting to vLLM server ...")
    client = build_client(args)
    check_vllm_connectivity(client, args.generator_model)

    # Save config
    atomic_write_json(
        {
            "target":                  args.target,
            "overgen_ratio":           args.overgen_ratio,
            "overgen_ratio_multihop":  args.overgen_ratio_multihop,
            "distribution":            distribution,
            "type_targets":            type_targets,
            "max_retries_per_type":    args.max_retries_per_type,
            "personas":                [p.name for p in AUTOSAR_PERSONAS],
            "generator_model":         args.generator_model,
            "vllm_url":                args.vllm_url,
            "q_temperature":           args.q_temperature,
            "a_temperature":           args.a_temperature,
            "seed":                    args.seed,
            "batch_size":              args.batch_size,
            "min_context_chars":       args.min_context_chars,
            "min_per_pdf":             args.min_per_pdf,
        },
        config_path,
    )

    # Generate
    print("\n[4/4] Generating ...")
    chunk_to_pdf = build_chunk_to_pdf_map(kg)
    run_generation(
        scenarios_by_type=scenarios_by_type,
        type_targets=type_targets,
        multihop_pairs=multihop_pairs,
        content_nodes=content_nodes,
        client=client,
        args=args,
        output_path=candidates_path,
        already_done_by_type=already_done_by_type,
        chunk_to_pdf=chunk_to_pdf,
        rng=rng,
    )

    print("\n" + "=" * 70)
    print(" Stage A complete.")
    print(f" Next: python validate_candidates.py --output-dir {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


    