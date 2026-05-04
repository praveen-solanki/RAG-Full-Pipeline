# """flat_to_hierarchy.py
# ====================
# Converts a flat-list structure JSON (_hierarchy.json) into the nested-tree
# format expected by run_rag_v3.py / run_rag_v4.py (_structure.json).

# Strategy — two passes:

#   Pass 1  HIERARCHY  (LLM)
#     The full flat node list (node_id + title + page range) is sent to the LLM
#     in a single prompt.  The LLM is asked to:
#       • infer parent–child relationships from section numbering and title patterns
#       • attach spec-item tags ([AP_TPS_...], [constr_...]) and table entries
#         as children of the nearest logical section
#       • skip/omit nodes it cannot meaningfully place (auto-generated change-log
#         rows, noise entries, etc.)
#       • return a clean nested JSON tree (node_ids only — page numbers are
#         merged in from the original flat list afterward so LLM errors don't
#         corrupt page ranges)

#   Pass 2  SUMMARIES  (LLM, batched)
#     For every node in the resulting tree the LLM writes a one-sentence
#     summary based on the node's title and its immediate children's titles.
#     Nodes whose content the LLM cannot meaningfully describe get summary="".

# Input modes
# -----------
#   Single file:
#     { "doc_name": "Foo.pdf", "total_pages": 75, "source": "...",
#       "hierarchy": [ ... ] }

#   Directory mode:
#     A directory containing one or more *_hierarchy.json files.
#     Each file is processed independently and the corresponding *_structure.json
#     file is written next to it unless --output-dir is provided.

# Output  (_structure.json)
#   { "doc_description": "Foo.pdf",
#     "structure": [
#       { "title": "...", "node_id": "0001",
#         "start_index": 2, "end_index": 3,
#         "summary": "...",
#         "nodes": [ ... ] },
#       ...
#     ] }

# Usage
# -----
#   Single file:
#     python flat_to_hierarchy.py --input PATH [--output PATH] [--no-summary]

#   Directory:
#     python flat_to_hierarchy.py --input-dir DIR [--output-dir DIR] [--no-summary]
# """

# import argparse
# import json
# import os
# import re
# import sys
# import time
# from pathlib import Path
# from openai import OpenAI

# # =============================================================================
# # CONFIG
# # =============================================================================

# # OLLAMA_BASE_URL = "http://localhost:8011/v1"
# # OLLAMA_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"

# OLLAMA_BASE_URL = "http://localhost:11434/v1"
# OLLAMA_MODEL = "qwen2.5:7b"

# SUMMARY_BATCH_SIZE = 64      # nodes per summary batch
# BATCH_SLEEP = 0.1            # seconds between batches
# MAX_RETRIES = 1
# RETRY_BACKOFF = 2.0          # seconds, doubles each retry

# # =============================================================================
# # CLIENT
# # =============================================================================

# client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)


# # =============================================================================
# # RETRY WRAPPER
# # =============================================================================

# def call_with_retry(fn, *args, **kwargs):
#     last_exc = None
#     for attempt in range(1, MAX_RETRIES + 2):
#         try:
#             return fn(*args, **kwargs)
#         except Exception as e:
#             last_exc = e
#             if attempt <= MAX_RETRIES:
#                 wait = RETRY_BACKOFF * (2 ** (attempt - 1))
#                 print(f"  ↺ attempt {attempt} failed ({e.__class__.__name__}: {e})"
#                       f" — retrying in {wait:.0f}s ...")
#                 time.sleep(wait)
#             else:
#                 print(f"  ✗ all {MAX_RETRIES + 1} attempts failed: {e}")
#     raise last_exc


# # =============================================================================
# # PASS 1 — LLM HIERARCHY BUILDER
# # =============================================================================

# HIERARCHY_SYSTEM_PROMPT = """\
# You are a document structure expert specialising in technical standards documents
# (AUTOSAR, ISO, IEEE, ETSI and similar).

# Your task: convert a FLAT list of document sections into a NESTED hierarchy that
# mirrors the document's logical table of contents.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RULES  (apply ALL rules; priority = order listed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# RULE 1 — NUMBERED SECTIONS define the skeleton.
#   Numeric prefixes like "1", "2.3", "4.1.2" determine depth.
#   A section is a child of the nearest shorter matching prefix above it.
#   Example: "2.4.1 Foo" → child of "2.4 Bar" → child of "2 Baz".

# RULE 2 — APPENDIX SECTIONS follow the same logic with letter prefixes.
#   "A Foo", "A.1 Bar", "B Baz", "C.1.2 Qux" …
#   A single capital letter (A, B, C …) is depth-1 appendix root.
#   "A.1", "A.2" → children of "A"; "C.1.1" → child of "C.1", etc.

# RULE 3 — SPEC-ITEM TAGS  ([AP_TPS_...], [constr_...])  are NOT automatically leaves.
#   Use your knowledge of AUTOSAR and technical standards to judge whether a
#   spec-item tag is a real, self-contained specification entry that belongs in
#   the document structure — if so, include it as a child of the nearest numbered/
#   appendix section above it, and allow other nodes to nest under it if they
#   logically belong there.  If a tag is a noise artifact, duplicate, or
#   auto-generated entry with no standalone retrieval value, skip it per RULE 6.

# RULE 4 — TABLE ENTRIES  ("Table N.M: …")  are NOT automatically leaves.
#   Use your judgment: if the table entry is a meaningful schema or data table
#   that readers would look up independently, include it as a child of the nearest
#   numbered section above it, and nest further nodes under it if appropriate.
#   If it is a spurious or unresolvable label with no clear parent context,
#   skip it per RULE 6.

# RULE 5 — FRONT-MATTER nodes that have no numeric/letter prefix and no
#   obvious parent ("Disclaimer", "Table of Contents", "References",
#   "Preface", "Scope" …) become top-level roots in their original order.

# RULE 6 — SKIP nodes you cannot meaningfully classify.
#   If a node matches none of the rules above AND its title gives no semantic
#   signal useful for retrieval (e.g. an auto-generated change-log row whose
#   parent section cannot be determined, or a bare label with no context),
#   OMIT it entirely from the output.
#   ▸ Do NOT invent parents.
#   ▸ Do NOT force-fit ambiguous noise into the tree.
#   ▸ Skipped nodes must NOT appear anywhere in the output JSON.

# RULE 7 — Every node_id you DO include must appear exactly ONCE.
#   No duplicates.  Do not rename or alter node_id strings.

# RULE 8 — Do NOT generate summaries here.
#   If the input node already carries a non-empty "summary" field, copy it
#   verbatim into the output node.  Otherwise set "summary": "".

# RULE 9 — Page range integrity.
#   Copy start_index / end_index verbatim from the flat input for each node.
#   Then ensure every parent's end_index >= the maximum end_index of all its
#   descendants (expand bottom-up after placing all children).

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT FORMAT — return ONLY valid JSON, no markdown fence, no preamble.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# {
#   "structure": [
#     {
#       "node_id":     "<string, copied from input>",
#       "title":       "<string, copied from input>",
#       "start_index": <int>,
#       "end_index":   <int>,
#       "summary":     "",
#       "nodes":       [ /* same schema, recursively */ ]
#     }
#   ]
# }
# """

# HIERARCHY_USER_TEMPLATE = """\
# Convert the flat node list below into a nested hierarchy following all rules \
# in the system prompt.

# Document : {doc_name}
# Pages    : {total_pages}

# Flat node list  (node_id | title | start_index | end_index)
# ------------------------------------------------------------
# {node_table}
# ------------------------------------------------------------

# Return ONLY the JSON object.  No explanation, no markdown.
# """


# def build_node_table(flat_nodes: list[dict]) -> str:
#     """Compact pipe-separated table for the LLM prompt."""
#     return "\n".join(
#         f"{n['node_id']} | {n['title']} | {n['start_index']} | {n['end_index']}"
#         for n in flat_nodes
#     )


# def llm_build_hierarchy(source_data: dict) -> list[dict]:
#     """
#     Pass 1: send the full flat list to the LLM and get back a nested tree.
#     Returns the list of root node dicts (the 'structure' array).
#     """
#     flat_nodes = source_data.get("hierarchy", [])
#     doc_name = source_data.get("doc_name", "unknown")
#     total_pages = source_data.get("total_pages", "unknown")

#     user_msg = HIERARCHY_USER_TEMPLATE.format(
#         doc_name=doc_name,
#         total_pages=total_pages,
#         node_table=build_node_table(flat_nodes),
#     )

#     print(f"  Sending {len(flat_nodes)} nodes to LLM for hierarchy construction …")

#     response = client.chat.completions.create(
#         model=OLLAMA_MODEL,
#         messages=[
#             {"role": "system", "content": HIERARCHY_SYSTEM_PROMPT},
#             {"role": "user", "content": user_msg},
#         ],
#         response_format={"type": "json_object"},
#         temperature=0,
#     )

#     raw = response.choices[0].message.content.strip()

#     # Strip accidental markdown fences
#     raw = re.sub(r"^```(?:json)?\s*", "", raw)
#     raw = re.sub(r"\s*```$", "", raw)

#     try:
#         parsed = json.loads(raw)
#     except json.JSONDecodeError as e:
#         raise ValueError(
#             f"LLM returned invalid JSON for hierarchy: {e}\n\nRaw (first 600 chars):\n{raw[:600]}"
#         )

#     # Tolerate LLM returning bare list instead of {"structure": [...]}
#     structure = parsed.get("structure", parsed)
#     if not isinstance(structure, list):
#         raise ValueError(
#             f"Expected a list under 'structure', got {type(structure).__name__}"
#         )

#     return structure


# # =============================================================================
# # POST-PROCESS — authoritative page data + validation
# # =============================================================================

# def build_flat_index(flat_nodes: list[dict]) -> dict[str, dict]:
#     """node_id → original flat node dict for O(1) lookup."""
#     return {n["node_id"]: n for n in flat_nodes}


# def merge_page_data(tree_nodes: list[dict], flat_index: dict[str, dict]) -> None:
#     """
#     Overwrite start_index / end_index on every node in the LLM-generated
#     tree with the authoritative values from the original flat list, then
#     re-expand parent end_index values bottom-up.

#     This guarantees page ranges are always correct even if the LLM made
#     arithmetic errors or hallucinated page numbers.
#     Mutates tree_nodes in place.
#     """

#     def _fix(node: dict) -> int:
#         nid = node["node_id"]
#         if nid in flat_index:
#             orig = flat_index[nid]
#             node["start_index"] = orig["start_index"]
#             node["end_index"] = orig["end_index"]
#         else:
#             print(f"  [WARN] node_id '{nid}' not in original flat list — keeping LLM values")

#         child_max = node["end_index"]
#         for child in node.get("nodes", []):
#             child_max = max(child_max, _fix(child))

#         node["end_index"] = child_max
#         return node["end_index"]

#     for root in tree_nodes:
#         _fix(root)


# def validate_tree(tree_nodes: list[dict], flat_index: dict[str, dict]) -> None:
#     """Report which nodes were included, skipped (RULE 6), or duplicated."""
#     seen: set[str] = set()

#     def _collect(nodes: list[dict]) -> None:
#         for n in nodes:
#             nid = n["node_id"]
#             if nid in seen:
#                 print(f"  [WARN] duplicate node_id '{nid}' in LLM output")
#             seen.add(nid)
#             _collect(n.get("nodes", []))

#     _collect(tree_nodes)

#     skipped = set(flat_index.keys()) - seen
#     if skipped:
#         print(f"  [INFO] {len(skipped)} node(s) omitted by LLM (RULE 6 — noise / unclassifiable):")
#         for nid in sorted(skipped):
#             print(f"         {nid} | {flat_index[nid]['title']}")

#     covered = seen & set(flat_index.keys())
#     print(f"  [OK]  {len(covered)}/{len(flat_index)} original nodes placed in tree")


# # =============================================================================
# # PASS 2 — LLM SUMMARY GENERATION
# # =============================================================================

# SUMMARY_SYSTEM_PROMPT = """\
# You are a technical documentation assistant for AUTOSAR and automotive software standards.

# For each section entry below write a SINGLE concise sentence (maximum 25 words)
# describing what the section covers.

# Base summaries ONLY on the section title and its children's titles.
# Never hallucinate technical details not clearly implied by the title.

# Special cases:
#   • Spec-item tags  ([AP_TPS_APMC_xxxxx], [constr_xxxxx]) — describe what the
#     rule or constraint defines; do NOT repeat the tag ID verbatim in the summary.
#   • Table entries  ("Table N.M: ClassName") — describe what the table documents.
#   • Change-log sections  ("Added/Changed/Deleted Specification Items in Rxx-xx",
#     "Traceable item history …") — set summary to "" (empty); these are
#     administrative records with no semantic retrieval value.
#   • Any section whose content you genuinely cannot infer from its title —
#     set summary to "" rather than guessing.
#   • If a node already has a non-empty "existing_summary" field in the input,
#     copy it verbatim as the summary for that node_id — do not regenerate it.

# Reply ONLY with a JSON object mapping node_id → summary string.
# No markdown, no preamble, no extra keys.
# Example:  {"0001": "Covers disclaimer and legal notices.", "0002": ""}
# """


# def summarise_batch(batch: list[dict]) -> dict[str, str]:
#     items = []
#     for node in batch:
#         entry: dict = {"node_id": node["node_id"], "title": node["title"]}
#         child_titles = [c["title"] for c in node.get("nodes", [])]
#         if child_titles:
#             entry["children"] = child_titles
#         items.append(entry)

#     user_msg = "Sections to summarise:\n" + json.dumps(items, indent=2)

#     response = client.chat.completions.create(
#         model=OLLAMA_MODEL,
#         messages=[
#             {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
#             {"role": "user", "content": user_msg},
#         ],
#         response_format={"type": "json_object"},
#         temperature=0,
#     )

#     raw = response.choices[0].message.content.strip()
#     raw = re.sub(r"^```(?:json)?\s*", "", raw)
#     raw = re.sub(r"\s*```$", "", raw)

#     try:
#         result = json.loads(raw)
#         return {str(k): str(v) for k, v in result.items()}
#     except json.JSONDecodeError as e:
#         print(f"  [WARN] JSON parse failed for summary batch: {e}")
#         return {}


# def flatten_tree(roots: list[dict]) -> list[dict]:
#     """BFS — collect every node in the tree in top-down order."""
#     all_nodes: list[dict] = []
#     queue = list(roots)
#     while queue:
#         node = queue.pop(0)
#         all_nodes.append(node)
#         queue.extend(node.get("nodes", []))
#     return all_nodes


# def add_summaries(roots: list[dict], skip: bool = False) -> None:
#     """Pass 2: fill 'summary' on every node via batched LLM calls.
#     Nodes that already carry a non-empty summary are left untouched."""
#     all_nodes = flatten_tree(roots)
#     total = len(all_nodes)

#     if skip:
#         print(f"  [summary] skipped (--no-summary) — setting empty string on nodes without existing summary")
#         for node in all_nodes:
#             if not node.get("summary"):
#                 node["summary"] = ""
#         return

#     needs_summary = [n for n in all_nodes if not n.get("summary")]
#     already_done = total - len(needs_summary)
#     if already_done:
#         print(f"  [summary] {already_done}/{total} nodes already have summaries — skipping them")

#     if not needs_summary:
#         print(f"  [summary] all {total} nodes already summarised — nothing to do")
#         return

#     print(f"  [summary] generating summaries for {len(needs_summary)} nodes "
#           f"in batches of {SUMMARY_BATCH_SIZE} …")

#     summaries: dict[str, str] = {}
#     total_batches = (len(needs_summary) + SUMMARY_BATCH_SIZE - 1) // SUMMARY_BATCH_SIZE

#     for batch_num, batch_start in enumerate(range(0, len(needs_summary), SUMMARY_BATCH_SIZE), 1):
#         batch = needs_summary[batch_start: batch_start + SUMMARY_BATCH_SIZE]
#         end = min(batch_start + SUMMARY_BATCH_SIZE, len(needs_summary))
#         print(f"    batch {batch_num}/{total_batches} "
#               f"(nodes {batch_start + 1}–{end}) … ", end="", flush=True)
#         try:
#             result = call_with_retry(summarise_batch, batch)
#             summaries.update(result)
#             filled = sum(1 for v in result.values() if v)
#             print(f"OK  ({filled}/{len(batch)} non-empty)")
#         except Exception as e:
#             print(f"FAILED — {e}  (empty summaries used for this batch)")

#         if batch_start + SUMMARY_BATCH_SIZE < len(needs_summary):
#             time.sleep(BATCH_SLEEP)

#     for node in needs_summary:
#         node["summary"] = summaries.get(node["node_id"], "")

#     total_filled = sum(1 for n in all_nodes if n.get("summary"))
#     print(f"  [summary] {total_filled}/{total} nodes have non-empty summaries")


# # =============================================================================
# # STATS
# # =============================================================================

# def print_stats(roots: list[dict]) -> None:
#     all_nodes = flatten_tree(roots)
#     depth_counts: dict[int, int] = {}

#     def _depth(node: dict, d: int) -> None:
#         depth_counts[d] = depth_counts.get(d, 0) + 1
#         for child in node.get("nodes", []):
#             _depth(child, d + 1)

#     for root in roots:
#         _depth(root, 0)

#     print(f"  Tree stats:")
#     print(f"    Total nodes : {len(all_nodes)}")
#     print(f"    Root nodes  : {len(roots)}")
#     for d in sorted(depth_counts):
#         print(f"    Depth {d}      : {depth_counts[d]} node(s)")


# # =============================================================================
# # FILE/DIR HELPERS
# # =============================================================================

# def derive_output_path(input_path: str, output_path: str | None = None) -> str:
#     """Derive the output path for one input file."""
#     if output_path:
#         return os.path.abspath(output_path)

#     base = re.sub(r"_structure\.json$", "", input_path)
#     base = re.sub(r"\.json$", "", base)
#     return base + "_structure.json"


# def process_one_file(input_path: str, output_path: str | None, no_summary: bool) -> int:
#     """Process a single hierarchy JSON file. Returns 0 on success, 1 on failure."""
#     input_path = os.path.abspath(input_path)
#     if not os.path.exists(input_path):
#         print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
#         return 1

#     final_output = derive_output_path(input_path, output_path)

#     print("=" * 64)
#     print(f"  Input  : {input_path}")
#     print(f"  Output : {final_output}")
#     print(f"  Model  : {OLLAMA_MODEL}  @  {OLLAMA_BASE_URL}")
#     print(f"  Pass 2 : {'DISABLED (--no-summary)' if no_summary else 'ENABLED'}")
#     print("=" * 64)

#     print("\n[1/4] Loading flat hierarchy …")
#     with open(input_path, "r", encoding="utf-8") as f:
#         source_data = json.load(f)

#     flat_nodes = source_data.get("hierarchy", [])
#     if not flat_nodes:
#         print("[ERROR] 'hierarchy' key is empty or missing.", file=sys.stderr)
#         return 1

#     flat_index = build_flat_index(flat_nodes)
#     print(f"  {len(flat_nodes)} flat nodes loaded from '{source_data.get('doc_name', '?')}'")

#     print("\n[2/4] Pass 1 — LLM hierarchy construction …")
#     structure = call_with_retry(llm_build_hierarchy, source_data)
#     print(f"  LLM returned {len(structure)} root node(s)")

#     merge_page_data(structure, flat_index)
#     validate_tree(structure, flat_index)
#     print_stats(structure)

#     print("\n[3/4] Pass 2 — LLM summary generation …")
#     add_summaries(structure, skip=no_summary)

#     print("\n[4/4] Writing output …")
#     output = {
#         "doc_description": source_data.get("doc_name", ""),
#         "total_pages": source_data.get("total_pages"),
#         "source_format": source_data.get("source", ""),
#         "structure": structure,
#     }

#     os.makedirs(os.path.dirname(final_output) or ".", exist_ok=True)
#     with open(final_output, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)

#     print(f"  Saved → {final_output}")
#     print(f"\n✅ Done.  '{os.path.basename(final_output)}' is ready for run_rag_v4.py.")
#     return 0


# def find_input_files(input_dir: str) -> list[str]:
#     """Return *_hierarchy.json files from a directory, sorted by name."""
#     p = Path(input_dir)
#     if not p.exists():
#         raise FileNotFoundError(f"Input directory not found: {input_dir}")
#     if not p.is_dir():
#         raise NotADirectoryError(f"Not a directory: {input_dir}")

#     files = sorted(str(x) for x in p.iterdir() if x.is_file() and x.name.endswith("_structure.json"))
#     return files


# # =============================================================================
# # MAIN
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Convert flat _hierarchy.json file(s) into nested _structure.json using LLM hierarchy inference + summaries (Ollama)."
#     )

#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument(
#         "--input", "-i",
#         help="Path to one flat _hierarchy.json file",
#     )
#     group.add_argument(
#         "--input-dir",
#         help="Path to a directory containing one or more *_hierarchy.json files",
#     )

#     parser.add_argument(
#         "--output", "-o",
#         default=None,
#         help="Output path for single-file mode only. If omitted, *_structure.json is derived automatically.",
#     )
#     parser.add_argument(
#         "--output-dir",
#         default=None,
#         help="Output directory for directory mode. If omitted, each output is written next to its input file.",
#     )
#     parser.add_argument(
#         "--no-summary",
#         action="store_true",
#         help="Skip Pass 2 summary generation (summaries will be empty strings)",
#     )
#     args = parser.parse_args()

#     # Single file mode
#     if args.input:
#         return process_one_file(args.input, args.output, args.no_summary)

#     # Directory mode
#     try:
#         input_files = find_input_files(args.input_dir)
#     except Exception as e:
#         print(f"[ERROR] {e}", file=sys.stderr)
#         return 1

#     if not input_files:
#         print(f"[ERROR] No *_hierarchy.json files found in directory: {args.input_dir}", file=sys.stderr)
#         return 1

#     output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)

#     print(f"[INFO] Found {len(input_files)} input file(s) in {os.path.abspath(args.input_dir)}")

#     overall_rc = 0
#     for idx, input_file in enumerate(input_files, 1):
#         print(f"\n--- Processing {idx}/{len(input_files)}: {os.path.basename(input_file)} ---")

#         if output_dir:
#             base = os.path.basename(input_file)
#             base = re.sub(r"_structure\.json$", "", base)
#             base = re.sub(r"\.json$", "", base)
#             out_path = os.path.join(output_dir, base + "_structure.json")
#         else:
#             out_path = None

#         rc = process_one_file(input_file, out_path, args.no_summary)
#         if rc != 0:
#             overall_rc = rc

#     return overall_rc


# if __name__ == "__main__":
#     raise SystemExit(main())





"""flat_to_hierarchy.py
====================
Converts a flat-list structure JSON (_hierarchy.json) into the nested-tree
format expected by run_rag_v3.py / run_rag_v4.py (_structure.json).

Strategy — two passes:

  Pass 1  HIERARCHY  (LLM)
    The full flat node list (node_id + title + page range) is sent to the LLM
    in a single prompt.  The LLM is asked to:
      • infer parent–child relationships from section numbering and title patterns
      • attach spec-item tags ([AP_TPS_...], [constr_...]) and table entries
        as children of the nearest logical section
      • skip/omit nodes it cannot meaningfully place (auto-generated change-log
        rows, noise entries, etc.)
      • return a clean nested JSON tree (node_ids only — page numbers are
        merged in from the original flat list afterward so LLM errors don't
        corrupt page ranges)

  Pass 2  SUMMARIES  (LLM, batched)
    For every node in the resulting tree the LLM writes a one-sentence
    summary based on the node's title and its immediate children's titles.
    Nodes whose content the LLM cannot meaningfully describe get summary="".

Input modes
-----------
  Single file:
    { "doc_name": "Foo.pdf", "total_pages": 75, "source": "...",
      "hierarchy": [ ... ] }

  Directory mode:
    A directory containing one or more *_hierarchy.json files.
    Each file is processed independently and the corresponding *_structure.json
    file is written next to it unless --output-dir is provided.

Output  (_structure.json)
  { "doc_description": "Foo.pdf",
    "structure": [
      { "title": "...", "node_id": "0001",
        "start_index": 2, "end_index": 3,
        "summary": "...",
        "nodes": [ ... ] },
      ...
    ] }

Usage
-----
  Single file:
    python flat_to_hierarchy.py --input PATH [--output PATH] [--no-summary]

  Directory:
    python flat_to_hierarchy.py --input-dir DIR [--output-dir DIR] [--no-summary]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

# =============================================================================
# CONFIG
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:8011/v1"
OLLAMA_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

SUMMARY_BATCH_SIZE = 256      # nodes per summary batch
HIERARCHY_CHUNK_SIZE = 250   # max nodes per Pass-1 LLM call (avoids output truncation)
BATCH_SLEEP = 0.1            # seconds between batches
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0          # seconds, doubles each retry

# =============================================================================
# CLIENT
# =============================================================================

client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)


# =============================================================================
# RETRY WRAPPER
# =============================================================================

def call_with_retry(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt <= MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"  ↺ attempt {attempt} failed ({e.__class__.__name__}: {e})"
                      f" — retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                print(f"  ✗ all {MAX_RETRIES + 1} attempts failed: {e}")
    raise last_exc


# =============================================================================
# PASS 1 — LLM HIERARCHY BUILDER
# =============================================================================

HIERARCHY_SYSTEM_PROMPT = """\
You are a document structure expert specialising in technical standards documents
(AUTOSAR, ISO, IEEE, ETSI and similar).

Your task: convert a FLAT list of document sections into a NESTED hierarchy that
mirrors the document's logical table of contents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES  (apply ALL rules; priority = order listed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — NUMBERED SECTIONS define the skeleton.
  Numeric prefixes like "1", "2.3", "4.1.2" determine depth.
  A section is a child of the nearest shorter matching prefix above it.
  Example: "2.4.1 Foo" → child of "2.4 Bar" → child of "2 Baz".

RULE 2 — APPENDIX SECTIONS follow the same logic with letter prefixes.
  "A Foo", "A.1 Bar", "B Baz", "C.1.2 Qux" …
  A single capital letter (A, B, C …) is depth-1 appendix root.
  "A.1", "A.2" → children of "A"; "C.1.1" → child of "C.1", etc.

RULE 3 — SPEC-ITEM TAGS  ([AP_TPS_...], [constr_...])  are NOT automatically leaves.
  Use your knowledge of AUTOSAR and technical standards to judge whether a
  spec-item tag is a real, self-contained specification entry that belongs in
  the document structure — if so, include it as a child of the nearest numbered/
  appendix section above it, and allow other nodes to nest under it if they
  logically belong there.  If a tag is a noise artifact, duplicate, or
  auto-generated entry with no standalone retrieval value, skip it per RULE 6.

RULE 4 — TABLE ENTRIES  ("Table N.M: …")  are NOT automatically leaves.
  Use your judgment: if the table entry is a meaningful schema or data table
  that readers would look up independently, include it as a child of the nearest
  numbered section above it, and nest further nodes under it if appropriate.
  If it is a spurious or unresolvable label with no clear parent context,
  skip it per RULE 6.

RULE 5 — FRONT-MATTER nodes that have no numeric/letter prefix and no
  obvious parent ("Disclaimer", "Table of Contents", "References",
  "Preface", "Scope" …) become top-level roots in their original order.

RULE 6 — SKIP nodes you cannot meaningfully classify.
  If a node matches none of the rules above AND its title gives no semantic
  signal useful for retrieval (e.g. an auto-generated change-log row whose
  parent section cannot be determined, or a bare label with no context),
  OMIT it entirely from the output.
  ▸ Do NOT invent parents.
  ▸ Do NOT force-fit ambiguous noise into the tree.
  ▸ Skipped nodes must NOT appear anywhere in the output JSON.

RULE 7 — Every node_id you DO include must appear exactly ONCE.
  No duplicates.  Do not rename or alter node_id strings.

RULE 8 — Do NOT generate summaries here.
  If the input node already carries a non-empty "summary" field, copy it
  verbatim into the output node.  Otherwise set "summary": "".

RULE 9 — Page range integrity.
  Copy start_index / end_index verbatim from the flat input for each node.
  Then ensure every parent's end_index >= the maximum end_index of all its
  descendants (expand bottom-up after placing all children).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — return ONLY valid JSON, no markdown fence, no preamble.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "structure": [
    {
      "node_id":     "<string, copied from input>",
      "title":       "<string, copied from input>",
      "start_index": <int>,
      "end_index":   <int>,
      "summary":     "",
      "nodes":       [ /* same schema, recursively */ ]
    }
  ]
}
"""

HIERARCHY_USER_TEMPLATE = """\
Convert the flat node list below into a nested hierarchy following all rules \
in the system prompt.

Document : {doc_name}
Pages    : {total_pages}

Flat node list  (node_id | title | start_index | end_index)
------------------------------------------------------------
{node_table}
------------------------------------------------------------

Return ONLY the JSON object.  No explanation, no markdown.
"""


def build_node_table(flat_nodes: list[dict]) -> str:
    """Compact pipe-separated table for the LLM prompt."""
    return "\n".join(
        f"{n['node_id']} | {n['title']} | {n['start_index']} | {n['end_index']}"
        for n in flat_nodes
    )


def _depth1_prefix(title: str) -> str:
    """
    Return the depth-1 section prefix from a title, i.e. the leading single
    number or single capital letter.
      '2 Foo'       → '2'
      '2.4.1 Foo'   → '2'
      'A Foo'        → 'A'
      'A.1.2 Bar'    → 'A'
      'Disclaimer'   → ''   (front-matter, no prefix)
    """
    m = re.match(r"^([A-Z]|\d+)(?:[.\s]|$)", title)
    return m.group(1) if m else ""


def split_into_section_chunks(flat_nodes: list[dict], max_chunk: int) -> list[list[dict]]:
    """
    Split flat_nodes into chunks that respect depth-1 section boundaries so
    that a parent section and all its subordinate nodes always land in the
    same chunk.

    Strategy:
      1. Identify every depth-1 section boundary by tracking when the leading
         top-level prefix changes (e.g. '1' → '2', or 'A' → 'B').
      2. Group consecutive whole top-level sections greedily: keep adding
         sections to the current chunk until the next section would push it
         over max_chunk.  Then start a new chunk.
      3. Front-matter nodes (no depth-1 prefix) before the first numbered
         section are prepended to the first chunk.
      4. If a single top-level section alone exceeds max_chunk, it is sent
         as its own oversized chunk (unavoidable — keeps it self-contained).
    """
    if not flat_nodes:
        return []

    # Step 1: group consecutive nodes that share the same depth-1 prefix.
    # Nodes with no prefix (front-matter) get their own group keyed by "".
    groups: list[tuple[str, list[dict]]] = []   # (prefix, nodes)
    for node in flat_nodes:
        prefix = _depth1_prefix(node["title"])
        if groups and groups[-1][0] == prefix:
            groups[-1][1].append(node)
        else:
            groups.append((prefix, [node]))

    # Step 2: merge front-matter (prefix=="") into the next group if possible,
    # otherwise keep as its own first chunk.
    merged_groups: list[list[dict]] = []
    i = 0
    while i < len(groups):
        prefix, nodes = groups[i]
        if prefix == "" and i + 1 < len(groups):
            # Attach front-matter to the following section group
            merged_groups.append(nodes + groups[i + 1][1])
            i += 2
        else:
            merged_groups.append(nodes)
            i += 1

    # Step 3: greedily pack section groups into chunks <= max_chunk.
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for group in merged_groups:
        if current and len(current) + len(group) > max_chunk:
            chunks.append(current)
            current = list(group)
        else:
            current.extend(group)
    if current:
        chunks.append(current)

    return chunks


def llm_build_hierarchy(source_data: dict) -> list[dict]:
    """
    Pass 1: send the full flat list to the LLM and get back a nested tree.
    For large documents (> HIERARCHY_CHUNK_SIZE nodes) the list is split at
    depth-1 section boundaries so no section is ever split across chunks.
    Each chunk is processed independently and results are concatenated in order.
    Returns the list of root node dicts (the 'structure' array).
    """
    flat_nodes = source_data.get("hierarchy", [])

    if len(flat_nodes) > HIERARCHY_CHUNK_SIZE:
        chunks_in = split_into_section_chunks(flat_nodes, HIERARCHY_CHUNK_SIZE)
        total_chunks = len(chunks_in)
        print(f"  Splitting {len(flat_nodes)} nodes into {total_chunks} section-boundary chunk(s) …")
        all_roots: list[dict] = []
        for idx, chunk_nodes in enumerate(chunks_in, 1):
            print(f"  Chunk {idx}/{total_chunks} ({len(chunk_nodes)} nodes) …", end=" ", flush=True)
            chunk_data = dict(source_data, hierarchy=chunk_nodes)
            sub_tree = call_with_retry(_llm_call_single_chunk, chunk_data)
            print(f"{len(sub_tree)} root(s)")
            all_roots.extend(sub_tree)
        return all_roots

    # Small document — single call as before
    return _llm_call_single_chunk(source_data)


def _llm_call_single_chunk(source_data: dict) -> list[dict]:
    """Send one flat node list to the LLM and return the parsed structure list."""
    flat_nodes = source_data.get("hierarchy", [])
    doc_name = source_data.get("doc_name", "unknown")
    total_pages = source_data.get("total_pages", "unknown")

    user_msg = HIERARCHY_USER_TEMPLATE.format(
        doc_name=doc_name,
        total_pages=total_pages,
        node_table=build_node_table(flat_nodes),
    )

    print(f"  Sending {len(flat_nodes)} nodes to LLM for hierarchy construction …")

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": HIERARCHY_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON for hierarchy: {e}\n\nRaw (first 600 chars):\n{raw[:600]}"
        )

    # Tolerate LLM returning bare list instead of {"structure": [...]}
    structure = parsed.get("structure", parsed)
    if not isinstance(structure, list):
        raise ValueError(
            f"Expected a list under 'structure', got {type(structure).__name__}"
        )

    return structure


# =============================================================================
# POST-PROCESS — authoritative page data + validation
# =============================================================================

def build_flat_index(flat_nodes: list[dict]) -> dict[str, dict]:
    """node_id → original flat node dict for O(1) lookup."""
    return {n["node_id"]: n for n in flat_nodes}


def merge_page_data(tree_nodes: list[dict], flat_index: dict[str, dict]) -> None:
    """
    Overwrite start_index / end_index on every node in the LLM-generated
    tree with the authoritative values from the original flat list, then
    re-expand parent end_index values bottom-up.

    This guarantees page ranges are always correct even if the LLM made
    arithmetic errors or hallucinated page numbers.
    Mutates tree_nodes in place.
    """

    def _fix(node: dict) -> int:
        nid = node["node_id"]
        if nid in flat_index:
            orig = flat_index[nid]
            node["start_index"] = orig["start_index"]
            node["end_index"] = orig["end_index"]
        else:
            print(f"  [WARN] node_id '{nid}' not in original flat list — keeping LLM values")

        child_max = node["end_index"]
        for child in node.get("nodes", []):
            child_max = max(child_max, _fix(child))

        node["end_index"] = child_max
        return node["end_index"]

    for root in tree_nodes:
        _fix(root)


def validate_tree(tree_nodes: list[dict], flat_index: dict[str, dict]) -> None:
    """Report which nodes were included, skipped (RULE 6), or duplicated."""
    seen: set[str] = set()

    def _collect(nodes: list[dict]) -> None:
        for n in nodes:
            nid = n["node_id"]
            if nid in seen:
                print(f"  [WARN] duplicate node_id '{nid}' in LLM output")
            seen.add(nid)
            _collect(n.get("nodes", []))

    _collect(tree_nodes)

    skipped = set(flat_index.keys()) - seen
    if skipped:
        print(f"  [INFO] {len(skipped)} node(s) omitted by LLM (RULE 6 — noise / unclassifiable):")
        for nid in sorted(skipped):
            print(f"         {nid} | {flat_index[nid]['title']}")

    covered = seen & set(flat_index.keys())
    print(f"  [OK]  {len(covered)}/{len(flat_index)} original nodes placed in tree")


# =============================================================================
# PASS 2 — LLM SUMMARY GENERATION
# =============================================================================

SUMMARY_SYSTEM_PROMPT = """\
You are a technical documentation assistant for AUTOSAR and automotive software standards.

For each section entry below write a SINGLE concise sentence (maximum 25 words)
describing what the section covers.

Base summaries ONLY on the section title and its children's titles.
Never hallucinate technical details not clearly implied by the title.

Special cases:
  • Spec-item tags  ([AP_TPS_APMC_xxxxx], [constr_xxxxx]) — describe what the
    rule or constraint defines; do NOT repeat the tag ID verbatim in the summary.
  • Table entries  ("Table N.M: ClassName") — describe what the table documents.
  • Change-log sections  ("Added/Changed/Deleted Specification Items in Rxx-xx",
    "Traceable item history …") — set summary to "" (empty); these are
    administrative records with no semantic retrieval value.
  • Any section whose content you genuinely cannot infer from its title —
    set summary to "" rather than guessing.
  • If a node already has a non-empty "existing_summary" field in the input,
    copy it verbatim as the summary for that node_id — do not regenerate it.

Reply ONLY with a JSON object mapping node_id → summary string.
No markdown, no preamble, no extra keys.
Example:  {"0001": "Covers disclaimer and legal notices.", "0002": ""}
"""


def summarise_batch(batch: list[dict]) -> dict[str, str]:
    items = []
    for node in batch:
        entry: dict = {"node_id": node["node_id"], "title": node["title"]}
        child_titles = [c["title"] for c in node.get("nodes", [])]
        if child_titles:
            entry["children"] = child_titles
        items.append(entry)

    user_msg = "Sections to summarise:\n" + json.dumps(items, indent=2)

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        return {str(k): str(v) for k, v in result.items()}
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse failed for summary batch: {e}")
        return {}


def flatten_tree(roots: list[dict]) -> list[dict]:
    """BFS — collect every node in the tree in top-down order."""
    all_nodes: list[dict] = []
    queue = list(roots)
    while queue:
        node = queue.pop(0)
        all_nodes.append(node)
        queue.extend(node.get("nodes", []))
    return all_nodes


def add_summaries(roots: list[dict], skip: bool = False) -> None:
    """Pass 2: fill 'summary' on every node via batched LLM calls.
    Nodes that already carry a non-empty summary are left untouched."""
    all_nodes = flatten_tree(roots)
    total = len(all_nodes)

    if skip:
        print(f"  [summary] skipped (--no-summary) — setting empty string on nodes without existing summary")
        for node in all_nodes:
            if not node.get("summary"):
                node["summary"] = ""
        return

    needs_summary = [n for n in all_nodes if not n.get("summary")]
    already_done = total - len(needs_summary)
    if already_done:
        print(f"  [summary] {already_done}/{total} nodes already have summaries — skipping them")

    if not needs_summary:
        print(f"  [summary] all {total} nodes already summarised — nothing to do")
        return

    print(f"  [summary] generating summaries for {len(needs_summary)} nodes "
          f"in batches of {SUMMARY_BATCH_SIZE} …")

    summaries: dict[str, str] = {}
    total_batches = (len(needs_summary) + SUMMARY_BATCH_SIZE - 1) // SUMMARY_BATCH_SIZE

    for batch_num, batch_start in enumerate(range(0, len(needs_summary), SUMMARY_BATCH_SIZE), 1):
        batch = needs_summary[batch_start: batch_start + SUMMARY_BATCH_SIZE]
        end = min(batch_start + SUMMARY_BATCH_SIZE, len(needs_summary))
        print(f"    batch {batch_num}/{total_batches} "
              f"(nodes {batch_start + 1}–{end}) … ", end="", flush=True)
        try:
            result = call_with_retry(summarise_batch, batch)
            summaries.update(result)
            filled = sum(1 for v in result.values() if v)
            print(f"OK  ({filled}/{len(batch)} non-empty)")
        except Exception as e:
            print(f"FAILED — {e}  (empty summaries used for this batch)")

        if batch_start + SUMMARY_BATCH_SIZE < len(needs_summary):
            time.sleep(BATCH_SLEEP)

    for node in needs_summary:
        node["summary"] = summaries.get(node["node_id"], "")

    total_filled = sum(1 for n in all_nodes if n.get("summary"))
    print(f"  [summary] {total_filled}/{total} nodes have non-empty summaries")


# =============================================================================
# STATS
# =============================================================================

def print_stats(roots: list[dict]) -> None:
    all_nodes = flatten_tree(roots)
    depth_counts: dict[int, int] = {}

    def _depth(node: dict, d: int) -> None:
        depth_counts[d] = depth_counts.get(d, 0) + 1
        for child in node.get("nodes", []):
            _depth(child, d + 1)

    for root in roots:
        _depth(root, 0)

    print(f"  Tree stats:")
    print(f"    Total nodes : {len(all_nodes)}")
    print(f"    Root nodes  : {len(roots)}")
    for d in sorted(depth_counts):
        print(f"    Depth {d}      : {depth_counts[d]} node(s)")


# =============================================================================
# FILE/DIR HELPERS
# =============================================================================

def derive_output_path(input_path: str, output_path: str | None = None) -> str:
    """Derive the output path for one input file."""
    if output_path:
        return os.path.abspath(output_path)

    base = re.sub(r"_structure\.json$", "", input_path)
    base = re.sub(r"\.json$", "", base)
    return base + "_structure.json"


def process_one_file(input_path: str, output_path: str | None, no_summary: bool) -> int:
    """Process a single hierarchy JSON file. Returns 0 on success, 1 on failure."""
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    final_output = derive_output_path(input_path, output_path)

    print("=" * 64)
    print(f"  Input  : {input_path}")
    print(f"  Output : {final_output}")
    print(f"  Model  : {OLLAMA_MODEL}  @  {OLLAMA_BASE_URL}")
    print(f"  Pass 2 : {'DISABLED (--no-summary)' if no_summary else 'ENABLED'}")
    print("=" * 64)

    print("\n[1/4] Loading flat hierarchy …")
    with open(input_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    flat_nodes = source_data.get("hierarchy", [])
    if not flat_nodes:
        print("[ERROR] 'hierarchy' key is empty or missing.", file=sys.stderr)
        return 1

    flat_index = build_flat_index(flat_nodes)
    print(f"  {len(flat_nodes)} flat nodes loaded from '{source_data.get('doc_name', '?')}'")

    print("\n[2/4] Pass 1 — LLM hierarchy construction …")
    structure = call_with_retry(llm_build_hierarchy, source_data)
    print(f"  LLM returned {len(structure)} root node(s)")

    merge_page_data(structure, flat_index)
    validate_tree(structure, flat_index)
    print_stats(structure)

    print("\n[3/4] Pass 2 — LLM summary generation …")
    add_summaries(structure, skip=no_summary)

    print("\n[4/4] Writing output …")
    output = {
        "doc_description": source_data.get("doc_name", ""),
        "total_pages": source_data.get("total_pages"),
        "source_format": source_data.get("source", ""),
        "structure": structure,
    }

    os.makedirs(os.path.dirname(final_output) or ".", exist_ok=True)
    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Saved → {final_output}")
    print(f"\n✅ Done.  '{os.path.basename(final_output)}' is ready for run_rag_v4.py.")
    return 0


def find_input_files(input_dir: str) -> list[str]:
    """Return *_hierarchy.json files from a directory, sorted by name."""
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    files = sorted(str(x) for x in p.iterdir() if x.is_file() and x.name.endswith("_structure.json"))
    return files


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert flat _hierarchy.json file(s) into nested _structure.json using LLM hierarchy inference + summaries (Ollama)."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i",
        help="Path to one flat _hierarchy.json file",
    )
    group.add_argument(
        "--input-dir",
        help="Path to a directory containing one or more *_hierarchy.json files",
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for single-file mode only. If omitted, *_structure.json is derived automatically.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for directory mode. If omitted, each output is written next to its input file.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip Pass 2 summary generation (summaries will be empty strings)",
    )
    args = parser.parse_args()

    # Single file mode
    if args.input:
        return process_one_file(args.input, args.output, args.no_summary)

    # Directory mode
    try:
        input_files = find_input_files(args.input_dir)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if not input_files:
        print(f"[ERROR] No *_hierarchy.json files found in directory: {args.input_dir}", file=sys.stderr)
        return 1

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Found {len(input_files)} input file(s) in {os.path.abspath(args.input_dir)}")

    overall_rc = 0
    for idx, input_file in enumerate(input_files, 1):
        print(f"\n--- Processing {idx}/{len(input_files)}: {os.path.basename(input_file)} ---")

        if output_dir:
            base = os.path.basename(input_file)
            base = re.sub(r"_structure\.json$", "", base)
            base = re.sub(r"\.json$", "", base)
            out_path = os.path.join(output_dir, base + "_structure.json")
        else:
            out_path = None

        rc = process_one_file(input_file, out_path, args.no_summary)
        if rc != 0:
            overall_rc = rc

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())