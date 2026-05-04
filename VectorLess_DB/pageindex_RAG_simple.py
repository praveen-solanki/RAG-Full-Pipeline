# #!/usr/bin/env python3
# # coding: utf-8
# """
# pageindex_RAG_simple.py
# =======================
# RAG pipeline using pre-built PageIndex tree structures. Two modes:

#   --mode eval   (default) — GT-based evaluation. source_document known from
#                             questions JSON. Computes all retrieval + answer metrics.
#   --mode infer            — Real-world inference. No GT needed. Doc-selection runs
#                             first using description strategy (tutorial: doc-search/
#                             description.md). No metrics computed.

# Assumptions (all files already exist — no ingestion happens here):

#   PDF input  — use ONE of:
#     --pdf_file   /data/pdfs/report.pdf       single PDF file
#     --pdf_dir    /data/pdfs/                 directory of PDFs

#   Tree input — use ONE of:
#     --tree_file  /data/trees/report_structure.json   single tree JSON
#     --tree_dir   /data/trees/                        directory of tree JSONs

#   --md_dir          /data/mds/         Markdown page folders (used with --use_md)
#   --query           questions.json     question bank (eval) or plain list (infer)
#   --provider        openai|nvidia|ollama   generation model
#   --model           model name (optional, provider default used if omitted)
#   --judge_provider  openai|nvidia|ollama   independent judge model
#   --judge_model     model name (optional)
#   --domain          autosar|none       enables AUTOSAR domain preference injection
#                                        into tree search prompt (tutorial: tree-search/
#                                        README.md expert knowledge section)
#   --pageindex_repo  /path/to/PageIndex/  adds repo root to sys.path so
#                                          pageindex.retrieve is used for page extraction.
#                                          If omitted PyPDF2 fallback is used instead.
#                                          Startup prints which extractor is active.

# Pipeline per question:
#   EVAL:  doc lookup (GT) → [preference] → tree search → resolve nodes
#          → page extract → answer gen → retrieval eval → evidence recall → LLM judge
#   INFER: doc-selection (LLM) → [preference] → tree search → resolve nodes
#          → page extract → answer gen → print answer

# Usage:
#   # Eval — multi PDF, AUTOSAR domain, separate judge
#   python3 pageindex_RAG_simple.py \
#       --mode eval --query /data/q.json \
#       --pdf_dir /data/pdfs/ --tree_dir /data/trees/ \
#       --domain autosar \
#       --provider nvidia --model kimi-k2 \
#       --judge_provider openai --judge_model gpt-4.1

#   # Infer — user question, doc auto-selected
#   python3 pageindex_RAG_simple.py \
#       --mode infer --query /data/user_q.json \
#       --pdf_dir /data/pdfs/ --tree_dir /data/trees/ \
#       --domain autosar --provider nvidia

#   # Single PDF eval
#   python3 pageindex_RAG_simple.py \
#       --mode eval --query /data/q.json \
#       --pdf_file /data/pdfs/report.pdf \
#       --tree_file /data/trees/report_structure.json \
#       --provider ollama --model llama3.1:8b --parallel 4
# """

# import argparse
# import json
# import os
# import re
# import sys
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from dotenv import load_dotenv
# from openai import OpenAI

# # =============================================================================
# # CLI ARGUMENT PARSING
# # =============================================================================

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="PageIndex retrieval-only RAG pipeline with metrics",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog=__doc__,
#     )

#     # ── Paths ─────────────────────────────────────────────────────────────────
#     parser.add_argument("--query",      required=True,
#                         help="Path to questions JSON file")

#     # Tree input — single file OR directory (at least one required)
#     tree_group = parser.add_mutually_exclusive_group(required=True)
#     tree_group.add_argument("--tree_file", default=None,
#                         help="Single tree JSON file (used for ALL questions). "
#                              "Use when processing one PDF.")
#     tree_group.add_argument("--tree_dir",  default=None,
#                         help="Directory containing *_structure.json tree files. "
#                              "Auto-mapped per question via source_document field.")

#     # PDF input — single file OR directory (at least one required)
#     pdf_group = parser.add_mutually_exclusive_group(required=True)
#     pdf_group.add_argument("--pdf_file",  default=None,
#                         help="Single PDF file path (used for ALL questions). "
#                              "Use when processing one PDF.")
#     pdf_group.add_argument("--pdf_dir",   default=None,
#                         help="Directory containing PDF files. "
#                              "Auto-mapped per question via source_document field.")

#     parser.add_argument("--md_dir",     default=None,
#                         help="Directory containing per-doc markdown page folders "
#                              "(used when --use_md flag is set)")
#     parser.add_argument("--output_dir", default="./results",
#                         help="Directory to write results.json and metrics_summary.json "
#                              "(default: ./results)")
#     parser.add_argument("--env_file",   default=".env",
#                         help="Path to .env file for API keys (default: .env)")

#     # ── Generation LLM provider ───────────────────────────────────────────────
#     parser.add_argument("--provider",   required=True,
#                         choices=["openai", "nvidia", "ollama"],
#                         help="LLM backend for tree search + answer generation")
#     parser.add_argument("--model",      default=None,
#                         help="Generation model name override. Defaults: "
#                              "openai=gpt-4.1, nvidia=moonshotai/kimi-k2-instruct-0905, "
#                              "ollama=llama3.1:8b")

#     # ── Judge LLM provider (independent from generation) ─────────────────────
#     # A separate model for judging avoids self-evaluation bias.
#     # If omitted, falls back to --provider / --model (same as generator).
#     parser.add_argument("--judge_provider", default=None,
#                         choices=["openai", "nvidia", "ollama"],
#                         help="LLM backend for the judge. Defaults to --provider if omitted. "
#                              "Recommended: use a stronger/different model than --model.")
#     parser.add_argument("--judge_model",    default=None,
#                         help="Judge model name override. Defaults to --model if omitted.")

#     # ── Content source ────────────────────────────────────────────────────────
#     parser.add_argument("--use_md",     action="store_true",
#                         help="Serve page content from MD files in --md_dir instead of PDFs")

#     # ── Runtime ───────────────────────────────────────────────────────────────
#     parser.add_argument("--parallel",   type=int, default=1,
#                         help="Number of parallel workers (>1 recommended only for Ollama). "
#                              "Default: 1 (sequential)")
#     parser.add_argument("--sleep",      type=float, default=0.5,
#                         help="Seconds to sleep between questions in sequential mode "
#                              "(default: 0.5)")
#     parser.add_argument("--max_retries", type=int, default=3,
#                         help="Max LLM call retries on failure (default: 3)")
#     parser.add_argument("--retry_backoff", type=float, default=2.0,
#                         help="Base backoff seconds for retries, doubles each attempt "
#                              "(default: 2.0)")

#     # ── PageIndex repo ────────────────────────────────────────────────────────
#     parser.add_argument("--pageindex_repo", default=None,
#                         help="Path to PageIndex repo root (added to sys.path). "
#                              "If set, pageindex.retrieve is used for page extraction. "
#                              "If omitted, PyPDF2 fallback is used. "
#                              "Startup log shows which extractor is active.")

#     # ── Pipeline mode ─────────────────────────────────────────────────────────
#     parser.add_argument("--mode", default="eval",
#                         choices=["eval", "infer"],
#                         help="eval: GT-based evaluation with full metrics (default). "
#                              "infer: real-world inference, doc-selection runs first, "
#                              "no metrics computed.")

#     # ── Domain preference injection ───────────────────────────────────────────
#     # Tutorial: tree-search/README.md — 'Expert Knowledge / Preference Injection'
#     # Adds domain-specific routing hints to the tree search prompt so the LLM
#     # navigates to the correct sections for known question patterns.
#     parser.add_argument("--domain", default=None,
#                         choices=["autosar", "none"],
#                         help="Enable domain preference injection into tree search. "
#                              "autosar: injects AUTOSAR-specific section hints. "
#                              "none / omitted: no preference injection (default).")

#     return parser.parse_args()


# # =============================================================================
# # GLOBALS — set after arg parsing
# # =============================================================================

# args          = None   # filled in main()
# client        = None   # OpenAI-compatible client (generation)
# MODEL         = None   # generation model string
# judge_client  = None   # OpenAI-compatible client (judge — independent from generator)
# JUDGE_MODEL   = None   # judge model string
# MAX_RETRIES   = 3
# RETRY_BACKOFF = 2.0


# # =============================================================================
# # LLM CLIENT SETUP
# # =============================================================================

# PROVIDER_DEFAULTS = {
#     "openai": {
#         "model":    "gpt-4.1",
#         "base_url": "https://api.openai.com/v1",
#         "key_env":  "OPENAI_API_KEY",
#     },
#     "nvidia": {
#         "model":    "moonshotai/kimi-k2-instruct-0905",
#         "base_url": "https://integrate.api.nvidia.com/v1",
#         "key_env":  "NVIDIA_API_KEY",
#     },
#     "ollama": {
#         "model":    "llama3.1:8b",
#         "base_url": "http://localhost:11434/v1",
#         "key_env":  None,   # Ollama doesn't need a real key
#     },
# }


# def setup_llm_client(provider: str, model_override: str | None) -> tuple:
#     """
#     Build the OpenAI-compatible client and model string for the chosen provider.
#     All three providers (OpenAI, Nvidia, Ollama) use the same OpenAI client —
#     only base_url and api_key differ.
#     Returns (client, model_string).
#     """
#     cfg      = PROVIDER_DEFAULTS[provider]
#     model    = model_override or cfg["model"]
#     base_url = os.getenv(f"{provider.upper()}_BASE_URL") or cfg["base_url"]

#     if provider == "ollama":
#         api_key = "ollama"   # Ollama accepts any non-empty string
#     else:
#         key_env = cfg["key_env"]
#         api_key = os.getenv(key_env)
#         if not api_key:
#             raise EnvironmentError(
#                 f"API key not found. Set {key_env} in your .env file or environment."
#             )

#     llm_client = OpenAI(api_key=api_key, base_url=base_url)
#     print(f"[backend] {provider.upper()}  base_url={base_url}  model={model}")
#     return llm_client, model


# # =============================================================================
# # FILE LOADERS
# # =============================================================================

# def load_structure(pdf_name: str) -> dict:
#     """
#     Load the pre-built tree/TOC JSON for a given source_document filename.

#     Resolution priority:
#       1. args.tree_file — use directly for ALL questions (single-doc mode)
#       2. args.tree_dir  — look up {tree_dir}/{docname}_structure.json (multi-doc mode)
#     """
#     # Single file mode — same tree used for every question
#     if args.tree_file:
#         if not os.path.exists(args.tree_file):
#             raise FileNotFoundError(f"Tree file not found: {args.tree_file}")
#         with open(args.tree_file, "r", encoding="utf-8") as f:
#             return json.load(f)

#     # Directory mode — map by source_document name
#     base           = os.path.splitext(pdf_name)[0]
#     structure_path = os.path.join(args.tree_dir, f"{base}_structure.json")
#     if not os.path.exists(structure_path):
#         raise FileNotFoundError(
#             f"Tree structure file not found: {structure_path}\n"
#             f"Expected: {{tree_dir}}/{base}_structure.json"
#         )
#     with open(structure_path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def load_md_pages(doc_name: str, md_dir: str) -> list | None:
#     """
#     Load per-page markdown files from:
#         {md_dir}/{doc_name_without_ext}/pages/page_1.md
#         {md_dir}/{doc_name_without_ext}/pages/page_2.md  ...

#     Returns a list of {'page': int, 'content': str} dicts sorted by page number,
#     or None if the folder doesn't exist or has no .md files.
#     """
#     doc_stem  = os.path.splitext(doc_name)[0]
#     pages_dir = os.path.join(md_dir, doc_stem, "pages")

#     if not os.path.isdir(pages_dir):
#         print(f"  [md] pages folder not found: {pages_dir} — falling back to PDF")
#         return None

#     pages = []
#     for fname in sorted(os.listdir(pages_dir)):
#         if not fname.endswith(".md"):
#             continue
#         nums = re.findall(r"\d+", fname)
#         if not nums:
#             continue
#         page_num = int(nums[-1])
#         with open(os.path.join(pages_dir, fname), "r", encoding="utf-8") as f:
#             content = f.read().strip()
#         pages.append({"page": page_num, "content": content})

#     if not pages:
#         print(f"  [md] no .md files found in: {pages_dir} — falling back to PDF")
#         return None

#     pages.sort(key=lambda x: x["page"])
#     print(f"  [md] loaded {len(pages)} markdown pages from {pages_dir}")
#     return pages


# def build_documents(pdf_name: str, structure: dict) -> tuple[dict, str]:
#     """
#     Build the documents dict expected by pageindex.retrieve.
#     Returns (documents_dict, doc_id).

#     PDF path resolution priority:
#       1. args.pdf_file — use directly for ALL questions (single-doc mode)
#       2. args.pdf_dir  — join with pdf_name (multi-doc mode)

#     When use_md=True and md_dir is set, per-page markdown files are injected
#     into doc_info['pages']. retrieve.py uses these cached pages instead of
#     opening the PDF via PyPDF2.
#     """
#     doc_id = os.path.splitext(pdf_name)[0]

#     # Resolve PDF path
#     if args.pdf_file:
#         pdf_path = args.pdf_file
#     elif args.pdf_dir:
#         pdf_path = os.path.join(args.pdf_dir, pdf_name)
#     else:
#         pdf_path = ""

#     cached_pages = None
#     if args.use_md and args.md_dir:
#         cached_pages = load_md_pages(pdf_name, args.md_dir)

#     documents = {
#         doc_id: {
#             "type":            "pdf",
#             "doc_name":        pdf_name,
#             "doc_description": structure.get("doc_description", ""),
#             "path":            pdf_path,
#             "structure":       structure.get("structure", []),
#             "pages":           cached_pages,   # None → retrieve.py opens PDF normally
#         }
#     }
#     return documents, doc_id


# # =============================================================================
# # TREE HELPERS
# # =============================================================================

# def add_prefix_summaries(nodes: list, parent_prefix: str = "") -> None:
#     """
#     Walk the tree and attach prefix_summary to every node.
#     prefix_summary = concatenation of all ancestor summaries above this node.
#     Gives LLM full parent context when reasoning about deep nodes (PageIndex style).
#     Mutates nodes in place.
#     """
#     for node in nodes:
#         node["prefix_summary"] = parent_prefix
#         own_summary = node.get("summary", "")
#         next_prefix = (parent_prefix + "\n" + own_summary).strip() if own_summary else parent_prefix
#         children    = node.get("nodes", [])
#         if children:
#             add_prefix_summaries(children, next_prefix)


# def build_node_index(nodes: list, index: dict | None = None) -> dict:
#     """
#     Flatten the tree into a dict keyed by node_id for O(1) lookup.
#     Used to resolve node IDs → page ranges without trusting LLM output.
#     """
#     if index is None:
#         index = {}
#     for node in nodes:
#         index[node["node_id"]] = node
#         children = node.get("nodes", [])
#         if children:
#             build_node_index(children, index)
#     return index


# def get_page_range_string(nodes: list[dict]) -> str:
#     """
#     Convert resolved node dicts → compact page range string for get_page_content.
#     Deduplicates and sorts pages to avoid fetching the same page twice.
#     Example output: "5-7,12,15-17"
#     """
#     pages = set()
#     for node in nodes:
#         start = node.get("start_index")
#         end   = node.get("end_index")
#         if start is not None and end is not None:
#             pages.update(range(start, end + 1))

#     if not pages:
#         return ""

#     sorted_pages = sorted(pages)
#     ranges       = []
#     rs = sorted_pages[0]
#     re_ = sorted_pages[0]

#     for p in sorted_pages[1:]:
#         if p == re_ + 1:
#             re_ = p
#         else:
#             ranges.append(f"{rs}-{re_}" if rs != re_ else str(rs))
#             rs = re_ = p
#     ranges.append(f"{rs}-{re_}" if rs != re_ else str(rs))
#     return ",".join(ranges)


# def resolve_nodes(node_ids: list, node_index: dict) -> list[dict]:
#     """
#     Convert list of node_id strings → full node dicts via the pre-built index.
#     Unknown node_ids are skipped with a warning.
#     """
#     resolved = []
#     for nid in node_ids:
#         node = node_index.get(str(nid))
#         if node:
#             resolved.append({
#                 "node_id":     node["node_id"],
#                 "title":       node.get("title", ""),
#                 "start_index": node["start_index"],
#                 "end_index":   node["end_index"],
#             })
#         else:
#             print(f"  [WARN] node_id '{nid}' not found in tree index — skipped")
#     return resolved


# # =============================================================================
# # PAGE REFERENCE PARSER
# # =============================================================================

# def parse_page_reference(page_reference: str) -> tuple[int | None, int | None]:
#     """
#     Parse a page_reference string into (start_page, end_page) integers.
#     Handles: "Pages 5-6", "Page 12", "5-6", "12", "pages 3, 8"
#     Returns (None, None) if parsing fails.
#     """
#     if not page_reference:
#         return None, None

#     cleaned = re.sub(r"(?i)^pages?\s*", "", page_reference.strip())

#     m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", cleaned)
#     if m:
#         return int(m.group(1)), int(m.group(2))

#     nums = re.findall(r"\d+", cleaned)
#     if len(nums) >= 2:
#         return int(nums[0]), int(nums[-1])
#     if len(nums) == 1:
#         n = int(nums[0])
#         return n, n

#     return None, None


# # =============================================================================
# # RETRY WRAPPER
# # =============================================================================

# def call_with_retry(fn, *fn_args, **fn_kwargs):
#     """
#     Call fn(*fn_args, **fn_kwargs) and retry up to MAX_RETRIES times on failure
#     with exponential backoff (RETRY_BACKOFF seconds base, doubles each attempt).
#     Re-raises the last exception after all attempts are exhausted.
#     """
#     last_exc = None
#     for attempt in range(1, MAX_RETRIES + 2):
#         try:
#             return fn(*fn_args, **fn_kwargs)
#         except Exception as e:
#             last_exc = e
#             if attempt <= MAX_RETRIES:
#                 wait = RETRY_BACKOFF * (2 ** (attempt - 1))
#                 print(f"  ↺ attempt {attempt} failed ({e.__class__.__name__}: {e}) "
#                       f"— retrying in {wait:.0f}s ...")
#                 time.sleep(wait)
#             else:
#                 print(f"  ✗ all {MAX_RETRIES + 1} attempts failed: {e}")
#     raise last_exc


# # =============================================================================
# # DOMAIN PREFERENCES  (tutorial: tree-search/README.md — Expert Knowledge)
# # =============================================================================

# # AUTOSAR preference rules: maps query keywords → section routing hints.
# # Each rule has:
# #   keywords : if ANY of these appear (case-insensitive) in the query → rule fires
# #   hint     : the routing guidance injected into the tree search prompt
# AUTOSAR_PREFERENCES = [
#     {
#         "keywords": ["timing", "schedule", "task", "preempt", "runnab"],
#         "hint": "Prioritize OS, SchM (Schedule Manager), and Timing sections. "
#                 "For task-level questions focus on OsTask, OsEvent, and OsAlarm nodes.",
#     },
#     {
#         "keywords": ["memory", "memmap", "section", "linker", "compiler abstraction"],
#         "hint": "Prioritize MemMap, Compiler Abstraction, and Platform Type sections.",
#     },
#     {
#         "keywords": ["api", "function", "prototype", "signature", "return", "parameter"],
#         "hint": "Prioritize API Specification chapters (SWS_* numbered requirements) "
#                 "and any node titled 'API', 'Function Definitions', or 'Interfaces'.",
#     },
#     {
#         "keywords": ["error", "det", "diagnostic", "fault", "dem", "dtc"],
#         "hint": "Prioritize Development Error Tracer (Det), Diagnostic Event Manager "
#                 "(Dem), and Error Handling sections.",
#     },
#     {
#         "keywords": ["configuration", "ecuc", "parameter", "container", "variant"],
#         "hint": "Prioritize Configuration Specification (EcucParam), EcucContainers, "
#                 "and post-build/pre-compile configuration sections.",
#     },
#     {
#         "keywords": ["communication", "com", "pdu", "signal", "ipdu", "message"],
#         "hint": "Prioritize COM, PduR (PDU Router), CanIf, LinIf, and Signal sections.",
#     },
#     {
#         "keywords": ["init", "initializ", "startup", "mode", "bsw"],
#         "hint": "Prioritize Initialization, Mode Management (BswM), and "
#                 "Basic Software Module Description sections.",
#     },
#     {
#         "keywords": ["nvm", "nvram", "non-volatile", "storage", "persist"],
#         "hint": "Prioritize NvM (Non-Volatile Memory Manager) and Ea/Fee sections.",
#     },
#     {
#         "keywords": ["eeprom", "flash", "fls", "ea", "fee"],
#         "hint": "Prioritize Flash Driver (Fls), EEPROM Abstraction (Ea), "
#                 "and Flash EEPROM Emulation (Fee) sections.",
#     },
#     {
#         "keywords": ["watchdog", "wdg", "alive", "trigger"],
#         "hint": "Prioritize Watchdog Driver (Wdg) and Watchdog Manager (WdgM) sections.",
#     },
#     {
#         "keywords": ["arti", "trace", "hook", "instrument"],
#         "hint": "Prioritize ARTI (AUTOSAR Run-Time Interface), tracing hooks, "
#                 "and instrumentation sections.",
#     },
#     {
#         "keywords": ["requirement", "srs", "sws", "tps", "constraint", "shall"],
#         "hint": "Focus on SWS_ numbered requirement nodes and any Constraints sections.",
#     },
# ]


# def get_domain_preference(query: str, domain: str | None) -> str | None:
#     """
#     Look up domain-specific preference hints for the given query.
#     Returns a combined hint string if any rules fire, or None if no match.

#     Tutorial reference: tree-search/README.md — Expert Knowledge section.
#     The returned string is injected into the tree search prompt as:
#       'Expert Knowledge of relevant sections: {preference}'
#     """
#     if not domain or domain == "none":
#         return None

#     if domain == "autosar":
#         rules     = AUTOSAR_PREFERENCES
#         q_lower   = query.lower()
#         fired     = [r["hint"] for r in rules
#                      if any(kw in q_lower for kw in r["keywords"])]
#         if fired:
#             return " ".join(fired)
#         return None

#     return None


# # =============================================================================
# # DOC SELECTION — INFER MODE  (tutorial: doc-search/description.md)
# # =============================================================================

# def doc_selection_infer(query: str, doc_registry: list[dict]) -> list[str]:
#     """
#     Select relevant doc_ids for a query using the description-based strategy.

#     Tutorial: doc-search/description.md
#     Uses an LLM to compare the query against pre-generated one-sentence descriptions
#     stored in each tree's 'doc_description' field.

#     Prompt is taken verbatim from the tutorial.

#     Args:
#       query        : the user's question
#       doc_registry : list of {doc_id, doc_name, doc_description} dicts built
#                      from all loaded tree files in --tree_dir

#     Returns list of selected doc_ids. Empty list = no relevant document found.
#     """
#     if not doc_registry:
#         return []

#     # Tutorial prompt — doc-search/description.md 'Search with LLM' section
#     docs_json = json.dumps(doc_registry, indent=2)
#     prompt = f"""You are given a list of documents with their IDs, file names, and descriptions. Your task is to select documents that may contain information relevant to answering the user query.

# Query: {query}

# Documents: {docs_json}

# Response Format:
# {{
#     "thinking": "<Your reasoning for document selection>",
#     "answer": <Python list of relevant doc_ids>, e.g. ["doc_id1", "doc_id2"]. Return [] if no documents are relevant.
# }}

# Return only the JSON structure, with no additional output."""

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"},
#     )
#     result = json.loads(response.choices[0].message.content)
#     selected = result.get("answer", [])
#     if not isinstance(selected, list):
#         selected = []
#     print(f"  [doc-select] selected {len(selected)} doc(s): {selected}")
#     return selected


# def build_doc_registry(tree_dir: str) -> list[dict]:
#     """
#     Build the doc registry for infer-mode doc-selection.
#     Reads all *_structure.json files in tree_dir and extracts
#     doc_id, doc_name, doc_description — exactly the fields the tutorial prompt uses.
#     """
#     registry = []
#     if not tree_dir or not os.path.isdir(tree_dir):
#         return registry
#     for fname in sorted(os.listdir(tree_dir)):
#         if not fname.endswith("_structure.json"):
#             continue
#         path = os.path.join(tree_dir, fname)
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             # doc_id = filename stem without _structure suffix
#             doc_id   = fname.replace("_structure.json", "")
#             doc_name = doc_id + ".pdf"
#             doc_desc = data.get("doc_description", "")
#             if not doc_desc:
#                 # Fallback: generate a description from root node titles
#                 nodes    = data.get("structure", [])
#                 titles   = [n.get("title", "") for n in nodes[:5] if n.get("title")]
#                 doc_desc = f"Document covering: {', '.join(titles)}" if titles else doc_id
#             registry.append({
#                 "doc_id":          doc_id,
#                 "doc_name":        doc_name,
#                 "doc_description": doc_desc,
#             })
#         except Exception as e:
#             print(f"  [WARN] could not load registry entry for {fname}: {e}")
#     print(f"[doc-registry] {len(registry)} documents indexed for infer-mode selection")
#     return registry


# # =============================================================================
# # PIPELINE STEPS
# # =============================================================================

# def tree_search(query: str, tree_structure_json: str,
#                preference: str | None = None) -> list:
#     """
#     Step 2 — LLM identifies relevant node_ids from the tree.
#     Returns only node_id strings — page numbers are resolved programmatically.

#     Base prompt matches the official PageIndex prompt (tutorial: tree-search/README.md).
#     When preference is provided, uses the Enhanced Tree Search with Expert Preference
#     prompt from the same tutorial.

#     Args:
#       query               : the user question
#       tree_structure_json : clean serialised PageIndex tree (no prefix_summary)
#       preference          : optional domain hint from get_domain_preference()
#     """
#     if preference:
#         # Tutorial: tree-search/README.md — 'Enhanced Tree Search with Expert Preference'
#         prompt = f"""You are given a question and a tree structure of a document.
# You need to find all nodes that are likely to contain the answer.

# Query: {query}

# Document tree structure: {tree_structure_json}

# Expert Knowledge of relevant sections: {preference}

# Reply in the following JSON format:
# {{
#   "thinking": "<your reasoning about which nodes are relevant>",
#   "node_list": [node_id1, node_id2, ...]
# }}"""
#     else:
#         # Tutorial: tree-search/README.md — basic LLM tree search
#         prompt = f"""You are given a query and the tree structure of a document.
# You need to find all nodes that are likely to contain the answer.

# Query: {query}

# Document tree structure: {tree_structure_json}

# Reply in the following JSON format:
# {{
#   "thinking": "<your reasoning about which nodes are relevant>",
#   "node_list": [node_id1, node_id2, ...]
# }}"""

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"},
#     )
#     result = json.loads(response.choices[0].message.content)
#     return result.get("node_list", [])


# def generate_answer(query: str, page_contents: list[dict]) -> str:
#     """
#     Step 4 — Generate final answer from extracted page content.
#     """
#     context = "\n\n".join(
#         f"[Page {p['page']}]\n{p['content']}"
#         for p in page_contents if p.get("content")
#     )
#     prompt = f"""Answer the following question using only the provided context.
# Be precise and cite the page number when possible.

# Question: {query}

# Context:
# {context}

# Answer:"""

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return response.choices[0].message.content.strip()


# # =============================================================================
# # EVALUATION
# # =============================================================================

# def check_retrieval_overlap(retrieved_nodes: list[dict],
#                             start_page: int, end_page: int) -> dict:
#     """
#     Step 5a — No LLM needed.
#     Compares retrieved page ranges against the gold start_page/end_page.
#     Returns: hit flag, recall, precision, F1, overlapping pages.
#     """
#     gold_pages = set(range(start_page, end_page + 1))

#     retrieved_pages = set()
#     for node in retrieved_nodes:
#         s = node.get("start_index")
#         e = node.get("end_index")
#         if s is not None and e is not None:
#             retrieved_pages.update(range(s, e + 1))

#     overlap   = gold_pages & retrieved_pages
#     recall    = round(len(overlap) / len(gold_pages),      2) if gold_pages      else 0.0
#     precision = round(len(overlap) / len(retrieved_pages), 2) if retrieved_pages else 0.0
#     f1        = round(
#         2 * precision * recall / (precision + recall), 2
#     ) if (precision + recall) > 0 else 0.0

#     return {
#         "gold_pages":      sorted(gold_pages),
#         "retrieved_pages": sorted(retrieved_pages),
#         "overlap_pages":   sorted(overlap),
#         "retrieval_hit":   len(overlap) > 0,
#         "recall":          recall,
#         "precision":       precision,
#         "f1":              f1,
#     }


# def check_evidence_recall(page_contents: list[dict], evidence_snippets: list) -> dict:
#     """
#     Step 5a-ii — No LLM needed.
#     Checks how many gold evidence_snippets appear (substring match) in the
#     full retrieved page text. This is the most direct retrieval quality signal:
#     did we actually retrieve the text that answers the question?

#     Returns:
#       total_snippets   : number of gold snippets in the question
#       matched_snippets : how many were found in retrieved content
#       evidence_recall  : matched / total  (None if no snippets — excluded from avg)
#       no_snippets      : True when evidence_snippets is empty
#     """
#     if not evidence_snippets:
#         return {
#             "total_snippets":   0,
#             "matched_snippets": 0,
#             "evidence_recall":  None,
#             "no_snippets":      True,
#         }

#     # Build one normalised string from all retrieved page content
#     full_text = " ".join(
#         p.get("content", "") for p in page_contents if p.get("content")
#     ).lower()

#     matched = 0
#     for snippet in evidence_snippets:
#         # Normalise whitespace + lowercase for robust substring match
#         normalised = re.sub(r"\s+", " ", snippet.strip()).lower()
#         if normalised and normalised in full_text:
#             matched += 1

#     total = len(evidence_snippets)
#     return {
#         "total_snippets":   total,
#         "matched_snippets": matched,
#         "evidence_recall":  round(matched / total, 4),
#         "no_snippets":      False,
#     }


# def llm_judge(question: str, ground_truth: str, generated_answer: str,
#               evidence_snippets: list, source_document: str) -> dict:
#     """
#     Step 5b — LLM as judge.
#     Evaluates the generated answer against ground truth and gold evidence snippets.
#     Scores: correctness, completeness, hallucination, verdict.
#     """
#     snippets_text = "\n".join(f"- {s}" for s in evidence_snippets) if evidence_snippets else "N/A"

#     prompt = f"""You are an expert evaluator for a RAG system.

# Document: {source_document}

# Question: {question}

# Ground Truth Answer:
# {ground_truth}

# Gold Evidence Snippets (from the source document):
# {snippets_text}

# Generated Answer:
# {generated_answer}

# Evaluate on these three criteria:
# 1. Factual correctness — does the generated answer convey the same facts as the ground truth?
# 2. Completeness — does it cover all key points in the ground truth?
# 3. Hallucination — does it add facts not supported by the ground truth or evidence?

# Reply ONLY in this JSON format with no extra text:
# {{
#   "verdict": "correct" | "partial" | "incorrect",
#   "correctness_score": <float 0.0 to 1.0>,
#   "completeness_score": <float 0.0 to 1.0>,
#   "hallucination": "none" | "minor" | "major",
#   "reasoning": "<brief explanation of your scores>"
# }}"""

#     # Use the independent judge client so the judge model is never the same
#     # call path as the generator — avoids self-evaluation bias.
#     response = judge_client.chat.completions.create(
#         model=JUDGE_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"},
#     )
#     return json.loads(response.choices[0].message.content)


# # =============================================================================
# # PER-QUESTION PIPELINE
# # =============================================================================

# def process_question(q: dict, index: int, total: int) -> dict:
#     """
#     Run the full pipeline for a single question.
#     Called directly (sequential) or from a ThreadPoolExecutor (parallel).
#     """
#     qid               = q.get("id", f"q{index:03d}")
#     query             = q.get("question", "")
#     pdf_name          = q.get("source_document", "")
#     ground_truth      = q.get("answer", "")
#     evidence_snippets = q.get("evidence_snippets", [])
#     page_reference    = q.get("page_reference", "")
#     difficulty        = q.get("difficulty", "")
#     question_type     = q.get("question_type", "")

#     start_page, end_page = parse_page_reference(page_reference)

#     print(f"\n[{index}/{total}] {qid} — {query[:80]}...")

#     if not query or not pdf_name:
#         print(f"  ✗ SKIP: missing 'question' or 'source_document'")
#         return {
#             "id":              qid,
#             "question":        query,
#             "source_document": pdf_name,
#             "question_type":   question_type,
#             "difficulty":      difficulty,
#             "page_reference":  page_reference,
#             "status":          "skipped",
#             "error":           "missing 'question' or 'source_document'",
#         }

#     try:
#         # ── Step 1: Load pre-built structure + build documents dict ───────────
#         structure         = load_structure(pdf_name)
#         documents, doc_id = build_documents(pdf_name, structure)

#         # ── Step 1b: Prepare clean tree for search + build node index ──────────
#         tree_nodes = documents[doc_id]["structure"]

#         # Serialise the CLEAN tree BEFORE mutation so the search prompt receives
#         # exactly what the tutorial shows: {PageIndex_Tree} with no injected fields.
#         # (tutorial: tree-search/README.md — "Document tree structure: {PageIndex_Tree}")
#         tree_json = json.dumps(tree_nodes, indent=2)

#         # Now mutate: add prefix_summaries for internal node resolution context
#         # and build the O(1) lookup index. These are never sent to the LLM prompt.
#         add_prefix_summaries(tree_nodes)
#         node_index = build_node_index(tree_nodes)

#         # ── Step 2a: Domain preference lookup (optional) ─────────────────────
#         preference = get_domain_preference(query, args.domain)
#         if preference:
#             print(f"  → [preference] domain={args.domain} hint injected into tree search")

#         # ── Step 2b: Tree search — LLM returns node_ids only ──────────────────
#         node_ids  = call_with_retry(tree_search, query, tree_json, preference)
#         print(f"  → {len(node_ids)} node_id(s) returned by LLM")

#         # ── Step 2b: Resolve node_ids → page ranges from tree (no LLM) ────────
#         relevant_nodes = resolve_nodes(node_ids, node_index)
#         print(f"  → {len(relevant_nodes)} node(s) resolved from tree index")

#         # ── Step 3: Extract page content ──────────────────────────────────────
#         page_range = get_page_range_string(relevant_nodes)
#         if not page_range:
#             raise ValueError("Tree search returned no resolvable nodes with page ranges.")

#         # Use pageindex.retrieve if available, otherwise fall back to inline extraction
#         try:
#             from pageindex.retrieve import get_document_structure, get_page_content
#             raw_content   = get_page_content(documents, doc_id, page_range)
#             page_contents = json.loads(raw_content)
#         except ImportError:
#             # Fallback: extract pages directly via PyPDF2
#             page_contents = extract_pages_pypdf2(documents[doc_id]["path"], page_range)

#         # ── Step 4: Generate answer ────────────────────────────────────────────
#         answer = call_with_retry(generate_answer, query, page_contents)
#         print(f"  → Answer: {answer[:120]}...")

#         # ── Step 5a: Retrieval evaluation (no LLM) ────────────────────────────
#         # 5a-i  Page-level overlap vs gold page_reference
#         if start_page is not None and end_page is not None:
#             retrieval_eval = check_retrieval_overlap(relevant_nodes, start_page, end_page)
#         else:
#             retrieval_eval = {
#                 "retrieval_hit":  None,
#                 "recall":         None,
#                 "precision":      None,
#                 "f1":             None,
#                 "page_ref_unparseable": True,
#                 "note": f"could not parse page range from: {page_reference!r}",
#             }
#         print(f"  → Retrieval hit: {retrieval_eval.get('retrieval_hit')} | "
#               f"Recall: {retrieval_eval.get('recall', 'N/A')}")

#         # 5a-ii Evidence-snippet recall — did retrieved text contain the gold snippets?
#         # Pure substring check: no LLM involved.
#         evidence_recall_result = check_evidence_recall(page_contents, evidence_snippets)
#         print(f"  → Evidence recall: {evidence_recall_result.get('evidence_recall', 'N/A')} "
#               f"({evidence_recall_result.get('matched_snippets', 0)}/"
#               f"{evidence_recall_result.get('total_snippets', 0)} snippets matched)")

#         # ── Step 5b: LLM judge ────────────────────────────────────────────────
#         evaluation = call_with_retry(
#             llm_judge,
#             question          = query,
#             ground_truth      = ground_truth,
#             generated_answer  = answer,
#             evidence_snippets = evidence_snippets,
#             source_document   = pdf_name,
#         )
#         print(f"  → Verdict: {evaluation.get('verdict')} | "
#               f"Correctness: {evaluation.get('correctness_score')}")

#         return {
#             "id":                   qid,
#             "question":             query,
#             "question_type":        question_type,
#             "difficulty":           difficulty,
#             "source_document":      pdf_name,
#             "page_reference":       page_reference,
#             "gold_start_page":      start_page,
#             "gold_end_page":        end_page,
#             "retrieved_nodes":      relevant_nodes,
#             "pages_used":           page_range,
#             "answer":               answer,
#             "ground_truth":         ground_truth,
#             "evidence_snippets":    evidence_snippets,
#             "retrieval_eval":       retrieval_eval,
#             "evidence_recall_eval": evidence_recall_result,
#             "evaluation":           evaluation,
#             "status":               "success",
#         }

#     except Exception as e:
#         print(f"  ✗ ERROR: {e}")
#         return {
#             "id":               qid,
#             "question":         query,
#             "question_type":    question_type,
#             "difficulty":       difficulty,
#             "source_document":  pdf_name,
#             "page_reference":   page_reference,
#             "gold_start_page":  start_page,
#             "gold_end_page":    end_page,
#             "answer":           "",
#             "ground_truth":     ground_truth,
#             "status":           "error",
#             "error":            str(e),
#         }




# def process_question_infer(q: dict, index: int, total: int,
#                            doc_registry: list[dict]) -> dict:
#     """
#     INFER MODE — process a single question without GT.
#     1. Doc-selection via description-based LLM (tutorial: doc-search/description.md)
#     2. Tree search with optional domain preference
#     3. Page extraction + answer generation
#     No metrics computed — no GT available.
#     """
#     qid   = q.get("id", f"q{index:03d}")
#     query = q.get("question", "")

#     print(f"\n[{index}/{total}] {qid} — {query[:80]}...")

#     if not query:
#         return {"id": qid, "question": query, "status": "skipped",
#                 "error": "missing 'question'"}

#     try:
#         # ── Doc-selection (tutorial: doc-search/description.md) ───────────────
#         selected_doc_ids = call_with_retry(doc_selection_infer, query, doc_registry)
#         if not selected_doc_ids:
#             raise ValueError("Doc-selection returned no relevant documents for query.")

#         # Process first selected document (highest relevance)
#         doc_id   = selected_doc_ids[0]
#         pdf_name = doc_id + ".pdf"

#         structure         = load_structure(pdf_name)
#         documents, doc_id = build_documents(pdf_name, structure)

#         tree_nodes = documents[doc_id]["structure"]
#         tree_json  = json.dumps(tree_nodes, indent=2)
#         add_prefix_summaries(tree_nodes)
#         node_index = build_node_index(tree_nodes)

#         # ── Domain preference ─────────────────────────────────────────────────
#         preference = get_domain_preference(query, args.domain)
#         if preference:
#             print(f"  → [preference] domain={args.domain} hint injected")

#         # ── Tree search ───────────────────────────────────────────────────────
#         node_ids       = call_with_retry(tree_search, query, tree_json, preference)
#         relevant_nodes = resolve_nodes(node_ids, node_index)
#         page_range     = get_page_range_string(relevant_nodes)

#         if not page_range:
#             raise ValueError("Tree search returned no resolvable nodes.")

#         try:
#             from pageindex.retrieve import get_page_content
#             page_contents = json.loads(get_page_content(documents, doc_id, page_range))
#         except ImportError:
#             page_contents = extract_pages_pypdf2(documents[doc_id]["path"], page_range)

#         # ── Answer generation ─────────────────────────────────────────────────
#         answer = call_with_retry(generate_answer, query, page_contents)
#         print(f"  → Answer: {answer[:120]}...")

#         return {
#             "id":              qid,
#             "question":        query,
#             "source_document": pdf_name,
#             "selected_docs":   selected_doc_ids,
#             "retrieved_nodes": relevant_nodes,
#             "pages_used":      page_range,
#             "answer":          answer,
#             "status":          "success",
#         }

#     except Exception as e:
#         print(f"  ✗ ERROR: {e}")
#         return {"id": qid, "question": query, "status": "error", "error": str(e)}

# # =============================================================================
# # FALLBACK PAGE EXTRACTOR (when pageindex.retrieve is not importable)
# # =============================================================================

# def extract_pages_pypdf2(pdf_path: str, page_range: str) -> list[dict]:
#     """
#     Fallback page extractor using PyPDF2 when pageindex.retrieve is unavailable.
#     page_range: compact string like "5-7,12,15-17" (1-indexed)
#     Returns list of {'page': int, 'content': str}
#     """
#     try:
#         import PyPDF2
#     except ImportError:
#         raise ImportError("PyPDF2 is required for PDF extraction. pip install PyPDF2")

#     page_nums = set()
#     for part in page_range.split(","):
#         part = part.strip()
#         m = re.match(r"^(\d+)-(\d+)$", part)
#         if m:
#             page_nums.update(range(int(m.group(1)), int(m.group(2)) + 1))
#         elif part.isdigit():
#             page_nums.add(int(part))

#     results = []
#     with open(pdf_path, "rb") as f:
#         reader = PyPDF2.PdfReader(f)
#         for pg in sorted(page_nums):
#             idx = pg - 1   # PyPDF2 is 0-indexed
#             if 0 <= idx < len(reader.pages):
#                 text = reader.pages[idx].extract_text() or ""
#                 results.append({"page": pg, "content": text.strip()})
#     return results


# # =============================================================================
# # METRICS AGGREGATION
# # =============================================================================

# def compute_metrics_summary(results: list[dict], dataset_info: dict) -> dict:
#     """
#     Aggregate per-question results into clean, correct summary metrics.

#     Retrieval metrics (page-level):
#       - Only computed over questions where page_reference was parseable.
#       - retrieval_metrics_excluded tracks how many were skipped.

#     Evidence recall:
#       - Only averaged over questions that actually have evidence_snippets.
#       - evidence_recall_excluded tracks how many had no snippets.

#     Answer quality (accuracy):
#       - Denominator is total questions (not just successful) so errors
#         count against the score — no inflation from excluding failures.

#     Breakdown by question_type and difficulty included.
#     """
#     total      = len(results)
#     successful = [r for r in results if r["status"] == "success"]
#     n_success  = len(successful)

#     # ── Retrieval page-level metrics ──────────────────────────────────────────
#     # Only include questions where page_reference was parseable (recall/prec/f1 exist)
#     ret_evaluable = [
#         r for r in successful
#         if isinstance(r.get("retrieval_eval", {}).get("recall"), float)
#     ]
#     ret_excluded = n_success - len(ret_evaluable)

#     ret_hits = sum(
#         1 for r in ret_evaluable
#         if r["retrieval_eval"].get("retrieval_hit")
#     )
#     avg_ret_recall    = (round(sum(r["retrieval_eval"]["recall"]    for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None)
#     avg_ret_precision = (round(sum(r["retrieval_eval"]["precision"] for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None)
#     avg_ret_f1        = (round(sum(r["retrieval_eval"]["f1"]        for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None)

#     # ── Evidence-snippet recall ───────────────────────────────────────────────
#     # Only average over questions that have snippets (no_snippets=False)
#     ev_evaluable = [
#         r for r in successful
#         if isinstance(r.get("evidence_recall_eval", {}).get("evidence_recall"), float)
#     ]
#     ev_excluded = n_success - len(ev_evaluable)
#     avg_evidence_recall = (
#         round(sum(r["evidence_recall_eval"]["evidence_recall"] for r in ev_evaluable) / max(len(ev_evaluable), 1), 4)
#         if ev_evaluable else None
#     )

#     # ── Answer quality (LLM judge) ────────────────────────────────────────────
#     correct   = sum(1 for r in successful if r.get("evaluation", {}).get("verdict") == "correct")
#     partial   = sum(1 for r in successful if r.get("evaluation", {}).get("verdict") == "partial")
#     incorrect = n_success - correct - partial

#     # Accuracy denominator = total questions (errors count as wrong — no inflation)
#     accuracy = round(correct / max(total, 1), 4)

#     judge_evaluable = [r for r in successful if r.get("evaluation", {}).get("correctness_score") is not None]
#     avg_correctness  = (round(sum(r["evaluation"]["correctness_score"]  for r in judge_evaluable) / max(len(judge_evaluable), 1), 4) if judge_evaluable else None)
#     avg_completeness = (round(sum(r["evaluation"]["completeness_score"] for r in judge_evaluable) / max(len(judge_evaluable), 1), 4) if judge_evaluable else None)

#     hallucination_counts = {"none": 0, "minor": 0, "major": 0}
#     for r in successful:
#         h = r.get("evaluation", {}).get("hallucination", "")
#         if h in hallucination_counts:
#             hallucination_counts[h] += 1

#     # ── Breakdown by question_type ────────────────────────────────────────────
#     def make_breakdown(results_list: list, group_key: str) -> dict:
#         groups: dict = {}
#         for r in results_list:
#             key     = r.get(group_key) or "unknown"
#             verdict = r.get("evaluation", {}).get("verdict", "unknown")
#             if key not in groups:
#                 groups[key] = {
#                     "total": 0, "correct": 0, "partial": 0, "incorrect": 0,
#                     "retrieval_hits": 0, "retrieval_evaluable": 0,
#                     "evidence_evaluable": 0,
#                     "_correctness_sum": 0.0, "_recall_sum": 0.0,
#                     "_ev_recall_sum": 0.0,
#                 }
#             g = groups[key]
#             g["total"] += 1
#             if verdict in ("correct", "partial", "incorrect"):
#                 g[verdict] += 1
#             # retrieval
#             ret = r.get("retrieval_eval", {})
#             if isinstance(ret.get("recall"), float):
#                 g["retrieval_evaluable"] += 1
#                 g["_recall_sum"] += ret["recall"]
#                 if ret.get("retrieval_hit"):
#                     g["retrieval_hits"] += 1
#             # evidence
#             ev = r.get("evidence_recall_eval", {})
#             if isinstance(ev.get("evidence_recall"), float):
#                 g["evidence_evaluable"] += 1
#                 g["_ev_recall_sum"] += ev["evidence_recall"]
#             # correctness
#             cs = r.get("evaluation", {}).get("correctness_score")
#             if isinstance(cs, float):
#                 g["_correctness_sum"] += cs

#         # Finalise — compute rates, delete accumulators
#         for key, g in groups.items():
#             g["accuracy"]           = round(g["correct"] / max(g["total"], 1), 4)
#             g["retrieval_hit_rate"] = round(g["retrieval_hits"] / max(g["retrieval_evaluable"], 1), 4) if g["retrieval_evaluable"] else None
#             g["avg_retrieval_recall"]    = round(g["_recall_sum"]    / max(g["retrieval_evaluable"], 1), 4) if g["retrieval_evaluable"] else None
#             g["avg_evidence_recall"]     = round(g["_ev_recall_sum"] / max(g["evidence_evaluable"],  1), 4) if g["evidence_evaluable"]  else None
#             g["avg_correctness"]         = round(g["_correctness_sum"] / max(g["total"], 1), 4)
#             del g["_correctness_sum"], g["_recall_sum"], g["_ev_recall_sum"]
#         return groups

#     return {
#         "dataset_info": dataset_info,
#         "total":        total,
#         "successful":   n_success,
#         "errors":       total - n_success,
#         "summary": {
#             # ── Retrieval (page-level) ──────────────────────────────────────
#             "retrieval_evaluable":         len(ret_evaluable),
#             "retrieval_metrics_excluded":  ret_excluded,
#             "retrieval_hits":              ret_hits,
#             "retrieval_hit_rate":          round(ret_hits / max(len(ret_evaluable), 1), 4) if ret_evaluable else None,
#             "avg_retrieval_recall":        avg_ret_recall,
#             "avg_retrieval_precision":     avg_ret_precision,
#             "avg_retrieval_f1":            avg_ret_f1,
#             # ── Evidence recall ─────────────────────────────────────────────
#             "evidence_evaluable":          len(ev_evaluable),
#             "evidence_recall_excluded":    ev_excluded,
#             "avg_evidence_recall":         avg_evidence_recall,
#             # ── Answer quality ──────────────────────────────────────────────
#             # accuracy = correct / TOTAL (errors count as wrong — no inflation)
#             "correct":                     correct,
#             "partial":                     partial,
#             "incorrect":                   incorrect,
#             "accuracy":                    accuracy,
#             "avg_correctness_score":       avg_correctness,
#             "avg_completeness_score":      avg_completeness,
#             "hallucination_counts":        hallucination_counts,
#         },
#         "breakdown_by_question_type": make_breakdown(successful, "question_type"),
#         "breakdown_by_difficulty":    make_breakdown(successful, "difficulty"),
#     }


# # =============================================================================
# # MAIN PIPELINE
# # =============================================================================

# def run_pipeline():
#     global args, client, MODEL, judge_client, JUDGE_MODEL, MAX_RETRIES, RETRY_BACKOFF

#     args = parse_args()

#     # ── Load environment variables ────────────────────────────────────────────
#     if os.path.exists(args.env_file):
#         load_dotenv(args.env_file)
#         print(f"[env] loaded {args.env_file}")
#     else:
#         print(f"[env] .env file not found at {args.env_file} — using system env vars")

#     # ── Add PageIndex repo to sys.path + report which extractor will be used ──
#     if args.pageindex_repo:
#         sys.path.insert(0, args.pageindex_repo)
#         # Verify the import actually works so user knows at startup, not mid-run
#         try:
#             from pageindex.retrieve import get_page_content  # noqa: F401
#             print(f"[extractor] pageindex.retrieve  ← from repo: {args.pageindex_repo}")
#         except ImportError as ie:
#             print(f"[extractor] pageindex.retrieve NOT found in {args.pageindex_repo} "
#                   f"({ie}) — falling back to PyPDF2")
#     else:
#         print("[extractor] PyPDF2 (fallback)  ← set --pageindex_repo to use "
#               "pageindex.retrieve instead")

#     # ── Set globals ───────────────────────────────────────────────────────────
#     MAX_RETRIES   = args.max_retries
#     RETRY_BACKOFF = args.retry_backoff

#     # ── Setup generation LLM client ───────────────────────────────────────────
#     print("[generation]")
#     client, MODEL = setup_llm_client(args.provider, args.model)

#     # ── Setup judge LLM client (independent from generator) ───────────────────
#     # Falls back to same provider/model as generator if --judge_provider not given.
#     judge_provider = args.judge_provider or args.provider
#     judge_model_override = args.judge_model or args.model
#     print("[judge]")
#     judge_client, JUDGE_MODEL = setup_llm_client(judge_provider, judge_model_override)
#     if judge_provider == args.provider and JUDGE_MODEL == MODEL:
#         print("  [WARN] judge model == generation model — consider using --judge_provider "
#               "/ --judge_model with a different model to avoid self-evaluation bias.")

#     # ── Validate required files/dirs ─────────────────────────────────────────
#     if not os.path.exists(args.query):
#         raise FileNotFoundError(f"Questions file not found: {args.query}")

#     # Tree input validation
#     if args.tree_file and not os.path.isfile(args.tree_file):
#         raise FileNotFoundError(f"Tree file not found: {args.tree_file}")
#     if args.tree_dir and not os.path.isdir(args.tree_dir):
#         raise FileNotFoundError(f"Tree dir not found: {args.tree_dir}")

#     # PDF input validation
#     if args.pdf_file and not os.path.isfile(args.pdf_file):
#         raise FileNotFoundError(f"PDF file not found: {args.pdf_file}")
#     if args.pdf_dir and not os.path.isdir(args.pdf_dir):
#         raise FileNotFoundError(f"PDF dir not found: {args.pdf_dir}")

#     # MD validation
#     if args.use_md and not args.md_dir:
#         raise ValueError("--use_md requires --md_dir to be set")
#     if args.use_md and args.md_dir and not os.path.isdir(args.md_dir):
#         raise FileNotFoundError(f"MD dir not found: {args.md_dir}")

#     # Log active mode clearly
#     tree_mode = f"single file: {args.tree_file}" if args.tree_file else f"dir: {args.tree_dir}"
#     pdf_mode  = f"single file: {args.pdf_file}"  if args.pdf_file  else f"dir: {args.pdf_dir}"
#     print(f"[tree] {tree_mode}")
#     print(f"[pdf]  {pdf_mode}")
#     if args.use_md:
#         print(f"[md]   {args.md_dir}")

#     # ── Load questions ────────────────────────────────────────────────────────
#     with open(args.query, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     if isinstance(data, list):
#         questions    = data
#         dataset_info = {}
#     else:
#         questions    = data.get("questions", [])
#         dataset_info = data.get("dataset_info", {})

#     total = len(questions)
#     print(f"\n[mode] {args.mode.upper()}")
#     print(f"Loaded {total} questions from {args.query}")
#     if dataset_info:
#         print(f"Dataset info: {dataset_info}")
#     if args.domain and args.domain != "none":
#         print(f"[domain] {args.domain} — preference injection enabled")

#     # ── Parallel vs sequential ────────────────────────────────────────────────
#     workers = None
#     if args.parallel and args.parallel > 1:
#         workers = args.parallel
#         print(f"[parallel] workers: {workers}")
#     else:
#         print(f"[sequential] processing {total} questions one at a time")

#     results_map: dict[int, dict] = {}

#     # ── INFER MODE ────────────────────────────────────────────────────────────
#     if args.mode == "infer":
#         # Build doc registry once from tree_dir for description-based selection
#         # (tutorial: doc-search/description.md)
#         if not args.tree_dir:
#             raise ValueError("--mode infer requires --tree_dir "
#                              "(used to build doc description registry)")
#         doc_registry = build_doc_registry(args.tree_dir)
#         if not doc_registry:
#             raise ValueError("No *_structure.json files found in tree_dir — "
#                              "cannot build doc registry for infer mode.")

#         if workers:
#             with ThreadPoolExecutor(max_workers=workers) as executor:
#                 futures = {
#                     executor.submit(
#                         process_question_infer, q, i, total, doc_registry
#                     ): i
#                     for i, q in enumerate(questions, 1)
#                 }
#                 for future in as_completed(futures):
#                     i = futures[future]
#                     try:
#                         results_map[i] = future.result()
#                     except Exception as e:
#                         q = questions[i - 1]
#                         results_map[i] = {
#                             "id": q.get("id", f"q{i:03d}"),
#                             "status": "error", "error": str(e),
#                         }
#         else:
#             for i, q in enumerate(questions, 1):
#                 results_map[i] = process_question_infer(q, i, total, doc_registry)
#                 time.sleep(args.sleep)

#         results = [results_map[i] for i in range(1, total + 1)]

#         # Save infer results — no metrics
#         os.makedirs(args.output_dir, exist_ok=True)
#         results_path = os.path.join(args.output_dir, "infer_results.json")
#         with open(results_path, "w", encoding="utf-8") as f:
#             json.dump({"total": total, "results": results}, f,
#                       indent=2, ensure_ascii=False)

#         success = sum(1 for r in results if r["status"] == "success")
#         print(f"\n{'='*65}")
#         print(f"✅ Infer done — {success}/{total} answered  "
#               f"({total - success} errors)")
#         print(f"   Results  → {results_path}")
#         print(f"{'='*65}")
#         return

#     # ── EVAL MODE ─────────────────────────────────────────────────────────────
#     if workers:
#         with ThreadPoolExecutor(max_workers=workers) as executor:
#             futures = {
#                 executor.submit(process_question, q, i, total): i
#                 for i, q in enumerate(questions, 1)
#             }
#             for future in as_completed(futures):
#                 i = futures[future]
#                 try:
#                     results_map[i] = future.result()
#                 except Exception as e:
#                     q = questions[i - 1]
#                     results_map[i] = {
#                         "id":     q.get("id", f"q{i:03d}"),
#                         "status": "error",
#                         "error":  str(e),
#                     }
#     else:
#         for i, q in enumerate(questions, 1):
#             results_map[i] = process_question(q, i, total)
#             time.sleep(args.sleep)

#     # ── Reassemble in original order ──────────────────────────────────────────
#     results = [results_map[i] for i in range(1, total + 1)]

#     # ── Compute metrics ───────────────────────────────────────────────────────
#     metrics = compute_metrics_summary(results, dataset_info)

#     # ── Save outputs ──────────────────────────────────────────────────────────
#     os.makedirs(args.output_dir, exist_ok=True)

#     results_path = os.path.join(args.output_dir, "results.json")
#     metrics_path = os.path.join(args.output_dir, "metrics_summary.json")

#     full_output = {**metrics, "results": results}
#     with open(results_path, "w", encoding="utf-8") as f:
#         json.dump(full_output, f, indent=2, ensure_ascii=False)

#     with open(metrics_path, "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2, ensure_ascii=False)

#     # ── Print summary ─────────────────────────────────────────────────────────
#     s = metrics["summary"]
#     print(f"\n{'='*65}")
#     print(f"✅ Eval done — {metrics['successful']}/{total} successful  "
#           f"({metrics['errors']} errors)")
#     print(f"")
#     print(f"   [Retrieval — page level]")
#     print(f"   Evaluable (parseable page_ref) : {s['retrieval_evaluable']}  "
#           f"(excluded: {s['retrieval_metrics_excluded']})")
#     print(f"   Retrieval hits                 : "
#           f"{s['retrieval_hits']}/{s['retrieval_evaluable']}")
#     print(f"   Avg recall / prec / F1         : "
#           f"{s['avg_retrieval_recall']} / "
#           f"{s['avg_retrieval_precision']} / "
#           f"{s['avg_retrieval_f1']}")
#     print(f"")
#     print(f"   [Retrieval — evidence snippets]")
#     print(f"   Evaluable (has snippets)       : {s['evidence_evaluable']}  "
#           f"(excluded: {s['evidence_recall_excluded']})")
#     print(f"   Avg evidence recall            : {s['avg_evidence_recall']}")
#     print(f"")
#     print(f"   [Answer quality — LLM judge]")
#     print(f"   Correct / Partial / Incorrect  : "
#           f"{s['correct']} / {s['partial']} / {s['incorrect']}")
#     print(f"   Accuracy (correct/total)       : {s['accuracy']}")
#     print(f"   Avg correctness score          : {s['avg_correctness_score']}")
#     print(f"   Avg completeness score         : {s['avg_completeness_score']}")
#     print(f"   Hallucination                  : {s['hallucination_counts']}")
#     print(f"")
#     print(f"   Results  → {results_path}")
#     print(f"   Metrics  → {metrics_path}")
#     print(f"{'='*65}")


# if __name__ == "__main__":
#     run_pipeline()



#!/usr/bin/env python3
# coding: utf-8
"""
pageindex_RAG_simple.py
=======================
RAG pipeline using pre-built PageIndex tree structures. Two modes:

  --mode eval   (default) — GT-based evaluation. source_document known from
                            questions JSON. Computes all retrieval + answer metrics.
  --mode infer            — Real-world inference. No GT needed. Doc-selection runs
                            first using description strategy (tutorial: doc-search/
                            description.md). No metrics computed.

Assumptions (all files already exist — no ingestion happens here):

  PDF input  — use ONE of:
    --pdf_file   /data/pdfs/report.pdf       single PDF file
    --pdf_dir    /data/pdfs/                 directory of PDFs

  Tree input — use ONE of:
    --tree_file  /data/trees/report_structure.json   single tree JSON
    --tree_dir   /data/trees/                        directory of tree JSONs

  --md_dir          /data/mds/         Markdown page folders (used with --use_md)
  --query           questions.json     question bank (eval) or plain list (infer)
  --provider        openai|nvidia|ollama   generation model
  --model           model name (optional, provider default used if omitted)
  --judge_provider  openai|nvidia|ollama   independent judge model
  --judge_model     model name (optional)
  --domain          autosar|none       enables AUTOSAR domain preference injection
                                       into tree search prompt (tutorial: tree-search/
                                       README.md expert knowledge section)
  --pageindex_repo  /path/to/PageIndex/  adds repo root to sys.path so
                                         pageindex.retrieve is used for page extraction.
                                         If omitted PyPDF2 fallback is used instead.
                                         Startup prints which extractor is active.

Pipeline per question:
  EVAL:  doc lookup (GT) → [preference] → tree search → resolve nodes
         → page extract → answer gen → retrieval eval → evidence recall → LLM judge
  INFER: doc-selection (LLM) → [preference] → tree search → resolve nodes
         → page extract → answer gen → print answer

Usage:
  # Eval — multi PDF, AUTOSAR domain, separate judge
  python3 pageindex_RAG_simple.py \
      --mode eval --query /data/q.json \
      --pdf_dir /data/pdfs/ --tree_dir /data/trees/ \
      --domain autosar \
      --provider nvidia --model kimi-k2 \
      --judge_provider openai --judge_model gpt-4.1

  # Infer — user question, doc auto-selected
  python3 pageindex_RAG_simple.py \
      --mode infer --query /data/user_q.json \
      --pdf_dir /data/pdfs/ --tree_dir /data/trees/ \
      --domain autosar --provider nvidia

  # Single PDF eval
  python3 pageindex_RAG_simple.py \
      --mode eval --query /data/q.json \
      --pdf_file /data/pdfs/report.pdf \
      --tree_file /data/trees/report_structure.json \
      --provider ollama --model llama3.1:8b --parallel 4
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PageIndex retrieval-only RAG pipeline with metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument("--query",      required=True,
                        help="Path to questions JSON file")

    # Tree input — single file OR directory (at least one required)
    tree_group = parser.add_mutually_exclusive_group(required=True)
    tree_group.add_argument("--tree_file", default=None,
                        help="Single tree JSON file (used for ALL questions). "
                             "Use when processing one PDF.")
    tree_group.add_argument("--tree_dir",  default=None,
                        help="Directory containing *_structure.json tree files. "
                             "Auto-mapped per question via source_document field.")

    # PDF input — single file OR directory (at least one required)
    pdf_group = parser.add_mutually_exclusive_group(required=True)
    pdf_group.add_argument("--pdf_file",  default=None,
                        help="Single PDF file path (used for ALL questions). "
                             "Use when processing one PDF.")
    pdf_group.add_argument("--pdf_dir",   default=None,
                        help="Directory containing PDF files. "
                             "Auto-mapped per question via source_document field.")

    parser.add_argument("--md_dir",     default=None,
                        help="Directory containing per-doc markdown page folders "
                             "(used when --use_md flag is set)")
    parser.add_argument("--output_dir", default="./results",
                        help="Directory to write results.json and metrics_summary.json "
                             "(default: ./results)")
    parser.add_argument("--env_file",   default=".env",
                        help="Path to .env file for API keys (default: .env)")

    # ── Generation LLM provider ───────────────────────────────────────────────
    parser.add_argument("--provider",   required=True,
                        choices=["openai", "nvidia", "ollama"],
                        help="LLM backend for tree search + answer generation")
    parser.add_argument("--model",      default=None,
                        help="Generation model name override. Defaults: "
                             "openai=gpt-4.1, nvidia=moonshotai/kimi-k2-instruct-0905, "
                             "ollama=llama3.1:8b")

    # ── Judge LLM provider (independent from generation) ─────────────────────
    # A separate model for judging avoids self-evaluation bias.
    # If omitted, falls back to --provider / --model (same as generator).
    parser.add_argument("--judge_provider", default=None,
                        choices=["openai", "nvidia", "ollama"],
                        help="LLM backend for the judge. Defaults to --provider if omitted. "
                             "Recommended: use a stronger/different model than --model.")
    parser.add_argument("--judge_model",    default=None,
                        help="Judge model name override. Defaults to --model if omitted.")

    # ── Content source ────────────────────────────────────────────────────────
    parser.add_argument("--use_md",     action="store_true",
                        help="Serve page content from MD files in --md_dir instead of PDFs")

    # ── Runtime ───────────────────────────────────────────────────────────────
    parser.add_argument("--parallel",   type=int, default=1,
                        help="Number of parallel workers (>1 recommended only for Ollama). "
                             "Default: 1 (sequential)")
    parser.add_argument("--sleep",      type=float, default=0.5,
                        help="Seconds to sleep between questions in sequential mode "
                             "(default: 0.5)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max LLM call retries on failure (default: 3)")
    parser.add_argument("--retry_backoff", type=float, default=2.0,
                        help="Base backoff seconds for retries, doubles each attempt "
                             "(default: 2.0)")

    # ── PageIndex repo ────────────────────────────────────────────────────────
    parser.add_argument("--pageindex_repo", default=None,
                        help="Path to PageIndex repo root (added to sys.path). "
                             "If set, pageindex.retrieve is used for page extraction. "
                             "If omitted, PyPDF2 fallback is used. "
                             "Startup log shows which extractor is active.")

    # ── Pipeline mode ─────────────────────────────────────────────────────────
    parser.add_argument("--mode", default="eval",
                        choices=["eval", "infer"],
                        help="eval: GT-based evaluation with full metrics (default). "
                             "infer: real-world inference, doc-selection runs first, "
                             "no metrics computed.")

    # ── Domain preference injection ───────────────────────────────────────────
    # Tutorial: tree-search/README.md — 'Expert Knowledge / Preference Injection'
    # Adds domain-specific routing hints to the tree search prompt so the LLM
    # navigates to the correct sections for known question patterns.
    parser.add_argument("--domain", default=None,
                        choices=["autosar", "none"],
                        help="Enable domain preference injection into tree search. "
                             "autosar: injects AUTOSAR-specific section hints. "
                             "none / omitted: no preference injection (default).")

    return parser.parse_args()


# =============================================================================
# GLOBALS — set after arg parsing
# =============================================================================

args          = None   # filled in main()
client        = None   # OpenAI-compatible client (generation)
MODEL         = None   # generation model string
judge_client  = None   # OpenAI-compatible client (judge — independent from generator)
JUDGE_MODEL   = None   # judge model string
MAX_RETRIES   = 3
RETRY_BACKOFF = 2.0

# Per-document cache — built once at startup, reused across all questions.
# Keys: pdf_name (source_document string)
# Values: dict with keys: documents, doc_id, tree_json, node_index
# Eliminates redundant disk reads, JSON parses, tree serialisation, and
# add_prefix_summaries / build_node_index work for every question.
DOC_CACHE: dict = {}


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

PROVIDER_DEFAULTS = {
    "openai": {
        "model":    "gpt-4.1",
        "base_url": "https://api.openai.com/v1",
        "key_env":  "OPENAI_API_KEY",
    },
    "nvidia": {
        "model":    "moonshotai/kimi-k2-instruct-0905",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env":  "NVIDIA_API_KEY",
    },
    "ollama": {
        "model":    "llama3.1:8b",
        "base_url": "http://localhost:11434/v1",
        "key_env":  None,   # Ollama doesn't need a real key
    },
}


def setup_llm_client(provider: str, model_override: str | None) -> tuple:
    """
    Build the OpenAI-compatible client and model string for the chosen provider.
    All three providers (OpenAI, Nvidia, Ollama) use the same OpenAI client —
    only base_url and api_key differ.
    Returns (client, model_string).
    """
    cfg      = PROVIDER_DEFAULTS[provider]
    model    = model_override or cfg["model"]
    base_url = os.getenv(f"{provider.upper()}_BASE_URL") or cfg["base_url"]

    if provider == "ollama":
        api_key = "ollama"   # Ollama accepts any non-empty string
    else:
        key_env = cfg["key_env"]
        api_key = os.getenv(key_env)
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set {key_env} in your .env file or environment."
            )

    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[backend] {provider.upper()}  base_url={base_url}  model={model}")
    return llm_client, model


# =============================================================================
# FILE LOADERS
# =============================================================================

def load_structure(pdf_name: str) -> dict:
    """
    Load the pre-built tree/TOC JSON for a given source_document filename.

    Resolution priority:
      1. args.tree_file — use directly for ALL questions (single-doc mode)
      2. args.tree_dir  — look up {tree_dir}/{docname}_structure.json (multi-doc mode)
    """
    # Single file mode — same tree used for every question
    if args.tree_file:
        if not os.path.exists(args.tree_file):
            raise FileNotFoundError(f"Tree file not found: {args.tree_file}")
        with open(args.tree_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Directory mode — map by source_document name
    base           = os.path.splitext(pdf_name)[0]
    structure_path = os.path.join(args.tree_dir, f"{base}_structure.json")
    if not os.path.exists(structure_path):
        raise FileNotFoundError(
            f"Tree structure file not found: {structure_path}\n"
            f"Expected: {{tree_dir}}/{base}_structure.json"
        )
    with open(structure_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_md_pages(doc_name: str, md_dir: str) -> list | None:
    """
    Load per-page markdown files from:
        {md_dir}/{doc_name_without_ext}/pages/page_1.md
        {md_dir}/{doc_name_without_ext}/pages/page_2.md  ...

    Returns a list of {'page': int, 'content': str} dicts sorted by page number,
    or None if the folder doesn't exist or has no .md files.
    """
    doc_stem  = os.path.splitext(doc_name)[0]
    pages_dir = os.path.join(md_dir, doc_stem, "pages")

    if not os.path.isdir(pages_dir):
        print(f"  [md] pages folder not found: {pages_dir} — falling back to PDF")
        return None

    pages = []
    for fname in sorted(os.listdir(pages_dir)):
        if not fname.endswith(".md"):
            continue
        nums = re.findall(r"\d+", fname)
        if not nums:
            continue
        page_num = int(nums[-1])
        with open(os.path.join(pages_dir, fname), "r", encoding="utf-8") as f:
            content = f.read().strip()
        pages.append({"page": page_num, "content": content})

    if not pages:
        print(f"  [md] no .md files found in: {pages_dir} — falling back to PDF")
        return None

    pages.sort(key=lambda x: x["page"])
    print(f"  [md] loaded {len(pages)} markdown pages from {pages_dir}")
    return pages


def build_documents(pdf_name: str, structure: dict) -> tuple[dict, str]:
    """
    Build the documents dict expected by pageindex.retrieve.
    Returns (documents_dict, doc_id).

    PDF path resolution priority:
      1. args.pdf_file — use directly for ALL questions (single-doc mode)
      2. args.pdf_dir  — join with pdf_name (multi-doc mode)

    When use_md=True and md_dir is set, per-page markdown files are injected
    into doc_info['pages']. retrieve.py uses these cached pages instead of
    opening the PDF via PyPDF2.
    """
    doc_id = os.path.splitext(pdf_name)[0]

    # Resolve PDF path
    if args.pdf_file:
        pdf_path = args.pdf_file
    elif args.pdf_dir:
        pdf_path = os.path.join(args.pdf_dir, pdf_name)
    else:
        pdf_path = ""

    cached_pages = None
    if args.use_md and args.md_dir:
        cached_pages = load_md_pages(pdf_name, args.md_dir)

    documents = {
        doc_id: {
            "type":            "pdf",
            "doc_name":        pdf_name,
            "doc_description": structure.get("doc_description", ""),
            "path":            pdf_path,
            "structure":       structure.get("structure", []),
            "pages":           cached_pages,   # None → retrieve.py opens PDF normally
        }
    }
    return documents, doc_id


# =============================================================================
# TREE HELPERS
# =============================================================================

def add_prefix_summaries(nodes: list, parent_prefix: str = "") -> None:
    """
    Walk the tree and attach prefix_summary to every node.
    prefix_summary = concatenation of all ancestor summaries above this node.
    Gives LLM full parent context when reasoning about deep nodes (PageIndex style).
    Mutates nodes in place.
    """
    for node in nodes:
        node["prefix_summary"] = parent_prefix
        own_summary = node.get("summary", "")
        next_prefix = (parent_prefix + "\n" + own_summary).strip() if own_summary else parent_prefix
        children    = node.get("nodes", [])
        if children:
            add_prefix_summaries(children, next_prefix)


def build_node_index(nodes: list, index: dict | None = None) -> dict:
    """
    Flatten the tree into a dict keyed by node_id for O(1) lookup.
    Used to resolve node IDs → page ranges without trusting LLM output.
    """
    if index is None:
        index = {}
    for node in nodes:
        index[node["node_id"]] = node
        children = node.get("nodes", [])
        if children:
            build_node_index(children, index)
    return index


def get_page_range_string(nodes: list[dict]) -> str:
    """
    Convert resolved node dicts → compact page range string for get_page_content.
    Deduplicates and sorts pages to avoid fetching the same page twice.
    Example output: "5-7,12,15-17"
    """
    pages = set()
    for node in nodes:
        start = node.get("start_index")
        end   = node.get("end_index")
        if start is not None and end is not None:
            pages.update(range(start, end + 1))

    if not pages:
        return ""

    sorted_pages = sorted(pages)
    ranges       = []
    rs = sorted_pages[0]
    re_ = sorted_pages[0]

    for p in sorted_pages[1:]:
        if p == re_ + 1:
            re_ = p
        else:
            ranges.append(f"{rs}-{re_}" if rs != re_ else str(rs))
            rs = re_ = p
    ranges.append(f"{rs}-{re_}" if rs != re_ else str(rs))
    return ",".join(ranges)


def resolve_nodes(node_ids: list, node_index: dict) -> list[dict]:
    """
    Convert list of node_id strings → full node dicts via the pre-built index.
    Unknown node_ids are skipped with a warning.
    """
    resolved = []
    for nid in node_ids:
        node = node_index.get(str(nid))
        if node:
            resolved.append({
                "node_id":     node["node_id"],
                "title":       node.get("title", ""),
                "start_index": node["start_index"],
                "end_index":   node["end_index"],
            })
        else:
            print(f"  [WARN] node_id '{nid}' not found in tree index — skipped")
    return resolved


# =============================================================================
# PAGE REFERENCE PARSER
# =============================================================================

def parse_page_reference(page_reference: str) -> tuple[int | None, int | None]:
    """
    Parse a page_reference string into (start_page, end_page) integers.
    Handles: "Pages 5-6", "Page 12", "5-6", "12", "pages 3, 8"
    Returns (None, None) if parsing fails.
    """
    if not page_reference:
        return None, None

    cleaned = re.sub(r"(?i)^pages?\s*", "", page_reference.strip())

    m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", cleaned)
    if m:
        return int(m.group(1)), int(m.group(2))

    nums = re.findall(r"\d+", cleaned)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[-1])
    if len(nums) == 1:
        n = int(nums[0])
        return n, n

    return None, None


# =============================================================================
# RETRY WRAPPER
# =============================================================================

def call_with_retry(fn, *fn_args, **fn_kwargs):
    """
    Call fn(*fn_args, **fn_kwargs) and retry up to MAX_RETRIES times on failure
    with exponential backoff (RETRY_BACKOFF seconds base, doubles each attempt).
    Re-raises the last exception after all attempts are exhausted.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return fn(*fn_args, **fn_kwargs)
        except Exception as e:
            last_exc = e
            if attempt <= MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"  ↺ attempt {attempt} failed ({e.__class__.__name__}: {e}) "
                      f"— retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                print(f"  ✗ all {MAX_RETRIES + 1} attempts failed: {e}")
    raise last_exc


# =============================================================================
# DOMAIN PREFERENCES  (tutorial: tree-search/README.md — Expert Knowledge)
# =============================================================================

# AUTOSAR preference rules: maps query keywords → section routing hints.
# Each rule has:
#   keywords : if ANY of these appear (case-insensitive) in the query → rule fires
#   hint     : the routing guidance injected into the tree search prompt
AUTOSAR_PREFERENCES = [
    {
        "keywords": ["timing", "schedule", "task", "preempt", "runnab"],
        "hint": "Prioritize OS, SchM (Schedule Manager), and Timing sections. "
                "For task-level questions focus on OsTask, OsEvent, and OsAlarm nodes.",
    },
    {
        "keywords": ["memory", "memmap", "section", "linker", "compiler abstraction"],
        "hint": "Prioritize MemMap, Compiler Abstraction, and Platform Type sections.",
    },
    {
        "keywords": ["api", "function", "prototype", "signature", "return", "parameter"],
        "hint": "Prioritize API Specification chapters (SWS_* numbered requirements) "
                "and any node titled 'API', 'Function Definitions', or 'Interfaces'.",
    },
    {
        "keywords": ["error", "det", "diagnostic", "fault", "dem", "dtc"],
        "hint": "Prioritize Development Error Tracer (Det), Diagnostic Event Manager "
                "(Dem), and Error Handling sections.",
    },
    {
        "keywords": ["configuration", "ecuc", "parameter", "container", "variant"],
        "hint": "Prioritize Configuration Specification (EcucParam), EcucContainers, "
                "and post-build/pre-compile configuration sections.",
    },
    {
        "keywords": ["communication", "com", "pdu", "signal", "ipdu", "message"],
        "hint": "Prioritize COM, PduR (PDU Router), CanIf, LinIf, and Signal sections.",
    },
    {
        "keywords": ["init", "initializ", "startup", "mode", "bsw"],
        "hint": "Prioritize Initialization, Mode Management (BswM), and "
                "Basic Software Module Description sections.",
    },
    {
        "keywords": ["nvm", "nvram", "non-volatile", "storage", "persist"],
        "hint": "Prioritize NvM (Non-Volatile Memory Manager) and Ea/Fee sections.",
    },
    {
        "keywords": ["eeprom", "flash", "fls", "ea", "fee"],
        "hint": "Prioritize Flash Driver (Fls), EEPROM Abstraction (Ea), "
                "and Flash EEPROM Emulation (Fee) sections.",
    },
    {
        "keywords": ["watchdog", "wdg", "alive", "trigger"],
        "hint": "Prioritize Watchdog Driver (Wdg) and Watchdog Manager (WdgM) sections.",
    },
    {
        "keywords": ["arti", "trace", "hook", "instrument"],
        "hint": "Prioritize ARTI (AUTOSAR Run-Time Interface), tracing hooks, "
                "and instrumentation sections.",
    },
    {
        "keywords": ["requirement", "srs", "sws", "tps", "constraint", "shall"],
        "hint": "Focus on SWS_ numbered requirement nodes and any Constraints sections.",
    },
]


def get_domain_preference(query: str, domain: str | None) -> str | None:
    """
    Look up domain-specific preference hints for the given query.
    Returns a combined hint string if any rules fire, or None if no match.

    Tutorial reference: tree-search/README.md — Expert Knowledge section.
    The returned string is injected into the tree search prompt as:
      'Expert Knowledge of relevant sections: {preference}'
    """
    if not domain or domain == "none":
        return None

    if domain == "autosar":
        rules     = AUTOSAR_PREFERENCES
        q_lower   = query.lower()
        fired     = [r["hint"] for r in rules
                     if any(kw in q_lower for kw in r["keywords"])]
        if fired:
            return " ".join(fired)
        return None

    return None


# =============================================================================
# DOC SELECTION — INFER MODE  (tutorial: doc-search/description.md)
# =============================================================================

def doc_selection_infer(query: str, doc_registry: list[dict]) -> list[str]:
    """
    Select relevant doc_ids for a query using the description-based strategy.

    Tutorial: doc-search/description.md
    Uses an LLM to compare the query against pre-generated one-sentence descriptions
    stored in each tree's 'doc_description' field.

    Prompt is taken verbatim from the tutorial.

    Args:
      query        : the user's question
      doc_registry : list of {doc_id, doc_name, doc_description} dicts built
                     from all loaded tree files in --tree_dir

    Returns list of selected doc_ids. Empty list = no relevant document found.
    """
    if not doc_registry:
        return []

    # Tutorial prompt — doc-search/description.md 'Search with LLM' section
    docs_json = json.dumps(doc_registry, indent=2)
    prompt = f"""You are given a list of documents with their IDs, file names, and descriptions. Your task is to select documents that may contain information relevant to answering the user query.

Query: {query}

Documents: {docs_json}

Response Format:
{{
    "thinking": "<Your reasoning for document selection>",
    "answer": <Python list of relevant doc_ids>, e.g. ["doc_id1", "doc_id2"]. Return [] if no documents are relevant.
}}

Return only the JSON structure, with no additional output."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    selected = result.get("answer", [])
    if not isinstance(selected, list):
        selected = []
    print(f"  [doc-select] selected {len(selected)} doc(s): {selected}")
    return selected


def build_doc_registry(tree_dir: str) -> list[dict]:
    """
    Build the doc registry for infer-mode doc-selection.
    Reads all *_structure.json files in tree_dir and extracts
    doc_id, doc_name, doc_description — exactly the fields the tutorial prompt uses.
    """
    registry = []
    if not tree_dir or not os.path.isdir(tree_dir):
        return registry
    for fname in sorted(os.listdir(tree_dir)):
        if not fname.endswith("_structure.json"):
            continue
        path = os.path.join(tree_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # doc_id = filename stem without _structure suffix
            doc_id   = fname.replace("_structure.json", "")
            doc_name = doc_id + ".pdf"
            doc_desc = data.get("doc_description", "")
            if not doc_desc:
                # Fallback: generate a description from root node titles
                nodes    = data.get("structure", [])
                titles   = [n.get("title", "") for n in nodes[:5] if n.get("title")]
                doc_desc = f"Document covering: {', '.join(titles)}" if titles else doc_id
            registry.append({
                "doc_id":          doc_id,
                "doc_name":        doc_name,
                "doc_description": doc_desc,
            })
        except Exception as e:
            print(f"  [WARN] could not load registry entry for {fname}: {e}")
    print(f"[doc-registry] {len(registry)} documents indexed for infer-mode selection")
    return registry


# =============================================================================
# DOCUMENT CACHE  — built once at startup, shared across all questions
# =============================================================================

def build_doc_cache(pdf_names: list[str]) -> None:
    """
    Pre-build the per-document cache for all unique source documents.
    Called once at pipeline startup so process_question never touches the disk
    or re-serialises the tree during the question loop.

    For each unique pdf_name:
      1. load_structure()         — read + parse the tree JSON from disk
      2. build_documents()        — assemble the documents dict
      3. json.dumps(tree_nodes)   — serialise CLEAN tree for LLM prompt
      4. add_prefix_summaries()   — mutate tree for internal context
      5. build_node_index()       — flatten tree for O(1) node lookup

    All five steps are done exactly once per document regardless of how many
    questions reference that document.
    """
    global DOC_CACHE
    unique = sorted(set(pdf_names))
    print(f"[cache] pre-building for {len(unique)} unique document(s) ...")
    for pdf_name in unique:
        if not pdf_name:
            continue
        try:
            structure         = load_structure(pdf_name)
            documents, doc_id = build_documents(pdf_name, structure)
            tree_nodes        = documents[doc_id]["structure"]

            # Serialise CLEAN tree before mutation (tutorial: tree-search/README.md)
            tree_json = json.dumps(tree_nodes, indent=2)

            # Mutate in place — prefix_summaries + node index for internal use only
            add_prefix_summaries(tree_nodes)
            node_index = build_node_index(tree_nodes)

            DOC_CACHE[pdf_name] = {
                "documents":  documents,
                "doc_id":     doc_id,
                "tree_json":  tree_json,
                "node_index": node_index,
            }
            print(f"  [cache] ✓ {pdf_name}  "
                  f"({len(node_index)} nodes, "
                  f"{len(tree_json):,} chars)")
        except Exception as e:
            print(f"  [cache] ✗ {pdf_name} — {e}  (will retry per-question)")
    print(f"[cache] ready — {len(DOC_CACHE)}/{len(unique)} documents cached")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def tree_search(query: str, tree_structure_json: str,
               preference: str | None = None) -> list:
    """
    Step 2 — LLM identifies relevant node_ids from the tree.
    Returns only node_id strings — page numbers are resolved programmatically.

    Base prompt matches the official PageIndex prompt (tutorial: tree-search/README.md).
    When preference is provided, uses the Enhanced Tree Search with Expert Preference
    prompt from the same tutorial.

    Args:
      query               : the user question
      tree_structure_json : clean serialised PageIndex tree (no prefix_summary)
      preference          : optional domain hint from get_domain_preference()
    """
    if preference:
        # Tutorial: tree-search/README.md — 'Enhanced Tree Search with Expert Preference'
        prompt = f"""You are given a question and a tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {tree_structure_json}

Expert Knowledge of relevant sections: {preference}

Reply in the following JSON format:
{{
  "thinking": "<your reasoning about which nodes are relevant>",
  "node_list": [node_id1, node_id2, ...]
}}"""
    else:
        # Tutorial: tree-search/README.md — basic LLM tree search
        prompt = f"""You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {tree_structure_json}

Reply in the following JSON format:
{{
  "thinking": "<your reasoning about which nodes are relevant>",
  "node_list": [node_id1, node_id2, ...]
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("node_list", [])


def generate_answer(query: str, page_contents: list[dict]) -> str:
    """
    Step 4 — Generate final answer from extracted page content.
    """
    context = "\n\n".join(
        f"[Page {p['page']}]\n{p['content']}"
        for p in page_contents if p.get("content")
    )
    prompt = f"""Answer the following question using only the provided context.
Be precise and cite the page number when possible.

Question: {query}

Context:
{context}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# EVALUATION
# =============================================================================

def check_retrieval_overlap(retrieved_nodes: list[dict],
                            start_page: int, end_page: int) -> dict:
    """
    Step 5a — No LLM needed.
    Compares retrieved page ranges against the gold start_page/end_page.
    Returns: hit flag, recall, precision, F1, overlapping pages.
    """
    gold_pages = set(range(start_page, end_page + 1))

    retrieved_pages = set()
    for node in retrieved_nodes:
        s = node.get("start_index")
        e = node.get("end_index")
        if s is not None and e is not None:
            retrieved_pages.update(range(s, e + 1))

    overlap   = gold_pages & retrieved_pages
    recall    = round(len(overlap) / len(gold_pages),      2) if gold_pages      else 0.0
    precision = round(len(overlap) / len(retrieved_pages), 2) if retrieved_pages else 0.0
    f1        = round(
        2 * precision * recall / (precision + recall), 2
    ) if (precision + recall) > 0 else 0.0

    return {
        "gold_pages":      sorted(gold_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "overlap_pages":   sorted(overlap),
        "retrieval_hit":   len(overlap) > 0,
        "recall":          recall,
        "precision":       precision,
        "f1":              f1,
    }


def check_evidence_recall(page_contents: list[dict], evidence_snippets: list) -> dict:
    """
    Step 5a-ii — No LLM needed.
    Checks how many gold evidence_snippets appear (substring match) in the
    full retrieved page text. This is the most direct retrieval quality signal:
    did we actually retrieve the text that answers the question?

    Returns:
      total_snippets   : number of gold snippets in the question
      matched_snippets : how many were found in retrieved content
      evidence_recall  : matched / total  (None if no snippets — excluded from avg)
      no_snippets      : True when evidence_snippets is empty
    """
    if not evidence_snippets:
        return {
            "total_snippets":   0,
            "matched_snippets": 0,
            "evidence_recall":  None,
            "no_snippets":      True,
        }

    # Build one normalised string from all retrieved page content
    full_text = " ".join(
        p.get("content", "") for p in page_contents if p.get("content")
    ).lower()
    norm_fulltext = re.sub(r"\s+", "", full_text)
    
    matched = 0
    for snippet in evidence_snippets:
        # Normalise whitespace + lowercase for robust substring match
        normalised    = re.sub(r"\s+", "", snippet.strip()).lower()
        norm_fulltext = re.sub(r"\s+", "", full_text).lower()
        if normalised and normalised in norm_fulltext:
            matched += 1

    total = len(evidence_snippets)
    return {
        "total_snippets":   total,
        "matched_snippets": matched,
        "evidence_recall":  round(matched / total, 4),
        "no_snippets":      False,
    }


def llm_judge(question: str, ground_truth: str, generated_answer: str,
              evidence_snippets: list, source_document: str) -> dict:
    """
    Step 5b — LLM as judge.
    Evaluates the generated answer against ground truth and gold evidence snippets.
    Scores: correctness, completeness, hallucination, verdict.
    """
    snippets_text = "\n".join(f"- {s}" for s in evidence_snippets) if evidence_snippets else "N/A"

    prompt = f"""You are an expert evaluator for a RAG system.

Document: {source_document}

Question: {question}

Ground Truth Answer:
{ground_truth}

Gold Evidence Snippets (from the source document):
{snippets_text}

Generated Answer:
{generated_answer}

Evaluate on these three criteria:
1. Factual correctness — does the generated answer convey the same facts as the ground truth?
2. Completeness — does it cover all key points in the ground truth?
3. Hallucination — does it add facts not supported by the ground truth or evidence?

Reply ONLY in this JSON format with no extra text:
{{
  "verdict": "correct" | "partial" | "incorrect",
  "correctness_score": <float 0.0 to 1.0>,
  "completeness_score": <float 0.0 to 1.0>,
  "hallucination": "none" | "minor" | "major",
  "reasoning": "<brief explanation of your scores>"
}}"""

    # Use the independent judge client so the judge model is never the same
    # call path as the generator — avoids self-evaluation bias.
    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PER-QUESTION PIPELINE
# =============================================================================

def process_question(q: dict, index: int, total: int) -> dict:
    """
    Run the full pipeline for a single question.
    Called directly (sequential) or from a ThreadPoolExecutor (parallel).
    """
    qid               = q.get("id", f"q{index:03d}")
    query             = q.get("question", "")
    pdf_name          = q.get("source_document", "")
    ground_truth      = q.get("answer", "")
    evidence_snippets = q.get("evidence_snippets", [])
    page_reference    = q.get("page_reference", "")
    difficulty        = q.get("difficulty", "")
    question_type     = q.get("question_type", "")

    start_page, end_page = parse_page_reference(page_reference)

    print(f"\n[{index}/{total}] {qid} — {query[:80]}...")

    if not query or not pdf_name:
        print(f"  ✗ SKIP: missing 'question' or 'source_document'")
        return {
            "id":              qid,
            "question":        query,
            "source_document": pdf_name,
            "question_type":   question_type,
            "difficulty":      difficulty,
            "page_reference":  page_reference,
            "status":          "skipped",
            "error":           "missing 'question' or 'source_document'",
        }

    try:
        # ── Step 1: Resolve from cache (built once at startup) ────────────────
        if pdf_name in DOC_CACHE:
            cached     = DOC_CACHE[pdf_name]
            documents  = cached["documents"]
            doc_id     = cached["doc_id"]
            tree_json  = cached["tree_json"]
            node_index = cached["node_index"]
        else:
            # Cache miss — fall back to per-question build (e.g. cache failed at startup)
            print(f"  [cache] MISS for {pdf_name} — building on the fly")
            structure         = load_structure(pdf_name)
            documents, doc_id = build_documents(pdf_name, structure)
            tree_nodes        = documents[doc_id]["structure"]
            tree_json         = json.dumps(tree_nodes, indent=2)
            add_prefix_summaries(tree_nodes)
            node_index        = build_node_index(tree_nodes)

        # ── Step 2a: Domain preference lookup (optional) ─────────────────────
        preference = get_domain_preference(query, args.domain)
        if preference:
            print(f"  → [preference] domain={args.domain} hint injected into tree search")

        # ── Step 2b: Tree search — LLM returns node_ids only ──────────────────
        node_ids  = call_with_retry(tree_search, query, tree_json, preference)
        print(f"  → {len(node_ids)} node_id(s) returned by LLM")

        # ── Step 2b: Resolve node_ids → page ranges from tree (no LLM) ────────
        relevant_nodes = resolve_nodes(node_ids, node_index)
        print(f"  → {len(relevant_nodes)} node(s) resolved from tree index")

        # ── Step 3: Extract page content ──────────────────────────────────────
        page_range = get_page_range_string(relevant_nodes)
        if not page_range:
            raise ValueError("Tree search returned no resolvable nodes with page ranges.")

        # Use pageindex.retrieve if available, otherwise fall back to inline extraction
        try:
            from pageindex.retrieve import get_document_structure, get_page_content
            raw_content   = get_page_content(documents, doc_id, page_range)
            page_contents = json.loads(raw_content)
        except ImportError:
            # Fallback: extract pages directly via PyPDF2
            page_contents = extract_pages_pypdf2(documents[doc_id]["path"], page_range)

        # ── Step 4: Generate answer ────────────────────────────────────────────
        answer = call_with_retry(generate_answer, query, page_contents)
        print(f"  → Answer: {answer[:120]}...")

        # ── Step 5a: Retrieval evaluation (no LLM) ────────────────────────────
        # 5a-i  Page-level overlap vs gold page_reference
        if start_page is not None and end_page is not None:
            retrieval_eval = check_retrieval_overlap(relevant_nodes, start_page, end_page)
        else:
            retrieval_eval = {
                "retrieval_hit":  None,
                "recall":         None,
                "precision":      None,
                "f1":             None,
                "page_ref_unparseable": True,
                "note": f"could not parse page range from: {page_reference!r}",
            }
        print(f"  → Retrieval hit: {retrieval_eval.get('retrieval_hit')} | "
              f"Recall: {retrieval_eval.get('recall', 'N/A')}")

        # 5a-ii Evidence-snippet recall — did retrieved text contain the gold snippets?
        # Pure substring check: no LLM involved.
        evidence_recall_result = check_evidence_recall(page_contents, evidence_snippets)
        print(f"  → Evidence recall: {evidence_recall_result.get('evidence_recall', 'N/A')} "
              f"({evidence_recall_result.get('matched_snippets', 0)}/"
              f"{evidence_recall_result.get('total_snippets', 0)} snippets matched)")

        # ── Step 5b: LLM judge ────────────────────────────────────────────────
        evaluation = call_with_retry(
            llm_judge,
            question          = query,
            ground_truth      = ground_truth,
            generated_answer  = answer,
            evidence_snippets = evidence_snippets,
            source_document   = pdf_name,
        )
        print(f"  → Verdict: {evaluation.get('verdict')} | "
              f"Correctness: {evaluation.get('correctness_score')}")

        return {
            "id":                   qid,
            "question":             query,
            "question_type":        question_type,
            "difficulty":           difficulty,
            "source_document":      pdf_name,
            "page_reference":       page_reference,
            "gold_start_page":      start_page,
            "gold_end_page":        end_page,
            "retrieved_nodes":      relevant_nodes,
            "pages_used":           page_range,
            "answer":               answer,
            "ground_truth":         ground_truth,
            "evidence_snippets":    evidence_snippets,
            "retrieval_eval":       retrieval_eval,
            "evidence_recall_eval": evidence_recall_result,
            "evaluation":           evaluation,
            "status":               "success",
        }

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return {
            "id":               qid,
            "question":         query,
            "question_type":    question_type,
            "difficulty":       difficulty,
            "source_document":  pdf_name,
            "page_reference":   page_reference,
            "gold_start_page":  start_page,
            "gold_end_page":    end_page,
            "answer":           "",
            "ground_truth":     ground_truth,
            "status":           "error",
            "error":            str(e),
        }




def process_question_infer(q: dict, index: int, total: int,
                           doc_registry: list[dict]) -> dict:
    """
    INFER MODE — process a single question without GT.
    1. Doc-selection via description-based LLM (tutorial: doc-search/description.md)
    2. Tree search with optional domain preference
    3. Page extraction + answer generation
    No metrics computed — no GT available.
    """
    qid   = q.get("id", f"q{index:03d}")
    query = q.get("question", "")

    print(f"\n[{index}/{total}] {qid} — {query[:80]}...")

    if not query:
        return {"id": qid, "question": query, "status": "skipped",
                "error": "missing 'question'"}

    try:
        # ── Doc-selection (tutorial: doc-search/description.md) ───────────────
        selected_doc_ids = call_with_retry(doc_selection_infer, query, doc_registry)
        if not selected_doc_ids:
            raise ValueError("Doc-selection returned no relevant documents for query.")

        # Process first selected document (highest relevance)
        doc_id   = selected_doc_ids[0]
        pdf_name = doc_id + ".pdf"

        # Resolve from cache (built once at startup)
        if pdf_name in DOC_CACHE:
            cached     = DOC_CACHE[pdf_name]
            documents  = cached["documents"]
            doc_id     = cached["doc_id"]
            tree_json  = cached["tree_json"]
            node_index = cached["node_index"]
        else:
            print(f"  [cache] MISS for {pdf_name} — building on the fly")
            structure         = load_structure(pdf_name)
            documents, doc_id = build_documents(pdf_name, structure)
            tree_nodes        = documents[doc_id]["structure"]
            tree_json         = json.dumps(tree_nodes, indent=2)
            add_prefix_summaries(tree_nodes)
            node_index        = build_node_index(tree_nodes)

        # ── Domain preference ─────────────────────────────────────────────────
        preference = get_domain_preference(query, args.domain)
        if preference:
            print(f"  → [preference] domain={args.domain} hint injected")

        # ── Tree search ───────────────────────────────────────────────────────
        node_ids       = call_with_retry(tree_search, query, tree_json, preference)
        relevant_nodes = resolve_nodes(node_ids, node_index)
        page_range     = get_page_range_string(relevant_nodes)

        if not page_range:
            raise ValueError("Tree search returned no resolvable nodes.")

        try:
            from pageindex.retrieve import get_page_content
            page_contents = json.loads(get_page_content(documents, doc_id, page_range))
        except ImportError:
            page_contents = extract_pages_pypdf2(documents[doc_id]["path"], page_range)

        # ── Answer generation ─────────────────────────────────────────────────
        answer = call_with_retry(generate_answer, query, page_contents)
        print(f"  → Answer: {answer[:120]}...")

        return {
            "id":              qid,
            "question":        query,
            "source_document": pdf_name,
            "selected_docs":   selected_doc_ids,
            "retrieved_nodes": relevant_nodes,
            "pages_used":      page_range,
            "answer":          answer,
            "status":          "success",
        }

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return {"id": qid, "question": query, "status": "error", "error": str(e)}

# =============================================================================
# FALLBACK PAGE EXTRACTOR (when pageindex.retrieve is not importable)
# =============================================================================

def extract_pages_pypdf2(pdf_path: str, page_range: str) -> list[dict]:
    """
    Fallback page extractor using PyPDF2 when pageindex.retrieve is unavailable.
    page_range: compact string like "5-7,12,15-17" (1-indexed)
    Returns list of {'page': int, 'content': str}
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF extraction. pip install PyPDF2")

    page_nums = set()
    for part in page_range.split(","):
        part = part.strip()
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            page_nums.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            page_nums.add(int(part))

    results = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for pg in sorted(page_nums):
            idx = pg - 1   # PyPDF2 is 0-indexed
            if 0 <= idx < len(reader.pages):
                text = reader.pages[idx].extract_text() or ""
                results.append({"page": pg, "content": text.strip()})
    return results


# =============================================================================
# METRICS AGGREGATION
# =============================================================================

def compute_metrics_summary(results: list[dict], dataset_info: dict) -> dict:
    """
    Aggregate per-question results into clean, correct summary metrics.

    Retrieval metrics (page-level):
      - Only computed over questions where page_reference was parseable.
      - retrieval_metrics_excluded tracks how many were skipped.

    Evidence recall:
      - Only averaged over questions that actually have evidence_snippets.
      - evidence_recall_excluded tracks how many had no snippets.

    Answer quality (accuracy):
      - Denominator is total questions (not just successful) so errors
        count against the score — no inflation from excluding failures.

    Breakdown by question_type and difficulty included.
    """
    total      = len(results)
    successful = [r for r in results if r["status"] == "success"]
    n_success  = len(successful)

    # ── Retrieval page-level metrics ──────────────────────────────────────────
    # Only include questions where page_reference was parseable (recall/prec/f1 exist)
    ret_evaluable = [
        r for r in successful
        if isinstance(r.get("retrieval_eval", {}).get("recall"), float)
    ]
    ret_excluded = n_success - len(ret_evaluable)

    ret_hits = sum(
        1 for r in ret_evaluable
        if r["retrieval_eval"].get("retrieval_hit")
    )
    avg_ret_recall    = (round(sum(r["retrieval_eval"]["recall"]    for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None)
    avg_ret_precision = (round(sum(r["retrieval_eval"]["precision"] for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None)
    avg_ret_f1        = (round(sum(r["retrieval_eval"]["f1"]        for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None)

    # ── Evidence-snippet recall ───────────────────────────────────────────────
    # Only average over questions that have snippets (no_snippets=False)
    ev_evaluable = [
        r for r in successful
        if isinstance(r.get("evidence_recall_eval", {}).get("evidence_recall"), float)
    ]
    ev_excluded = n_success - len(ev_evaluable)
    avg_evidence_recall = (
        round(sum(r["evidence_recall_eval"]["evidence_recall"] for r in ev_evaluable) / max(len(ev_evaluable), 1), 4)
        if ev_evaluable else None
    )

    # ── Answer quality (LLM judge) ────────────────────────────────────────────
    correct   = sum(1 for r in successful if r.get("evaluation", {}).get("verdict") == "correct")
    partial   = sum(1 for r in successful if r.get("evaluation", {}).get("verdict") == "partial")
    incorrect = n_success - correct - partial

    # Accuracy denominator = total questions (errors count as wrong — no inflation)
    accuracy = round(correct / max(total, 1), 4)

    judge_evaluable = [r for r in successful if r.get("evaluation", {}).get("correctness_score") is not None]
    avg_correctness  = (round(sum(r["evaluation"]["correctness_score"]  for r in judge_evaluable) / max(len(judge_evaluable), 1), 4) if judge_evaluable else None)
    avg_completeness = (round(sum(r["evaluation"]["completeness_score"] for r in judge_evaluable) / max(len(judge_evaluable), 1), 4) if judge_evaluable else None)

    hallucination_counts = {"none": 0, "minor": 0, "major": 0}
    for r in successful:
        h = r.get("evaluation", {}).get("hallucination", "")
        if h in hallucination_counts:
            hallucination_counts[h] += 1

    # ── Breakdown by question_type ────────────────────────────────────────────
    def make_breakdown(results_list: list, group_key: str) -> dict:
        groups: dict = {}
        for r in results_list:
            key     = r.get(group_key) or "unknown"
            verdict = r.get("evaluation", {}).get("verdict", "unknown")
            if key not in groups:
                groups[key] = {
                    "total": 0, "correct": 0, "partial": 0, "incorrect": 0,
                    "retrieval_hits": 0, "retrieval_evaluable": 0,
                    "evidence_evaluable": 0,
                    "_correctness_sum": 0.0, "_recall_sum": 0.0,
                    "_ev_recall_sum": 0.0,
                }
            g = groups[key]
            g["total"] += 1
            if verdict in ("correct", "partial", "incorrect"):
                g[verdict] += 1
            # retrieval
            ret = r.get("retrieval_eval", {})
            if isinstance(ret.get("recall"), float):
                g["retrieval_evaluable"] += 1
                g["_recall_sum"] += ret["recall"]
                if ret.get("retrieval_hit"):
                    g["retrieval_hits"] += 1
            # evidence
            ev = r.get("evidence_recall_eval", {})
            if isinstance(ev.get("evidence_recall"), float):
                g["evidence_evaluable"] += 1
                g["_ev_recall_sum"] += ev["evidence_recall"]
            # correctness
            cs = r.get("evaluation", {}).get("correctness_score")
            if isinstance(cs, float):
                g["_correctness_sum"] += cs

        # Finalise — compute rates, delete accumulators
        for key, g in groups.items():
            g["accuracy"]           = round(g["correct"] / max(g["total"], 1), 4)
            g["retrieval_hit_rate"] = round(g["retrieval_hits"] / max(g["retrieval_evaluable"], 1), 4) if g["retrieval_evaluable"] else None
            g["avg_retrieval_recall"]    = round(g["_recall_sum"]    / max(g["retrieval_evaluable"], 1), 4) if g["retrieval_evaluable"] else None
            g["avg_evidence_recall"]     = round(g["_ev_recall_sum"] / max(g["evidence_evaluable"],  1), 4) if g["evidence_evaluable"]  else None
            g["avg_correctness"]         = round(g["_correctness_sum"] / max(g["total"], 1), 4)
            del g["_correctness_sum"], g["_recall_sum"], g["_ev_recall_sum"]
        return groups

    return {
        "dataset_info": dataset_info,
        "total":        total,
        "successful":   n_success,
        "errors":       total - n_success,
        "summary": {
            # ── Retrieval (page-level) ──────────────────────────────────────
            "retrieval_evaluable":         len(ret_evaluable),
            "retrieval_metrics_excluded":  ret_excluded,
            "retrieval_hits":              ret_hits,
            "retrieval_hit_rate":          round(ret_hits / max(len(ret_evaluable), 1), 4) if ret_evaluable else None,
            "avg_retrieval_recall":        avg_ret_recall,
            "avg_retrieval_precision":     avg_ret_precision,
            "avg_retrieval_f1":            avg_ret_f1,
            # ── Evidence recall ─────────────────────────────────────────────
            "evidence_evaluable":          len(ev_evaluable),
            "evidence_recall_excluded":    ev_excluded,
            "avg_evidence_recall":         avg_evidence_recall,
            # ── Answer quality ──────────────────────────────────────────────
            # accuracy = correct / TOTAL (errors count as wrong — no inflation)
            "correct":                     correct,
            "partial":                     partial,
            "incorrect":                   incorrect,
            "accuracy":                    accuracy,
            "avg_correctness_score":       avg_correctness,
            "avg_completeness_score":      avg_completeness,
            "hallucination_counts":        hallucination_counts,
        },
        "breakdown_by_question_type": make_breakdown(successful, "question_type"),
        "breakdown_by_difficulty":    make_breakdown(successful, "difficulty"),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    global args, client, MODEL, judge_client, JUDGE_MODEL, MAX_RETRIES, RETRY_BACKOFF

    args = parse_args()

    # ── Load environment variables ────────────────────────────────────────────
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        print(f"[env] loaded {args.env_file}")
    else:
        print(f"[env] .env file not found at {args.env_file} — using system env vars")

    # ── Add PageIndex repo to sys.path + report which extractor will be used ──
    if args.pageindex_repo:
        sys.path.insert(0, args.pageindex_repo)
        # Verify the import actually works so user knows at startup, not mid-run
        try:
            from pageindex.retrieve import get_page_content  # noqa: F401
            print(f"[extractor] pageindex.retrieve  ← from repo: {args.pageindex_repo}")
        except ImportError as ie:
            print(f"[extractor] pageindex.retrieve NOT found in {args.pageindex_repo} "
                  f"({ie}) — falling back to PyPDF2")
    else:
        print("[extractor] PyPDF2 (fallback)  ← set --pageindex_repo to use "
              "pageindex.retrieve instead")

    # ── Set globals ───────────────────────────────────────────────────────────
    MAX_RETRIES   = args.max_retries
    RETRY_BACKOFF = args.retry_backoff

    # ── Setup generation LLM client ───────────────────────────────────────────
    print("[generation]")
    client, MODEL = setup_llm_client(args.provider, args.model)

    # ── Setup judge LLM client (independent from generator) ───────────────────
    # Falls back to same provider/model as generator if --judge_provider not given.
    judge_provider = args.judge_provider or args.provider
    judge_model_override = args.judge_model or args.model
    print("[judge]")
    judge_client, JUDGE_MODEL = setup_llm_client(judge_provider, judge_model_override)
    if judge_provider == args.provider and JUDGE_MODEL == MODEL:
        print("  [WARN] judge model == generation model — consider using --judge_provider "
              "/ --judge_model with a different model to avoid self-evaluation bias.")

    # ── Validate required files/dirs ─────────────────────────────────────────
    if not os.path.exists(args.query):
        raise FileNotFoundError(f"Questions file not found: {args.query}")

    # Tree input validation
    if args.tree_file and not os.path.isfile(args.tree_file):
        raise FileNotFoundError(f"Tree file not found: {args.tree_file}")
    if args.tree_dir and not os.path.isdir(args.tree_dir):
        raise FileNotFoundError(f"Tree dir not found: {args.tree_dir}")

    # PDF input validation
    if args.pdf_file and not os.path.isfile(args.pdf_file):
        raise FileNotFoundError(f"PDF file not found: {args.pdf_file}")
    if args.pdf_dir and not os.path.isdir(args.pdf_dir):
        raise FileNotFoundError(f"PDF dir not found: {args.pdf_dir}")

    # MD validation
    if args.use_md and not args.md_dir:
        raise ValueError("--use_md requires --md_dir to be set")
    if args.use_md and args.md_dir and not os.path.isdir(args.md_dir):
        raise FileNotFoundError(f"MD dir not found: {args.md_dir}")

    # Log active mode clearly
    tree_mode = f"single file: {args.tree_file}" if args.tree_file else f"dir: {args.tree_dir}"
    pdf_mode  = f"single file: {args.pdf_file}"  if args.pdf_file  else f"dir: {args.pdf_dir}"
    print(f"[tree] {tree_mode}")
    print(f"[pdf]  {pdf_mode}")
    if args.use_md:
        print(f"[md]   {args.md_dir}")

    # ── Load questions ────────────────────────────────────────────────────────
    with open(args.query, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        questions    = data
        dataset_info = {}
    else:
        questions    = data.get("questions", [])
        dataset_info = data.get("dataset_info", {})

    total = len(questions)
    print(f"\n[mode] {args.mode.upper()}")
    print(f"Loaded {total} questions from {args.query}")
    if dataset_info:
        print(f"Dataset info: {dataset_info}")
    if args.domain and args.domain != "none":
        print(f"[domain] {args.domain} — preference injection enabled")

    # ── Pre-build document cache ──────────────────────────────────────────────
    # Collect all unique source_document values and build the cache once.
    # In eval mode: from the questions JSON.
    # In infer mode: from the tree_dir registry (done inside the infer block).
    if args.mode == "eval":
        pdf_names = [q.get("source_document", "") for q in questions]
        build_doc_cache(pdf_names)

    # ── Parallel vs sequential ────────────────────────────────────────────────
    workers = None
    if args.parallel and args.parallel > 1:
        workers = args.parallel
        print(f"[parallel] workers: {workers}")
    else:
        print(f"[sequential] processing {total} questions one at a time")

    results_map: dict[int, dict] = {}

    # ── INFER MODE ────────────────────────────────────────────────────────────
    if args.mode == "infer":
        # Build doc registry once from tree_dir for description-based selection
        # (tutorial: doc-search/description.md)
        if not args.tree_dir:
            raise ValueError("--mode infer requires --tree_dir "
                             "(used to build doc description registry)")
        doc_registry = build_doc_registry(args.tree_dir)
        if not doc_registry:
            raise ValueError("No *_structure.json files found in tree_dir — "
                             "cannot build doc registry for infer mode.")

        # Pre-build cache for all documents in the registry
        build_doc_cache([entry["doc_name"] for entry in doc_registry])

        if workers:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_question_infer, q, i, total, doc_registry
                    ): i
                    for i, q in enumerate(questions, 1)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        results_map[i] = future.result()
                    except Exception as e:
                        q = questions[i - 1]
                        results_map[i] = {
                            "id": q.get("id", f"q{i:03d}"),
                            "status": "error", "error": str(e),
                        }
        else:
            for i, q in enumerate(questions, 1):
                results_map[i] = process_question_infer(q, i, total, doc_registry)
                time.sleep(args.sleep)

        results = [results_map[i] for i in range(1, total + 1)]

        # Save infer results — no metrics
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "infer_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"total": total, "results": results}, f,
                      indent=2, ensure_ascii=False)

        success = sum(1 for r in results if r["status"] == "success")
        print(f"\n{'='*65}")
        print(f"✅ Infer done — {success}/{total} answered  "
              f"({total - success} errors)")
        print(f"   Results  → {results_path}")
        print(f"{'='*65}")
        return

    # ── EVAL MODE ─────────────────────────────────────────────────────────────
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_question, q, i, total): i
                for i, q in enumerate(questions, 1)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results_map[i] = future.result()
                except Exception as e:
                    q = questions[i - 1]
                    results_map[i] = {
                        "id":     q.get("id", f"q{i:03d}"),
                        "status": "error",
                        "error":  str(e),
                    }
    else:
        for i, q in enumerate(questions, 1):
            results_map[i] = process_question(q, i, total)
            time.sleep(args.sleep)

    # ── Reassemble in original order ──────────────────────────────────────────
    results = [results_map[i] for i in range(1, total + 1)]

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics_summary(results, dataset_info)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "results.json")
    metrics_path = os.path.join(args.output_dir, "metrics_summary.json")

    full_output = {**metrics, "results": results}
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    s = metrics["summary"]
    print(f"\n{'='*65}")
    print(f"✅ Eval done — {metrics['successful']}/{total} successful  "
          f"({metrics['errors']} errors)")
    print(f"")
    print(f"   [Retrieval — page level]")
    print(f"   Evaluable (parseable page_ref) : {s['retrieval_evaluable']}  "
          f"(excluded: {s['retrieval_metrics_excluded']})")
    print(f"   Retrieval hits                 : "
          f"{s['retrieval_hits']}/{s['retrieval_evaluable']}")
    print(f"   Avg recall / prec / F1         : "
          f"{s['avg_retrieval_recall']} / "
          f"{s['avg_retrieval_precision']} / "
          f"{s['avg_retrieval_f1']}")
    print(f"")
    print(f"   [Retrieval — evidence snippets]")
    print(f"   Evaluable (has snippets)       : {s['evidence_evaluable']}  "
          f"(excluded: {s['evidence_recall_excluded']})")
    print(f"   Avg evidence recall            : {s['avg_evidence_recall']}")
    print(f"")
    print(f"   [Answer quality — LLM judge]")
    print(f"   Correct / Partial / Incorrect  : "
          f"{s['correct']} / {s['partial']} / {s['incorrect']}")
    print(f"   Accuracy (correct/total)       : {s['accuracy']}")
    print(f"   Avg correctness score          : {s['avg_correctness_score']}")
    print(f"   Avg completeness score         : {s['avg_completeness_score']}")
    print(f"   Hallucination                  : {s['hallucination_counts']}")
    print(f"")
    print(f"   Results  → {results_path}")
    print(f"   Metrics  → {metrics_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_pipeline()