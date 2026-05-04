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
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sentence_transformers import SentenceTransformer
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
                        choices=["openai", "nvidia", "ollama", "vllm"],
                        help="LLM backend for tree search + answer generation")
    parser.add_argument("--model",      default=None,
                        help="Generation model name override. Defaults: "
                             "openai=gpt-4.1, nvidia=moonshotai/kimi-k2-instruct-0905, "
                             "ollama=llama3.1:8b, vllm=Qwen/Qwen2.5-72B-Instruct-AWQ")

    # ── Judge LLM provider (independent from generation) ─────────────────────
    # A separate model for judging avoids self-evaluation bias.
    # If omitted, falls back to --provider / --model (same as generator).
    parser.add_argument("--judge_provider", default=None,
                        choices=["openai", "nvidia", "ollama", "vllm"],
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
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Maximum number of questions to process (default: all)")
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

    # ── Embedding backend ─────────────────────────────────────────────────────
    parser.add_argument("--embed_backend", default="sentence_transformer",
                        choices=["sentence_transformer", "ollama"],
                        help="Embedding backend for context_recall and context_precision. "
                             "sentence_transformer (default): BAAI/bge-small-en-v1.5 runs "
                             "locally via sentence-transformers, no extra server needed. "
                             "ollama: nomic-embed-text:latest via local Ollama HTTP API.")

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

# ── Embedding backend config ──────────────────────────────────────────────────
# Two backends are supported, selected at runtime via --embed_backend:
#   sentence_transformer  (default) — BAAI/bge-small-en-v1.5, runs locally, no server needed
#   ollama                          — nomic-embed-text:latest via local Ollama HTTP API
OLLAMA_EMBED_URL    = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_MODEL  = "nomic-embed-text:latest"
ST_EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
_st_model           = None   # lazy-loaded SentenceTransformer instance (set in run_pipeline)

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
        "model":    "meta/llama-3.3-70b-instruct",
        # "base_url": "http://localhost:11434/v1",
        "base_url": "http://127.0.0.1:8111/v1",
        "key_env":  None,   # Ollama doesn't need a real key
    },
    "vllm": {
        "model":    "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "base_url": "http://localhost:8011/v1",
        "key_env":  None,   # vLLM local server — no API key required
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

    if provider in ("ollama", "vllm"):
        api_key = "vllm" if provider == "vllm" else "ollama"
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
                "tree_nodes": tree_nodes,   # kept for hierarchical search drilldown
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

def _tree_search_one_level(query: str, nodes: list, preference: str | None,
                           depth: int) -> list:
    """
    Single LLM call for one level of the hierarchical tree search.
    Sends only the nodes at the current level (title + summary + node_id).
    Children are stripped so the prompt stays small.

    Returns a list of node_id strings the LLM selected as relevant.
    """
    # Build a lean representation — no children, no prefix_summary
    slim = [
        {
            "node_id": n["node_id"],
            "title":   n.get("title", ""),
            "summary": n.get("summary", ""),
        }
        for n in nodes
    ]
    level_json = json.dumps(slim, indent=2)

    pref_line = (f"\nExpert Knowledge of relevant sections: {preference}\n"
                 if preference else "")

    prompt = f"""You are navigating a document tree to answer a question.
You are at depth {depth} of the tree. Select ALL nodes at this level that are
likely to contain — or lead to — the answer. Be inclusive: if unsure, include it.

Query: {query}
{pref_line}
Nodes at this level:
{level_json}

Reply ONLY in this JSON format:
{{
  "thinking": "<brief reasoning>",
  "node_list": [node_id1, node_id2, ...]
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    selected = result.get("node_list", [])
    return [str(nid) for nid in selected]


def tree_search(query: str, tree_structure_json: str,
                preference: str | None = None,
                tree_nodes: list | None = None,
                max_depth: int = 4) -> list:
    """
    Step 2 — Hierarchical multi-stage tree search.

    Instead of sending the entire flattened tree in one prompt, this function
    drills down level by level:

      Stage 1 : send only top-level nodes  → LLM picks relevant branches
      Stage 2 : expand each selected branch's children → LLM picks again
      Stage N : repeat until leaf nodes (no children) or max_depth reached

    Benefits over flat search:
      - Far fewer tokens per call on large documents
      - LLM attention stays focused on the current branching decision
      - Naturally prunes irrelevant subtrees early

    Falls back to flat single-call search when tree_nodes is not supplied
    (e.g. cache miss path) to preserve backward compatibility.

    Args:
      query               : the user question
      tree_structure_json : full serialised tree — used only for flat fallback
      preference          : optional domain hint injected at every level
      tree_nodes          : live tree node list (with nested 'nodes' children).
                            When supplied, hierarchical search is used.
      max_depth           : safety cap on recursion depth (default 4)
    """
    # ── Flat fallback — tree_nodes not available ──────────────────────────────
    if tree_nodes is None:
        pref_line = (f"\nExpert Knowledge of relevant sections: {preference}\n"
                     if preference else "")
        prompt = f"""You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}
{pref_line}
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

    # ── Hierarchical search ───────────────────────────────────────────────────
    # Build a fast node_id → node dict so we can look up children instantly
    all_nodes_index: dict = {}
    def _index(nodes: list) -> None:
        for n in nodes:
            all_nodes_index[str(n["node_id"])] = n
            if n.get("nodes"):
                _index(n["nodes"])
    _index(tree_nodes)

    collected_leaf_ids: list[str] = []   # final answer — nodes with no children
    frontier: list = tree_nodes          # current level to present to LLM

    for depth in range(1, max_depth + 1):
        if not frontier:
            break

        selected_ids = call_with_retry(
            _tree_search_one_level, query, frontier, preference, depth
        )
        print(f"  → [tree-search depth {depth}] {len(frontier)} nodes shown, "
              f"{len(selected_ids)} selected")

        if not selected_ids:
            # LLM found nothing useful at this level — stop drilling
            break

        next_frontier: list = []
        for nid in selected_ids:
            node = all_nodes_index.get(nid)
            if node is None:
                print(f"  [WARN] hierarchical search: node_id '{nid}' not found — skipped")
                continue
            children = node.get("nodes", [])
            if children:
                # Has children — expand them at the next level
                next_frontier.extend(children)
            else:
                # Leaf node — collect directly
                collected_leaf_ids.append(nid)

        frontier = next_frontier

    # If we exhausted max_depth with remaining frontier nodes, collect them as-is
    # (treat the deepest selected non-leaf nodes as the answer)
    if frontier and not collected_leaf_ids:
        collected_leaf_ids = [str(n["node_id"]) for n in frontier]

    # Deduplicate while preserving order
    seen: set = set()
    final_ids: list[str] = []
    for nid in collected_leaf_ids:
        if nid not in seen:
            seen.add(nid)
            final_ids.append(nid)

    return final_ids


def generate_answer(query: str, page_contents: list[dict]) -> str:
    """
    Step 4 — Generate final answer from extracted page content.

    Each page dict may carry an optional 'source_doc' key (added by the
    multi-doc loop) so the LLM can distinguish same-numbered pages from
    different documents.  Single-doc pages without 'source_doc' fall back
    to the plain "[Page N]" label — backward compatible.
    """
    context_parts = []
    for p in page_contents:
        if not p.get("content"):
            continue
        doc_label = p.get("source_doc")
        if doc_label:
            header = f"[Document: {doc_label} | Page {p['page']}]"
        else:
            header = f"[Page {p['page']}]"
        context_parts.append(f"{header}\n{p['content']}")
    context = "\n\n".join(context_parts)

    prompt = f"""Answer the following question using only the provided context.
Be precise and cite the document name and page number when possible.

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
    Returns: hit flag and overlapping pages.
    """
    gold_pages = set(range(start_page, end_page + 1))

    retrieved_pages = set()
    for node in retrieved_nodes:
        s = node.get("start_index")
        e = node.get("end_index")
        if s is not None and e is not None:
            retrieved_pages.update(range(s, e + 1))

    overlap = gold_pages & retrieved_pages

    return {
        "gold_pages":      sorted(gold_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "overlap_pages":   sorted(overlap),
        "retrieval_hit":   len(overlap) > 0,
    }


def _get_embedding(text: str) -> list[float] | None:
    """
    Returns an embedding vector for the given text using the backend selected
    via --embed_backend:

      sentence_transformer (default):
        BAAI/bge-small-en-v1.5 via sentence-transformers.  Runs entirely locally,
        no extra server required.  Model is loaded once at startup into _st_model.

      ollama:
        nomic-embed-text:latest via Ollama HTTP /api/embeddings.  Requires a
        running Ollama server with the model already pulled.

    Returns the embedding list, or None on failure so callers can degrade
    gracefully (embed_failed=True) rather than crash.
    """
    global _st_model
    try:
        if args is not None and args.embed_backend == "ollama":
            # ── Ollama path ───────────────────────────────────────────────────
            resp = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            time.sleep(0.05)
            resp.raise_for_status()
            return resp.json()["embedding"]
        else:
            # ── SentenceTransformer path (default) ────────────────────────────
            if _st_model is None:
                # Lazy-load fallback — should have been loaded in run_pipeline,
                # but guard here for tests / direct function calls.
                _st_model = SentenceTransformer(ST_EMBED_MODEL)
            vec = _st_model.encode(text, normalize_embeddings=True)
            return vec.tolist()
    except Exception as e:
        print(f"  [WARN] embedding call failed ({getattr(args, 'embed_backend', 'st')}): {e}")
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Cosine similarity threshold above which a snippet is considered "present"
# in a retrieved page or context block.
#   BAAI/bge-small-en-v1.5 (sentence_transformer backend): 0.80 is a good default.
#   nomic-embed-text (ollama backend):                      0.80 also works well.
# Increase toward 0.90 to be stricter; decrease toward 0.70 to be more lenient.
EMBED_SIM_THRESHOLD = 0.70


def check_evidence_recall(page_contents: list[dict], evidence_snippets: list) -> dict:
    """
    Step 5a-ii — Semantic context recall via embeddings.

    For each gold evidence snippet, embeds the snippet and checks its cosine
    similarity against each retrieved page individually.  A snippet is counted
    as "matched" if its max similarity across all pages >= EMBED_SIM_THRESHOLD.

    Per-page comparison (not whole-context concatenation) avoids similarity
    dilution on large retrieved contexts.

    Returns:
      total_snippets        : number of gold snippets
      matched_snippets      : snippets whose max per-page similarity >= threshold
      context_recall        : matched / embeddable_total  (None if nothing embeddable)
      similarity_scores     : per-snippet max cosine similarity (None for failed embeds)
      no_snippets           : True when evidence_snippets is empty
      embed_failed          : True when NO page could be embedded (full abort)
      embed_partial_failure : True when some (not all) snippet embeds failed
      failed_snippet_embeds : count of snippet embedding failures
    """
    if not evidence_snippets:
        return {
            "total_snippets":        0,
            "matched_snippets":      0,
            "context_recall":        None,
            "similarity_scores":     [],
            "no_snippets":           True,
            "embed_failed":          False,
            "embed_partial_failure": False,
            "failed_snippet_embeds": 0,
        }

    # Embed each retrieved page individually.
    # A snippet is "matched" if its similarity to ANY single page >= threshold.
    # Symmetric with compute_context_precision; avoids whole-context dilution.
    page_vecs: list[list[float]] = []
    for p in page_contents:
        content = p.get("content", "")
        if not content:
            continue
        vec = _get_embedding(content)
        if vec is not None:
            page_vecs.append(vec)

    # If no page could be embedded there is nothing to compare against — full abort
    if not page_vecs:
        return {
            "total_snippets":        len(evidence_snippets),
            "matched_snippets":      0,
            "context_recall":        None,
            "similarity_scores":     [],
            "no_snippets":           False,
            "embed_failed":          True,
            "embed_partial_failure": False,
            "failed_snippet_embeds": 0,
        }

    matched         = 0
    failed_snippets = 0
    scores          = []
    for snippet in evidence_snippets:
        snippet_vec = _get_embedding(snippet.strip())
        if snippet_vec is None:
            # Track failure but do NOT count this snippet in the denominator —
            # scoring it as 0 would silently deflate recall toward zero.
            scores.append(None)
            failed_snippets += 1
            continue
        # Max similarity across all individual pages — not diluted by context length
        max_sim = max(_cosine_similarity(snippet_vec, pv) for pv in page_vecs)
        sim = round(max_sim, 4)
        scores.append(sim)
        if sim >= EMBED_SIM_THRESHOLD:
            matched += 1

    # Denominator = only the snippets we could actually embed.
    # Using len(evidence_snippets) would silently deflate recall toward zero
    # when some embeds fail, with embed_failed still False — a silent metric error.
    embeddable = len(evidence_snippets) - failed_snippets
    partial    = failed_snippets > 0

    return {
        "total_snippets":        len(evidence_snippets),
        "matched_snippets":      matched,
        "context_recall":        round(matched / embeddable, 4) if embeddable else None,
        "similarity_scores":     scores,
        "no_snippets":           False,
        "embed_failed":          False,
        "embed_partial_failure": partial,
        "failed_snippet_embeds": failed_snippets,
    }


def compute_context_precision(page_contents: list[dict], evidence_snippets: list) -> dict:
    """
    Step 5a-iii — Semantic context precision via nomic-embed-text embeddings.

    For each retrieved page, embeds the page content and checks its cosine
    similarity against every gold snippet.  A page is counted as "relevant"
    if it exceeds EMBED_SIM_THRESHOLD against at least one snippet.

    context_precision = relevant_pages / total_pages

    Returns:
      total_pages          : number of retrieved pages
      relevant_pages       : pages semantically similar to at least one snippet
      context_precision    : relevant_pages / total_pages  (None if no pages)
      no_snippets          : True when evidence_snippets is empty
      embed_failed         : True when Ollama embedding call failed
    """
    if not evidence_snippets:
        return {
            "total_pages":       0,
            "relevant_pages":    0,
            "context_precision": None,
            "no_snippets":       True,
            "embed_failed":      False,
        }

    total_pages = len(page_contents)
    if total_pages == 0:
        return {
            "total_pages":       0,
            "relevant_pages":    0,
            "context_precision": None,
            "no_snippets":       False,
            "embed_failed":      False,
        }

    # Embed all snippets — skip failures instead of aborting the whole metric
    snippet_vecs     = []
    failed_snippets  = 0
    for snippet in evidence_snippets:
        vec = _get_embedding(snippet.strip())
        if vec is None:
            failed_snippets += 1
            continue
        snippet_vecs.append(vec)

    # If every snippet failed there is nothing to compare against — full abort
    if not snippet_vecs:
        return {
            "total_pages":       total_pages,
            "relevant_pages":    0,
            "context_precision": None,
            "no_snippets":       False,
            "embed_failed":      True,
        }

    relevant_pages  = 0
    failed_pages    = 0
    for page in page_contents:
        content = page.get("content", "")
        if not content:
            continue
        page_vec = _get_embedding(content)
        if page_vec is None:
            # Skip this page — don't abort; continue scoring the rest
            failed_pages += 1
            continue
        # Page is relevant if it is similar to ANY gold snippet
        if any(_cosine_similarity(page_vec, sv) >= EMBED_SIM_THRESHOLD
               for sv in snippet_vecs):
            relevant_pages += 1

    partial = failed_snippets > 0 or failed_pages > 0
    return {
        "total_pages":          total_pages,
        "relevant_pages":       relevant_pages,
        "context_precision":    round(relevant_pages / total_pages, 4),
        "no_snippets":          False,
        "embed_failed":         False,
        "embed_partial_failure": partial,           # approximation flag
        "failed_snippet_embeds": failed_snippets,  # how many snippets were skipped
        "failed_page_embeds":    failed_pages,     # how many pages were skipped
    }
# Max chars for generated_answer in the judge prompt.
# RAG answers state key facts upfront — truncating tail elaboration does not
# affect correctness/completeness/hallucination scoring.
# Ground truth and evidence snippets are never truncated.
JUDGE_MAX_ANSWER_CHARS = 4000


def llm_judge(question: str, ground_truth: str, generated_answer: str,
              evidence_snippets: list, source_document: str | list) -> dict:
    """
    Step 5b — LLM as judge.
    Evaluates the generated answer against ground truth and gold evidence snippets.
    Scores: correctness, completeness, hallucination, verdict.

    source_document may be a str (single-doc) or a list of str (multi-doc).
    Both are normalised to a readable string before being inserted into the prompt
    so the judge never sees a raw Python list repr.

    generated_answer is truncated to JUDGE_MAX_ANSWER_CHARS before being inserted
    into the prompt to prevent context-limit failures on the judge model (which
    return empty responses and cause JSONDecodeError on all retry attempts).
    ground_truth and evidence_snippets are never truncated.
    """
    # Normalise source_document to a clean human-readable string for the prompt
    if isinstance(source_document, list):
        doc_label = ", ".join(source_document)
    else:
        doc_label = source_document or "unknown"

    # Truncate only the generated answer — ground truth and snippets stay intact
    safe_answer = generated_answer[:JUDGE_MAX_ANSWER_CHARS]
    if len(generated_answer) > JUDGE_MAX_ANSWER_CHARS:
        safe_answer += "\n... [truncated for judge prompt]"

    snippets_text = "\n".join(f"- {s}" for s in evidence_snippets) if evidence_snippets else "N/A"

    prompt = f"""You are an expert evaluator for a RAG system.

Document: {doc_label}

Question: {question}

Ground Truth Answer:
{ground_truth}

Gold Evidence Snippets (from the source document):
{snippets_text}

Generated Answer:
{safe_answer}

Evaluate on these three criteria:
1. Factual correctness — does the generated answer convey the same facts as the ground truth?
2. Completeness — does it cover all key points in the ground truth?
3. Hallucination — does it add facts not supported by the ground truth or evidence?

Reply ONLY in this JSON format with no extra text:
{{
  "verdict": "correct" | "incorrect",
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
    # Guard against empty response (e.g. judge model context limit still exceeded)
    raw = response.choices[0].message.content.strip()
    if not raw:
        print("  [WARN] judge returned empty response — recording as failed evaluation")
        return {
            "verdict":            "incorrect",
            "correctness_score":  0.0,
            "completeness_score": 0.0,
            "hallucination":      "none",
            "reasoning":          "judge returned empty response — evaluation failed",
        }
    return json.loads(raw)

# =============================================================================
# PER-QUESTION PIPELINE
# =============================================================================

def process_question(q: dict, index: int, total: int) -> dict:
    """
    Run the full pipeline for a single question.
    Called directly (sequential) or from a ThreadPoolExecutor (parallel).
    """
    qid               = q.get("id", f"q{index:03d}")
    # GT format uses "user_input"; legacy format uses "question" — support both
    query             = q.get("user_input") or q.get("question", "")
    # GT format: source_documents is a list of full paths — extract filename only
    # Legacy format: source_document is already a bare filename
    raw_src           = q.get("source_documents") or q.get("source_document", "")
    if isinstance(raw_src, list):
        pdf_names_list = [os.path.basename(r) for r in raw_src if r]
    else:
        pdf_names_list = [os.path.basename(raw_src)] if raw_src else []
    # Keep pdf_name as the primary doc for backward-compatible result fields
    pdf_name          = pdf_names_list[0] if pdf_names_list else ""
    # GT format uses "reference" for the gold answer; legacy uses "answer"
    ground_truth      = q.get("reference") or q.get("answer", "")
    # GT format uses "reference_contexts" for gold evidence; legacy uses "evidence_snippets"
    evidence_snippets = q.get("reference_contexts") or q.get("evidence_snippets", [])
    page_reference    = q.get("page_reference", "")   # absent in GT → gracefully None,None
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
        # ── Steps 1-3: Per-document loop — sequential, one doc at a time ──────
        # Each doc is searched and extracted independently. Results are merged
        # before answer generation so there is no cross-doc token explosion and
        # no single-prompt context limit concern.
        all_page_contents:  list[dict]       = []   # merged pages from all docs
        all_retrieved_nodes: dict[str, list] = {}   # pdf_name → resolved node list
        all_pages_used:      dict[str, str]  = {}   # pdf_name → page range string

        is_multi_doc = len(pdf_names_list) > 1
        if is_multi_doc:
            print(f"  → [multi-doc] {len(pdf_names_list)} source docs: {pdf_names_list}")

        for cur_pdf_name in pdf_names_list:
            print(f"  → [doc] processing: {cur_pdf_name}")

            # ── Step 1: Resolve from cache (built once at startup) ────────────
            if cur_pdf_name in DOC_CACHE:
                cached      = DOC_CACHE[cur_pdf_name]
                documents   = cached["documents"]
                cur_doc_id  = cached["doc_id"]
                tree_json   = cached["tree_json"]
                tree_nodes  = cached["tree_nodes"]
                node_index  = cached["node_index"]
            else:
                # Cache miss — fall back to per-question build
                print(f"  [cache] MISS for {cur_pdf_name} — building on the fly")
                structure          = load_structure(cur_pdf_name)
                documents, cur_doc_id = build_documents(cur_pdf_name, structure)
                tree_nodes         = documents[cur_doc_id]["structure"]
                tree_json          = json.dumps(tree_nodes, indent=2)
                add_prefix_summaries(tree_nodes)
                node_index         = build_node_index(tree_nodes)

            # ── Step 2a: Domain preference lookup (optional) ──────────────────
            preference = get_domain_preference(query, args.domain)
            if preference:
                print(f"  → [preference] domain={args.domain} hint injected into tree search")

            # ── Step 2b: Tree search — hierarchical multi-stage LLM drilldown ─
            node_ids = call_with_retry(tree_search, query, tree_json, preference, tree_nodes)
            print(f"  → {len(node_ids)} node_id(s) returned by tree search ({cur_pdf_name})")

            # ── Step 2c: Resolve node_ids → page ranges from tree (no LLM) ────
            relevant_nodes = resolve_nodes(node_ids, node_index)
            print(f"  → {len(relevant_nodes)} node(s) resolved from tree index ({cur_pdf_name})")

            # ── Step 3: Extract page content ──────────────────────────────────
            page_range = get_page_range_string(relevant_nodes)
            if not page_range:
                print(f"  [WARN] no resolvable page ranges for {cur_pdf_name} — skipping doc")
                continue

            try:
                from pageindex.retrieve import get_document_structure, get_page_content
                raw_content       = get_page_content(documents, cur_doc_id, page_range)
                cur_page_contents = json.loads(raw_content)
            except ImportError:
                cur_page_contents = extract_pages_pypdf2(documents[cur_doc_id]["path"], page_range)

            all_page_contents.extend(
                {**pg, "source_doc": cur_pdf_name} for pg in cur_page_contents
            )
            all_retrieved_nodes[cur_pdf_name] = relevant_nodes
            all_pages_used[cur_pdf_name]      = page_range

        if not all_page_contents:
            raise ValueError("Tree search returned no resolvable nodes across all source documents.")

        # ── Flatten result fields for single-doc backward compatibility ───────
        # Single-doc: keep the same scalar types as before so existing consumers
        # of results.json don't break. Multi-doc: use dicts keyed by pdf_name.
        if not is_multi_doc:
            retrieved_nodes_out = all_retrieved_nodes.get(pdf_name, [])
            pages_used_out      = all_pages_used.get(pdf_name, "")
            source_doc_out      = pdf_name
        else:
            retrieved_nodes_out = all_retrieved_nodes
            pages_used_out      = all_pages_used
            source_doc_out      = pdf_names_list

        # ── Step 4: Generate answer (single call over merged context) ─────────
        answer = call_with_retry(generate_answer, query, all_page_contents)
        print(f"  → Answer: {answer[:120]}...")

        # ── Step 5a: Retrieval evaluation (no LLM) ────────────────────────────
        # 5a-i  Page-level overlap vs gold page_reference
        # Multi-doc: page_reference is per-doc — skip overlap check, flag clearly.
        if is_multi_doc:
            retrieval_eval = {
                "retrieval_hit":  None,
                "multi_doc":      True,
                "note": "page-level retrieval eval skipped — multi-document question; "
                        "page_reference covers only one doc",
            }
        elif start_page is not None and end_page is not None:
            retrieval_eval = check_retrieval_overlap(retrieved_nodes_out, start_page, end_page)
        else:
            retrieval_eval = {
                "retrieval_hit":        None,
                "page_ref_unparseable": True,
                "note": f"could not parse page range from: {page_reference!r}",
            }
        print(f"  → Retrieval hit: {retrieval_eval.get('retrieval_hit')}")

        # 5a-ii Context recall — semantic similarity via embeddings
        evidence_recall_result = check_evidence_recall(all_page_contents, evidence_snippets)
        _cr_note = ("embed_failed" if evidence_recall_result.get("embed_failed")
                    else f"{evidence_recall_result.get('matched_snippets', 0)}/"
                         f"{evidence_recall_result.get('total_snippets', 0)} snippets "
                         f"sims={evidence_recall_result.get('similarity_scores', [])}")
        print(f"  → Context recall: {evidence_recall_result.get('context_recall', 'N/A')} "
              f"({_cr_note})")

        # 5a-iii Context precision — what fraction of retrieved pages are relevant?
        context_precision_result = compute_context_precision(all_page_contents, evidence_snippets)
        print(f"  → Context precision: {context_precision_result.get('context_precision', 'N/A')} "
              f"({context_precision_result.get('relevant_pages', 0)}/"
              f"{context_precision_result.get('total_pages', 0)} pages relevant)")

        # ── Step 5b: LLM judge ────────────────────────────────────────────────
        evaluation = call_with_retry(
            llm_judge,
            question          = query,
            ground_truth      = ground_truth,
            generated_answer  = answer,
            evidence_snippets = evidence_snippets,
            source_document   = source_doc_out,
        )
        print(f"  → Verdict: {evaluation.get('verdict')} | "
              f"Correctness: {evaluation.get('correctness_score')}")

        return {
            "id":                     qid,
            "question":               query,
            "question_type":          question_type,
            "difficulty":             difficulty,
            "source_document":        source_doc_out,   # str for single, list for multi
            "page_reference":         page_reference,
            "gold_start_page":        start_page,
            "gold_end_page":          end_page,
            "retrieved_nodes":        retrieved_nodes_out,  # list for single, dict for multi
            "pages_used":             pages_used_out,        # str for single, dict for multi
            "answer":                 answer,
            "ground_truth":           ground_truth,
            "evidence_snippets":      evidence_snippets,
            "retrieval_eval":         retrieval_eval,
            "context_recall_eval":    evidence_recall_result,
            "context_precision_eval": context_precision_result,
            "evaluation":             evaluation,
            "status":                 "success",
        }

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        # Use pdf_names_list if available (defined before the try block);
        # for single-doc keep a string, for multi-doc use the full list so the
        # error record accurately reflects which documents were involved.
        err_source = pdf_names_list if len(pdf_names_list) > 1 else pdf_name
        return {
            "id":               qid,
            "question":         query,
            "question_type":    question_type,
            "difficulty":       difficulty,
            "source_document":  err_source,
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
    query = q.get("user_input") or q.get("question", "")

    print(f"\n[{index}/{total}] {qid} — {query[:80]}...")

    if not query:
        return {"id": qid, "question": query, "status": "skipped",
                "error": "missing 'question'"}

    try:
        # ── Doc-selection (tutorial: doc-search/description.md) ───────────────
        selected_doc_ids = call_with_retry(doc_selection_infer, query, doc_registry)
        if not selected_doc_ids:
            raise ValueError("Doc-selection returned no relevant documents for query.")

        # ── Per-document loop — sequential, one doc at a time ────────────────
        # All selected docs are searched independently; page contents are merged
        # into one list before answer generation.
        all_page_contents:   list[dict]       = []
        all_retrieved_nodes: dict[str, list]  = {}
        all_pages_used:      dict[str, str]   = {}

        print(f"  → [multi-doc] processing {len(selected_doc_ids)} selected doc(s): {selected_doc_ids}")

        for sel_doc_id in selected_doc_ids:
            cur_pdf_name = sel_doc_id + ".pdf"
            print(f"  → [doc] {cur_pdf_name}")

            # Resolve from cache (built once at startup)
            if cur_pdf_name in DOC_CACHE:
                cached      = DOC_CACHE[cur_pdf_name]
                documents   = cached["documents"]
                cur_doc_id  = cached["doc_id"]
                tree_json   = cached["tree_json"]
                tree_nodes  = cached["tree_nodes"]
                node_index  = cached["node_index"]
            else:
                print(f"  [cache] MISS for {cur_pdf_name} — building on the fly")
                structure              = load_structure(cur_pdf_name)
                documents, cur_doc_id  = build_documents(cur_pdf_name, structure)
                tree_nodes             = documents[cur_doc_id]["structure"]
                tree_json              = json.dumps(tree_nodes, indent=2)
                add_prefix_summaries(tree_nodes)
                node_index             = build_node_index(tree_nodes)

            # ── Domain preference ─────────────────────────────────────────────
            preference = get_domain_preference(query, args.domain)
            if preference:
                print(f"  → [preference] domain={args.domain} hint injected")

            # ── Tree search — hierarchical multi-stage LLM drilldown ─────────
            node_ids       = call_with_retry(tree_search, query, tree_json, preference, tree_nodes)
            relevant_nodes = resolve_nodes(node_ids, node_index)
            page_range     = get_page_range_string(relevant_nodes)

            if not page_range:
                print(f"  [WARN] no resolvable page ranges for {cur_pdf_name} — skipping doc")
                continue

            try:
                from pageindex.retrieve import get_page_content
                cur_page_contents = json.loads(get_page_content(documents, cur_doc_id, page_range))
            except ImportError:
                cur_page_contents = extract_pages_pypdf2(documents[cur_doc_id]["path"], page_range)

            all_page_contents.extend(
                {**pg, "source_doc": cur_pdf_name} for pg in cur_page_contents
            )
            all_retrieved_nodes[cur_pdf_name] = relevant_nodes
            all_pages_used[cur_pdf_name]      = page_range

        if not all_page_contents:
            raise ValueError("No resolvable nodes across any selected document.")

        # ── Answer generation ─────────────────────────────────────────────────
        answer = call_with_retry(generate_answer, query, all_page_contents)
        print(f"  → Answer: {answer[:120]}...")

        # Normalise source_document: string for single-doc (consistent with eval
        # mode), list for multi-doc.  Avoids breaking downstream isinstance checks.
        selected_names = list(all_retrieved_nodes.keys())
        source_doc_out = selected_names[0] if len(selected_names) == 1 else selected_names

        return {
            "id":              qid,
            "question":        query,
            "source_document": source_doc_out,
            "selected_docs":   selected_doc_ids,
            "retrieved_nodes": all_retrieved_nodes,
            "pages_used":      all_pages_used,
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
      - Only hit rate computed (no recall/precision/F1 — page_reference rarely populated).

    Context recall & precision:
      - context_recall    : fraction of gold snippets found in retrieved pages.
      - context_precision : fraction of retrieved pages containing a gold snippet.
      - Both averaged only over questions that have evidence_snippets.

    Answer quality (accuracy):
      - partial verdicts are merged into correct (partial → correct).
      - Denominator is total questions so errors count against the score.
    """
    total      = len(results)
    successful = [r for r in results if r["status"] == "success"]
    n_success  = len(successful)

    # ── Retrieval page-level hit rate ─────────────────────────────────────────
    ret_evaluable = [
        r for r in successful
        if r.get("retrieval_eval", {}).get("retrieval_hit") is not None
        and not r.get("retrieval_eval", {}).get("page_ref_unparseable")
    ]
    ret_hits = sum(1 for r in ret_evaluable if r["retrieval_eval"]["retrieval_hit"])
    retrieval_hit_rate = (
        round(ret_hits / len(ret_evaluable), 4) if ret_evaluable else None
    )

    # ── Context recall ────────────────────────────────────────────────────────
    cr_evaluable = [
        r for r in successful
        if isinstance(r.get("context_recall_eval", {}).get("context_recall"), float)
    ]
    avg_context_recall = (
        round(sum(r["context_recall_eval"]["context_recall"] for r in cr_evaluable)
              / len(cr_evaluable), 4)
        if cr_evaluable else None
    )

    # ── Context precision ─────────────────────────────────────────────────────
    cp_evaluable = [
        r for r in successful
        if isinstance(r.get("context_precision_eval", {}).get("context_precision"), float)
    ]
    avg_context_precision = (
        round(sum(r["context_precision_eval"]["context_precision"] for r in cp_evaluable)
              / len(cp_evaluable), 4)
        if cp_evaluable else None
    )

    # ── Answer quality (LLM judge) ────────────────────────────────────────────
    # The judge prompt only emits "correct" | "incorrect".
    # "partial" is kept in the guard solely for backward-compatibility with
    # results.json files produced before the prompt was updated — it is never
    # emitted by the current judge and does NOT appear anywhere in the summary
    # output or print block.  Removing it would silently mis-count legacy data.
    correct = sum(
        1 for r in successful
        if r.get("evaluation", {}).get("verdict") in ("correct", "partial")
    )
    incorrect = sum(
        1 for r in successful
        if r.get("evaluation", {}).get("verdict") == "incorrect"
    )
    # Accuracy denominator = total (errors count as wrong — no inflation)
    accuracy = round(correct / max(total, 1), 4)

    judge_evaluable  = [r for r in successful if r.get("evaluation", {}).get("correctness_score") is not None]
    avg_correctness  = (round(sum(r["evaluation"]["correctness_score"]  for r in judge_evaluable) / max(len(judge_evaluable), 1), 4) if judge_evaluable else None)
    avg_completeness = (round(sum(r["evaluation"]["completeness_score"] for r in judge_evaluable) / max(len(judge_evaluable), 1), 4) if judge_evaluable else None)

    hallucination_counts = {"none": 0, "minor": 0, "major": 0}
    for r in successful:
        h = r.get("evaluation", {}).get("hallucination", "")
        if h in hallucination_counts:
            hallucination_counts[h] += 1

    return {
        "dataset_info": dataset_info,
        "total":        total,
        "successful":   n_success,
        "errors":       total - n_success,
        "summary": {
            # ── Retrieval (page-level hit rate only) ────────────────────────
            "retrieval_evaluable":  len(ret_evaluable),
            "retrieval_hits":       ret_hits,
            "retrieval_hit_rate":   retrieval_hit_rate,
            # ── Context recall & precision ───────────────────────────────────
            "context_recall_evaluable":    len(cr_evaluable),
            "avg_context_recall":          avg_context_recall,
            "context_precision_evaluable": len(cp_evaluable),
            "avg_context_precision":       avg_context_precision,
            # ── Answer quality ───────────────────────────────────────────────
            # partial is merged into correct; accuracy = correct / total
            "correct":               correct,
            "incorrect":             incorrect,
            "accuracy":              accuracy,
            "avg_correctness_score":  avg_correctness,
            "avg_completeness_score": avg_completeness,
            "hallucination_counts":   hallucination_counts,
        },
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

    # ── Setup embedding backend ───────────────────────────────────────────────
    global _st_model
    if args.embed_backend == "sentence_transformer":
        print(f"[embeddings] backend=sentence_transformer  model={ST_EMBED_MODEL}")
        try:
            _st_model = SentenceTransformer(ST_EMBED_MODEL)
            test_vec  = _get_embedding("test")
            print(f"  [embeddings] OK — vector dim={len(test_vec)}")
        except Exception as e:
            print(f"  [WARN] SentenceTransformer load failed: {e}  "
                  "context_recall/precision will show embed_failed=True.")
    else:
        print(f"[embeddings] backend=ollama  model={OLLAMA_EMBED_MODEL}  url={OLLAMA_EMBED_URL}")
        try:
            test_vec = _get_embedding("test")
            if test_vec:
                print(f"  [embeddings] OK — vector dim={len(test_vec)}")
            else:
                print("  [WARN] embedding test returned empty — "
                      "is nomic-embed-text pulled in Ollama?")
        except Exception as e:
            print(f"  [WARN] could not reach Ollama embedding endpoint: {e}")

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
    if args.max_questions is not None:
        questions = questions[:args.max_questions]
        total = len(questions)
        print(f"[max_questions] limiting to {total} questions")
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
        def _extract_pdf_names(q: dict) -> list[str]:
            raw = q.get("source_documents") or q.get("source_document", "")
            if isinstance(raw, list):
                return [os.path.basename(r) for r in raw if r]
            return [os.path.basename(raw)] if raw else []
        # Flatten — every doc referenced by every question gets cached once
        pdf_names = [name for q in questions for name in _extract_pdf_names(q)]
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
    print(f"   Evaluable (parseable page_ref) : {s['retrieval_evaluable']}")
    print(f"   Retrieval hits                 : "
          f"{s['retrieval_hits']}/{s['retrieval_evaluable']}")
    print(f"   Retrieval hit rate             : {s['retrieval_hit_rate']}")
    print(f"")
    print(f"   [RAG context quality]")
    print(f"   Context recall evaluable       : {s['context_recall_evaluable']}")
    print(f"   Avg context recall             : {s['avg_context_recall']}")
    print(f"   Context precision evaluable    : {s['context_precision_evaluable']}")
    print(f"   Avg context precision          : {s['avg_context_precision']}")
    print(f"")
    print(f"   [Answer quality — LLM judge]")
    print(f"   Correct / Incorrect            : "
          f"{s['correct']} / {s['incorrect']}")
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