#!/usr/bin/env python3
# coding: utf-8
"""
pageindex_RAG_levelrag_v2.py
============================
PageIndex RAG pipeline with exact LevelRAG paper implementation.

LevelRAG Paper Method (all 4 stages faithfully implemented):
  STAGE 1 — High-Level Query Planner
              LLM decomposes the original query into atomic sub-queries.
              Each sub-query is self-contained and independently answerable.

  STAGE 2 — Per-Sub-Query Retrieval
              For every atomic sub-query: tree_search() -> resolve nodes ->
              extract pages -> produce a partial context independently.
              Each sub-query retrieval is ISOLATED — no merging yet.

  STAGE 3 — Per-Sub-Query Partial Answer Generation
              For every atomic sub-query: generate_partial_answer() is
              called with ONLY that sub-query's retrieved pages as context.
              The LLM answers the sub-query in isolation.
              KEY DIFFERENCE from v1: LLM sees focused context per sub-query.

  STAGE 4 — Final Synthesis
              All partial answers (one per sub-query) are assembled and
              passed to synthesize_final_answer() with the ORIGINAL query.
              The synthesis LLM combines partial answers into one final answer.

Per-Step File Outputs (written on pipeline completion):
  {output_dir}/
  |-- step1_decomposition.json      <- Stage 1: sub-queries per question
  |-- step2_retrieval.json          <- Stage 2: per-sub-query nodes + pages
  |-- step3_partial_answers.json    <- Stage 3: per-sub-query partial answers
  |-- step4_final_answers.json      <- Stage 4: synthesized final answers
  |-- results.json                  <- full pipeline output (eval mode)
  |-- metrics_summary.json          <- aggregated metrics (eval mode)
  |-- infer_results.json            <- full output (infer mode)

  Live scratch files (one JSON line per question, written as each completes):
  |-- step1_decomposition.jsonl
  |-- step2_retrieval.jsonl
  |-- step3_partial_answers.jsonl
  |-- step4_final_answers.jsonl

Standard (non-LevelRAG) mode:
  When --use_level_rag is NOT passed, behaves exactly as the original
  pageindex_RAG_simple.py. All step files are still written with single-entry
  records so you can always inspect the same files regardless of mode.

Usage:
  # Eval with LevelRAG (4-stage paper method)
  python3 pageindex_RAG_levelrag_v2.py \\
      --mode eval --query /data/q.json \\
      --pdf_dir /data/pdfs/ --tree_dir /data/trees/ \\
      --provider openai --model gpt-4.1 \\
      --judge_provider openai --judge_model gpt-4.1 \\
      --use_level_rag --level_rag_max_subqueries 4 \\
      --output_dir ./results_levelrag

  # Eval standard (original single-pass behaviour)
  python3 pageindex_RAG_levelrag_v2.py \\
      --mode eval --query /data/q.json \\
      --pdf_dir /data/pdfs/ --tree_dir /data/trees/ \\
      --provider openai --output_dir ./results_standard

  # Infer with LevelRAG
  python3 pageindex_RAG_levelrag_v2.py \\
      --mode infer --query /data/user_q.json \\
      --pdf_dir /data/pdfs/ --tree_dir /data/trees/ \\
      --provider nvidia --use_level_rag
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PageIndex RAG — exact LevelRAG 4-stage paper implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--query", required=True,
                        help="Path to questions JSON file")

    tree_group = parser.add_mutually_exclusive_group(required=True)
    tree_group.add_argument("--tree_file", default=None,
                            help="Single tree JSON file (used for ALL questions).")
    tree_group.add_argument("--tree_dir",  default=None,
                            help="Directory of *_structure.json tree files.")

    pdf_group = parser.add_mutually_exclusive_group(required=True)
    pdf_group.add_argument("--pdf_file", default=None,
                           help="Single PDF file path (used for ALL questions).")
    pdf_group.add_argument("--pdf_dir",  default=None,
                           help="Directory of PDF files.")

    parser.add_argument("--md_dir",     default=None,
                        help="Directory of per-doc markdown page folders (--use_md)")
    parser.add_argument("--output_dir", default="./results",
                        help="Directory to write all output files (default: ./results)")
    parser.add_argument("--env_file",   default=".env",
                        help="Path to .env file for API keys (default: .env)")

    parser.add_argument("--provider", required=True,
                        choices=["openai", "nvidia", "ollama"],
                        help="LLM backend for tree search + answer generation")
    parser.add_argument("--model", default=None,
                        help="Generation model name override.")

    parser.add_argument("--judge_provider", default=None,
                        choices=["openai", "nvidia", "ollama"],
                        help="LLM backend for judge. Defaults to --provider.")
    parser.add_argument("--judge_model", default=None,
                        help="Judge model name override. Defaults to --model.")

    parser.add_argument("--use_md", action="store_true",
                        help="Serve page content from MD files in --md_dir")

    parser.add_argument("--parallel",      type=int,   default=1,
                        help="Parallel workers (default: 1 sequential)")
    parser.add_argument("--sleep",         type=float, default=0.5,
                        help="Sleep between questions in sequential mode (default: 0.5)")
    parser.add_argument("--max_retries",   type=int,   default=3,
                        help="Max LLM call retries (default: 3)")
    parser.add_argument("--retry_backoff", type=float, default=2.0,
                        help="Base backoff seconds for retries (default: 2.0)")

    parser.add_argument("--pageindex_repo", default=None,
                        help="Path to PageIndex repo root for pageindex.retrieve. "
                             "If omitted, PyPDF2 fallback is used.")

    parser.add_argument("--mode", default="eval", choices=["eval", "infer"],
                        help="eval: GT-based evaluation (default). "
                             "infer: real-world inference with doc-selection.")

    parser.add_argument("--domain", default=None, choices=["autosar", "none"],
                        help="Domain preference injection into tree search.")

    # LevelRAG flags
    parser.add_argument("--use_level_rag", action="store_true",
                        help="Enable exact LevelRAG 4-stage pipeline: "
                             "Stage1=decompose, Stage2=retrieve per sub-query, "
                             "Stage3=partial answer per sub-query, Stage4=synthesize.")
    parser.add_argument("--level_rag_max_subqueries", type=int, default=4,
                        help="Max atomic sub-queries LLM may produce (default: 4).")

    return parser.parse_args()


# =============================================================================
# GLOBALS
# =============================================================================

args          = None
client        = None
MODEL         = None
judge_client  = None
JUDGE_MODEL   = None
MAX_RETRIES   = 3
RETRY_BACKOFF = 2.0
DOC_CACHE: dict = {}

# LevelRAG config
USE_LEVEL_RAG: bool           = False
LEVEL_RAG_MAX_SUBQUERIES: int = 4
LEVEL_RAG_MIN_SUBQUERIES: int = 1
LEVEL_RAG_LOG: bool           = True


# =============================================================================
# STEP FILE SYSTEM
# Buffers records in memory; writes live .jsonl scratch files as each question
# completes; flushes to proper .json arrays at end of run.
# =============================================================================

_STEP_BUFFERS: dict[str, list] = {
    "step1_decomposition":    [],
    "step2_retrieval":        [],
    "step3_partial_answers":  [],
    "step4_final_answers":    [],
}
_STEP_LOCK = threading.Lock()


def _append_step(step_name: str, record: dict) -> None:
    """Thread-safe in-memory append + live .jsonl scratch write."""
    with _STEP_LOCK:
        _STEP_BUFFERS[step_name].append(record)

    if args and args.output_dir:
        scratch = os.path.join(args.output_dir, f"{step_name}.jsonl")
        try:
            with _STEP_LOCK:
                with open(scratch, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # never let logging break the pipeline


def _flush_step_files() -> None:
    """Convert all in-memory buffers to final .json arrays. Called once at end."""
    if not (args and args.output_dir):
        return
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n   [step files written]")
    for step_name, records in _STEP_BUFFERS.items():
        path = os.path.join(args.output_dir, f"{step_name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
        print(f"   {step_name + '.json':<38} ({len(records)} records)  ->  {path}")


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

PROVIDER_DEFAULTS = {
    "openai": {"model": "gpt-4.1",
               "base_url": "https://api.openai.com/v1",
               "key_env":  "OPENAI_API_KEY"},
    "nvidia": {"model": "moonshotai/kimi-k2-instruct-0905",
               "base_url": "https://integrate.api.nvidia.com/v1",
               "key_env":  "NVIDIA_API_KEY"},
    "ollama": {"model": "llama3.1:8b",
               "base_url": "http://localhost:11434/v1",
               "key_env":  None},
}


def setup_llm_client(provider: str, model_override: str | None) -> tuple:
    cfg      = PROVIDER_DEFAULTS[provider]
    model    = model_override or cfg["model"]
    base_url = os.getenv(f"{provider.upper()}_BASE_URL") or cfg["base_url"]
    if provider == "ollama":
        api_key = "ollama"
    else:
        api_key = os.getenv(cfg["key_env"])
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set {cfg['key_env']} in your .env or environment.")
    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[backend] {provider.upper()}  base_url={base_url}  model={model}")
    return llm_client, model


# =============================================================================
# FILE LOADERS
# =============================================================================

def load_structure(pdf_name: str) -> dict:
    if args.tree_file:
        if not os.path.exists(args.tree_file):
            raise FileNotFoundError(f"Tree file not found: {args.tree_file}")
        with open(args.tree_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    base = os.path.splitext(pdf_name)[0]
    path = os.path.join(args.tree_dir, f"{base}_structure.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tree structure not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_md_pages(doc_name: str, md_dir: str) -> list | None:
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
        with open(os.path.join(pages_dir, fname), "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        pages.append({"page": page_num, "content": content})
    if not pages:
        print(f"  [md] no .md files found in {pages_dir} — falling back to PDF")
        return None
    pages.sort(key=lambda x: x["page"])
    return pages


def build_documents(pdf_name: str, structure: dict) -> tuple[dict, str]:
    doc_id = os.path.splitext(pdf_name)[0]
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
            "pages":           cached_pages,
        }
    }
    return documents, doc_id


# =============================================================================
# TREE HELPERS
# =============================================================================

def add_prefix_summaries(nodes: list, parent_prefix: str = "") -> None:
    for node in nodes:
        node["prefix_summary"] = parent_prefix
        own = node.get("summary", "")
        nxt = (parent_prefix + "\n" + own).strip() if own else parent_prefix
        if node.get("nodes"):
            add_prefix_summaries(node["nodes"], nxt)


def build_node_index(nodes: list, index: dict | None = None) -> dict:
    if index is None:
        index = {}
    for node in nodes:
        index[node["node_id"]] = node
        if node.get("nodes"):
            build_node_index(node["nodes"], index)
    return index


def get_page_range_string(nodes: list[dict]) -> str:
    pages = set()
    for node in nodes:
        s = node.get("start_index")
        e = node.get("end_index")
        if s is not None and e is not None:
            pages.update(range(s, e + 1))
    if not pages:
        return ""
    sp = sorted(pages)
    ranges, rs, re_ = [], sp[0], sp[0]
    for p in sp[1:]:
        if p == re_ + 1:
            re_ = p
        else:
            ranges.append(f"{rs}-{re_}" if rs != re_ else str(rs))
            rs = re_ = p
    ranges.append(f"{rs}-{re_}" if rs != re_ else str(rs))
    return ",".join(ranges)


def resolve_nodes(node_ids: list, node_index: dict) -> list[dict]:
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
            print(f"  [WARN] node_id '{nid}' not in tree index — skipped")
    return resolved


# =============================================================================
# PAGE REFERENCE PARSER
# =============================================================================

def parse_page_reference(page_reference: str) -> tuple[int | None, int | None]:
    if not page_reference:
        return None, None
    cleaned = re.sub(r"(?i)^pages?\s*", "", page_reference.strip())
    m = re.match(r"^(\d+)\s*[-\u2013]\s*(\d+)$", cleaned)
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
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return fn(*fn_args, **fn_kwargs)
        except Exception as e:
            last_exc = e
            if attempt <= MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"  retry {attempt} failed ({e.__class__.__name__}: {e}) "
                      f"— retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                print(f"  all {MAX_RETRIES + 1} attempts failed: {e}")
    raise last_exc


# =============================================================================
# DOMAIN PREFERENCES
# =============================================================================

AUTOSAR_PREFERENCES = [
    {"keywords": ["timing", "schedule", "task", "preempt", "runnab"],
     "hint": "Prioritize OS, SchM, and Timing sections. "
             "For task questions focus on OsTask, OsEvent, OsAlarm nodes."},
    {"keywords": ["memory", "memmap", "section", "linker", "compiler abstraction"],
     "hint": "Prioritize MemMap, Compiler Abstraction, and Platform Type sections."},
    {"keywords": ["api", "function", "prototype", "signature", "return", "parameter"],
     "hint": "Prioritize API Specification chapters (SWS_* requirements) "
             "and nodes titled API, Function Definitions, or Interfaces."},
    {"keywords": ["error", "det", "diagnostic", "fault", "dem", "dtc"],
     "hint": "Prioritize Det, Dem, and Error Handling sections."},
    {"keywords": ["configuration", "ecuc", "parameter", "container", "variant"],
     "hint": "Prioritize EcucParam, EcucContainers, and configuration sections."},
    {"keywords": ["communication", "com", "pdu", "signal", "ipdu", "message"],
     "hint": "Prioritize COM, PduR, CanIf, LinIf, and Signal sections."},
    {"keywords": ["init", "initializ", "startup", "mode", "bsw"],
     "hint": "Prioritize Initialization, BswM, and BSW Module Description sections."},
    {"keywords": ["nvm", "nvram", "non-volatile", "storage", "persist"],
     "hint": "Prioritize NvM and Ea/Fee sections."},
    {"keywords": ["eeprom", "flash", "fls", "ea", "fee"],
     "hint": "Prioritize Fls, Ea, and Fee sections."},
    {"keywords": ["watchdog", "wdg", "alive", "trigger"],
     "hint": "Prioritize Wdg and WdgM sections."},
    {"keywords": ["arti", "trace", "hook", "instrument"],
     "hint": "Prioritize ARTI, tracing hooks, and instrumentation sections."},
    {"keywords": ["requirement", "srs", "sws", "tps", "constraint", "shall"],
     "hint": "Focus on SWS_ numbered requirement nodes and Constraints sections."},
]


def get_domain_preference(query: str, domain: str | None) -> str | None:
    if not domain or domain == "none":
        return None
    if domain == "autosar":
        q_lower = query.lower()
        fired   = [r["hint"] for r in AUTOSAR_PREFERENCES
                   if any(kw in q_lower for kw in r["keywords"])]
        return " ".join(fired) if fired else None
    return None


# =============================================================================
# DOC SELECTION — INFER MODE
# =============================================================================

def doc_selection_infer(query: str, doc_registry: list[dict]) -> list[str]:
    if not doc_registry:
        return []
    prompt = f"""You are given a list of documents with their IDs, file names, and descriptions. Your task is to select documents that may contain information relevant to answering the user query.

Query: {query}

Documents: {json.dumps(doc_registry, indent=2)}

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
    result   = json.loads(response.choices[0].message.content)
    selected = result.get("answer", [])
    if not isinstance(selected, list):
        selected = []
    print(f"  [doc-select] {len(selected)} doc(s) selected: {selected}")
    return selected


def build_doc_registry(tree_dir: str) -> list[dict]:
    registry = []
    if not tree_dir or not os.path.isdir(tree_dir):
        return registry
    for fname in sorted(os.listdir(tree_dir)):
        if not fname.endswith("_structure.json"):
            continue
        path = os.path.join(tree_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            doc_id   = fname.replace("_structure.json", "")
            doc_name = doc_id + ".pdf"
            doc_desc = data.get("doc_description", "")
            if not doc_desc:
                nodes    = data.get("structure", [])
                titles   = [n.get("title", "") for n in nodes[:5] if n.get("title")]
                doc_desc = f"Document covering: {', '.join(titles)}" if titles else doc_id
            registry.append({"doc_id": doc_id, "doc_name": doc_name,
                              "doc_description": doc_desc})
        except Exception as e:
            print(f"  [WARN] could not load registry entry for {fname}: {e}")
    print(f"[doc-registry] {len(registry)} documents indexed")
    return registry


# =============================================================================
# DOCUMENT CACHE
# =============================================================================

def build_doc_cache(pdf_names: list[str]) -> None:
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
            tree_json         = json.dumps(tree_nodes, indent=2)
            add_prefix_summaries(tree_nodes)
            node_index        = build_node_index(tree_nodes)
            DOC_CACHE[pdf_name] = {
                "documents":  documents,
                "doc_id":     doc_id,
                "tree_json":  tree_json,
                "node_index": node_index,
            }
            print(f"  [cache] OK {pdf_name}  "
                  f"({len(node_index)} nodes, {len(tree_json):,} chars)")
        except Exception as e:
            print(f"  [cache] FAIL {pdf_name} — {e}  (will retry per-question)")
    print(f"[cache] ready — {len(DOC_CACHE)}/{len(unique)} documents cached")


# =============================================================================
# SHARED RETRIEVAL PRIMITIVE
# =============================================================================

def tree_search(query: str, tree_structure_json: str,
                preference: str | None = None) -> list:
    """
    LLM identifies relevant node_ids from the document tree.
    Returns a list of node_id values only.
    Page numbers are resolved programmatically from the tree — never trusted from LLM.
    """
    if preference:
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


def extract_page_content(documents: dict, doc_id: str,
                         page_range: str) -> list[dict]:
    """Extract page text. Uses pageindex.retrieve if available, else PyPDF2."""
    try:
        from pageindex.retrieve import get_page_content
        return json.loads(get_page_content(documents, doc_id, page_range))
    except ImportError:
        return extract_pages_pypdf2(documents[doc_id]["path"], page_range)


# =============================================================================
# LEVELRAG — EXACT 4-STAGE PAPER IMPLEMENTATION
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: High-Level Query Planner (Decomposition)
# ─────────────────────────────────────────────────────────────────────────────

def levelrag_stage1_decompose(qid: str, query: str) -> tuple[list[str], dict]:
    """
    LevelRAG Paper — Stage 1: High-Level Query Planner.

    The LLM reads the original complex question and decomposes it into the
    minimum number of atomic sub-queries needed to answer it fully.

    Rules enforced in the prompt:
      - If already atomic: return it unchanged as a single-item list.
      - Max = LEVEL_RAG_MAX_SUBQUERIES.
      - Each sub-query is self-contained (no cross-dependency).
      - Order: foundational facts first, derived facts last.

    Returns:
        sub_queries : list[str] — atomic sub-queries to process
        record      : dict — complete Stage 1 record written to step1_decomposition
    """
    prompt = f"""You are a high-level query planner for a document retrieval system.

Analyse the following question and decompose it into the minimum number of
atomic, self-contained sub-queries needed to fully answer it.

Definition of atomic: a sub-query asks for exactly ONE fact and can be answered
independently without needing the answer to any other sub-query.

Rules:
1. If the question is ALREADY ATOMIC, return it as a single-item list unchanged.
   Do NOT invent unnecessary sub-queries.
2. Produce AT MOST {LEVEL_RAG_MAX_SUBQUERIES} sub-queries.
3. Each sub-query must be a complete, grammatically correct English question.
4. Sub-queries must be non-overlapping — do not ask the same thing twice.
5. Preserve all technical terms, identifiers, and names exactly as they appear.
6. Order sub-queries logically: foundational/definitional facts first.

Original question: {query}

Reply ONLY in this exact JSON format, no extra text:
{{
  "thinking": "<step-by-step decomposition reasoning>",
  "is_already_atomic": <true or false>,
  "sub_queries": ["<sub-query 1>", "<sub-query 2>", ...]
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result        = json.loads(response.choices[0].message.content)
    thinking      = result.get("thinking", "")
    is_atomic     = result.get("is_already_atomic", False)
    raw_subs      = result.get("sub_queries", [query])

    raw_subs = [s.strip() for s in raw_subs if isinstance(s, str) and s.strip()]
    if not raw_subs:
        raw_subs = [query]

    clamped   = len(raw_subs) > LEVEL_RAG_MAX_SUBQUERIES
    subs_used = raw_subs[:LEVEL_RAG_MAX_SUBQUERIES]
    if len(subs_used) < LEVEL_RAG_MIN_SUBQUERIES:
        subs_used = [query]

    record = {
        "qid":                  qid,
        "original_query":       query,
        "stage":                "1_decomposition",
        "llm_thinking":         thinking,
        "is_already_atomic":    is_atomic,
        "llm_raw_sub_queries":  raw_subs,
        "sub_queries_used":     subs_used,
        "n_sub_queries":        len(subs_used),
        "clamped":              clamped,
        "clamped_from":         len(raw_subs) if clamped else None,
    }

    if LEVEL_RAG_LOG:
        tag = "(already atomic — no decomposition)" if is_atomic else f"-> {len(subs_used)} sub-quer{'y' if len(subs_used)==1 else 'ies'}"
        print(f"  [S1] {tag}" + (f"  (clamped from {len(raw_subs)})" if clamped else ""))
        for i, sq in enumerate(subs_used, 1):
            print(f"       [{i}] {sq}")

    return subs_used, record


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Per-Sub-Query Independent Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def levelrag_stage2_retrieve_one(sub_query: str, sub_idx: int,
                                  tree_json: str, node_index: dict,
                                  documents: dict, doc_id: str,
                                  preference: str | None) -> dict:
    """
    LevelRAG Paper — Stage 2: Independent retrieval for ONE sub-query.

    Completely isolated: tree_search(sub_query) -> valid node_ids ->
    resolve_nodes() -> get_page_range_string() -> extract_page_content().

    No information from other sub-queries' retrievals is used here.
    This isolation is the core of LevelRAG Stage 2 — each sub-question
    retrieves exactly the pages relevant to its own scope.
    """
    if LEVEL_RAG_LOG:
        print(f"  [S2] retrieve sub-query [{sub_idx}]: {sub_query[:80]}")

    raw_node_ids = call_with_retry(tree_search, sub_query, tree_json, preference)

    valid_ids   = [str(nid) for nid in raw_node_ids if str(nid) in node_index]
    invalid_ids = [str(nid) for nid in raw_node_ids if str(nid) not in node_index]
    if invalid_ids and LEVEL_RAG_LOG:
        print(f"       WARN: {len(invalid_ids)} unknown node_id(s) dropped: {invalid_ids}")

    resolved_nodes = resolve_nodes(valid_ids, node_index)
    page_range     = get_page_range_string(resolved_nodes)

    if LEVEL_RAG_LOG:
        print(f"       {len(raw_node_ids)} node_id(s) from LLM  |  "
              f"{len(valid_ids)} valid  |  pages: {page_range or 'NONE'}")

    page_contents = []
    if page_range:
        page_contents = extract_page_content(documents, doc_id, page_range)

    return {
        "sub_idx":           sub_idx,
        "sub_query":         sub_query,
        "raw_node_ids":      raw_node_ids,
        "valid_node_ids":    valid_ids,
        "invalid_node_ids":  invalid_ids,
        "resolved_nodes":    resolved_nodes,
        "page_range":        page_range,
        "n_pages_retrieved": len(page_contents),
        "page_contents":     page_contents,
    }


def levelrag_stage2_retrieve_all(qid: str, query: str,
                                  sub_queries: list[str],
                                  tree_json: str, node_index: dict,
                                  documents: dict, doc_id: str,
                                  preference: str | None) -> tuple[list[dict], dict]:
    """
    Stage 2 orchestrator — runs retrieve_one for each sub-query in order.
    Returns the list of per-sub-query retrieval dicts and the step2 record.
    """
    t0 = time.time()
    retrievals = []
    for i, sq in enumerate(sub_queries, 1):
        ret = levelrag_stage2_retrieve_one(sq, i, tree_json, node_index,
                                           documents, doc_id, preference)
        retrievals.append(ret)
    elapsed = round(time.time() - t0, 2)

    if LEVEL_RAG_LOG:
        total_pages = sum(r["n_pages_retrieved"] for r in retrievals)
        print(f"  [S2] done — {len(retrievals)} sub-queries  |  "
              f"{total_pages} total page-slots retrieved  |  {elapsed}s")

    record = {
        "qid":            qid,
        "original_query": query,
        "stage":          "2_retrieval",
        "n_sub_queries":  len(sub_queries),
        "retrievals": [
            {
                "sub_idx":           r["sub_idx"],
                "sub_query":         r["sub_query"],
                "raw_node_ids":      r["raw_node_ids"],
                "valid_node_ids":    r["valid_node_ids"],
                "invalid_node_ids":  r["invalid_node_ids"],
                "resolved_nodes":    r["resolved_nodes"],
                "page_range":        r["page_range"],
                "n_pages_retrieved": r["n_pages_retrieved"],
                "page_contents":     r["page_contents"],
            }
            for r in retrievals
        ],
        "elapsed_s": elapsed,
    }
    return retrievals, record


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: Per-Sub-Query Partial Answer Generation
# ─────────────────────────────────────────────────────────────────────────────

def levelrag_stage3_partial_answer_one(sub_query: str, sub_idx: int,
                                        page_contents: list[dict]) -> dict:
    """
    LevelRAG Paper — Stage 3: Partial answer for ONE sub-query.

    KEY PAPER DETAIL:
    The LLM is given ONLY this sub-query's retrieved pages as context.
    Pages from other sub-queries are NOT visible here. This forces the model
    to answer each atomic question in complete isolation before synthesis,
    preventing context bleed and ensuring each partial answer is traceable
    to a specific scope of retrieved evidence.

    If no pages were retrieved for this sub-query, the partial answer is
    explicitly marked so Stage 4 synthesis can acknowledge the gap honestly.
    """
    if LEVEL_RAG_LOG:
        print(f"  [S3] partial answer sub-query [{sub_idx}]: {sub_query[:80]}")

    if not page_contents:
        partial_answer = "[NO CONTENT RETRIEVED — CANNOT ANSWER THIS SUB-QUESTION]"
        if LEVEL_RAG_LOG:
            print(f"       -> no pages retrieved — marked as unanswerable")
        return {
            "sub_idx":        sub_idx,
            "sub_query":      sub_query,
            "pages_used":     [],
            "partial_answer": partial_answer,
            "had_content":    False,
        }

    context = "\n\n".join(
        f"[Page {p['page']}]\n{p['content']}"
        for p in page_contents if p.get("content")
    )

    prompt = f"""You are answering one focused sub-question using ONLY the context provided below.
The context contains pages retrieved specifically for this sub-question and nothing else.

Instructions:
- Answer directly and concisely.
- Cite page numbers where relevant.
- If the context does not contain enough information, say so explicitly.
- Do NOT guess or add information not present in the context.

Sub-question: {sub_query}

Context (pages retrieved for this sub-question only):
{context}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    partial_answer = response.choices[0].message.content.strip()
    pages_used     = [p["page"] for p in page_contents if p.get("content")]

    if LEVEL_RAG_LOG:
        print(f"       -> {partial_answer[:100]}...")

    return {
        "sub_idx":        sub_idx,
        "sub_query":      sub_query,
        "pages_used":     pages_used,
        "partial_answer": partial_answer,
        "had_content":    True,
    }


def levelrag_stage3_partial_answers_all(qid: str, query: str,
                                         retrievals: list[dict]) -> tuple[list[dict], dict]:
    """
    Stage 3 orchestrator — runs partial_answer_one for each retrieval in order.
    Each call receives ONLY its own sub-query's page_contents.
    Returns the list of partial answer dicts and the step3 record.
    """
    t0 = time.time()
    partials = []
    for ret in retrievals:
        pa = call_with_retry(
            levelrag_stage3_partial_answer_one,
            sub_query     = ret["sub_query"],
            sub_idx       = ret["sub_idx"],
            page_contents = ret["page_contents"],
        )
        partials.append(pa)
    elapsed = round(time.time() - t0, 2)

    if LEVEL_RAG_LOG:
        answered = sum(1 for p in partials if p["had_content"])
        print(f"  [S3] done — {answered}/{len(partials)} partial answers had content  |  {elapsed}s")

    record = {
        "qid":             qid,
        "original_query":  query,
        "stage":           "3_partial_answers",
        "n_sub_queries":   len(retrievals),
        "partial_answers": [
            {
                "sub_idx":        p["sub_idx"],
                "sub_query":      p["sub_query"],
                "pages_used":     p["pages_used"],
                "partial_answer": p["partial_answer"],
                "had_content":    p["had_content"],
            }
            for p in partials
        ],
        "elapsed_s": elapsed,
    }
    return partials, record


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: Final Answer Synthesis
# ─────────────────────────────────────────────────────────────────────────────

def levelrag_stage4_synthesize(qid: str, query: str,
                                partials: list[dict]) -> tuple[str, dict]:
    """
    LevelRAG Paper — Stage 4: Final synthesis.

    The synthesis LLM receives:
      - The ORIGINAL complex query (not sub-queries)
      - All partial answers, each labelled by its sub-question

    It synthesizes these into one coherent, complete, non-redundant final answer.

    This cleanly separates "information gathering" (Stages 1-3) from
    "answer composition" (Stage 4), following the paper's architecture.
    The synthesis LLM never sees raw page text — only the structured
    partial answers produced by Stage 3.
    """
    if LEVEL_RAG_LOG:
        print(f"  [S4] synthesizing {len(partials)} partial answer(s) into final answer ...")

    partials_text = "\n\n".join(
        f"Sub-question {p['sub_idx']}: {p['sub_query']}\n"
        f"Partial answer: {p['partial_answer']}"
        for p in partials
    )

    prompt = f"""You are synthesizing a final answer to a complex question.
You have been given partial answers, each addressing one atomic sub-question.

Your task: combine the partial answers into a single, coherent, complete response
to the ORIGINAL question.

Instructions:
1. Use ONLY information present in the partial answers — do not add new facts.
2. If a partial answer says no content was retrieved, acknowledge that gap honestly.
3. Do not repeat the same fact from multiple partial answers — merge them cleanly.
4. Write in a clear, direct style. Preserve page number citations from partial answers.
5. The final answer must directly and completely address the original question.

Original question: {query}

Partial answers (from atomic sub-questions):
{partials_text}

Final synthesized answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    final_answer = response.choices[0].message.content.strip()

    if LEVEL_RAG_LOG:
        print(f"  [S4] -> {final_answer[:120]}...")

    record = {
        "qid":             qid,
        "original_query":  query,
        "stage":           "4_synthesis",
        "n_partial_answers": len(partials),
        "partial_answers_used": [
            {"sub_idx":        p["sub_idx"],
             "sub_query":      p["sub_query"],
             "partial_answer": p["partial_answer"],
             "pages_used":     p["pages_used"],
             "had_content":    p["had_content"]}
            for p in partials
        ],
        "final_answer": final_answer,
    }
    return final_answer, record


# ─────────────────────────────────────────────────────────────────────────────
# LevelRAG Top-Level Orchestrator for one question
# ─────────────────────────────────────────────────────────────────────────────

def run_levelrag_pipeline(qid: str, query: str,
                           tree_json: str, node_index: dict,
                           documents: dict, doc_id: str,
                           preference: str | None) -> tuple[str, list[dict], list[dict], dict]:
    """
    Runs all 4 LevelRAG stages for one question.
    Writes to all 4 step files.

    Returns:
        final_answer      : str
        all_resolved_nodes: list[dict] — deduplicated union of nodes across all
                            sub-queries (used for retrieval evaluation)
        all_page_contents : list[dict] — union of pages across all sub-queries
                            (used for evidence recall evaluation)
        levelrag_summary  : dict — compact summary embedded in results.json
    """
    # Stage 1
    sub_queries, s1_record = call_with_retry(levelrag_stage1_decompose, qid, query)
    _append_step("step1_decomposition", s1_record)

    # Stage 2
    retrievals, s2_record = levelrag_stage2_retrieve_all(
        qid, query, sub_queries,
        tree_json, node_index, documents, doc_id, preference
    )
    _append_step("step2_retrieval", s2_record)

    # Stage 3
    partials, s3_record = levelrag_stage3_partial_answers_all(qid, query, retrievals)
    _append_step("step3_partial_answers", s3_record)

    # Stage 4
    final_answer, s4_record = call_with_retry(
        levelrag_stage4_synthesize, qid, query, partials
    )
    _append_step("step4_final_answers", s4_record)

    # Deduplicated union of all retrieved nodes (for retrieval_eval)
    seen_nids, all_resolved_nodes = set(), []
    for ret in retrievals:
        for node in ret["resolved_nodes"]:
            if node["node_id"] not in seen_nids:
                seen_nids.add(node["node_id"])
                all_resolved_nodes.append(node)

    # Union of all page contents (for evidence recall)
    seen_pages, all_page_contents = set(), []
    for ret in retrievals:
        for pc in ret["page_contents"]:
            if pc["page"] not in seen_pages:
                seen_pages.add(pc["page"])
                all_page_contents.append(pc)

    levelrag_summary = {
        "n_sub_queries":      s1_record["n_sub_queries"],
        "sub_queries":        s1_record["sub_queries_used"],
        "is_already_atomic":  s1_record["is_already_atomic"],
        "total_unique_nodes": len(all_resolved_nodes),
        "total_unique_pages": len(all_page_contents),
        "per_sub_retrieval": [
            {
                "sub_idx":        ret["sub_idx"],
                "sub_query":      ret["sub_query"],
                "page_range":     ret["page_range"],
                "n_pages":        ret["n_pages_retrieved"],
                "partial_answer": pa["partial_answer"],
            }
            for ret, pa in zip(retrievals, partials)
        ],
    }

    return final_answer, all_resolved_nodes, all_page_contents, levelrag_summary


# =============================================================================
# STANDARD (NON-LEVELRAG) SINGLE-PASS PIPELINE
# Also writes to all 4 step files so output structure is always identical.
# =============================================================================

def run_standard_pipeline(qid: str, query: str,
                           tree_json: str, node_index: dict,
                           documents: dict, doc_id: str,
                           preference: str | None) -> tuple[str, list[dict], list[dict]]:
    """
    Original single-pass retrieval + answer generation.
    Writes placeholder records to all 4 step files for structural consistency.

    Returns:
        answer         : str
        resolved_nodes : list[dict]
        page_contents  : list[dict]
    """
    raw_node_ids   = call_with_retry(tree_search, query, tree_json, preference)
    valid_ids      = [str(nid) for nid in raw_node_ids if str(nid) in node_index]
    resolved_nodes = resolve_nodes(valid_ids, node_index)
    page_range     = get_page_range_string(resolved_nodes)

    print(f"  -> {len(raw_node_ids)} node_id(s) from LLM  |  "
          f"{len(resolved_nodes)} resolved  |  pages: {page_range or 'NONE'}")

    page_contents = []
    if page_range:
        page_contents = extract_page_content(documents, doc_id, page_range)

    # Standard answer generation — original query + all pages combined
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
    answer = response.choices[0].message.content.strip()
    print(f"  -> Answer: {answer[:120]}...")

    # Write to all 4 step files (standard-mode placeholders)
    _append_step("step1_decomposition", {
        "qid": qid, "original_query": query, "stage": "1_decomposition",
        "mode": "standard_no_decomposition",
        "sub_queries_used": [query], "n_sub_queries": 1,
        "is_already_atomic": True,
    })
    _append_step("step2_retrieval", {
        "qid": qid, "original_query": query, "stage": "2_retrieval",
        "mode": "standard_single_pass",
        "retrievals": [{
            "sub_idx": 1, "sub_query": query,
            "raw_node_ids": raw_node_ids, "valid_node_ids": valid_ids,
            "resolved_nodes": resolved_nodes, "page_range": page_range,
            "n_pages_retrieved": len(page_contents),
            "page_contents": page_contents,
        }],
    })
    _append_step("step3_partial_answers", {
        "qid": qid, "original_query": query, "stage": "3_partial_answers",
        "mode": "standard_no_partial_answers",
        "partial_answers": [{
            "sub_idx": 1, "sub_query": query,
            "pages_used": [p["page"] for p in page_contents],
            "partial_answer": answer, "had_content": bool(page_contents),
        }],
    })
    _append_step("step4_final_answers", {
        "qid": qid, "original_query": query, "stage": "4_synthesis",
        "mode": "standard_direct_answer",
        "final_answer": answer,
    })

    return answer, resolved_nodes, page_contents


# =============================================================================
# EVALUATION
# =============================================================================

def check_retrieval_overlap(retrieved_nodes: list[dict],
                             start_page: int, end_page: int) -> dict:
    gold_pages      = set(range(start_page, end_page + 1))
    retrieved_pages = set()
    for node in retrieved_nodes:
        s = node.get("start_index")
        e = node.get("end_index")
        if s is not None and e is not None:
            retrieved_pages.update(range(s, e + 1))
    overlap   = gold_pages & retrieved_pages
    recall    = round(len(overlap) / len(gold_pages),      2) if gold_pages      else 0.0
    precision = round(len(overlap) / len(retrieved_pages), 2) if retrieved_pages else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 2) \
                if (precision + recall) > 0 else 0.0
    return {
        "gold_pages":      sorted(gold_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "overlap_pages":   sorted(overlap),
        "retrieval_hit":   len(overlap) > 0,
        "recall":          recall,
        "precision":       precision,
        "f1":              f1,
    }


def check_evidence_recall(page_contents: list[dict],
                           evidence_snippets: list) -> dict:
    if not evidence_snippets:
        return {"total_snippets": 0, "matched_snippets": 0,
                "evidence_recall": None, "no_snippets": True}
    full_text     = " ".join(p.get("content", "") for p in page_contents
                             if p.get("content")).lower()
    norm_fulltext = re.sub(r"\s+", "", full_text)
    matched = 0
    for snippet in evidence_snippets:
        normalised = re.sub(r"\s+", "", snippet.strip()).lower()
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
    snippets_text = "\n".join(f"- {s}" for s in evidence_snippets) \
                    if evidence_snippets else "N/A"
    prompt = f"""You are an expert evaluator for a RAG system.

Document: {source_document}
Question: {question}

Ground Truth Answer:
{ground_truth}

Gold Evidence Snippets (from the source document):
{snippets_text}

Generated Answer:
{generated_answer}

Evaluate on three criteria:
1. Factual correctness — does the generated answer convey the same facts as ground truth?
2. Completeness — does it cover all key points in the ground truth?
3. Hallucination — does it add facts not supported by ground truth or evidence?

Reply ONLY in this JSON format with no extra text:
{{
  "verdict": "correct" | "partial" | "incorrect",
  "correctness_score": <float 0.0 to 1.0>,
  "completeness_score": <float 0.0 to 1.0>,
  "hallucination": "none" | "minor" | "major",
  "reasoning": "<brief explanation>"
}}"""

    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PER-QUESTION ORCHESTRATORS
# =============================================================================

def process_question(q: dict, index: int, total: int) -> dict:
    """EVAL MODE — run full pipeline for one question."""
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
        print(f"  SKIP: missing 'question' or 'source_document'")
        return {
            "id": qid, "question": query, "source_document": pdf_name,
            "question_type": question_type, "difficulty": difficulty,
            "page_reference": page_reference,
            "status": "skipped", "error": "missing 'question' or 'source_document'",
        }

    try:
        # Resolve document from cache
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

        preference = get_domain_preference(query, args.domain)
        if preference:
            print(f"  -> [preference] domain={args.domain} hint injected")

        # ── Dispatch ──────────────────────────────────────────────────────────
        levelrag_summary = None

        if USE_LEVEL_RAG:
            print(f"  -> [levelrag] 4-stage pipeline ...")
            answer, relevant_nodes, all_page_contents, levelrag_summary = \
                run_levelrag_pipeline(qid, query, tree_json, node_index,
                                      documents, doc_id, preference)
        else:
            answer, relevant_nodes, all_page_contents = \
                run_standard_pipeline(qid, query, tree_json, node_index,
                                      documents, doc_id, preference)

        page_range = get_page_range_string(relevant_nodes)
        print(f"  -> {len(relevant_nodes)} unique node(s)  |  "
              f"{len(all_page_contents)} unique page(s)  |  answer: {len(answer)} chars")

        # ── Retrieval evaluation ──────────────────────────────────────────────
        if start_page is not None and end_page is not None:
            retrieval_eval = check_retrieval_overlap(relevant_nodes, start_page, end_page)
        else:
            retrieval_eval = {
                "retrieval_hit": None, "recall": None,
                "precision": None, "f1": None,
                "page_ref_unparseable": True,
                "note": f"could not parse: {page_reference!r}",
            }
        print(f"  -> Retrieval hit: {retrieval_eval.get('retrieval_hit')} | "
              f"Recall: {retrieval_eval.get('recall', 'N/A')}")

        # ── Evidence recall ───────────────────────────────────────────────────
        evidence_recall_result = check_evidence_recall(all_page_contents, evidence_snippets)
        print(f"  -> Evidence recall: {evidence_recall_result.get('evidence_recall', 'N/A')} "
              f"({evidence_recall_result.get('matched_snippets', 0)}/"
              f"{evidence_recall_result.get('total_snippets', 0)} snippets)")

        # ── LLM Judge ─────────────────────────────────────────────────────────
        evaluation = call_with_retry(
            llm_judge,
            question          = query,
            ground_truth      = ground_truth,
            generated_answer  = answer,
            evidence_snippets = evidence_snippets,
            source_document   = pdf_name,
        )
        print(f"  -> Verdict: {evaluation.get('verdict')} | "
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
            "levelrag":             levelrag_summary,
            "status":               "success",
        }

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {
            "id": qid, "question": query, "question_type": question_type,
            "difficulty": difficulty, "source_document": pdf_name,
            "page_reference": page_reference,
            "gold_start_page": start_page, "gold_end_page": end_page,
            "answer": "", "ground_truth": ground_truth,
            "status": "error", "error": str(e),
        }


def process_question_infer(q: dict, index: int, total: int,
                            doc_registry: list[dict]) -> dict:
    """INFER MODE — process one question without GT."""
    qid   = q.get("id", f"q{index:03d}")
    query = q.get("question", "")
    print(f"\n[{index}/{total}] {qid} — {query[:80]}...")

    if not query:
        return {"id": qid, "question": query, "status": "skipped",
                "error": "missing 'question'"}

    try:
        selected_doc_ids = call_with_retry(doc_selection_infer, query, doc_registry)
        if not selected_doc_ids:
            raise ValueError("Doc-selection returned no relevant documents.")

        doc_id   = selected_doc_ids[0]
        pdf_name = doc_id + ".pdf"

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

        preference = get_domain_preference(query, args.domain)
        if preference:
            print(f"  -> [preference] domain={args.domain} hint injected")

        levelrag_summary = None
        if USE_LEVEL_RAG:
            print(f"  -> [levelrag] 4-stage pipeline ...")
            answer, relevant_nodes, _, levelrag_summary = \
                run_levelrag_pipeline(qid, query, tree_json, node_index,
                                      documents, doc_id, preference)
        else:
            answer, relevant_nodes, _ = \
                run_standard_pipeline(qid, query, tree_json, node_index,
                                      documents, doc_id, preference)

        page_range = get_page_range_string(relevant_nodes)
        print(f"  -> Answer: {answer[:120]}...")

        return {
            "id":              qid,
            "question":        query,
            "source_document": pdf_name,
            "selected_docs":   selected_doc_ids,
            "retrieved_nodes": relevant_nodes,
            "pages_used":      page_range,
            "answer":          answer,
            "levelrag":        levelrag_summary,
            "status":          "success",
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"id": qid, "question": query, "status": "error", "error": str(e)}


# =============================================================================
# FALLBACK PAGE EXTRACTOR
# =============================================================================

def extract_pages_pypdf2(pdf_path: str, page_range: str) -> list[dict]:
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 required. pip install PyPDF2")
    page_nums = set()
    for part in page_range.split(","):
        part = part.strip()
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            page_nums.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            page_nums.add(int(part))
    results = []
    with open(pdf_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for pg in sorted(page_nums):
            idx = pg - 1
            if 0 <= idx < len(reader.pages):
                text = reader.pages[idx].extract_text() or ""
                results.append({"page": pg, "content": text.strip()})
    return results


# =============================================================================
# METRICS AGGREGATION
# =============================================================================

def compute_metrics_summary(results: list[dict], dataset_info: dict) -> dict:
    total      = len(results)
    successful = [r for r in results if r["status"] == "success"]
    n_success  = len(successful)

    ret_evaluable = [r for r in successful
                     if isinstance(r.get("retrieval_eval", {}).get("recall"), float)]
    ret_excluded  = n_success - len(ret_evaluable)
    ret_hits      = sum(1 for r in ret_evaluable
                        if r["retrieval_eval"].get("retrieval_hit"))
    avg_rr  = round(sum(r["retrieval_eval"]["recall"]    for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None
    avg_rp  = round(sum(r["retrieval_eval"]["precision"] for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None
    avg_rf1 = round(sum(r["retrieval_eval"]["f1"]        for r in ret_evaluable) / max(len(ret_evaluable), 1), 4) if ret_evaluable else None

    ev_evaluable = [r for r in successful
                    if isinstance(r.get("evidence_recall_eval", {}).get("evidence_recall"), float)]
    ev_excluded  = n_success - len(ev_evaluable)
    avg_ev_recall = round(sum(r["evidence_recall_eval"]["evidence_recall"] for r in ev_evaluable) /
                          max(len(ev_evaluable), 1), 4) if ev_evaluable else None

    correct   = sum(1 for r in successful if r.get("evaluation", {}).get("verdict") == "correct")
    partial   = sum(1 for r in successful if r.get("evaluation", {}).get("verdict") == "partial")
    incorrect = n_success - correct - partial
    accuracy  = round(correct / max(total, 1), 4)

    judge_ev  = [r for r in successful if r.get("evaluation", {}).get("correctness_score") is not None]
    avg_corr  = round(sum(r["evaluation"]["correctness_score"]  for r in judge_ev) / max(len(judge_ev), 1), 4) if judge_ev else None
    avg_comp  = round(sum(r["evaluation"]["completeness_score"] for r in judge_ev) / max(len(judge_ev), 1), 4) if judge_ev else None

    hall_counts = {"none": 0, "minor": 0, "major": 0}
    for r in successful:
        h = r.get("evaluation", {}).get("hallucination", "")
        if h in hall_counts:
            hall_counts[h] += 1

    def make_breakdown(results_list, group_key):
        groups: dict = {}
        for r in results_list:
            key     = r.get(group_key) or "unknown"
            verdict = r.get("evaluation", {}).get("verdict", "unknown")
            if key not in groups:
                groups[key] = {"total": 0, "correct": 0, "partial": 0, "incorrect": 0,
                               "retrieval_hits": 0, "retrieval_evaluable": 0,
                               "evidence_evaluable": 0,
                               "_cs": 0.0, "_rr": 0.0, "_er": 0.0}
            g = groups[key]
            g["total"] += 1
            if verdict in ("correct", "partial", "incorrect"):
                g[verdict] += 1
            ret = r.get("retrieval_eval", {})
            if isinstance(ret.get("recall"), float):
                g["retrieval_evaluable"] += 1
                g["_rr"] += ret["recall"]
                if ret.get("retrieval_hit"):
                    g["retrieval_hits"] += 1
            ev = r.get("evidence_recall_eval", {})
            if isinstance(ev.get("evidence_recall"), float):
                g["evidence_evaluable"] += 1
                g["_er"] += ev["evidence_recall"]
            cs = r.get("evaluation", {}).get("correctness_score")
            if isinstance(cs, float):
                g["_cs"] += cs
        for key, g in groups.items():
            g["accuracy"]             = round(g["correct"] / max(g["total"], 1), 4)
            g["retrieval_hit_rate"]   = round(g["retrieval_hits"] / max(g["retrieval_evaluable"], 1), 4) if g["retrieval_evaluable"] else None
            g["avg_retrieval_recall"] = round(g["_rr"] / max(g["retrieval_evaluable"], 1), 4) if g["retrieval_evaluable"] else None
            g["avg_evidence_recall"]  = round(g["_er"] / max(g["evidence_evaluable"],  1), 4) if g["evidence_evaluable"]  else None
            g["avg_correctness"]      = round(g["_cs"] / max(g["total"], 1), 4)
            del g["_cs"], g["_rr"], g["_er"]
        return groups

    return {
        "dataset_info": dataset_info,
        "total":        total,
        "successful":   n_success,
        "errors":       total - n_success,
        "summary": {
            "retrieval_evaluable":        len(ret_evaluable),
            "retrieval_metrics_excluded": ret_excluded,
            "retrieval_hits":             ret_hits,
            "retrieval_hit_rate":         round(ret_hits / max(len(ret_evaluable), 1), 4) if ret_evaluable else None,
            "avg_retrieval_recall":       avg_rr,
            "avg_retrieval_precision":    avg_rp,
            "avg_retrieval_f1":           avg_rf1,
            "evidence_evaluable":         len(ev_evaluable),
            "evidence_recall_excluded":   ev_excluded,
            "avg_evidence_recall":        avg_ev_recall,
            "correct":                    correct,
            "partial":                    partial,
            "incorrect":                  incorrect,
            "accuracy":                   accuracy,
            "avg_correctness_score":      avg_corr,
            "avg_completeness_score":     avg_comp,
            "hallucination_counts":       hall_counts,
        },
        "breakdown_by_question_type": make_breakdown(successful, "question_type"),
        "breakdown_by_difficulty":    make_breakdown(successful, "difficulty"),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    global args, client, MODEL, judge_client, JUDGE_MODEL, MAX_RETRIES, RETRY_BACKOFF
    global USE_LEVEL_RAG, LEVEL_RAG_MAX_SUBQUERIES

    args = parse_args()

    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        print(f"[env] loaded {args.env_file}")
    else:
        print(f"[env] {args.env_file} not found — using system env vars")

    if args.pageindex_repo:
        sys.path.insert(0, args.pageindex_repo)
        try:
            from pageindex.retrieve import get_page_content  # noqa: F401
            print(f"[extractor] pageindex.retrieve <- from repo: {args.pageindex_repo}")
        except ImportError as ie:
            print(f"[extractor] pageindex.retrieve NOT found ({ie}) — falling back to PyPDF2")
    else:
        print("[extractor] PyPDF2 (fallback) <- set --pageindex_repo to use pageindex.retrieve")

    MAX_RETRIES              = args.max_retries
    RETRY_BACKOFF            = args.retry_backoff
    USE_LEVEL_RAG            = args.use_level_rag
    LEVEL_RAG_MAX_SUBQUERIES = args.level_rag_max_subqueries

    print(f"\n[levelrag] {'ENABLED — exact 4-stage paper pipeline' if USE_LEVEL_RAG else 'DISABLED — standard single-pass'}")
    if USE_LEVEL_RAG:
        print(f"           Stage 1: query decomposition (max {LEVEL_RAG_MAX_SUBQUERIES} sub-queries)")
        print(f"           Stage 2: per-sub-query independent retrieval")
        print(f"           Stage 3: per-sub-query partial answer (isolated context)")
        print(f"           Stage 4: final synthesis (original query + all partial answers)")

    print("\n[generation]")
    client, MODEL = setup_llm_client(args.provider, args.model)

    judge_provider       = args.judge_provider or args.provider
    judge_model_override = args.judge_model or args.model
    print("[judge]")
    judge_client, JUDGE_MODEL = setup_llm_client(judge_provider, judge_model_override)
    if judge_provider == args.provider and JUDGE_MODEL == MODEL:
        print("  [WARN] judge model == generation model — self-evaluation bias risk.")

    # Validate paths
    if not os.path.exists(args.query):
        raise FileNotFoundError(f"Questions file not found: {args.query}")
    if args.tree_file and not os.path.isfile(args.tree_file):
        raise FileNotFoundError(f"Tree file not found: {args.tree_file}")
    if args.tree_dir and not os.path.isdir(args.tree_dir):
        raise FileNotFoundError(f"Tree dir not found: {args.tree_dir}")
    if args.pdf_file and not os.path.isfile(args.pdf_file):
        raise FileNotFoundError(f"PDF file not found: {args.pdf_file}")
    if args.pdf_dir and not os.path.isdir(args.pdf_dir):
        raise FileNotFoundError(f"PDF dir not found: {args.pdf_dir}")
    if args.use_md and not args.md_dir:
        raise ValueError("--use_md requires --md_dir")
    if args.use_md and args.md_dir and not os.path.isdir(args.md_dir):
        raise FileNotFoundError(f"MD dir not found: {args.md_dir}")

    tree_mode = f"single file: {args.tree_file}" if args.tree_file else f"dir: {args.tree_dir}"
    pdf_mode  = f"single file: {args.pdf_file}"  if args.pdf_file  else f"dir: {args.pdf_dir}"
    print(f"\n[tree] {tree_mode}")
    print(f"[pdf]  {pdf_mode}")

    # Prepare output dir and clear any stale .jsonl scratch files
    os.makedirs(args.output_dir, exist_ok=True)
    for step_name in _STEP_BUFFERS:
        scratch = os.path.join(args.output_dir, f"{step_name}.jsonl")
        if os.path.exists(scratch):
            os.remove(scratch)

    # Load questions
    with open(args.query, "r", encoding="utf-8") as fh:
        data = json.load(fh)
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

    if args.mode == "eval":
        build_doc_cache([q.get("source_document", "") for q in questions])

    workers = args.parallel if args.parallel and args.parallel > 1 else None
    if workers:
        print(f"[parallel] workers: {workers}")
    else:
        print(f"[sequential] {total} questions one at a time")

    results_map: dict[int, dict] = {}

    # ── INFER MODE ────────────────────────────────────────────────────────────
    if args.mode == "infer":
        if not args.tree_dir:
            raise ValueError("--mode infer requires --tree_dir")
        doc_registry = build_doc_registry(args.tree_dir)
        if not doc_registry:
            raise ValueError("No *_structure.json files found in tree_dir.")
        build_doc_cache([entry["doc_name"] for entry in doc_registry])

        if workers:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_question_infer, q, i, total, doc_registry): i
                    for i, q in enumerate(questions, 1)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        results_map[i] = future.result()
                    except Exception as e:
                        q = questions[i - 1]
                        results_map[i] = {"id": q.get("id", f"q{i:03d}"),
                                          "status": "error", "error": str(e)}
        else:
            for i, q in enumerate(questions, 1):
                results_map[i] = process_question_infer(q, i, total, doc_registry)
                time.sleep(args.sleep)

        results = [results_map[i] for i in range(1, total + 1)]
        results_path = os.path.join(args.output_dir, "infer_results.json")
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump({"total": total, "results": results}, fh,
                      indent=2, ensure_ascii=False)

        _flush_step_files()

        success = sum(1 for r in results if r["status"] == "success")
        print(f"\n{'='*65}")
        print(f"Infer done — {success}/{total} answered  ({total - success} errors)")
        print(f"   infer_results.json  ->  {results_path}")
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
                    results_map[i] = {"id": q.get("id", f"q{i:03d}"),
                                      "status": "error", "error": str(e)}
    else:
        for i, q in enumerate(questions, 1):
            results_map[i] = process_question(q, i, total)
            time.sleep(args.sleep)

    results = [results_map[i] for i in range(1, total + 1)]
    metrics = compute_metrics_summary(results, dataset_info)

    results_path = os.path.join(args.output_dir, "results.json")
    metrics_path = os.path.join(args.output_dir, "metrics_summary.json")

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump({**metrics, "results": results}, fh, indent=2, ensure_ascii=False)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    _flush_step_files()

    s = metrics["summary"]
    print(f"\n{'='*65}")
    print(f"Eval done — {metrics['successful']}/{total} successful  "
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
    print(f"   [Output files ->  {args.output_dir}/]")
    print(f"   results.json           (full pipeline output + metrics)")
    print(f"   metrics_summary.json   (aggregated metrics only)")
    print(f"   step1_decomposition.json   (Stage 1: query decomposition per question)")
    print(f"   step2_retrieval.json       (Stage 2: per-sub-query nodes + pages)")
    print(f"   step3_partial_answers.json (Stage 3: per-sub-query partial answers)")
    print(f"   step4_final_answers.json   (Stage 4: synthesized final answers)")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_pipeline()