"""
PageIndex RAG Pipeline v4
=========================
Upgraded from v3 to support flat-list structure files (_hierarchy.json)
in addition to the original nested-tree format (_structure.json).

Changes from v3:
  1. load_structure() now resolves both _hierarchy.json and _structure.json filenames
     automatically — tries _hierarchy.json first, falls back to _structure.json.
  2. build_documents() reads the correct top-level key: "hierarchy" (flat format)
     OR "structure" (tree format) — whichever is present.
  3. doc_description falls back to "doc_name" field when the dedicated key is absent
     (the flat format omits doc_description).
  4. tree_search() prompt updated: describes the input as a flat section index when
     all nodes are leaf-level, so the LLM reasons over a wide list rather than
     navigating a tree hierarchy.
  5. detect_flat_structure() helper added: inspects the loaded nodes and returns True
     when every node has an empty "nodes" list — used to select the right LLM prompt.

Everything from v3 is preserved:
  - USE_MD / GLM_OCR_DIR markdown page content injection
  - USE_OLLAMA / NVIDIA backend toggle
  - Parallel processing with OLLAMA_PARALLEL
  - Retry logic with exponential backoff
  - add_prefix_summaries() / build_node_index() (both already handle flat data correctly)
  - LLM judge evaluation
  - Breakdown by question_type and difficulty

New input format (per question):
  {
    "id":               "doc001_q001",          # used as output id directly
    "question":         "What is ...",           # natural-language query
    "source_document":  "SomePdf.pdf",           # PDF filename
    "answer":           "Ground truth ...",      # reference answer
    "evidence_snippets": ["snippet 1", ...],     # gold evidence
    "page_reference":   "Pages 5-6",             # gold page range string e.g. "Pages 5-6" or "5-6"
    "difficulty":       "medium",                # metadata (passed through)
    "question_type":    "multi-hop"              # metadata (passed through)
  }

The top-level JSON may also contain a "dataset_info" block — it is ignored by
the pipeline and simply echoed into the output for traceability.

Pipeline steps:
  1. Load structure + build documents dict
  2. Tree Search  (LLM picks relevant nodes)
  3. Page Extraction
  4. Answer Generation
  5a. Retrieval Evaluation  (page overlap vs gold page_reference — no LLM)
  5b. LLM Judge             (generated answer vs answer + evidence_snippets)

To change anything, edit ONLY the CONFIG section below.
"""

import json
import re
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# =============================================================================
# CONFIG — Edit these to change paths, model, or behaviour
# =============================================================================

ENV_FILE        = "/home/olj3kor/praveen/Github_copilot/.env"

PAGEINDEX_REPO  = "/home/olj3kor/praveen/PageIndex"
QUERIES_FILE    = "/home/olj3kor/praveen/Github_copilot/Q.json"
STRUCTURE_DIR   = "/home/olj3kor/praveen/DocLing/structured"
PDF_DIR         = "/home/olj3kor/praveen/Image_dataset_generation/pdfs"
OUTPUT_FILE     = "/home/olj3kor/praveen/results/rag_Nemo_v10_results.json"

SLEEP_BETWEEN_Q = 0.5   # seconds between questions (avoids rate limits)

# -----------------------------------------------------------------------------
# Retry settings — applied to every LLM API call (tree_search, generate_answer,
#                  llm_judge). On failure the call is retried up to MAX_RETRIES
#                  times with exponential backoff (RETRY_BACKOFF seconds base,
#                  doubles each attempt: 2s → 4s → 8s ...).
#                  Set MAX_RETRIES = 0 to disable retries entirely.
# -----------------------------------------------------------------------------
MAX_RETRIES   = 3     # number of retry attempts after the first failure
RETRY_BACKOFF = 2.0   # base wait in seconds (doubles each retry)

# -----------------------------------------------------------------------------
# Content source — set USE_MD = True to serve page content from GLM OCR
#                  markdown files instead of extracting via PyPDF2 from the PDF.
#                  Set USE_MD = False to keep the original PDF extraction.
# -----------------------------------------------------------------------------
USE_MD          = False

GLM_OCR_DIR     = "/home/olj3kor/praveen/GLM_OCR_OUTPUT"
# Expected layout:
#   {GLM_OCR_DIR}/{doc_name_without_pdf}/pages/page_1.md
#   {GLM_OCR_DIR}/{doc_name_without_pdf}/pages/page_2.md  ...

# -----------------------------------------------------------------------------
# Backend selector — set USE_OLLAMA = True to use a local Ollama model,
#                    set USE_OLLAMA = False to use the NVIDIA API.
# -----------------------------------------------------------------------------
USE_OLLAMA      = False

# NVIDIA API settings (used when USE_OLLAMA = False)
NVIDIA_MODEL    = "moonshotai/kimi-k2-instruct-0905"

# Ollama settings (used when USE_OLLAMA = True)
OLLAMA_MODEL    = "llama3.1:8b"        # any model you have pulled in Ollama
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_PARALLEL = 4     # number of questions to process in parallel when USE_OLLAMA = True
                        # set to None or 1 to process sequentially (same as NVIDIA path)

# =============================================================================
# SETUP
# =============================================================================

load_dotenv(ENV_FILE)

sys.path.insert(0, PAGEINDEX_REPO)
from pageindex.retrieve import get_document_structure, get_page_content

if USE_OLLAMA:
    MODEL  = OLLAMA_MODEL
    client = OpenAI(
        api_key="ollama",               # Ollama ignores the key; any non-empty string works
        base_url=OLLAMA_BASE_URL,
    )
    print(f"[backend] Ollama  →  {OLLAMA_BASE_URL}  model: {MODEL}")
else:
    MODEL  = NVIDIA_MODEL
    client = OpenAI(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    print(f"[backend] NVIDIA API  model: {MODEL}")


# =============================================================================
# HELPERS
# =============================================================================

def load_structure(pdf_name: str) -> dict:
    """
    Load the structure/hierarchy JSON for a given PDF filename.

    Tries the following suffixes in order:
      1. _hierarchy.json  — flat-list format produced by granite_docling_vlm
      2. _structure.json  — original nested-tree format

    Raises FileNotFoundError if neither file exists.
    """
    base = os.path.splitext(pdf_name)[0]
    for suffix in ("_hierarchy.json", "_structure.json"):
        candidate = os.path.join(STRUCTURE_DIR, f"{base}{suffix}")
        if os.path.exists(candidate):
            print(f"  [structure] loading {suffix} → {candidate}")
            with open(candidate, "r") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No structure file found for '{pdf_name}' in {STRUCTURE_DIR}. "
        f"Tried: {base}_hierarchy.json, {base}_structure.json"
    )


def load_md_pages(doc_name: str) -> list | None:
    """
    Load per-page markdown files from:
        {GLM_OCR_DIR}/{doc_name}/pages/page_1.md, page_2.md, ...

    Returns a list of {'page': int, 'content': str} dicts sorted by page number,
    or None if the folder doesn't exist or has no .md files.
    Injected into doc_info['pages'] so retrieve.py uses these instead of PyPDF2.
    """
    pages_dir = os.path.join(GLM_OCR_DIR, doc_name, "pages")
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
        print(f"  [md] no .md files in: {pages_dir} — falling back to PDF")
        return None

    pages.sort(key=lambda x: x["page"])
    print(f"  [md] loaded {len(pages)} markdown pages from {pages_dir}")
    return pages


def detect_flat_structure(nodes: list) -> bool:
    """
    Return True when every node in the list has an empty 'nodes' list,
    indicating a flat section index rather than a nested tree.
    Used to select the appropriate LLM prompt in tree_search().
    """
    return all(not node.get("nodes") for node in nodes)


def build_documents(pdf_name: str, structure: dict) -> tuple[dict, str]:
    """
    Build the documents dict expected by retrieve.py.
    Returns (documents_dict, doc_id).

    Supports two JSON layouts:
      - Flat format  (_hierarchy.json): top-level key is "hierarchy"
      - Tree format  (_structure.json): top-level key is "structure"
    The correct key is selected automatically; if both are absent an empty
    list is used and a warning is printed.

    doc_description falls back to the "doc_name" field (present in the flat
    format) when the dedicated "doc_description" key is missing.

    When USE_MD = True, per-page markdown files from GLM_OCR_DIR are injected
    into doc_info['pages']. retrieve.py checks this field first and uses the
    cached pages instead of opening the PDF via PyPDF2.
    """
    doc_id   = os.path.splitext(pdf_name)[0]
    pdf_path = os.path.join(PDF_DIR, pdf_name)

    # ── Resolve node list — handle both key names ──────────────────────────
    if "hierarchy" in structure:
        nodes = structure["hierarchy"]           # flat format
    elif "structure" in structure:
        nodes = structure["structure"]           # original tree format
    else:
        nodes = []
        print(f"  [WARN] Neither 'hierarchy' nor 'structure' key found in structure file "
              f"for '{pdf_name}' — node list will be empty.")

    # ── doc_description — absent in flat format, fall back to doc_name ─────
    doc_description = (
        structure.get("doc_description")
        or structure.get("doc_name", "")
    )

    cached_pages = None
    if USE_MD:
        cached_pages = load_md_pages(doc_id)

    documents = {
        doc_id: {
            "type":            "pdf",
            "doc_name":        pdf_name,
            "doc_description": doc_description,
            "path":            pdf_path,
            "structure":       nodes,
            "pages":           cached_pages,
        }
    }
    return documents, doc_id


def parse_page_reference(page_reference: str) -> tuple[int | None, int | None]:
    """
    Parse a page_reference string into (start_page, end_page) integers.

    Handles formats:
      "Pages 5-6"   -> (5, 6)
      "Page 12"     -> (12, 12)
      "5-6"         -> (5, 6)
      "12"          -> (12, 12)
      "pages 3, 8"  -> (3, 8)   # treats first and last as range
    Returns (None, None) if parsing fails.
    """
    if not page_reference:
        return None, None

    # Strip leading label like "Pages ", "Page ", "pages " etc.
    cleaned = re.sub(r"(?i)^pages?\s*", "", page_reference.strip())

    # Try "N-M" range
    m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", cleaned)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Try comma-separated list — use first and last as range bounds
    nums = re.findall(r"\d+", cleaned)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[-1])
    if len(nums) == 1:
        n = int(nums[0])
        return n, n

    return None, None


def call_with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) and retry up to MAX_RETRIES times on any exception,
    with exponential backoff starting at RETRY_BACKOFF seconds.

    Covers all three LLM calls: tree_search, generate_answer, llm_judge.
    If all attempts fail the last exception is re-raised so process_question
    can record it as an error.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):   # +2: 1 first try + MAX_RETRIES retries
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt <= MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))   # 2s, 4s, 8s ...
                print(f"  ↺ attempt {attempt} failed ({e.__class__.__name__}: {e}) "
                      f"— retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                print(f"  ✗ all {MAX_RETRIES + 1} attempts failed: {e}")
    raise last_exc


def add_prefix_summaries(nodes: list, parent_prefix: str = "") -> None:
    """
    Walk the tree and attach a prefix_summary to every node.
    prefix_summary = concatenation of all ancestor summaries above this node.
    This gives the LLM full parent context when reasoning about deep nodes,
    exactly as PageIndex does internally.
    Mutates the nodes in place.
    """
    for node in nodes:
        node["prefix_summary"] = parent_prefix
        own_summary  = node.get("summary", "")
        next_prefix  = (parent_prefix + "\n" + own_summary).strip() if own_summary else parent_prefix
        children     = node.get("nodes", [])
        if children:
            add_prefix_summaries(children, next_prefix)


def build_node_index(nodes: list, index: dict | None = None) -> dict:
    """
    Flatten the tree into a dict keyed by node_id for O(1) lookup.
    Used to resolve node IDs returned by tree_search into page ranges
    without trusting the LLM to produce correct page numbers.
    """
    if index is None:
        index = {}
    for node in nodes:
        index[node["node_id"]] = node
        children = node.get("nodes", [])
        if children:
            build_node_index(children, index)
    return index


def tree_search(query: str, tree_structure_json: str, is_flat: bool = False) -> list[str]:
    """
    Step 2 — Ask LLM to identify relevant nodes from the document structure.
    Returns a list of node_id strings ONLY — page numbers are looked up
    programmatically afterward, never trusted from LLM output.

    When is_flat=True the prompt describes the input as a flat section index
    (no parent/child nesting) so the LLM scans the full list rather than
    trying to navigate a tree hierarchy.  When is_flat=False the original
    tree-navigation prompt is used, matching the official PageIndex prompt.
    """
    if is_flat:
        prompt = f"""You are given a query and a flat section index of a document.
Each entry has a node_id, title, and page range (start_index to end_index).
There is no parent-child nesting — every entry is an independent section.
Identify ALL sections whose title or page range are likely to contain the answer.
Be inclusive: when in doubt, include the section.

Query: {query}

Document sections (flat list): {tree_structure_json}

Reply in the following JSON format:
{{
  "thinking": <your reasoning about which sections are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}"""
    else:
        prompt = f"""You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {tree_structure_json}

Reply in the following JSON format:
{{
  "thinking": <your reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("node_list", [])


def resolve_nodes(node_ids: list[str], node_index: dict) -> list[dict]:
    """
    Convert a list of node_id strings into full node dicts
    by looking them up in the pre-built node_index.
    Any node_id not found in the index is silently skipped.
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


def get_page_range_string(nodes: list[dict]) -> str:
    """
    Convert resolved node dicts into a compact pages string for get_page_content.
    Deduplicates and sorts pages to avoid fetching the same page twice,
    which directly reduces over-retrieval and improves precision.
    """
    pages = set()
    for node in nodes:
        start = node.get("start_index")
        end   = node.get("end_index")
        if start is not None and end is not None:
            pages.update(range(start, end + 1))

    if not pages:
        return ""

    # Build compact ranges from sorted unique pages
    sorted_pages = sorted(pages)
    ranges       = []
    range_start  = sorted_pages[0]
    range_end    = sorted_pages[0]

    for p in sorted_pages[1:]:
        if p == range_end + 1:
            range_end = p
        else:
            ranges.append(f"{range_start}-{range_end}" if range_start != range_end else str(range_start))
            range_start = p
            range_end   = p
    ranges.append(f"{range_start}-{range_end}" if range_start != range_end else str(range_start))

    return ",".join(ranges)


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


def check_retrieval_overlap(retrieved_nodes: list[dict], start_page: int, end_page: int) -> dict:
    """
    Step 5a — No LLM needed.
    Compares retrieved page ranges against the gold start_page/end_page.
    Returns hit flag, recall, precision, F1, and which pages overlapped.
    """
    gold_pages = set(range(start_page, end_page + 1))

    retrieved_pages = set()
    for node in retrieved_nodes:
        s = node.get("start_index")
        e = node.get("end_index")
        if s is not None and e is not None:
            retrieved_pages.update(range(s, e + 1))

    overlap   = gold_pages & retrieved_pages
    recall    = round(len(overlap) / len(gold_pages),       2) if gold_pages       else 0.0
    precision = round(len(overlap) / len(retrieved_pages),  2) if retrieved_pages  else 0.0
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


def llm_judge(
    question:          str,
    ground_truth:      str,
    generated_answer:  str,
    evidence_snippets: list[str],
    source_document:   str,
) -> dict:
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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_question(q: dict, index: int, total: int) -> dict:
    """
    Run the full pipeline for a single question and return its result dict.
    Called either directly (sequential) or from a thread (parallel).
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
        # Step 1 — Load structure + build documents dict
        structure = load_structure(pdf_name)
        documents, doc_id = build_documents(pdf_name, structure)

        # Step 1b — Build node index + inject prefix_summaries into tree
        # prefix_summary propagates parent context to every node so the LLM
        # can reason about deep nodes with full ancestor context (PageIndex style).
        # For flat structures every prefix_summary will be "" (no ancestors) — correct.
        tree_nodes = documents[doc_id]["structure"]
        add_prefix_summaries(tree_nodes)
        node_index = build_node_index(tree_nodes)

        # Detect whether structure is a flat list (all nodes have empty "nodes")
        # so tree_search() can use the appropriate LLM prompt.
        is_flat = detect_flat_structure(tree_nodes)
        print(f"  [structure] {'flat section index' if is_flat else 'nested tree'} "
              f"— {len(tree_nodes)} top-level node(s), {len(node_index)} total node(s)")

        # Step 2 — Tree/section search: LLM returns only node_ids (no page numbers)
        tree_json = get_document_structure(documents, doc_id)
        node_ids  = call_with_retry(tree_search, query, tree_json, is_flat)
        print(f"  → {len(node_ids)} node_id(s) returned by LLM")

        # Step 2b — Resolve node_ids → page ranges programmatically from tree
        # Never trust the LLM for page numbers — look them up from the index
        relevant_nodes = resolve_nodes(node_ids, node_index)
        print(f"  → {len(relevant_nodes)} node(s) resolved from tree index")

        # Step 3 — Extract page content for the retrieved nodes
        page_range = get_page_range_string(relevant_nodes)
        if not page_range:
            raise ValueError("Tree search returned no resolvable nodes with page ranges.")
        raw_content   = get_page_content(documents, doc_id, page_range)
        page_contents = json.loads(raw_content)

        # Step 4 — Generate answer from extracted pages
        answer = call_with_retry(generate_answer, query, page_contents)
        print(f"  → Answer: {answer[:120]}...")

        # Step 5a — Retrieval evaluation (no LLM)
        if start_page is not None and end_page is not None:
            retrieval_eval = check_retrieval_overlap(relevant_nodes, start_page, end_page)
        else:
            retrieval_eval = {
                "retrieval_hit": None,
                "note": f"could not parse page range from: {page_reference!r}",
            }
        print(f"  → Retrieval hit: {retrieval_eval.get('retrieval_hit')} | "
              f"Recall: {retrieval_eval.get('recall', 'N/A')}")

        # Step 5b — LLM judge
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
            "id":               qid,
            "question":         query,
            "question_type":    question_type,
            "difficulty":       difficulty,
            "source_document":  pdf_name,
            "page_reference":   page_reference,
            "gold_start_page":  start_page,
            "gold_end_page":    end_page,
            "retrieved_nodes":  relevant_nodes,
            "pages_used":       page_range,
            "answer":           answer,
            "ground_truth":     ground_truth,
            "evidence_snippets": evidence_snippets,
            "retrieval_eval":   retrieval_eval,
            "evaluation":       evaluation,
            "status":           "success",
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


def run_pipeline():
    with open(QUERIES_FILE, "r") as f:
        data = json.load(f)

    # Support bare list OR {"questions": [...]} OR {"dataset_info": ..., "questions": [...]}
    if isinstance(data, list):
        questions    = data
        dataset_info = {}
    else:
        questions    = data.get("questions", [])
        dataset_info = data.get("dataset_info", {})

    total = len(questions)
    print(f"Loaded {total} questions from {QUERIES_FILE}")
    if dataset_info:
        print(f"Dataset info: {dataset_info}")

    # ── Decide parallel vs sequential ────────────────────────────────────────
    # Parallel is only available when USE_OLLAMA = True and OLLAMA_PARALLEL > 1.
    # Ollama serves multiple requests concurrently out of the box; the NVIDIA API
    # path is left sequential to respect rate limits.
    workers = None
    if USE_OLLAMA and OLLAMA_PARALLEL and OLLAMA_PARALLEL > 1:
        workers = OLLAMA_PARALLEL
        print(f"[parallel] Ollama parallel workers: {workers}")
    else:
        print(f"[sequential] processing {total} questions one at a time")

    results_map: dict[int, dict] = {}   # index → result, preserves original order

    if workers:
        # ── Parallel path ─────────────────────────────────────────────────
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
        # ── Sequential path ───────────────────────────────────────────────
        for i, q in enumerate(questions, 1):
            results_map[i] = process_question(q, i, total)
            time.sleep(SLEEP_BETWEEN_Q)

    # Re-assemble in original question order
    results = [results_map[i] for i in range(1, total + 1)]

    # ── Summary stats ────────────────────────────────────────────────────────
    success  = sum(1 for r in results if r["status"] == "success")
    correct  = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "correct")
    partial  = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "partial")
    ret_hits = sum(1 for r in results if r.get("retrieval_eval", {}).get("retrieval_hit"))

    avg_correctness = round(
        sum(r.get("evaluation", {}).get("correctness_score", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )
    avg_completeness = round(
        sum(r.get("evaluation", {}).get("completeness_score", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )
    avg_recall = round(
        sum(r.get("retrieval_eval", {}).get("recall", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )
    avg_precision = round(
        sum(r.get("retrieval_eval", {}).get("precision", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )
    avg_f1 = round(
        sum(r.get("retrieval_eval", {}).get("f1", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )

    # Breakdown by question_type
    qtypes: dict = {}
    for r in results:
        if r["status"] != "success":
            continue
        qt      = r.get("question_type", "unknown") or "unknown"
        verdict = r.get("evaluation", {}).get("verdict", "unknown")
        if qt not in qtypes:
            qtypes[qt] = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
        qtypes[qt]["total"] += 1
        if verdict in qtypes[qt]:
            qtypes[qt][verdict] += 1

    # Breakdown by difficulty (new in v2)
    difficulties: dict = {}
    for r in results:
        if r["status"] != "success":
            continue
        diff    = r.get("difficulty", "unknown") or "unknown"
        verdict = r.get("evaluation", {}).get("verdict", "unknown")
        if diff not in difficulties:
            difficulties[diff] = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
        difficulties[diff]["total"] += 1
        if verdict in difficulties[diff]:
            difficulties[diff][verdict] += 1

    output = {
        "dataset_info": dataset_info,
        "total":        total,
        "successful":   success,
        "errors":       total - success,
        "summary": {
            "retrieval_hits":          ret_hits,
            "retrieval_hit_rate":      round(ret_hits / max(success, 1), 3),
            "avg_retrieval_recall":    avg_recall,
            "avg_retrieval_precision": avg_precision,
            "avg_retrieval_f1":        avg_f1,
            "correct":                 correct,
            "partial":                 partial,
            "incorrect":               success - correct - partial,
            "avg_correctness_score":   avg_correctness,
            "avg_completeness_score":  avg_completeness,
        },
        "breakdown_by_question_type": qtypes,
        "breakdown_by_difficulty":    difficulties,
        "results": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Done — {success}/{total} successful  ({total - success} errors)")
    print(f"   Retrieval hit rate    : {ret_hits}/{success}")
    print(f"   Avg recall / prec / F1: {avg_recall} / {avg_precision} / {avg_f1}")
    print(f"   Correct / Partial     : {correct} / {partial}")
    print(f"   Avg correctness       : {avg_correctness}")
    print(f"   Avg completeness      : {avg_completeness}")
    print(f"   Results saved to      : {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_pipeline()