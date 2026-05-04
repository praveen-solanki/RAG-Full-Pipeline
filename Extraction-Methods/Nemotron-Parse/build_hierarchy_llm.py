"""
Hierarchical JSON Builder — Method 1 (LLM-based)
==================================================
Uses an LLM to read the *_markdown_bbox.json produced by
nemotron_parse_pipeline.py and directly generate a hierarchical
structure JSON matching the reference schema.

Supports two providers:
  --provider nvidia   → NVIDIA NIM API (cloud, needs API key in .env)
  --provider ollama   → Ollama (local, needs Ollama running)

Output schema (identical to build_hierarchy.py / Method 2):
  {
    "doc_name":  "filename.pdf",
    "structure": [
      {
        "title":       string,
        "node_id":     "0000",
        "start_index": int,
        "end_index":   int,
        "summary":     string,
        "nodes":       [ ... ]   ← only present if non-empty
      }
    ]
  }

.env file (same directory as this script):
  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx

Usage:

  # Single file, using Ollama (local)
  python build_hierarchy_llm.py --input output/MyDoc/MyDoc_markdown_bbox.json

  # Single file, using NVIDIA NIM
  python build_hierarchy_llm.py --input output/MyDoc/MyDoc_markdown_bbox.json \\
      --provider nvidia

  # Entire results directory, using Ollama
  python build_hierarchy_llm.py --results_dir output/

  # Entire results directory, using NVIDIA NIM with a specific model
  python build_hierarchy_llm.py --results_dir output/ \\
      --provider nvidia --nvidia_model mistral-nemo

  # Custom Ollama model
  python build_hierarchy_llm.py --results_dir output/ \\
      --provider ollama --ollama_model llama3

  # Process in chunks (for large documents that exceed context window)
  python build_hierarchy_llm.py --results_dir output/ --chunk_size 30

Requirements:
  pip install requests python-dotenv
  For Ollama: ollama pull mistral  (or any model)
  For NVIDIA: NVIDIA_API_KEY in .env
"""

import json
import re
import sys
import argparse
import requests
import time
from pathlib import Path
from typing import Optional

# ─── Load .env ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_OLLAMA_HOST    = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL   = "mistral"
DEFAULT_NVIDIA_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NVIDIA_MODEL   = "moonshotai/kimi-k2-instruct-0905"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD  markdown_bbox  JSON
# ══════════════════════════════════════════════════════════════════════════════

def load_markdown_bbox_json(json_path: Path) -> tuple:
    """
    Load a *_markdown_bbox.json file.
    Returns (doc_name: str, pages: list, flat_elements: list).
    flat_elements: each item is {type, text, page} — compact, no bbox.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "pages" not in data:
        raise ValueError(
            f"'{json_path.name}' has no 'pages' key — "
            "pass a *_markdown_bbox.json file."
        )

    doc_name      = data.get("doc_name", json_path.stem)
    flat_elements = []

    for page_entry in data.get("pages", []):
        page_num = page_entry.get("page", 0)
        result   = page_entry.get("result")
        if result is None:
            continue
        if isinstance(result, list):
            for elem in result:
                t = elem.get("type", "")
                x = elem.get("text", "").strip()
                if x and t not in ("Page-header", "Page-footer"):
                    flat_elements.append({"type": t, "text": x, "page": page_num})
        elif isinstance(result, dict) and "text" in result:
            x = result["text"].strip()
            if x:
                flat_elements.append({"type": "Text", "text": x, "page": page_num})

    return doc_name, flat_elements


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CHUNK ELEMENTS  (for large documents)
# ══════════════════════════════════════════════════════════════════════════════

def chunk_elements(flat_elements: list, chunk_size: int) -> list:
    """
    Split flat_elements into chunks of `chunk_size` pages each.
    Each chunk is a list of elements whose page falls in that range.
    """
    if chunk_size <= 0:
        return [flat_elements]

    if not flat_elements:
        return []

    max_page = max(e["page"] for e in flat_elements)
    chunks   = []

    for start in range(1, max_page + 1, chunk_size):
        end   = start + chunk_size - 1
        chunk = [e for e in flat_elements if start <= e["page"] <= end]
        if chunk:
            chunks.append(chunk)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a technical documentation structure extractor.

You will receive a list of document elements extracted from a PDF. Each element has:
- "type": the element type (Title, Section-header, Text, List-item, Table, Caption, etc.)
- "text": the extracted text content
- "page": the page number it appears on

Your task is to build a hierarchical JSON structure from these elements.

Rules:
1. Identify headings from their type (Title, Section-header) and text patterns (numbered sections like "1 Introduction", "2.1 Overview", "A Reference Material").
2. Group body content (Text, List-item, Table, etc.) under the nearest preceding heading.
3. Build a nested hierarchy based on section numbering depth:
   - "1 Introduction" → depth 0 (top-level)
   - "1.1 Sub-section" → depth 1 (child of "1 Introduction")
   - "1.1.1 Detail" → depth 2 (child of "1.1 Sub-section")
   - "A Appendix" → depth 0
   - "A.1 Sub" → depth 1
4. All pages before the first real numbered section heading are "Preface".
5. Do NOT create nodes for: TOC entries (lines with "....." or trailing page numbers), constraint labels like [AP_TPS_...] or [constr_...], running page headers/footers.
6. node_id values are zero-padded 4-digit integers starting from 0000, incrementing in document order.
7. start_index and end_index are page numbers (integers).
8. summary: write a concise 3-5 sentence technical summary of the content under each node.
9. nodes key: only include it if the node has children. Leaf nodes must NOT have a nodes key.
10. Keep section numbers in titles exactly as they appear (do not strip them).

Output ONLY valid JSON. No explanations, no markdown code fences, no preamble.
The output must exactly follow this schema:
{
  "doc_name": "filename.pdf",
  "structure": [
    {
      "title": "...",
      "node_id": "0000",
      "start_index": 1,
      "end_index": 5,
      "summary": "...",
      "nodes": [ ... ]
    }
  ]
}"""


def build_prompt(doc_name: str, elements: list, node_id_start: int = 0) -> str:
    """Build the user prompt from document elements."""
    # Compact representation — only type, text, page (no bbox to save tokens)
    compact = [
        {"type": e["type"], "text": e["text"], "page": e["page"]}
        for e in elements
    ]
    elements_json = json.dumps(compact, ensure_ascii=False, indent=None)

    return (
        f"Document name: {doc_name}\n"
        f"Node ID counter starts at: {node_id_start:04d}\n\n"
        f"Document elements:\n{elements_json}\n\n"
        f"Build the hierarchical structure JSON for this document."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LLM CALL  — OLLAMA
# ══════════════════════════════════════════════════════════════════════════════

def _check_ollama(host: str, model: str) -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if model.split(":")[0] not in available:
            print(f"  [WARN] Model '{model}' not in Ollama. Available: {available}")
            print(f"         Run:  ollama pull {model}")
            return False
        return True
    except Exception as e:
        print(f"  [WARN] Cannot reach Ollama at {host}: {e}")
        return False


def call_ollama(system: str, user: str,
                host: str, model: str,
                max_retries: int = 3) -> str:
    """Call Ollama chat API. Returns raw response text."""
    payload = {
        "model":    model,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 8192},
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(f"{host}/api/chat",
                                 json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            print(f"    [!] Ollama attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LLM CALL  — NVIDIA NIM
# ══════════════════════════════════════════════════════════════════════════════

def call_nvidia(system: str, user: str,
                api_key: str, model: str,
                max_retries: int = 3) -> str:
    """Call NVIDIA NIM chat completions API. Returns raw response text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept":        "application/json",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        "temperature": 0.1,
        "max_tokens":  8192,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(DEFAULT_NVIDIA_URL,
                                 headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            print(f"    [!] NVIDIA attempt {attempt}: "
                  f"{resp.status_code} — {resp.text[:200]}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise
        except Exception as e:
            print(f"    [!] NVIDIA attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PARSE LLM RESPONSE → VALID JSON
# ══════════════════════════════════════════════════════════════════════════════

def parse_llm_response(raw: str) -> dict:
    """
    Extract and parse JSON from LLM response.
    Handles cases where the LLM wraps output in markdown code fences.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned.strip())

    # Find the first { to last } in case there's any preamble left
    start = cleaned.find('{')
    end   = cleaned.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")

    json_str = cleaned[start:end+1]
    return json.loads(json_str)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MERGE CHUNKS  (when document is processed in multiple chunks)
# ══════════════════════════════════════════════════════════════════════════════

def merge_chunk_structures(doc_name: str, chunk_results: list) -> dict:
    """
    Merge multiple partial structure results (from chunked processing)
    into one final structure. Renumbers node_ids sequentially.
    """
    merged_structure = []
    for result in chunk_results:
        merged_structure.extend(result.get("structure", []))

    # Renumber node_ids sequentially across all chunks
    counter = [0]

    def renumber(nodes: list):
        for node in nodes:
            node["node_id"] = f"{counter[0]:04d}"
            counter[0] += 1
            if "nodes" in node:
                renumber(node["nodes"])

    renumber(merged_structure)

    return {"doc_name": doc_name, "structure": merged_structure}


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PROCESS ONE FILE
# ══════════════════════════════════════════════════════════════════════════════

def process_one(json_path: Path,
                output_dir: Optional[Path],
                provider: str,
                ollama_host: str,
                ollama_model: str,
                nvidia_api_key: str,
                nvidia_model: str,
                chunk_size: int) -> bool:
    """
    Full LLM pipeline for one *_markdown_bbox.json file.
    Returns True on success, False on failure.
    """
    try:
        print(f"\n  [→] Loading : {json_path.name}")
        doc_name, flat_elements = load_markdown_bbox_json(json_path)

        if not flat_elements:
            print(f"  [WARN] No elements — skipping {json_path.name}")
            return False

        print(f"  [✓] Elements : {len(flat_elements)} | Doc: {doc_name}")

        # ── Split into chunks if needed ───────────────────────────────────────
        chunks = chunk_elements(flat_elements, chunk_size)
        print(f"  [✓] Chunks   : {len(chunks)} "
              f"({'single pass' if len(chunks) == 1 else f'{chunk_size} pages each'})")

        chunk_results = []
        node_id_offset = 0

        for chunk_idx, chunk in enumerate(chunks, 1):
            if len(chunks) > 1:
                pages = [e["page"] for e in chunk]
                print(f"  [→] Chunk {chunk_idx}/{len(chunks)} "
                      f"(pages {min(pages)}–{max(pages)}) …")
            else:
                print(f"  [→] Calling {provider.upper()} …")

            user_prompt = build_prompt(doc_name, chunk, node_id_offset)

            # ── Call the selected provider ────────────────────────────────────
            if provider == "ollama":
                raw = call_ollama(SYSTEM_PROMPT, user_prompt,
                                  ollama_host, ollama_model)
            else:  # nvidia
                raw = call_nvidia(SYSTEM_PROMPT, user_prompt,
                                  nvidia_api_key, nvidia_model)

            # ── Parse response ────────────────────────────────────────────────
            result = parse_llm_response(raw)
            chunk_results.append(result)

            # Advance node_id counter for next chunk
            def _count_nodes(nodes):
                total = 0
                for n in nodes:
                    total += 1
                    total += _count_nodes(n.get("nodes", []))
                return total
            node_id_offset += _count_nodes(result.get("structure", []))

        # ── Merge chunks ──────────────────────────────────────────────────────
        if len(chunk_results) == 1:
            final = chunk_results[0]
            # Ensure doc_name is correct
            final["doc_name"] = doc_name
        else:
            print(f"  [→] Merging {len(chunk_results)} chunks …")
            final = merge_chunk_structures(doc_name, chunk_results)

        total_nodes = sum(1 for _ in _iter_nodes(final.get("structure", [])))
        print(f"  [✓] Nodes    : {total_nodes}")

        # ── Save ──────────────────────────────────────────────────────────────
        save_dir = output_dir if output_dir else json_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        stem     = json_path.stem.replace("_markdown_bbox", "")
        out_path = save_dir / f"{stem}_structure.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

        print(f"  [✓] Saved    : {out_path}")
        return True

    except (FileNotFoundError, ValueError) as e:
        print(f"  [ERROR] {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Unexpected: {e}")
        return False


def _iter_nodes(nodes: list):
    """Flatten all nodes recursively for counting."""
    for node in nodes:
        yield node
        yield from _iter_nodes(node.get("nodes", []))


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DISCOVER FILES
# ══════════════════════════════════════════════════════════════════════════════

def discover_bbox_files(results_dir: str) -> list:
    """Find all *_markdown_bbox.json files one level deep in results_dir."""
    root = Path(results_dir)
    if not root.is_dir():
        print(f"[ERROR] Not a valid directory: {results_dir}")
        sys.exit(1)
    found = sorted(root.glob("*/*_markdown_bbox.json"))
    if not found:
        print(f"[ERROR] No *_markdown_bbox.json files found under: {results_dir}")
        sys.exit(1)
    return found


# ══════════════════════════════════════════════════════════════════════════════
# 10.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Method 1: LLM-based hierarchical JSON builder.\n"
            "Supports NVIDIA NIM (cloud) and Ollama (local)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
examples:
  # Single file, Ollama (default)
  python build_hierarchy_llm.py --input output/MyDoc/MyDoc_markdown_bbox.json

  # Single file, NVIDIA NIM
  python build_hierarchy_llm.py --input output/MyDoc/MyDoc_markdown_bbox.json \\
      --provider nvidia

  # Entire results directory, Ollama
  python build_hierarchy_llm.py --results_dir /path/to/results/

  # Entire results directory, NVIDIA NIM
  python build_hierarchy_llm.py --results_dir /path/to/results/ --provider nvidia

  # Large documents: process 30 pages at a time
  python build_hierarchy_llm.py --results_dir /path/to/results/ --chunk_size 30

  # Custom models
  python build_hierarchy_llm.py --results_dir /path/to/results/ \\
      --provider ollama --ollama_model llama3
  python build_hierarchy_llm.py --results_dir /path/to/results/ \\
      --provider nvidia --nvidia_model mistralai/mistral-nemo

.env file (same directory as this script):
  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
        """
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", metavar="PATH",
        help="Path to a single *_markdown_bbox.json file"
    )
    input_group.add_argument(
        "--results_dir", metavar="DIR",
        help="Results directory — processes all *_markdown_bbox.json files one level deep"
    )

    # ── Provider ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--provider",
        choices=["ollama", "nvidia"],
        default="ollama",
        help="LLM provider to use (default: ollama)"
    )

    # ── Ollama options ────────────────────────────────────────────────────────
    parser.add_argument(
        "--ollama_model", default=DEFAULT_OLLAMA_MODEL, metavar="MODEL",
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--ollama_host", default=DEFAULT_OLLAMA_HOST, metavar="URL",
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_HOST})"
    )

    # ── NVIDIA options ────────────────────────────────────────────────────────
    parser.add_argument(
        "--nvidia_model", default=DEFAULT_NVIDIA_MODEL, metavar="MODEL",
        help=f"NVIDIA NIM model name (default: {DEFAULT_NVIDIA_MODEL})"
    )

    # ── Other ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", default=None, metavar="DIR",
        help="Where to save output files (default: same folder as input)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=0, metavar="N",
        help=(
            "Process N pages at a time (default: 0 = entire document at once).\n"
            "Use this for large documents that exceed the model context window.\n"
            "Example: --chunk_size 30"
        )
    )

    args = parser.parse_args()

    # ── Validate provider setup ───────────────────────────────────────────────
    nvidia_api_key = ""
    if args.provider == "nvidia":
        nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not nvidia_api_key:
            print("\n[ERROR] NVIDIA_API_KEY not found in .env")
            print("        Add to .env:  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx")
            sys.exit(1)
        print(f"[✓] NVIDIA API key loaded (ends ...{nvidia_api_key[-6:]})")
    else:
        print(f"[→] Checking Ollama at {args.ollama_host} …")
        if not _check_ollama(args.ollama_host, args.ollama_model):
            print("[ERROR] Ollama not ready. Fix the issue above and retry.")
            sys.exit(1)
        print(f"[✓] Ollama ready — model '{args.ollama_model}'")

    # ── Collect files ─────────────────────────────────────────────────────────
    if args.input:
        files = [Path(args.input)]
    else:
        files = discover_bbox_files(args.results_dir)

    total   = len(files)
    out_dir = Path(args.output_dir) if args.output_dir else None

    print(f"\n{'='*60}")
    print(f"  Build Hierarchy — LLM Method")
    print(f"  Provider  : {args.provider.upper()}")
    if args.provider == "nvidia":
        print(f"  Model     : {args.nvidia_model}")
    else:
        print(f"  Model     : {args.ollama_model}")
    print(f"  Files     : {total}")
    print(f"  Chunk size: {'full document' if args.chunk_size == 0 else f'{args.chunk_size} pages'}")
    if out_dir:
        print(f"  Output    : {out_dir.resolve()}")
    else:
        print(f"  Output    : same folder as each input")
    print(f"{'='*60}")

    for i, f in enumerate(files, 1):
        print(f"  [{i}/{total}] {f.parent.name}/{f.name}")

    # ── Process ───────────────────────────────────────────────────────────────
    success_count = 0
    failed        = []

    for idx, json_path in enumerate(files, 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{total}]  {json_path.parent.name}")
        print(f"{'─'*60}")

        ok = process_one(
            json_path      = json_path,
            output_dir     = out_dir,
            provider       = args.provider,
            ollama_host    = args.ollama_host,
            ollama_model   = args.ollama_model,
            nvidia_api_key = nvidia_api_key,
            nvidia_model   = args.nvidia_model,
            chunk_size     = args.chunk_size,
        )

        if ok:
            success_count += 1
        else:
            failed.append(json_path.parent.name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  DONE")
    print(f"  Processed : {success_count} / {total} successfully")
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for name in failed:
            print(f"    ✗  {name}")
    print(f"{'#'*60}\n")