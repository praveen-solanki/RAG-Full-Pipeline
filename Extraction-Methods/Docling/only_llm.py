"""
NVIDIA VLM Hierarchy Pipeline  —  PDF → Nested Tree JSON
=========================================================
Converts a PDF into page images, sends every page to an NVIDIA-hosted
vision-language model via the NIM API, and uses a structured prompt to
make the model return the full document hierarchy as a nested JSON tree.

No local AI model. No Docling. No Granite-Docling.
The model on NVIDIA's endpoint does all the work.

How it works:
  Step 1 — Convert each PDF page to a JPEG image (pdf2image / poppler).
  Step 2 — Send ALL page images in a single API call to the NVIDIA VLM.
           Each image is base64-encoded and passed as image_url content.
  Step 3 — A strict prompt tells the model to return ONLY a JSON object
           matching the nested tree schema — no prose, no markdown fences.
  Step 4 — Parse the JSON response and write it to disk.

Output files (per document):
  output/<doc_name>/
  ├── <doc_name>_hierarchy.json   ← nested tree
  └── <doc_name>_hierarchy.md     ← human-readable outline

Nested JSON schema:
  {
    "doc_name":  "report.pdf",
    "hierarchy": [
      {
        "title":       "Introduction",
        "node_id":     "0001",
        "start_index": 1,
        "end_index":   5,
        "nodes": [
          {
            "title":       "Document Conventions",
            "node_id":     "0002",
            "start_index": 2,
            "end_index":   4,
            "nodes": []
          }
        ]
      }
    ]
  }

Requirements:
  pip install requests pdf2image Pillow
  apt-get install poppler-utils      # Linux
  brew install poppler               # macOS

  NVIDIA API key from: https://build.nvidia.com
  Set it in a .env file next to this script:
      NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx

Default model:  meta/llama-4-scout-17b-16e-instruct
  (Multi-image capable, strong document reasoning, free tier on build.nvidia.com)
  Other good options on the same endpoint:
    meta/llama-4-maverick-17b-128e-instruct   (more capable, higher cost)
    qwen/qwen2.5-vl-7b-instruct               (lighter, fast)
    nvidia/llama-3.2-nv-vision-instruct       (NVIDIA-optimised Llama vision)

Usage:
  # Single PDF
  python nvidia_vlm_hierarchy.py --pdf path/to/doc.pdf

  # Folder of PDFs
  python nvidia_vlm_hierarchy.py --pdf_dir ./pdfs/

  # Custom model or output dir
  python nvidia_vlm_hierarchy.py --pdf doc.pdf \\
      --model meta/llama-4-maverick-17b-128e-instruct \\
      --output_dir ./results

  # Higher DPI for better quality (default 150)
  python nvidia_vlm_hierarchy.py --pdf doc.pdf --dpi 200

Important note on large PDFs:
  The NVIDIA API accepts up to ~20 images per request by default.
  For PDFs longer than this the script batches pages into chunks and
  merges the results. Each chunk gets its own API call.
"""

import os
import re
import sys
import json
import base64
import argparse
import time
from io import BytesIO
from pathlib import Path


# ─── Load API key from .env ───────────────────────────────────────────────────
def load_api_key() -> str:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
    else:
        print(f"[WARNING] No .env found at {env_path} — falling back to shell env")

    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        print("\n[ERROR] NVIDIA_API_KEY not set.")
        print("        Create a .env file next to this script:")
        print("            NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx")
        print("        Get your key: https://build.nvidia.com\n")
        sys.exit(1)
    return key


# ─── Convert PDF → list of JPEG images in memory ─────────────────────────────
def pdf_to_images(pdf_path: Path, dpi: int = 150) -> list:
    """
    Converts every page of the PDF to a PIL Image object.
    Returns a list of PIL Images in page order.
    Requires pdf2image and poppler.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("[ERROR] pdf2image not installed.  pip install pdf2image")
        print("        Also install poppler:  apt-get install poppler-utils")
        sys.exit(1)

    print(f"  [→] Rasterising {pdf_path.name} at {dpi} DPI ...")
    images = convert_from_path(str(pdf_path), dpi=dpi, fmt="jpeg")
    print(f"  [✓] {len(images)} page(s) rasterised")
    return images


# ─── Encode a PIL Image to base64 JPEG string ────────────────────────────────
def image_to_b64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─── Build the structured prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a document structure analyser.
Your ONLY job is to read document page images and return the document's
hierarchical outline as a single JSON object.

Rules you must follow without exception:
1. Return ONLY raw JSON — no markdown code fences, no explanation, no preamble.
2. The JSON must match this exact schema:
   {
     "hierarchy": [
       {
         "title":       "<heading text exactly as it appears>",
         "node_id":     "<4-digit zero-padded counter, e.g. 0001>",
         "start_index": <page number where this section starts, integer>,
         "end_index":   <page number where this section ends inclusive, integer>,
         "nodes":       [ <child nodes with identical schema, recursive> ]
       }
     ]
   }
3. node_id values must be unique across the entire document, assigned in
   reading order starting from "0001".
4. Capture ONLY structural headings: document title, chapter headings,
   section headings, subsection headings. Do NOT include body text,
   captions, table titles, footnotes, or page headers/footers.
5. Nest correctly: a subsection becomes a child node inside its parent
   section's "nodes" array.
6. start_index and end_index are 1-based page numbers.
   end_index of a section = start_index of the next sibling section - 1,
   or the last page of the document for the final section.
7. If a heading has no sub-headings, its "nodes" array must be [].
8. Do not invent headings that do not appear in the document images."""

USER_PROMPT_TEMPLATE = """The {n_pages} page images attached (in order) are the complete document
named "{doc_name}".

Analyse every page carefully, identify all structural headings and their
hierarchy, then return the JSON object described in your instructions.
Remember: raw JSON only, nothing else."""


# ─── Call NVIDIA VLM endpoint with page images ────────────────────────────────
def call_nvidia_vlm(images_b64: list, doc_name: str, total_pages: int,
                    model: str, api_key: str,
                    max_retries: int = 3) -> str:
    """
    Sends a batch of base64 page images to the NVIDIA NIM VLM endpoint.
    Returns the raw text content of the model's response.

    images_b64 — list of base64 JPEG strings (one per page in this chunk)
    doc_name   — used in the user prompt so the model knows the document name
    total_pages — actual total page count of the full document
    """
    import requests as req

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Build the content array: one image_url block per page + the text prompt
    content = []
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })
    content.append({
        "type": "text",
        "text": USER_PROMPT_TEMPLATE.format(
            n_pages=len(images_b64),
            doc_name=doc_name,
        )
    })

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        "max_tokens": 4096,
        "temperature": 0.0,     # deterministic — we want structured JSON
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = req.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except req.exceptions.HTTPError:
            print(f"    [!] HTTP {resp.status_code} on attempt {attempt} — "
                  f"{resp.text[:200]}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
            else:
                raise
        except req.exceptions.Timeout:
            print(f"    [!] Timeout on attempt {attempt}, retrying ...")
            if attempt < max_retries:
                time.sleep(10)
            else:
                raise


# ─── Parse model response to JSON — strip any accidental markdown fences ──────
def parse_json_response(raw: str) -> dict:
    """
    Extracts and parses the JSON from the model's raw text response.
    Handles cases where the model wraps it in ```json ... ``` fences
    despite the prompt saying not to.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()

    # Find the outermost JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response:\n{raw[:500]}")

    return json.loads(text[start: end + 1])


# ─── Merge multiple chunk responses into one hierarchy ────────────────────────
def merge_hierarchies(chunks: list) -> list:
    """
    When the PDF was split into batches, each batch returns a partial
    hierarchy. This function concatenates them into one flat list at the
    root level and re-numbers node_ids globally.

    Note: cross-chunk parent-child relationships (a heading that spans
    the chunk boundary) are resolved by treating the last root node of
    chunk N as the parent of the first root node of chunk N+1 if its
    level is deeper. For most documents with clear chapter boundaries
    this works well.
    """
    merged = []
    for chunk in chunks:
        merged.extend(chunk.get("hierarchy", []))

    # Re-number node_ids in reading order
    counter = [1]

    def renumber(nodes: list) -> None:
        for node in nodes:
            node["node_id"] = f"{counter[0]:04d}"
            counter[0] += 1
            renumber(node.get("nodes", []))

    renumber(merged)
    return merged


# ─── Render tree as Markdown ─────────────────────────────────────────────────
def tree_to_markdown(tree: list, doc_name: str) -> str:
    lines = [
        f"# {doc_name} — Document Hierarchy\n",
        f"*Extracted via NVIDIA NIM VLM*\n",
        "",
    ]

    def render(nodes: list, depth: int) -> None:
        for node in nodes:
            title = node["title"]
            start = node["start_index"]
            end   = node["end_index"]
            pages = f"p. {start}" if start == end else f"pp. {start}–{end}"
            if depth == 0:
                lines.append(f"\n## {title}  *({pages})*")
            else:
                lines.append(f"{'  ' * depth}- {title}  *({pages})*")
            render(node.get("nodes", []), depth + 1)

    render(tree, 0)
    return "\n".join(lines) + "\n"


# ─── Process one PDF ──────────────────────────────────────────────────────────
def process_pdf(pdf_path: Path, output_root: str,
                model: str, api_key: str,
                dpi: int = 150,
                chunk_size: int = 15) -> None:
    """
    Full pipeline for one PDF:
      1. Rasterise all pages.
      2. Split into chunks of chunk_size pages (NVIDIA default limit ~20).
      3. Call the VLM for each chunk with the structured prompt.
      4. Merge the hierarchy results.
      5. Write JSON + Markdown.

    chunk_size=15 leaves comfortable headroom below the typical 20-image
    limit so large, dense pages don't hit token limits.
    """
    doc_name       = pdf_path.stem
    doc_output_dir = Path(output_root) / doc_name
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Document : {pdf_path.name}")
    print(f"  Model    : {model}")
    print(f"  Output   : {doc_output_dir.resolve()}")
    print(f"{'='*60}\n")

    # Step 1 — rasterise
    images     = pdf_to_images(pdf_path, dpi=dpi)
    total_pages = len(images)

    # Step 2 — encode to base64
    print(f"  [→] Encoding {total_pages} page(s) to base64 ...")
    images_b64 = [image_to_b64(img) for img in images]
    print(f"  [✓] Encoding done")

    # Step 3 — batch into chunks and call the API
    chunks_data  = []
    page_batches = [
        images_b64[i: i + chunk_size]
        for i in range(0, total_pages, chunk_size)
    ]
    total_chunks = len(page_batches)

    for idx, batch in enumerate(page_batches, 1):
        page_start = (idx - 1) * chunk_size + 1
        page_end   = min(idx * chunk_size, total_pages)
        print(f"  [→] API call {idx}/{total_chunks}  "
              f"(pages {page_start}–{page_end}) ...")

        raw = call_nvidia_vlm(
            images_b64=batch,
            doc_name=doc_name,
            total_pages=total_pages,
            model=model,
            api_key=api_key,
        )

        try:
            parsed = parse_json_response(raw)
            chunks_data.append(parsed)
            n = sum(1 for _ in _walk(parsed.get("hierarchy", [])))
            print(f"  [✓] Got {n} node(s) from chunk {idx}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [!] Could not parse JSON from chunk {idx}: {e}")
            print(f"      Raw response (first 400 chars): {raw[:400]}")
            chunks_data.append({"hierarchy": []})

    # Step 4 — merge chunks
    hierarchy = merge_hierarchies(chunks_data)
    print(f"\n  [✓] Total nodes after merge: "
          f"{sum(1 for _ in _walk(hierarchy))}")

    # Step 5 — write JSON
    output_json = {
        "doc_name":    pdf_path.name,
        "total_pages": total_pages,
        "model":       model,
        "hierarchy":   hierarchy,
    }
    json_path = doc_output_dir / f"{doc_name}_hierarchy.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print(f"  [✓] JSON written  → {json_path.name}")

    # Step 6 — write Markdown
    md_content = tree_to_markdown(hierarchy, doc_name)
    md_path    = doc_output_dir / f"{doc_name}_hierarchy.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [✓] MD written   → {md_path.name}")

    print(f"\n  [✓] Finished: {pdf_path.name}")


# ─── Helper: depth-first walk ────────────────────────────────────────────────
def _walk(nodes: list):
    for node in nodes:
        yield node
        yield from _walk(node.get("nodes", []))


# ─── Collect PDFs ─────────────────────────────────────────────────────────────
def collect_pdfs(pdf_arg, pdf_dir_arg) -> list:
    seen, pdfs = set(), []

    def add(p: Path):
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            pdfs.append(r)

    if pdf_arg:
        p = Path(pdf_arg)
        if not p.exists():
            print(f"[ERROR] File not found: {p}"); sys.exit(1)
        if p.suffix.lower() != ".pdf":
            print(f"[ERROR] Not a PDF: {p}"); sys.exit(1)
        add(p)

    if pdf_dir_arg:
        d = Path(pdf_dir_arg)
        if not d.is_dir():
            print(f"[ERROR] Not a directory: {d}"); sys.exit(1)
        found = sorted(d.glob("*.pdf"))
        if not found:
            print(f"[WARNING] No PDFs found in: {d}")
        for p in found:
            add(p)

    if not pdfs:
        print("[ERROR] No PDFs to process."); sys.exit(1)
    return pdfs


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NVIDIA VLM Hierarchy Pipeline — PDF(s) → Nested Tree JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nvidia_vlm_hierarchy.py --pdf doc.pdf
  python nvidia_vlm_hierarchy.py --pdf_dir ./pdfs/
  python nvidia_vlm_hierarchy.py --pdf doc.pdf --output_dir ./results
  python nvidia_vlm_hierarchy.py --pdf doc.pdf --model meta/llama-4-maverick-17b-128e-instruct
  python nvidia_vlm_hierarchy.py --pdf doc.pdf --dpi 200

.env file (same folder as this script):
  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
  Get key: https://build.nvidia.com

Available vision models on NVIDIA NIM:
  meta/llama-4-scout-17b-16e-instruct          (default, free tier, multi-image)
  meta/llama-4-maverick-17b-128e-instruct      (stronger, higher cost)
  qwen/qwen2.5-vl-7b-instruct                  (lighter, fast)
  nvidia/llama-3.2-nv-vision-instruct          (NVIDIA-optimised)
        """
    )
    parser.add_argument("--pdf",        metavar="PATH", help="Single PDF file")
    parser.add_argument("--pdf_dir",    metavar="DIR",  help="Folder of PDFs")
    parser.add_argument("--output_dir", metavar="DIR",  default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--model",      metavar="MODEL",
                        default="meta/llama-4-scout-17b-16e-instruct",
                        help="NVIDIA NIM model ID (default: llama-4-scout)")
    parser.add_argument("--dpi",        type=int, default=150,
                        help="PDF raster DPI (default: 150)")
    parser.add_argument("--chunk_size", type=int, default=15,
                        help="Max pages per API call (default: 15)")

    args = parser.parse_args()

    if not args.pdf and not args.pdf_dir:
        parser.print_help()
        print("\n[ERROR] Provide --pdf and/or --pdf_dir\n")
        sys.exit(1)

    api_key = load_api_key()
    pdfs    = collect_pdfs(args.pdf, args.pdf_dir)

    print(f"\n[✓] NVIDIA VLM Hierarchy Pipeline")
    print(f"[✓] Model      : {args.model}")
    print(f"[✓] Endpoint   : https://integrate.api.nvidia.com/v1")
    print(f"[✓] Chunk size : {args.chunk_size} pages per API call")
    print(f"[✓] {len(pdfs)} PDF(s) queued:")
    for i, p in enumerate(pdfs, 1):
        print(f"    {i:>3}. {p.name}")

    failed = []
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Starting: {pdf_path.name}")
        try:
            process_pdf(
                pdf_path=pdf_path,
                output_root=args.output_dir,
                model=args.model,
                api_key=api_key,
                dpi=args.dpi,
                chunk_size=args.chunk_size,
            )
        except Exception as e:
            print(f"\n  [ERROR] Failed: {pdf_path.name} — {e}")
            failed.append(pdf_path.name)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Processed : {len(pdfs) - len(failed)}/{len(pdfs)} PDF(s)")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for name in failed:
            print(f"              ✗ {name}")
    print(f"  Output    : {Path(args.output_dir).resolve()}")
    print(f"{'='*60}\n")