"""
Nemotron Parse Pipeline — PDF → JSON + MD
==========================================
Converts one PDF or an entire folder of PDFs into structured outputs
using NVIDIA's Nemotron Parse NIM API.

Modes run per page:
  - markdown_bbox    → JSON array with bbox + type + text  (recommended)
  - markdown_no_bbox → Markdown text only
  - detection_only   → JSON array with bbox + type only (no text)

Output folder structure (per document):
  output/<doc_name>/
  ├── pages/
  │   ├── page_001_markdown_bbox.json
  │   ├── page_001_markdown_no_bbox.json
  │   ├── page_001_detection_only.json
  │   ├── page_002_markdown_bbox.json
  │   └── ...
  ├── <doc_name>_markdown_bbox.json       ← all pages merged
  ├── <doc_name>_markdown_bbox.md
  ├── <doc_name>_markdown_no_bbox.json
  ├── <doc_name>_markdown_no_bbox.md
  ├── <doc_name>_detection_only.json
  └── <doc_name>_detection_only.md

.env file (place in the same directory as this script):
  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx

Usage:
  # Single PDF
  python nemotron_parse_pipeline.py --pdf path/to/doc.pdf

  # Entire folder of PDFs
  python nemotron_parse_pipeline.py --pdf_dir path/to/pdf_folder/

  # Both flags together
  python nemotron_parse_pipeline.py --pdf doc.pdf --pdf_dir folder/

  # Optional flags
  python nemotron_parse_pipeline.py --pdf_dir ./pdfs --output_dir ./results --dpi 200

Requirements:
  pip install requests pdf2image Pillow python-dotenv
  apt-get install poppler-utils
"""

import os
import sys
import base64
import json
import mimetypes
import requests
import argparse
import time
from pathlib import Path


# ─── Load API key from .env ───────────────────────────────────────────────────
def load_api_key() -> str:
    """
    Loads NVIDIA_API_KEY from a .env file in the same directory as this script.
    Falls back to the environment variable if already set in the shell.
    """
    env_path = Path(__file__).parent / ".env"

    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key   = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
    else:
        print(f"[WARNING] No .env file found at: {env_path}")
        print("          Falling back to shell environment variable NVIDIA_API_KEY")

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print("\n[ERROR] NVIDIA_API_KEY not found.")
        print("        Create a .env file next to this script:")
        print("            NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx")
        print("        Get your key from: https://build.nvidia.com/nvidia/nemotron-parse\n")
        sys.exit(1)

    return api_key


# ─── NVIDIA NIM API config ────────────────────────────────────────────────────
NVAI_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

TOOLS = [
    "markdown_bbox",
    # "markdown_no_bbox",
    # "detection_only",
]


# ─── Exact NVIDIA sample: read image as base64 ───────────────────────────────
def _read_image_as_base64(path: str):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    return b64, mime


# ─── Exact NVIDIA sample: build content + tool spec ──────────────────────────
def _generate_content(tool_name: str, b64_str: str, mime: str):
    media_tag = f'<img src="data:{mime};base64,{b64_str}" />'
    content   = f"{media_tag}"
    tool_spec = [{"type": "function", "function": {"name": tool_name}}]
    return content, tool_spec


# ─── Call Nemotron Parse API for one image + one tool mode ───────────────────
def call_nemotron_parse(image_path: str, tool_name: str, api_key: str,
                        max_retries: int = 3) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    b64_str, mime   = _read_image_as_base64(image_path)
    content, tool_spec = _generate_content(tool_name, b64_str, mime)

    inputs = {
        "model": "nvidia/nemotron-parse",
        "messages": [{"role": "user", "content": content}],
        "tools": tool_spec,
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
        "max_tokens": 8192,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(NVAI_URL, headers=headers,
                                     json=inputs, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError:
            print(f"      [!] HTTP {response.status_code} on attempt {attempt} — "
                  f"{response.text[:200]}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise
        except requests.Timeout:
            print(f"      [!] Timeout on attempt {attempt}, retrying ...")
            if attempt < max_retries:
                time.sleep(5)
            else:
                raise


# ─── Extract useful payload from API response ─────────────────────────────────
def extract_result(api_response: dict, tool_name: str):
    """
    - markdown_bbox / detection_only → list of dicts {bbox, type, text}
    - markdown_no_bbox               → dict {"text": "..."}
    """
    try:
        choices = api_response.get("choices", [])
        if not choices:
            return None
        message    = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            args_str = tool_calls[0].get("function", {}).get("arguments", "[]")
            parsed   = json.loads(args_str)
            # API occasionally wraps result in an extra list
            if (isinstance(parsed, list) and len(parsed) > 0
                    and isinstance(parsed[0], list)):
                return parsed[0]
            return parsed

        # Fallback: plain text content (markdown_no_bbox)
        content = message.get("content", "")
        if content:
            return {"text": content}

        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"      [!] Could not parse response: {e}")
        return None


# ─── Convert PDF → PNG images ─────────────────────────────────────────────────
def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 150) -> list:
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("[ERROR] pdf2image is not installed.")
        print("        pip install pdf2image")
        print("        apt-get install poppler-utils")
        sys.exit(1)

    print(f"  [→] Converting PDF to images at {dpi} DPI ...")
    images      = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    image_paths = []

    for i, img in enumerate(images, start=1):
        img_path = os.path.join(output_dir, f"page_{i:03d}.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)

    print(f"  [✓] {len(images)} page(s) converted\n")
    return image_paths


# ─── Build Markdown from all page results ─────────────────────────────────────
def results_to_markdown(all_page_results: list, tool_name: str,
                        doc_name: str) -> str:
    lines = [
        f"# {doc_name}\n",
        f"*Extracted using Nemotron Parse — mode: `{tool_name}`*\n",
    ]

    for page_data in all_page_results:
        page_num = page_data["page"]
        result   = page_data["result"]
        lines.append(f"\n---\n\n## Page {page_num}\n")

        if result is None:
            lines.append("*[No content extracted for this page]*\n")
            continue

        # markdown_no_bbox → plain text blob
        if isinstance(result, dict) and "text" in result:
            lines.append(result["text"] + "\n")

        # markdown_bbox / detection_only → list of elements
        elif isinstance(result, list):
            for element in result:
                elem_type = element.get("type", "")
                text      = element.get("text", "")

                if not text and tool_name == "detection_only":
                    bbox = element.get("bbox", {})
                    lines.append(
                        f"- **{elem_type}** "
                        f"[x: {bbox.get('xmin',0):.3f}–{bbox.get('xmax',0):.3f}, "
                        f"y: {bbox.get('ymin',0):.3f}–{bbox.get('ymax',0):.3f}]\n"
                    )
                    continue

                if elem_type in ("Title", "Section-header"):
                    lines.append(f"\n{text}\n")
                elif elem_type == "Table":
                    lines.append(f"\n{text}\n")
                elif elem_type == "Figure":
                    lines.append(f"\n*[Figure detected]*\n")
                elif elem_type == "Caption":
                    lines.append(f"*{text}*\n")
                elif elem_type in ("Footnote", "Page-footer", "Page-header"):
                    lines.append(f"> {text}\n")
                elif elem_type == "List-item":
                    lines.append(f"- {text}\n")
                else:
                    lines.append(f"{text}\n")

    return "\n".join(lines)


# ─── Process one PDF file ─────────────────────────────────────────────────────
def process_pdf(pdf_path: Path, api_key: str, output_root: str, dpi: int):
    doc_name = pdf_path.stem

    doc_output_dir = Path(output_root) / doc_name
    pages_dir      = doc_output_dir / "pages"
    images_dir     = doc_output_dir / "_page_images"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Document : {pdf_path.name}")
    print(f"  Output   : {doc_output_dir.resolve()}")
    print(f"{'='*60}")

    # Step 1: PDF → images
    image_paths = pdf_to_images(str(pdf_path), str(images_dir), dpi=dpi)
    total_pages = len(image_paths)

    # Step 2: All 3 tool modes
    for tool_name in TOOLS:
        print(f"\n  [MODE] {tool_name}")
        print(f"  {'─'*38}")

        all_page_results = []

        for page_idx, image_path in enumerate(image_paths, start=1):
            print(f"    [→] Page {page_idx}/{total_pages} ...", end=" ", flush=True)

            raw_response = call_nemotron_parse(image_path, tool_name, api_key)
            result       = extract_result(raw_response, tool_name)

            page_data = {
                "page":   page_idx,
                "image":  os.path.basename(image_path),
                "result": result,
            }
            all_page_results.append(page_data)

            # Per-page JSON
            page_json_path = pages_dir / f"page_{page_idx:03d}_{tool_name}.json"
            with open(page_json_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, ensure_ascii=False, indent=2)
            print(f"✓  {page_json_path.name}")

        # Merged document JSON
        doc_json = {
            "doc_name":    pdf_path.name,
            "tool_mode":   tool_name,
            "total_pages": total_pages,
            "pages":       all_page_results,
        }
        doc_json_path = doc_output_dir / f"{doc_name}_{tool_name}.json"
        with open(doc_json_path, "w", encoding="utf-8") as f:
            json.dump(doc_json, f, ensure_ascii=False, indent=2)
        print(f"\n  [✓] Merged JSON  → {doc_json_path.name}")

        # Markdown
        md_content    = results_to_markdown(all_page_results, tool_name, doc_name)
        doc_md_path   = doc_output_dir / f"{doc_name}_{tool_name}.md"
        with open(doc_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  [✓] Markdown     → {doc_md_path.name}")

    print(f"\n  [✓] Finished: {pdf_path.name}")


# ─── Collect all PDFs from --pdf and/or --pdf_dir ────────────────────────────
def collect_pdfs(pdf_arg, pdf_dir_arg) -> list:
    seen = set()
    pdfs = []

    def add(p: Path):
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            pdfs.append(resolved)

    if pdf_arg:
        p = Path(pdf_arg)
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)
        if p.suffix.lower() != ".pdf":
            print(f"[ERROR] Not a PDF: {p}")
            sys.exit(1)
        add(p)

    if pdf_dir_arg:
        d = Path(pdf_dir_arg)
        if not d.is_dir():
            print(f"[ERROR] Not a valid directory: {d}")
            sys.exit(1)
        found = sorted(d.glob("*.pdf"))
        if not found:
            print(f"[WARNING] No .pdf files found in: {d}")
        for p in found:
            add(p)

    if not pdfs:
        print("[ERROR] No PDFs to process. Provide --pdf and/or --pdf_dir.")
        sys.exit(1)

    return pdfs


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nemotron Parse Pipeline — PDF(s) → JSON + MD (all 3 modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF
  python nemotron_parse_pipeline.py --pdf doc.pdf

  # Entire folder of PDFs
  python nemotron_parse_pipeline.py --pdf_dir ./my_pdfs/

  # Both (single file + entire folder)
  python nemotron_parse_pipeline.py --pdf extra.pdf --pdf_dir ./my_pdfs/

  # Custom output dir + higher quality DPI
  python nemotron_parse_pipeline.py --pdf_dir ./pdfs/ --output_dir ./results --dpi 200

.env file (same folder as this script):
  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
  Get key: https://build.nvidia.com/nvidia/nemotron-parse
        """
    )

    parser.add_argument(
        "--pdf",
        metavar="PATH",
        help="Path to a single PDF file"
    )
    parser.add_argument(
        "--pdf_dir",
        metavar="DIR",
        help="Path to a folder — all .pdf files inside will be processed"
    )
    parser.add_argument(
        "--output_dir",
        metavar="DIR",
        default="output",
        help="Root output directory (default: ./output)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PDF→image conversion (default: 150; higher = better quality, slower)"
    )

    args = parser.parse_args()

    if not args.pdf and not args.pdf_dir:
        parser.print_help()
        print("\n[ERROR] You must provide --pdf and/or --pdf_dir\n")
        sys.exit(1)

    # Load API key from .env
    api_key = load_api_key()

    # Collect all PDFs to process
    pdfs = collect_pdfs(args.pdf, args.pdf_dir)

    print(f"\n[✓] API key loaded from .env")
    print(f"[✓] {len(pdfs)} PDF(s) queued for processing:")
    for i, p in enumerate(pdfs, 1):
        print(f"    {i:>3}. {p.name}")

    # Process each PDF one by one
    failed = []
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Starting: {pdf_path.name}")
        try:
            process_pdf(
                pdf_path=pdf_path,
                api_key=api_key,
                output_root=args.output_dir,
                dpi=args.dpi,
            )
        except Exception as e:
            print(f"\n  [ERROR] Failed: {pdf_path.name} — {e}")
            failed.append(pdf_path.name)
            continue

    # Final summary
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Processed : {len(pdfs) - len(failed)}/{len(pdfs)} PDF(s) successfully")
    if failed:
        print(f"  Failed    : {len(failed)} PDF(s)")
        for name in failed:
            print(f"              ✗ {name}")
    print(f"  Output    : {Path(args.output_dir).resolve()}")
    print(f"{'='*60}\n")