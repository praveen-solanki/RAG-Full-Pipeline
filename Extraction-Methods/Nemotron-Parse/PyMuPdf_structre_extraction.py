"""
PDF TOC Extractor — PyMuPDF
============================
Extracts the Table of Contents (structural hierarchy) from one PDF or
an entire folder of PDFs using PyMuPDF (fitz).

For each PDF, three strategies are tried in order:
  1. PDF bookmark metadata  → get_toc() from the internal outline tree
  2. Heading numbering      → regex on Section-header / Title text (1. / 1.1 / 1.1.1)
  3. Font-size heuristic    → clusters heading sizes across all pages

Output per document:
  <output_dir>/<doc_name>_structure.json

JSON schema:
  {
    "doc_name"      : "report.pdf",
    "total_pages"   : 42,
    "toc_strategy"  : "metadata" | "numbering" | "font_size" | "none",
    "hierarchy"     : [
      {
        "level"     : 1,          # 1 = top, 2 = sub, 3 = sub-sub, ...
        "text"      : "Introduction",
        "page"      : 3,          # 1-based
        "number"    : "1."        # prefix if detected, else null
      },
      ...
    ]
  }

Usage:
  # Single PDF
  python pdf_toc_extractor.py --pdf path/to/doc.pdf

  # Folder of PDFs
  python pdf_toc_extractor.py --pdf_dir path/to/folder/

  # Both + custom output dir
  python pdf_toc_extractor.py --pdf doc.pdf --pdf_dir folder/ --output_dir ./toc_output

Requirements:
  pip install pymupdf
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[ERROR] PyMuPDF is not installed.")
    print("        Run:  pip install pymupdf")
    sys.exit(1)


# ─── Strategy 1: PDF Bookmark Metadata ───────────────────────────────────────

def extract_toc_from_metadata(doc: fitz.Document) -> list:
    """
    Uses PyMuPDF's get_toc() to pull the internal outline tree.
    Returns a list of dicts: {level, text, page} or [] if no TOC exists.
    """
    raw_toc = doc.get_toc(simple=True)  # [[level, title, page], ...]
    if not raw_toc:
        return []

    result = []
    for entry in raw_toc:
        level, title, page = entry[0], entry[1], entry[2]
        title = title.strip()
        if not title:
            continue
        result.append({
            "level"  : level,
            "text"   : title,
            "page"   : max(1, page),   # fitz can return 0 for missing pages
            "number" : _detect_number_prefix(title),
        })
    return result


# ─── Strategy 2: Heading Numbering Heuristic ─────────────────────────────────

# Matches prefixes like: 1   1.   1.2   1.2.3   A.   I.   I.1
_NUMBER_RE = re.compile(
    r"^(?:"
    r"(\d+(?:\.\d+)*)\.?"          # arabic: 1 / 1.2 / 1.2.3
    r"|([A-Z]\.)"                   # letter: A. B. C.
    r"|([IVXLCDM]+\.)"             # roman:  I. II. III.
    r")\s+"
)

def _detect_number_prefix(text: str):
    m = _NUMBER_RE.match(text.strip())
    if not m:
        return None
    return m.group(0).strip()

def _numbering_depth(text: str) -> int | None:
    """
    Returns the depth implied by a numbering prefix, or None if not found.
    '1.'        → 1
    '1.2'       → 2
    '1.2.3'     → 3
    'A.'        → 1
    'I.'        → 1
    """
    m = _NUMBER_RE.match(text.strip())
    if not m:
        return None
    arabic = m.group(1)
    if arabic:
        return len(arabic.split("."))
    return 1   # letter / roman → treat as level 1


def extract_toc_from_numbering(doc: fitz.Document) -> list:
    """
    Collects blocks across all pages, filters to those that look like
    numbered headings, and assigns depth from the numbering scheme.
    Returns [] if fewer than 2 numbered headings are found.
    """
    candidates = []

    for page_num in range(len(doc)):
        page   = doc[page_num]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block["type"] != 0:   # 0 = text block
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if not text:
                    continue
                depth = _numbering_depth(text)
                if depth is not None:
                    candidates.append({
                        "level"  : depth,
                        "text"   : text,
                        "page"   : page_num + 1,
                        "number" : _detect_number_prefix(text),
                    })

    if len(candidates) < 2:
        return []

    return candidates


# ─── Strategy 3: Font-Size Heuristic ─────────────────────────────────────────

def extract_toc_from_font_size(doc: fitz.Document) -> list:
    """
    Collects all text spans, identifies the body font size (most common),
    then treats spans significantly larger than body text as headings.
    Clusters the heading sizes into levels (largest = level 1, etc.).
    Returns [] if no clear heading sizes are found.
    """
    # Pass 1: collect (size, text, page, y0) for every span
    all_spans = []
    size_freq = defaultdict(int)

    for page_num in range(len(doc)):
        page   = doc[page_num]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = round(span["size"], 1)
                    size_freq[size] += len(text)
                    all_spans.append({
                        "size" : size,
                        "text" : text,
                        "page" : page_num + 1,
                        "y0"   : span["bbox"][1],
                        "flags": span.get("flags", 0),  # bold = flags & 16
                    })

    if not size_freq:
        return []

    # Body size = size with most total characters
    body_size = max(size_freq, key=size_freq.get)

    # Collect heading sizes: clearly bigger than body (threshold: +1.5pt)
    heading_sizes = sorted(
        {s for s in size_freq if s > body_size + 1.5},
        reverse=True
    )

    if not heading_sizes:
        return []

    # Map each heading size to a level (largest font = level 1)
    size_to_level = {s: i + 1 for i, s in enumerate(heading_sizes)}

    # Pass 2: collect heading spans, merge consecutive same-line spans
    result = []
    prev = None

    for span in all_spans:
        if span["size"] not in size_to_level:
            prev = None
            continue

        level = size_to_level[span["size"]]

        # Merge adjacent spans on the same page+line into one heading
        if (prev and
                prev["level"] == level and
                prev["page"] == span["page"] and
                abs(prev["_y0"] - span["y0"]) < 5):
            prev["text"] += " " + span["text"]
            prev["text"]  = prev["text"].strip()
        else:
            entry = {
                "level"  : level,
                "text"   : span["text"],
                "page"   : span["page"],
                "number" : _detect_number_prefix(span["text"]),
                "_y0"    : span["y0"],   # internal, stripped before saving
            }
            result.append(entry)
            prev = entry

    # Strip internal _y0 key
    for entry in result:
        entry.pop("_y0", None)

    # Only keep if we have a reasonable number of headings
    if len(result) < 2:
        return []

    return result


# ─── Orchestrator: Try all strategies in order ────────────────────────────────

def extract_structure(pdf_path: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    hierarchy     = []
    toc_strategy  = "none"

    # Strategy 1 — PDF metadata bookmarks
    hierarchy = extract_toc_from_metadata(doc)
    if hierarchy:
        toc_strategy = "metadata"
        print(f"    [✓] Strategy: PDF metadata bookmarks ({len(hierarchy)} entries)")
    else:
        # Strategy 2 — Numbering heuristic
        hierarchy = extract_toc_from_numbering(doc)
        if hierarchy:
            toc_strategy = "numbering"
            print(f"    [✓] Strategy: Heading numbering ({len(hierarchy)} entries)")
        else:
            # Strategy 3 — Font size clustering
            hierarchy = extract_toc_from_font_size(doc)
            if hierarchy:
                toc_strategy = "font_size"
                print(f"    [✓] Strategy: Font-size heuristic ({len(hierarchy)} entries)")
            else:
                toc_strategy = "none"
                print(f"    [!] No structural hierarchy could be detected")

    doc.close()

    return {
        "doc_name"     : pdf_path.name,
        "total_pages"  : total_pages,
        "toc_strategy" : toc_strategy,
        "hierarchy"    : hierarchy,
    }


# ─── Process one PDF ──────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, output_dir: Path):
    print(f"\n  [→] {pdf_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    structure     = extract_structure(pdf_path)
    out_filename  = f"{pdf_path.stem}_structure.json"
    out_path      = output_dir / out_filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

    print(f"    [✓] Saved → {out_path.resolve()}")
    return structure


# ─── Collect PDFs from CLI args ───────────────────────────────────────────────

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
        print("[ERROR] No PDFs to process. Use --pdf and/or --pdf_dir.")
        sys.exit(1)

    return pdfs


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF TOC Extractor — Extracts structural hierarchy using PyMuPDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies (tried in order):
  1. metadata   — PDF internal bookmark outline (fastest, most accurate)
  2. numbering  — Regex on heading numbers like 1. / 1.2 / 1.2.3 / A. / I.
  3. font_size  — Font size clustering (largest font = top level)
  4. none       — No hierarchy detected (flat document or scanned image PDF)

Examples:
  python pdf_toc_extractor.py --pdf report.pdf
  python pdf_toc_extractor.py --pdf_dir ./my_pdfs/
  python pdf_toc_extractor.py --pdf doc.pdf --pdf_dir ./pdfs/ --output_dir ./toc_out/
        """
    )

    parser.add_argument("--pdf",        metavar="PATH", help="Path to a single PDF file")
    parser.add_argument("--pdf_dir",    metavar="DIR",  help="Folder of PDFs to process")
    parser.add_argument("--output_dir", metavar="DIR",  default="toc_output",
                        help="Output directory for _structure.json files (default: ./toc_output)")

    args = parser.parse_args()

    if not args.pdf and not args.pdf_dir:
        parser.print_help()
        print("\n[ERROR] Provide at least --pdf or --pdf_dir\n")
        sys.exit(1)

    pdfs       = collect_pdfs(args.pdf, args.pdf_dir)
    output_dir = Path(args.output_dir)

    print(f"\n[✓] {len(pdfs)} PDF(s) queued:")
    for i, p in enumerate(pdfs, 1):
        print(f"    {i:>3}. {p.name}")

    failed = []
    for pdf_path in pdfs:
        try:
            process_pdf(pdf_path, output_dir)
        except Exception as e:
            print(f"    [ERROR] {pdf_path.name} — {e}")
            failed.append(pdf_path.name)

    print(f"\n{'='*55}")
    print(f"  DONE — {len(pdfs) - len(failed)}/{len(pdfs)} succeeded")
    if failed:
        for name in failed:
            print(f"  ✗ {name}")
    print(f"  Output → {output_dir.resolve()}")
    print(f"{'='*55}\n")