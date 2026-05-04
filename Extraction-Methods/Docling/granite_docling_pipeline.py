"""
Granite-Docling Hierarchy Pipeline  —  PDF → Nested Tree JSON
==============================================================
Extracts the structural hierarchy (Title + Section-header elements)
from each PDF using IBM's Granite-Docling 258M vision-language model
running locally via the Docling library.

Pipeline (4 stages per document):
  Stage 1 — Docling receives the PDF and rasterises each page internally.
             No manual image conversion is needed.
  Stage 2 — Each page image is processed by the Granite-Docling VLM
             (VlmPipeline + granite_docling preset). The model outputs
             DocTags, which Docling parses into a DoclingDocument object.
  Stage 3 — iterate_items() walks the DoclingDocument tree. Only
             TitleItem and SectionHeaderItem elements are kept.
             From each: item.level, item.text, item.prov[0].page_no.
  Stage 4 — A stack-based tree builder converts the flat ordered list
             into the nested {title, node_id, start_index, end_index,
             nodes[]} structure and writes it to disk.

Output folder structure (per document):
  output/<doc_name>/
  ├── <doc_name>_hierarchy.json   ← nested tree (primary output)
  └── <doc_name>_hierarchy.md     ← human-readable outline

Nested JSON schema:
  {
    "doc_name":    "report.pdf",
    "total_pages": 24,
    "source":      "granite_docling_vlm",
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

Installation:
  pip install "docling[vlm]"

  On first run, Docling will automatically download the Granite-Docling
  258M model weights from HuggingFace (~258MB, cached for subsequent runs).

  For Apple Silicon (MPS acceleration):
  pip install "docling[vlm,mlx]"

Usage:
  # Single PDF
  python granite_docling_pipeline.py --pdf path/to/doc.pdf

  # Entire folder of PDFs
  python granite_docling_pipeline.py --pdf_dir path/to/pdf_folder/

  # Both flags together
  python granite_docling_pipeline.py --pdf doc.pdf --pdf_dir folder/

  # Custom output directory
  python granite_docling_pipeline.py --pdf_dir ./pdfs --output_dir ./results
"""

import sys
import json
import argparse
import platform
from pathlib import Path


# ─── Docling availability check ───────────────────────────────────────────────
def _check_docling():
    """
    Verifies that docling and its VLM extras are installed.
    Prints a clear install command and exits if not found.
    """
    try:
        import docling                          # noqa: F401
    except ImportError:
        print("\n[ERROR] docling is not installed.")
        print("        Install it with VLM support:")
        print('            pip install "docling[vlm]"')
        print("        On Apple Silicon (MPS acceleration):")
        print('            pip install "docling[vlm,mlx]"\n')
        sys.exit(1)

    try:
        from docling.pipeline.vlm_pipeline import VlmPipeline  # noqa: F401
    except ImportError:
        print("\n[ERROR] docling VLM extras are missing.")
        print('        Reinstall with:  pip install "docling[vlm]"\n')
        sys.exit(1)


# ─── Stage 2 + 3 — run Granite-Docling and extract structural elements ────────
def run_granite_docling(pdf_path: Path) -> tuple[list, int]:
    """
    Runs the Granite-Docling VLM pipeline on a single PDF.

    Stage 2: DocumentConverter with VlmPipeline rasterises each page and
             passes it through Granite-Docling 258M. The model outputs
             DocTags which Docling parses into a DoclingDocument.

    Stage 3: iterate_items() walks the document tree. Only TitleItem and
             SectionHeaderItem are kept. From each element we read:
               item.level       — hierarchy depth (1 = top)
               item.text        — heading text
               item.prov[0].page_no — 1-based page number

    Returns:
        flat_headers  — ordered list of dicts:
                        {level: int, type: str, text: str, page: int}
        total_pages   — number of pages in the document
    """
    # ── imports live here so the import-error message above fires first ───────
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling_core.types.doc import TitleItem, SectionHeaderItem

    # ── detect platform and pick the right backend automatically ─────────────
    # Apple Silicon → MLX (native MPS, fastest on Mac)
    # CUDA available  → Transformers with bfloat16 on GPU
    # CPU fallback    → Transformers on CPU (slower but works everywhere)
    is_apple_silicon = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
    )

    if is_apple_silicon:
        # Try MLX backend first; fall back gracefully if not installed
        try:
            from docling.datamodel import vlm_model_specs
            from docling.datamodel.pipeline_options import VlmPipelineOptions

            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
            )
            print("  [i] Backend : Granite-Docling MLX (Apple Silicon MPS)")
        except (ImportError, AttributeError):
            pipeline_options = VlmPipelineOptions()
            print("  [i] Backend : Granite-Docling Transformers (MLX not available)")
    else:
        # Default: let Docling auto-select GPU (CUDA) or CPU
        pipeline_options = VlmPipelineOptions()
        print("  [i] Backend : Granite-Docling Transformers (auto GPU/CPU)")

    # ── build the converter — VlmPipeline uses granite_docling by default ─────
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    # ── Stage 2: convert — Docling rasterises pages and runs the VLM ─────────
    print(f"  [→] Running Granite-Docling on {pdf_path.name} ...")
    result      = converter.convert(source=str(pdf_path))
    doc         = result.document
    total_pages = len(doc.pages) if doc.pages else 0

    print(f"  [✓] Conversion complete — {total_pages} page(s) processed")

    # ── Stage 3: walk the document tree, filter structural elements ───────────
    flat_headers = []

    for item, _depth in doc.iterate_items():

        # Keep only heading-type elements
        if not isinstance(item, (TitleItem, SectionHeaderItem)):
            continue

        text = (item.text or "").strip()
        if not text:
            continue

        # item.level is the hierarchy depth set by Granite-Docling via DocTags
        # TitleItem always gets level=1; SectionHeaderItem gets 1..N
        level = getattr(item, "level", 1)
        if isinstance(item, TitleItem):
            level = 1

        # prov carries provenance — page number is 1-based in Docling
        page = 1
        if item.prov:
            page = item.prov[0].page_no

        flat_headers.append({
            "level": int(level),
            "type":  "Title" if isinstance(item, TitleItem) else "Section-header",
            "text":  text,
            "page":  page,
        })

    found = len(flat_headers)
    print(f"  [✓] Structural elements found: {found} "
          f"(Title + Section-header only)\n")

    return flat_headers, total_pages


# ─── Stage 4a — stack-based tree builder ─────────────────────────────────────
def build_tree(flat_headers: list, total_pages: int) -> list:
    """
    Converts the flat ordered list of structural elements into a recursive
    nested tree that matches the target JSON schema:

      {
        "title":       str,
        "node_id":     str  (zero-padded 4-digit counter, e.g. "0001"),
        "start_index": int  (page where this section starts),
        "end_index":   int  (page where this section ends, exclusive),
        "nodes":       list (child nodes, same schema, recursive)
      }

    Algorithm — single-pass stack:
      For each incoming element (ordered by reading order, so by page):
        1. Pop the stack until the top has a strictly lower level than
           the current element.  The remaining top is the parent.
        2. If the stack is empty → this is a root node.
        3. Append the new node as a child of the parent (or as a root).
        4. Push the new node onto the stack.

    After the pass, a second pass fills end_index:
      end_index of a node = start_index of its next sibling,
                            OR start_index of the parent's next sibling,
                            OR total_pages (for the last node in the tree).
    """
    if not flat_headers:
        return []

    counter = [1]   # mutable so the inner closure can update it

    def make_node(entry: dict) -> dict:
        node_id = f"{counter[0]:04d}"
        counter[0] += 1
        return {
            "title":       entry["text"],
            "node_id":     node_id,
            "start_index": entry["page"],
            "end_index":   entry["page"],   # placeholder; filled in second pass
            "nodes":       [],
            "_level":      entry["level"],  # temp field removed after building
        }

    roots: list  = []
    stack: list  = []   # list of (level: int, node: dict)

    for entry in flat_headers:
        node          = make_node(entry)
        current_level = entry["level"]

        # Pop everything at the same depth or deeper
        while stack and stack[-1][0] >= current_level:
            stack.pop()

        if stack:
            # Attach as child of the current stack top
            stack[-1][1]["nodes"].append(node)
        else:
            # No ancestor → this is a root node
            roots.append(node)

        stack.append((current_level, node))

    # ── Second pass: fill end_index correctly ─────────────────────────────────
    # end_index for a node is the start_index of the next sibling at the same
    # level (or the parent's next sibling, bubbling up), or total_pages as the
    # final fallback.
    def fill_end_index(nodes: list, next_available: int) -> None:
        for i, node in enumerate(nodes):
            # The next sibling's start_index caps this node's range
            sibling_start = (
                nodes[i + 1]["start_index"]
                if i + 1 < len(nodes)
                else next_available
            )
            # Recurse into children before setting our own end_index
            fill_end_index(node["nodes"], sibling_start)
            # Our end_index is the minimum of our children's extent and sibling
            node["end_index"] = sibling_start

    fill_end_index(roots, total_pages)

    # ── Clean up temp _level field ────────────────────────────────────────────
    def remove_temp(nodes: list) -> None:
        for n in nodes:
            n.pop("_level", None)
            remove_temp(n["nodes"])

    remove_temp(roots)
    return roots


# ─── Stage 4b — render tree as Markdown outline ───────────────────────────────
def tree_to_markdown(tree: list, doc_name: str) -> str:
    """
    Renders the nested tree as a clean indented Markdown outline.

    Level 1  →  ## heading  (no indent)
    Level 2  →    - bullet  (2-space indent per level after 1)

    Each entry is followed by a page-range reference: (p. X–Y)
    """
    lines = [
        f"# {doc_name} — Document Hierarchy\n",
        "*Extracted using Granite-Docling 258M VLM*\n",
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
                indent = "  " * depth
                lines.append(f"{indent}- {title}  *({pages})*")

            render(node["nodes"], depth + 1)

    render(tree, 0)
    return "\n".join(lines) + "\n"


# ─── Process one PDF ──────────────────────────────────────────────────────────
def process_pdf(pdf_path: Path, output_root: str) -> None:
    """
    Full pipeline for a single PDF:
      Stage 1+2 : Docling rasterises pages → Granite-Docling VLM runs
      Stage 3   : iterate_items() filters TitleItem + SectionHeaderItem
      Stage 4   : stack-based tree builder → write JSON + Markdown
    """
    doc_name       = pdf_path.stem
    doc_output_dir = Path(output_root) / doc_name
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Document : {pdf_path.name}")
    print(f"  Output   : {doc_output_dir.resolve()}")
    print(f"{'='*60}\n")

    # ── Stages 1–3 : Granite-Docling + element extraction ────────────────────
    flat_headers, total_pages = run_granite_docling(pdf_path)

    # ── Stage 4 : build nested tree ───────────────────────────────────────────
    print("  [→] Building hierarchy tree ...")
    tree = build_tree(flat_headers, total_pages)
    print(f"  [✓] Tree built — {len(tree)} root node(s), "
          f"{sum(1 for _ in _walk(tree))} total node(s)")

    # ── Write JSON ────────────────────────────────────────────────────────────
    output_json = {
        "doc_name":    pdf_path.name,
        "total_pages": total_pages,
        "source":      "granite_docling_vlm",
        "hierarchy":   tree,
    }
    json_path = doc_output_dir / f"{doc_name}_hierarchy.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print(f"  [✓] JSON written  → {json_path.name}")

    # ── Write Markdown ────────────────────────────────────────────────────────
    md_content = tree_to_markdown(tree, doc_name)
    md_path    = doc_output_dir / f"{doc_name}_hierarchy.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [✓] MD written   → {md_path.name}")

    print(f"\n  [✓] Finished: {pdf_path.name}")


# ─── Helper: walk entire tree depth-first ────────────────────────────────────
def _walk(nodes: list):
    """Generator that yields every node in the tree depth-first."""
    for node in nodes:
        yield node
        yield from _walk(node["nodes"])


# ─── Collect PDFs from --pdf and/or --pdf_dir ─────────────────────────────────
def collect_pdfs(pdf_arg: str | None, pdf_dir_arg: str | None) -> list:
    """
    Gathers all PDF paths from the provided arguments.
    Deduplicates by resolved absolute path.
    """
    seen: set  = set()
    pdfs: list = []

    def add(p: Path) -> None:
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
            print(f"[ERROR] Not a PDF file: {p}")
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


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Verify docling is installed before doing anything else
    _check_docling()

    parser = argparse.ArgumentParser(
        description="Granite-Docling Hierarchy Pipeline — PDF(s) → Nested Tree JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF
  python granite_docling_pipeline.py --pdf doc.pdf

  # Entire folder of PDFs
  python granite_docling_pipeline.py --pdf_dir ./my_pdfs/

  # Both (single file + entire folder)
  python granite_docling_pipeline.py --pdf extra.pdf --pdf_dir ./my_pdfs/

  # Custom output directory
  python granite_docling_pipeline.py --pdf_dir ./pdfs/ --output_dir ./results

Installation:
  pip install "docling[vlm]"
  pip install "docling[vlm,mlx]"   # Apple Silicon (MPS acceleration)

Model:
  Granite-Docling 258M — downloaded automatically from HuggingFace on first run.
  Apache 2.0 licence. Runs fully locally, no API key required.
        """
    )

    parser.add_argument(
        "--pdf",
        metavar="PATH",
        help="Path to a single PDF file",
    )
    parser.add_argument(
        "--pdf_dir",
        metavar="DIR",
        help="Path to a folder — all .pdf files inside will be processed",
    )
    parser.add_argument(
        "--output_dir",
        metavar="DIR",
        default="output",
        help="Root output directory (default: ./output)",
    )

    args = parser.parse_args()

    if not args.pdf and not args.pdf_dir:
        parser.print_help()
        print("\n[ERROR] You must provide --pdf and/or --pdf_dir\n")
        sys.exit(1)

    pdfs = collect_pdfs(args.pdf, args.pdf_dir)

    print(f"\n[✓] Granite-Docling Hierarchy Pipeline")
    print(f"[✓] Model   : Granite-Docling 258M (local, no API key)")
    print(f"[✓] {len(pdfs)} PDF(s) queued:")
    for i, p in enumerate(pdfs, 1):
        print(f"    {i:>3}. {p.name}")

    failed: list = []

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Starting: {pdf_path.name}")
        try:
            process_pdf(
                pdf_path=pdf_path,
                output_root=args.output_dir,
            )
        except Exception as e:
            print(f"\n  [ERROR] Failed: {pdf_path.name} — {e}")
            failed.append(pdf_path.name)
            continue

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Processed : {len(pdfs) - len(failed)}/{len(pdfs)} PDF(s) successfully")
    if failed:
        print(f"  Failed    : {len(failed)} PDF(s)")
        for name in failed:
            print(f"              ✗ {name}")
    print(f"  Output    : {Path(args.output_dir).resolve()}")
    print(f"{'='*60}\n")