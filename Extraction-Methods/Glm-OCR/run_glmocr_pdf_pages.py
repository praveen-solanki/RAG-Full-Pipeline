# #!/usr/bin/env python3
# """
# run_glmocr_pdf_pages.py

# Input: a PDF file OR a directory containing PDFs.
# Output:
#   OUT_ROOT/
#     <pdf_name>/
#       <pdf_name>.pdf                 (optional copy)
#       page_0001/
#         result.json
#         result.md
#         page_0001.png                (if --keep-images)
#       page_0002/
#         ...

# Works with:
# - MaaS (cloud): --mode maas --api-key ...
# - Self-hosted: --mode selfhosted --ocr-host ... --ocr-port ... (vLLM/SGLang)
# """

# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import shutil
# import sys
# from pathlib import Path

# import fitz  # PyMuPDF
# from PIL import Image

# from glmocr import GlmOcr


# def sanitize_folder_name(name: str) -> str:
#     # Keep it readable but safe across filesystems
#     name = name.strip()
#     name = name.replace(os.sep, "_")
#     name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)  # windows-illegal + control chars
#     name = re.sub(r"\s+", " ", name)  # collapse whitespace
#     return name or "document"


# def render_page_to_image(doc: fitz.Document, page_index: int, out_image_path: Path, dpi: int) -> None:
#     """Render a single PDF page to an image file."""
#     page = doc.load_page(page_index)
#     zoom = dpi / 72.0
#     mat = fitz.Matrix(zoom, zoom)
#     pix = page.get_pixmap(matrix=mat, alpha=False)

#     out_image_path.parent.mkdir(parents=True, exist_ok=True)
#     ext = out_image_path.suffix.lower()

#     if ext == ".png":
#         pix.save(str(out_image_path))
#     elif ext in [".jpg", ".jpeg"]:
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         img.save(str(out_image_path), format="JPEG", quality=95, optimize=True)
#     else:
#         raise ValueError(f"Unsupported image extension: {ext}")


# def safe_write_json(path: Path, obj) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# def process_one_pdf(
#     pdf_path: Path,
#     out_root: Path,
#     dpi: int,
#     image_format: str,
#     start_page: int,
#     end_page: int,
#     keep_images: bool,
#     fail_fast: bool,
#     copy_pdf: bool,
#     ocr: GlmOcr,
# ) -> None:
#     pdf_path = pdf_path.resolve()
#     pdf_name = sanitize_folder_name(pdf_path.stem)
#     pdf_out_dir = out_root / pdf_name
#     pdf_out_dir.mkdir(parents=True, exist_ok=True)

#     if copy_pdf:
#         # Copy PDF into its output folder (so outputs are self-contained)
#         dst_pdf = pdf_out_dir / pdf_path.name
#         if dst_pdf.resolve() != pdf_path:
#             shutil.copy2(pdf_path, dst_pdf)

#     doc = fitz.open(pdf_path)
#     try:
#         n_pages = doc.page_count
#         s = max(1, start_page)
#         e = end_page if end_page and end_page > 0 else n_pages
#         e = min(e, n_pages)

#         if s > e:
#             raise ValueError(f"Invalid page range: start={s}, end={e}, total_pages={n_pages}")

#         print(f"\n[INFO] Processing PDF: {pdf_path.name} | pages={n_pages} | range={s}..{e}")
#         print(f"[INFO] Output folder: {pdf_out_dir}")

#         for p in range(s, e + 1):
#             page_index0 = p - 1
#             page_dir = pdf_out_dir / f"page_{p:04d}"
#             page_dir.mkdir(parents=True, exist_ok=True)

#             img_ext = ".png" if image_format == "png" else ".jpg"
#             img_path = page_dir / f"page_{p:04d}{img_ext}"

#             try:
#                 print(f"\n[PAGE {p}/{e}] render -> {img_path.name}")
#                 render_page_to_image(doc, page_index0, img_path, dpi=dpi)

#                 print(f"[PAGE {p}/{e}] GLM-OCR parse...")
#                 result = ocr.parse(str(img_path))

#                 # Saves: result.json, result.md, imgs/ (if layout enabled)
#                 result.save(output_dir=str(page_dir))

#                 # Optional extra debug dump (ignore if not supported)
#                 try:
#                     safe_write_json(page_dir / "result.full.to_dict.json", result.to_dict())
#                 except Exception:
#                     pass

#                 if not keep_images:
#                     try:
#                         img_path.unlink(missing_ok=True)
#                     except Exception:
#                         pass

#                 print(f"[PAGE {p}/{e}] done ✅ -> {page_dir}")

#             except Exception as ex:
#                 print(f"[ERROR] Page {p} failed: {ex}", file=sys.stderr)
#                 if fail_fast:
#                     raise
#                 continue

#     finally:
#         doc.close()


# def main() -> int:
#     ap = argparse.ArgumentParser(description="Run GLM-OCR on PDFs (page-by-page).")

#     inp = ap.add_mutually_exclusive_group(required=True)
#     inp.add_argument("--pdf", help="Path to a single PDF")
#     inp.add_argument("--pdf-dir", help="Path to a directory containing PDFs")

#     ap.add_argument("--out", default="./outputs", help="Output ROOT folder (default: ./outputs)")
#     ap.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
#     ap.add_argument("--image-format", choices=["png", "jpg"], default="png", help="Rendered page image format")
#     ap.add_argument("--start-page", type=int, default=1, help="1-based start page (default: 1)")
#     ap.add_argument("--end-page", type=int, default=0, help="1-based end page inclusive (0 = till last)")
#     ap.add_argument("--keep-images", action="store_true", help="Keep rendered page images inside each page folder")
#     ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
#     ap.add_argument("--copy-pdf", action="store_true", help="Copy input PDF into its output folder")

#     # GLM-OCR runtime
#     ap.add_argument("--mode", choices=["maas", "selfhosted"], default=None, help="GLM-OCR mode")
#     ap.add_argument("--api-key", default=None, help="MaaS API key (if mode=maas)")
#     ap.add_argument("--config", default=None, help="Path to config.yaml (optional)")
#     ap.add_argument("--enable-layout", action="store_true", help="Enable layout detection (if available)")
#     ap.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")

#     # Self-hosted OCR API location (vLLM/SGLang)
#     ap.add_argument("--ocr-host", default=None, help="Selfhosted OCR API host (e.g., localhost)")
#     ap.add_argument("--ocr-port", type=int, default=None, help="Selfhosted OCR API port (e.g., 8080)")

#     args = ap.parse_args()

#     out_root = Path(args.out).expanduser().resolve()
#     out_root.mkdir(parents=True, exist_ok=True)

#     # Set env overrides (SDK reads GLMOCR_*). Works even if you don't pass kwargs.
#     if args.mode:
#         os.environ["GLMOCR_MODE"] = args.mode
#     if args.api_key:
#         os.environ["GLMOCR_API_KEY"] = args.api_key
#     if args.log_level:
#         os.environ["GLMOCR_LOG_LEVEL"] = args.log_level
#     if args.enable_layout:
#         os.environ["GLMOCR_ENABLE_LAYOUT"] = "true"
#     if args.ocr_host:
#         os.environ["GLMOCR_OCR_API_HOST"] = args.ocr_host
#     if args.ocr_port is not None:
#         os.environ["GLMOCR_OCR_API_PORT"] = str(args.ocr_port)

#     # Build parser once and reuse
#     parser_kwargs = {}
#     if args.config:
#         parser_kwargs["config_path"] = str(Path(args.config).expanduser().resolve())
#     if args.api_key:
#         parser_kwargs["api_key"] = args.api_key
#     if args.mode:
#         parser_kwargs["mode"] = args.mode
#     if args.enable_layout:
#         parser_kwargs["enable_layout"] = True
#     if args.log_level:
#         parser_kwargs["log_level"] = args.log_level

#     # Collect PDFs
#     pdfs: list[Path] = []
#     if args.pdf:
#         p = Path(args.pdf).expanduser()
#         if not p.exists():
#             print(f"[ERROR] PDF not found: {p}", file=sys.stderr)
#             return 2
#         pdfs = [p]
#     else:
#         d = Path(args.pdf_dir).expanduser()
#         if not d.exists() or not d.is_dir():
#             print(f"[ERROR] Not a directory: {d}", file=sys.stderr)
#             return 2
#         pdfs = sorted(d.glob("*.pdf"))
#         if not pdfs:
#             print(f"[ERROR] No PDFs found in: {d}", file=sys.stderr)
#             return 2

#     print(f"[INFO] Output ROOT: {out_root}")
#     print(f"[INFO] PDFs to process: {len(pdfs)}")

#     try:
#         with GlmOcr(**parser_kwargs) as ocr:
#             for pdf_path in pdfs:
#                 try:
#                     process_one_pdf(
#                         pdf_path=pdf_path,
#                         out_root=out_root,
#                         dpi=args.dpi,
#                         image_format=args.image_format,
#                         start_page=args.start_page,
#                         end_page=args.end_page,
#                         keep_images=args.keep_images,
#                         fail_fast=args.fail_fast,
#                         copy_pdf=args.copy_pdf,
#                         ocr=ocr,
#                     )
#                 except Exception as e:
#                     print(f"[ERROR] PDF failed: {pdf_path.name}: {e}", file=sys.stderr)
#                     if args.fail_fast:
#                         return 1
#                     continue

#         print("\n[INFO] All done ✅")
#         return 0

#     except Exception as e:
#         print(f"[FATAL] {e}", file=sys.stderr)
#         return 1


# if __name__ == "__main__":
#     raise SystemExit(main())


#!/usr/bin/env python3
"""
run_glmocr_pdf_pages.py

Input: a PDF file OR a directory containing PDFs.
Output:
  OUT_ROOT/
    <pdf_name>/
      <pdf_name>.pdf                 (optional copy)
      images/
        pages/
          page_0001.png        (rendered page, saved with --save-images pages|both)
          page_0002.png
          ...
        layout_vis/
          page_0001.jpg        (bbox visualization, saved with --save-images layout|both)
          page_0002.jpg
          ...
      markdown/
        page_0001.md
        page_0002.md
        ...
      json/
        page_0001.json
        page_0002.json
        ...

Works with:
- MaaS (cloud): --mode maas --api-key ...
- Self-hosted: --mode selfhosted --ocr-host ... --ocr-port ... (vLLM/SGLang)

Parallelism:
- --page-workers N  : N concurrent OCR calls per PDF (HTTP I/O to vLLM). Default: 4.
- --pdf-workers  N  : N PDFs processed in parallel (when using --pdf-dir).  Default: 1.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from glmocr import GlmOcr

# Thread-safe print so logs from concurrent workers don't interleave
_print_lock = threading.Lock()


def tprint(*args, file=None, **kwargs) -> None:
    with _print_lock:
        print(*args, file=file, **kwargs)


def sanitize_folder_name(name: str) -> str:
    name = name.strip()
    name = name.replace(os.sep, "_")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name or "document"


def render_page_to_image(doc: fitz.Document, page_index: int, out_image_path: Path, dpi: int) -> None:
    """Render a single PDF page to an image file."""
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_image_path.suffix.lower()

    if ext == ".png":
        pix.save(str(out_image_path))
    elif ext in [".jpg", ".jpeg"]:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(str(out_image_path), format="JPEG", quality=95, optimize=True)
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


def process_one_page(
    pdf_path: Path,
    page_num: int,        # 1-based
    total_pages: int,
    pages_dir: Path,
    layout_vis_dir: Path,
    markdown_dir: Path,
    json_dir: Path,
    dpi: int,
    image_format: str,
    save_images: str,
    ocr: GlmOcr,
) -> dict:
    """Render + OCR + save a single page. Safe to call from multiple threads.

    Each call opens its own fitz.Document so there is no shared mutable state
    between threads (PyMuPDF is not thread-safe on a shared Document object).

    Returns:
        dict with page_num, json_result, markdown_result for cross-page assembly.
    """
    page_name = f"page_{page_num:04d}"
    img_ext = ".png" if image_format == "png" else ".jpg"
    img_path = pages_dir / f"{page_name}{img_ext}"

    # Render page to image (always needed — used as input to OCR even if not saving)
    tprint(f"  [PAGE {page_num}/{total_pages}] rendering...")
    doc = fitz.open(pdf_path)
    try:
        render_page_to_image(doc, page_num - 1, img_path, dpi=dpi)
    finally:
        doc.close()

    # OCR — HTTP call to vLLM, purely I/O bound, safe to parallelise
    tprint(f"  [PAGE {page_num}/{total_pages}] OCR...")
    result = ocr.parse(str(img_path))

    # Delete rendered page if user does not want pages saved
    if save_images not in ("pages", "both"):
        try:
            img_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Save JSON
    json_file = json_dir / f"{page_name}.json"
    if isinstance(result.json_result, (dict, list)):
        json_file.write_text(
            json.dumps(result.json_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        json_file.write_text(str(result.json_result), encoding="utf-8")

    # Save Markdown
    if result.markdown_result and result.markdown_result.strip():
        md_file = markdown_dir / f"{page_name}.md"
        md_file.write_text(result.markdown_result, encoding="utf-8")

    # Save layout visualization image (only present when --enable-layout is on)
    if save_images in ("layout", "both") and hasattr(result, "layout_vis_dir") and result.layout_vis_dir:
        temp_path = Path(result.layout_vis_dir)
        if temp_path.exists():
            indices = getattr(result, "layout_image_indices", None)
            if indices is not None:
                candidates = []
                for idx in indices:
                    for ext in (".jpg", ".png"):
                        p = temp_path / f"layout_page{idx}{ext}"
                        if p.exists():
                            candidates.append(p)
                            break
            else:
                candidates = sorted(temp_path.glob("layout_page*.jpg"))
                candidates += sorted(temp_path.glob("layout_page*.png"))
            for src in candidates:
                dst = layout_vis_dir / f"{page_name}{src.suffix}"
                shutil.move(str(src), str(dst))
            # Clean up empty temp dir
            if indices is None:
                try:
                    temp_path.rmdir()
                except Exception:
                    pass

    tprint(f"  [PAGE {page_num}/{total_pages}] done ✅")

    # Return page data for cross-page assembly
    return {
        "page_num": page_num,
        "json_result": result.json_result,
        "markdown_result": result.markdown_result or "",
    }


def process_one_pdf(
    pdf_path: Path,
    out_root: Path,
    dpi: int,
    image_format: str,
    start_page: int,
    end_page: int,
    fail_fast: bool,
    copy_pdf: bool,
    page_workers: int,
    save_images: str,
    ocr: GlmOcr,
) -> None:
    pdf_path = pdf_path.resolve()
    pdf_name = sanitize_folder_name(pdf_path.stem)
    pdf_out_dir = out_root / pdf_name
    pdf_out_dir.mkdir(parents=True, exist_ok=True)

    # Flat output folders (page-wise — preserved as-is)
    pages_dir      = pdf_out_dir / "images" / "pages"
    layout_vis_dir = pdf_out_dir / "images" / "layout_vis"
    markdown_dir   = pdf_out_dir / "markdown"
    json_dir       = pdf_out_dir / "json"
    if save_images in ("pages", "both"):
        pages_dir.mkdir(parents=True, exist_ok=True)
    if save_images in ("layout", "both"):
        layout_vis_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Also create pages/ directory for normalized page-level markdown (alias)
    pages_md_dir = pdf_out_dir / "pages"
    pages_md_dir.mkdir(parents=True, exist_ok=True)

    if copy_pdf:
        dst_pdf = pdf_out_dir / pdf_path.name
        if dst_pdf.resolve() != pdf_path:
            shutil.copy2(pdf_path, dst_pdf)

    # Open once just to count pages, then close
    with fitz.open(pdf_path) as doc:
        n_pages = doc.page_count

    s = max(1, start_page)
    e = end_page if end_page and end_page > 0 else n_pages
    e = min(e, n_pages)

    if s > e:
        raise ValueError(f"Invalid page range: start={s}, end={e}, total_pages={n_pages}")

    tprint(f"\n[PDF] {pdf_path.name}  |  pages={n_pages}  |  range={s}..{e}  |  page_workers={page_workers}")
    tprint(f"[PDF] Output -> {pdf_out_dir}")

    failed_pages: list[int] = []
    page_results: dict[int, dict] = {}  # page_num -> result dict

    with ThreadPoolExecutor(max_workers=page_workers) as executor:
        futures = {
            executor.submit(
                process_one_page,
                pdf_path, p, e,
                pages_dir, layout_vis_dir, markdown_dir, json_dir,
                dpi, image_format, save_images, ocr,
            ): p
            for p in range(s, e + 1)
        }

        for future in as_completed(futures):
            p = futures[future]
            try:
                result_data = future.result()
                if result_data is not None:
                    page_results[p] = result_data
            except Exception as ex:
                tprint(f"  [ERROR] Page {p} failed: {ex}", file=sys.stderr)
                failed_pages.append(p)
                if fail_fast:
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(f"Page {p} failed (--fail-fast): {ex}") from ex

    if failed_pages:
        tprint(f"[PDF] {pdf_path.name} — {len(failed_pages)} page(s) failed: {sorted(failed_pages)}", file=sys.stderr)
    else:
        tprint(f"[PDF] {pdf_path.name} — all pages done ✅")

    # =====================================================================
    # NEW: Cross-page assembly + section-based output
    # =====================================================================
    if page_results:
        tprint(f"[PDF] {pdf_path.name} — assembling sections...")
        try:
            from glmocr.postprocess.document_assembler import DocumentAssembler

            assembler = DocumentAssembler()

            # Feed pages in order
            for page_num in sorted(page_results.keys()):
                pr = page_results[page_num]
                assembler.add_page(
                    page_num=pr["page_num"],
                    json_result=pr["json_result"],
                    markdown_result=pr["markdown_result"],
                )

                # Write raw page-level markdown to pages/ (faithful copy for debug)
                md_content = pr.get("markdown_result", "")
                if md_content and md_content.strip():
                    page_md_file = pages_md_dir / f"page_{page_num:03d}.md"
                    page_md_file.write_text(md_content, encoding="utf-8")

            # Run assembly: produces sections/, sections_json/, document_index.json
            # Normalization happens ONCE inside assembler (#3, #17)
            doc_index = assembler.assemble(
                output_dir=str(pdf_out_dir),
                source_file=pdf_path.name,
            )

            n_sections = doc_index.get("total_sections", 0)
            tprint(f"[PDF] {pdf_path.name} — assembly done: {n_sections} sections ✅")

        except Exception as ex:
            tprint(f"  [WARN] Section assembly failed (page outputs are fine): {ex}", file=sys.stderr)
            import traceback
            traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run GLM-OCR on PDFs (page-by-page), with optional parallelism.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--pdf",     help="Path to a single PDF file")
    inp.add_argument("--pdf-dir", help="Path to a directory — all *.pdf files inside will be processed")

    # Output
    ap.add_argument("--out",         default="./outputs", help="Output root folder")
    ap.add_argument("--copy-pdf",    action="store_true", help="Copy input PDF into its output folder")
    ap.add_argument(
        "--save-images",
        choices=["pages", "layout", "both", "none"],
        default="both",
        help=(
            "Which images to save. "
            "'pages' = rendered page PNGs/JPGs only; "
            "'layout' = layout bbox visualization only (requires --enable-layout); "
            "'both' = save both (default); "
            "'none' = skip all images."
        ),
    )

    # Rendering
    ap.add_argument("--dpi",          type=int, default=300,  help="Render DPI")
    ap.add_argument("--image-format", choices=["png", "jpg"], default="png", help="Page image format")
    ap.add_argument("--start-page",   type=int, default=1,    help="1-based start page")
    ap.add_argument("--end-page",     type=int, default=0,    help="1-based end page (0 = last)")

    # Parallelism
    ap.add_argument(
        "--page-workers", type=int, default=4,
        help="Concurrent OCR workers per PDF. Each worker sends one HTTP request to vLLM. "
             "Raise this if your vLLM server can handle more concurrent requests.",
    )
    ap.add_argument(
        "--pdf-workers", type=int, default=1,
        help="PDFs to process in parallel (useful with --pdf-dir). "
             "Total OCR concurrency = pdf-workers x page-workers.",
    )

    # Behaviour
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure")

    # GLM-OCR runtime
    ap.add_argument("--mode",          choices=["maas", "selfhosted"], default=None)
    ap.add_argument("--api-key",       default=None, help="MaaS API key")
    ap.add_argument("--config",        default=None, help="Path to config.yaml")
    ap.add_argument("--enable-layout", action="store_true", help="Enable layout detection")
    ap.add_argument("--log-level",     default=None, help="DEBUG / INFO / WARNING / ERROR")

    # Self-hosted vLLM / SGLang
    ap.add_argument("--ocr-host", default=None, help="Self-hosted OCR API host (e.g. localhost)")
    ap.add_argument("--ocr-port", type=int, default=None, help="Self-hosted OCR API port (e.g. 8090)")

    args = ap.parse_args()

    # Env overrides (SDK reads GLMOCR_*)
    if args.mode:              os.environ["GLMOCR_MODE"]          = args.mode
    if args.api_key:           os.environ["GLMOCR_API_KEY"]       = args.api_key
    if args.log_level:         os.environ["GLMOCR_LOG_LEVEL"]     = args.log_level
    if args.enable_layout:     os.environ["GLMOCR_ENABLE_LAYOUT"] = "true"
    if args.ocr_host:          os.environ["GLMOCR_OCR_API_HOST"]  = args.ocr_host
    if args.ocr_port is not None: os.environ["GLMOCR_OCR_API_PORT"] = str(args.ocr_port)

    # GlmOcr constructor kwargs
    parser_kwargs: dict = {}
    if args.config:        parser_kwargs["config_path"]   = str(Path(args.config).expanduser().resolve())
    if args.api_key:       parser_kwargs["api_key"]       = args.api_key
    if args.mode:          parser_kwargs["mode"]          = args.mode
    if args.enable_layout: parser_kwargs["enable_layout"] = True
    if args.log_level:     parser_kwargs["log_level"]     = args.log_level

    # Collect PDFs
    pdfs: list[Path] = []
    if args.pdf:
        p = Path(args.pdf).expanduser()
        if not p.exists():
            print(f"[ERROR] PDF not found: {p}", file=sys.stderr)
            return 2
        pdfs = [p]
    else:
        d = Path(args.pdf_dir).expanduser()
        if not d.exists() or not d.is_dir():
            print(f"[ERROR] Not a directory: {d}", file=sys.stderr)
            return 2
        pdfs = sorted(d.glob("*.pdf"))
        if not pdfs:
            print(f"[ERROR] No PDFs found in: {d}", file=sys.stderr)
            return 2

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output root  : {out_root}")
    print(f"[INFO] PDFs found   : {len(pdfs)}")
    print(f"[INFO] PDF workers  : {args.pdf_workers}")
    print(f"[INFO] Page workers : {args.page_workers}")

    pdf_kwargs = dict(
        out_root=out_root,
        dpi=args.dpi,
        image_format=args.image_format,
        start_page=args.start_page,
        end_page=args.end_page,
        fail_fast=args.fail_fast,
        copy_pdf=args.copy_pdf,
        page_workers=args.page_workers,
        save_images=args.save_images,
    )

    try:
        with GlmOcr(**parser_kwargs) as ocr:

            def _run(pdf_path: Path) -> None:
                process_one_pdf(pdf_path=pdf_path, ocr=ocr, **pdf_kwargs)

            if args.pdf_workers > 1 and len(pdfs) > 1:
                # Parallel across PDFs
                with ThreadPoolExecutor(max_workers=args.pdf_workers) as executor:
                    futures = {executor.submit(_run, pdf): pdf for pdf in pdfs}
                    for future in as_completed(futures):
                        pdf = futures[future]
                        try:
                            future.result()
                        except Exception as ex:
                            tprint(f"[ERROR] PDF failed: {pdf.name}: {ex}", file=sys.stderr)
                            if args.fail_fast:
                                return 1
            else:
                # Sequential across PDFs (default)
                for pdf in pdfs:
                    try:
                        _run(pdf)
                    except Exception as ex:
                        tprint(f"[ERROR] PDF failed: {pdf.name}: {ex}", file=sys.stderr)
                        if args.fail_fast:
                            return 1

        print("\n[INFO] All done ✅")
        return 0

    except Exception as ex:
        print(f"[FATAL] {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())