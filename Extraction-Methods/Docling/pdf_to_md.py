#!/usr/bin/env python3
"""
AUTOSAR PDF to Markdown Converter using MinerU v3.

Usage:
    python convert.py --input spec.pdf --output ./output
    python convert.py --input ./pdfs/ --output ./output --backend hybrid
    python convert.py --input ./pdfs/ --output ./output --multi-gpu --gpu-ids 0,1
    python convert.py --input spec.pdf --output ./output --cpu

Backend names (your installed MinerU version):
    pipeline             - fastest, GPU/CPU, best for batch (default)
    hybrid               - alias for hybrid-auto-engine (recommended for AUTOSAR)
    hybrid-auto-engine   - pipeline layout + VLM for complex blocks, best quality
    vlm                  - alias for vlm-auto-engine
    vlm-auto-engine      - VLM model, best heading detection
    hybrid-http-client   - hybrid via remote server
    vlm-http-client      - vlm via remote server
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("autosar-converter")


# ─────────────────────────────────────────────────────────────────────────────
# AUTOSAR-specific post-processing
# ─────────────────────────────────────────────────────────────────────────────

# Matches numbered section headings that MinerU sometimes writes as plain text
# e.g.  "7.3.2 RTE Interface\n" should become "### 7.3.2 RTE Interface"
_NUMBERED_HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+){0,4})\s{1,4}([A-Z][^\n]{3,120})$",
    re.MULTILINE,
)

# AUTOSAR constraint / requirement IDs  e.g. [SWS_Com_00325]  [RS_BSW_00102]
_CONSTRAINT_ID_RE = re.compile(
    r"(\[(?:SWS|RS|TPS|SRS|ECUC|ASWS)_[A-Za-z0-9_]+\])"
)

# Page headers / footers that OCR sometimes drags in
# e.g. "AUTOSAR AP R23-11  Document Title  Page 3 of 47"
_PAGE_ARTIFACT_RE = re.compile(
    r"^(AUTOSAR\s+\w+\s+R\d{2}-\d{2}.*|Document ID.*|Page \d+ of \d+.*)\n",
    re.MULTILINE | re.IGNORECASE,
)

# Heading level assignment based on numbering depth
def _numbering_depth(num_str: str) -> int:
    return len(num_str.split("."))


def _assign_heading(match: re.Match) -> str:
    """Convert a numbered plain-text line to a proper Markdown heading."""
    num, title = match.group(1), match.group(2)
    depth = _numbering_depth(num)
    hashes = "#" * min(depth, 4)          # cap at h4 to avoid clutter
    return f"{hashes} {num} {title}"


def postprocess_markdown(raw_md: str, source_pdf_name: str) -> str:
    """
    Apply AUTOSAR-specific cleaning and heading normalisation to raw MinerU output.

    Steps:
      1. Remove page-level header/footer artefacts
      2. Ensure double newlines between paragraphs (fixes 1-chunk problem)
      3. Detect numbered headings that MinerU missed and inject # markers
      4. Preserve AUTOSAR constraint IDs in backtick formatting
      5. Normalise excessive blank lines
    """
    md = raw_md

    # ── 1. Strip page artefacts ──────────────────────────────────────────────
    md = _PAGE_ARTIFACT_RE.sub("", md)

    # ── 2. Ensure double newlines between paragraphs ─────────────────────────
    # Replace single newline between non-blank lines with double newline
    # This is the fix for the "1-chunk paragraph problem" found in benchmarks
    md = re.sub(r"(?<!\n)\n(?!\n)(?!#)(?![-*|])", "\n\n", md)

    # ── 3. Inject headings for numbered sections MinerU missed ───────────────
    # Only apply when the line is NOT already a heading (doesn't start with #)
    lines = md.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            new_lines.append(line)
            continue
        m = _NUMBERED_HEADING_RE.match(line)
        if m:
            new_lines.append(_assign_heading(m))
        else:
            new_lines.append(line)
    md = "\n".join(new_lines)

    # ── 4. Wrap AUTOSAR constraint IDs in backticks ──────────────────────────
    md = _CONSTRAINT_ID_RE.sub(r"`\1`", md)

    # ── 5. Collapse 3+ consecutive blank lines to 2 ─────────────────────────
    md = re.sub(r"\n{3,}", "\n\n", md)

    # ── 6. Add document source comment at top ───────────────────────────────
    header = f"<!-- source: {source_pdf_name} -->\n\n"
    if not md.startswith("<!--"):
        md = header + md

    return md.strip() + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion logic
# ─────────────────────────────────────────────────────────────────────────────

def build_mineru_command(
    input_path: Path,
    output_dir: Path,
    backend: str,
    api_url: str | None = None,
) -> list[str]:
    """
    Build the mineru CLI command.

    MinerU v3 syntax:
        mineru -p <input> -o <output> -b <backend> [--api-url <url>]

    Backends:
        pipeline      – local GPU/CPU, fastest for batch, uses layout models
        vlm           – vision-language model, best heading detection
        hybrid        – pipeline for layout + VLM for tricky blocks (best quality)
    """
    cmd = [
        "mineru",
        "-p", str(input_path),
        "-o", str(output_dir),
        "-b", backend,
    ]
    if api_url:
        cmd += ["--api-url", api_url]
    return cmd


def find_output_md(output_dir: Path, pdf_stem: str) -> Path | None:
    """
    MinerU writes the .md file to:
        <output_dir>/<pdf_stem>/<pdf_stem>.md

    This function searches the output tree for the right file.
    """
    # Primary expected location
    candidate = output_dir / pdf_stem / f"{pdf_stem}.md"
    if candidate.exists():
        return candidate

    # Fallback: search recursively for any .md that isn't a debug file
    for p in output_dir.rglob("*.md"):
        if p.stem == pdf_stem and "_layout" not in p.stem:
            return p

    # Last resort: any .md in the output dir
    mds = [p for p in output_dir.rglob("*.md") if "_layout" not in p.name]
    if mds:
        return mds[0]

    return None


def convert_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    backend: str,
    final_dir: Path,
    api_url: str | None = None,
) -> bool:
    """
    Convert one PDF to a single, post-processed Markdown file.

    Returns True on success, False on failure.
    """
    pdf_stem = pdf_path.stem
    log.info(f"Converting  {pdf_path.name} ...")
    t0 = time.time()

    # ── Run MinerU ───────────────────────────────────────────────────────────
    raw_output_dir = output_dir / "mineru_raw" / pdf_stem
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_mineru_command(pdf_path, raw_output_dir, backend, api_url)
    log.debug(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,               # 30-min timeout for huge AUTOSAR specs
    )

    if result.returncode != 0:
        log.error(f"  MinerU failed for {pdf_path.name}")
        log.error(f"  STDERR: {result.stderr[-1000:]}")
        return False

    elapsed = time.time() - t0

    # ── Find the output .md file ─────────────────────────────────────────────
    md_file = find_output_md(raw_output_dir, pdf_stem)
    if md_file is None:
        log.error(f"  No .md output found for {pdf_path.name} in {raw_output_dir}")
        return False

    # ── Post-process ─────────────────────────────────────────────────────────
    raw_md = md_file.read_text(encoding="utf-8", errors="replace")
    clean_md = postprocess_markdown(raw_md, pdf_path.name)

    # ── Write final file ─────────────────────────────────────────────────────
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{pdf_stem}.md"
    final_path.write_text(clean_md, encoding="utf-8")

    size_kb = final_path.stat().st_size / 1024
    log.info(
        f"  Done  {pdf_path.name}  →  {final_path.name}  "
        f"({size_kb:.1f} KB, {elapsed:.1f}s)"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU router startup
# ─────────────────────────────────────────────────────────────────────────────

def start_mineru_router(gpu_ids: str = "auto") -> subprocess.Popen:
    """
    Launch mineru-router for multi-GPU load balancing.
    Returns the Popen handle so the caller can terminate it on exit.
    """
    cmd = [
        "mineru-router",
        "--host", "127.0.0.1",
        "--port", "8002",
        "--local-gpus", gpu_ids,
        "--enable-vlm-preload", "true",
    ]
    log.info(f"Starting mineru-router on GPU(s): {gpu_ids}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for router to become healthy
    import urllib.request
    health_url = "http://127.0.0.1:8002/health"
    for _ in range(60):                     # up to 60 seconds
        try:
            urllib.request.urlopen(health_url, timeout=1)
            log.info("  Router is healthy")
            return proc
        except Exception:
            time.sleep(1)

    log.warning("  Router did not become healthy within 60s — continuing anyway")
    return proc


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert AUTOSAR PDF(s) to structured Markdown using MinerU v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a single PDF file OR a directory containing PDFs"
    )
    parser.add_argument(
        "--output", "-o", default="./md_output",
        help="Root output directory (default: ./md_output)"
    )
    parser.add_argument(
        "--backend", "-b", default="pipeline",
        choices=["pipeline", "vlm", "hybrid", "vlm-auto-engine", "hybrid-auto-engine", "vlm-http-client", "hybrid-http-client"],
        help=(
            "MinerU backend to use:\n"
            "  pipeline – fastest, GPU/CPU, best for batch (default)\n"
            "  vlm      – vision-language model, best heading detection\n"
            "  hybrid   – pipeline layout + VLM for complex blocks (best quality)"
        )
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU-only mode (forces pipeline backend)"
    )
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Launch mineru-router for multi-GPU load balancing (Linux only)"
    )
    parser.add_argument(
        "--gpu-ids", default="auto",
        help="GPU IDs for multi-GPU mode, e.g. '0,1' or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip PDFs that already have a .md output file"
    )
    parser.add_argument(
        "--no-postprocess", action="store_true",
        help="Skip AUTOSAR-specific post-processing (use raw MinerU output)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Resolve short aliases → actual MinerU backend names
    _BACKEND_ALIAS = {
        "hybrid": "hybrid-auto-engine",
        "vlm":    "vlm-auto-engine",
    }
    args.backend = _BACKEND_ALIAS.get(args.backend, args.backend)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Validate MinerU is installed ─────────────────────────────────────────
    if shutil.which("mineru") is None:
        log.error(
            "mineru not found in PATH.\n"
            "Install with:  pip install 'mineru[all]'\n"
            "Then download models:  mineru-models-download"
        )
        sys.exit(1)

    # ── Resolve input ─────────────────────────────────────────────────────────
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(input_path.rglob("*.pdf"))
        if not pdf_files:
            log.error(f"No PDF files found in {input_path}")
            sys.exit(1)
        log.info(f"Found {len(pdf_files)} PDF file(s) in {input_path}")
    else:
        log.error(f"Input must be a .pdf file or directory: {input_path}")
        sys.exit(1)

    # ── Resolve output ────────────────────────────────────────────────────────
    output_root = Path(args.output).expanduser().resolve()
    final_dir   = output_root / "final_md"
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # ── Backend selection ─────────────────────────────────────────────────────
    backend = "pipeline" if args.cpu else args.backend

    # CPU mode: tell MinerU to use pipeline (no GPU needed)
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log.info("CPU mode enabled — using pipeline backend")

    # ── Multi-GPU router ──────────────────────────────────────────────────────
    router_proc = None
    api_url = None

    if args.multi_gpu and not args.cpu:
        router_proc = start_mineru_router(args.gpu_ids)
        api_url = "http://127.0.0.1:8002"
        log.info(f"Multi-GPU router started at {api_url}")

    # ── Process files ─────────────────────────────────────────────────────────
    total   = len(pdf_files)
    success = 0
    failed  = []

    log.info(f"Backend: {backend}  |  Output: {final_dir}")
    log.info("─" * 60)

    try:
        for i, pdf_path in enumerate(pdf_files, 1):
            log.info(f"[{i}/{total}]  {pdf_path.name}")

            # Skip if already done
            if args.skip_existing:
                target = final_dir / f"{pdf_path.stem}.md"
                if target.exists():
                    log.info(f"  Skipped (already exists): {target.name}")
                    success += 1
                    continue

            ok = convert_single_pdf(
                pdf_path   = pdf_path,
                output_dir = output_root,
                backend    = backend,
                final_dir  = final_dir,
                api_url    = api_url,
            )

            if ok:
                success += 1
                # Apply post-processing if not already done inside convert_single_pdf
                # (post-processing is always applied inside, but if --no-postprocess
                #  is set we re-read raw MinerU output)
                if args.no_postprocess:
                    raw_md_path = find_output_md(
                        output_root / "mineru_raw" / pdf_path.stem,
                        pdf_path.stem
                    )
                    if raw_md_path:
                        raw = raw_md_path.read_text(encoding="utf-8", errors="replace")
                        (final_dir / f"{pdf_path.stem}.md").write_text(raw, encoding="utf-8")
            else:
                failed.append(pdf_path.name)

    finally:
        if router_proc:
            log.info("Shutting down mineru-router ...")
            router_proc.terminate()
            router_proc.wait(timeout=10)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info(f"Completed:  {success}/{total} succeeded")
    if failed:
        log.warning(f"Failed ({len(failed)}):")
        for f in failed:
            log.warning(f"  - {f}")

    log.info(f"Output Markdown files → {final_dir}")

    # Write a conversion log
    log_path = output_root / "conversion_log.txt"
    with open(log_path, "w") as lf:
        lf.write(f"Converted: {success}/{total}\n")
        lf.write(f"Backend:   {backend}\n")
        if failed:
            lf.write("Failed:\n")
            for f in failed:
                lf.write(f"  {f}\n")
    log.info(f"Log written to {log_path}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()