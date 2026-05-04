#!/usr/bin/env python3
"""
eval_glmocr_overlap.py

Simple overlap evaluation: checks how many ground-truth leaf values are present
in the merged GLM-OCR output text (substring match with normalization).

- Page-aware: if GT entry comes from shipping_bill.pages[*] (has page_number),
  it matches against that page's OCR text; otherwise matches against full doc text.
- Normalization: HTML entity decode, uppercase, whitespace collapse.

Outputs (saved inside --doc-dir):
  eval_report.json
  eval_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from html import unescape as html_unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# --------------------------- Normalization ---------------------------

_ws_re = re.compile(r"\s+")
_nonprint_re = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")

def norm(s: str) -> str:
    """Normalize text for robust substring matching."""
    s = html_unescape(s)                # ✅ decode &#x27; &amp; etc.
    s = s.replace("\u00a0", " ")        # NBSP -> space
    s = _nonprint_re.sub(" ", s)
    s = _ws_re.sub(" ", s.strip())
    return s.upper()

def is_leaf_value(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool))

def leaf_to_string(v: Any) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        # keep original numeric form readable
        return str(v)
    return str(v)


# --------------------------- GT flattening (page-aware) ---------------------------

@dataclass
class GTLeaf:
    path: str
    value: str
    page_number: Optional[int] = None

def flatten_gt(obj: Any, prefix: str = "", page_ctx: Optional[int] = None) -> List[GTLeaf]:
    """
    Flatten GT JSON into leaf nodes.
    If traversing inside shipping_bill.pages[*] and that element has page_number,
    propagate it as page_ctx to all leaves under that page.
    """
    out: List[GTLeaf] = []

    if isinstance(obj, dict):
        # If this dict itself contains page_number, treat it as context
        if "page_number" in obj and isinstance(obj["page_number"], int):
            page_ctx = obj["page_number"]

        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            out.extend(flatten_gt(v, new_prefix, page_ctx))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            # Special: if list item is a dict and contains page_number, set ctx for that branch
            branch_ctx = page_ctx
            if isinstance(item, dict) and "page_number" in item and isinstance(item["page_number"], int):
                branch_ctx = item["page_number"]
            out.extend(flatten_gt(item, new_prefix, branch_ctx))

    else:
        # leaf
        if obj is None:
            return out
        if is_leaf_value(obj):
            out.append(GTLeaf(path=prefix, value=leaf_to_string(obj), page_number=page_ctx))

    return out


# --------------------------- OCR merged parsing ---------------------------

def looks_like_merged_glmocr(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    if "pages" not in d or not isinstance(d["pages"], list):
        return False
    # require at least one page with data
    for p in d["pages"]:
        if isinstance(p, dict) and "data" in p and isinstance(p["data"], dict):
            return True
    return False

def pick_merged_file(doc_dir: Path) -> Path:
    """
    Auto-select a merged json inside doc_dir.
    Heuristic:
      - consider *.json files in doc_dir (not inside page folders)
      - keep those whose JSON has top-level "pages" list with "data"
      - pick the most recently modified among candidates
    """
    candidates = []
    for fp in doc_dir.glob("*.json"):
        try:
            with fp.open("r", encoding="utf-8") as f:
                d = json.load(f)
            if looks_like_merged_glmocr(d):
                candidates.append(fp)
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"No merged GLM-OCR JSON found in {doc_dir}. "
            f"Expected a *.json with top-level 'pages' containing 'data'."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def collect_strings_from_ocr_data(data: Dict[str, Any]) -> List[str]:
    """
    Pull text from:
      - markdown_result (string)
      - json_result (nested list of regions with 'content')
    """
    parts: List[str] = []

    md = data.get("markdown_result")
    if isinstance(md, str) and md.strip():
        parts.append(md)

    jr = data.get("json_result")
    # json_result is usually: [[{region}, {region}, ...]]
    if isinstance(jr, list):
        for block in jr:
            if isinstance(block, list):
                for region in block:
                    if isinstance(region, dict):
                        c = region.get("content")
                        if isinstance(c, str) and c.strip():
                            parts.append(c)

    return parts

def build_page_texts(merged: Dict[str, Any]) -> Tuple[Dict[int, str], str]:
    """
    Returns:
      page_text_norm: {page_number: normalized text}
      full_text_norm: normalized concat of all pages
    """
    page_texts: Dict[int, str] = {}
    all_parts: List[str] = []

    for p in merged.get("pages", []):
        if not isinstance(p, dict):
            continue
        pn = p.get("page_number")
        data = p.get("data")
        if not isinstance(pn, int) or not isinstance(data, dict):
            continue

        parts = collect_strings_from_ocr_data(data)
        raw = "\n".join(parts)
        page_texts[pn] = norm(raw)
        all_parts.append(raw)

    full = norm("\n".join(all_parts))
    return page_texts, full


# --------------------------- Matching & reporting ---------------------------

@dataclass
class MatchResult:
    path: str
    value: str
    page_number: Optional[int]
    matched: bool

def evaluate_overlap(gt: Dict[str, Any], merged: Dict[str, Any]) -> Dict[str, Any]:
    gt_leaves = flatten_gt(gt)

    page_texts_norm, full_text_norm = build_page_texts(merged)

    results: List[MatchResult] = []
    for leaf in gt_leaves:
        v = leaf.value
        v_norm = norm(v)
        if not v_norm:
            continue

        # pick search text
        if leaf.page_number is not None and leaf.page_number in page_texts_norm:
            hay = page_texts_norm[leaf.page_number]
        else:
            hay = full_text_norm

        matched = v_norm in hay
        results.append(MatchResult(leaf.path, leaf.value, leaf.page_number, matched))

    total = len(results)
    hit = sum(1 for r in results if r.matched)
    recall = (hit / total) if total else 0.0

    misses = [r for r in results if not r.matched]

    # Some helpful breakdowns
    by_page: Dict[str, Dict[str, int]] = {}
    for r in results:
        key = str(r.page_number) if r.page_number is not None else "DOC"
        if key not in by_page:
            by_page[key] = {"total": 0, "hit": 0}
        by_page[key]["total"] += 1
        by_page[key]["hit"] += 1 if r.matched else 0

    report = {
        "metric": "substring_overlap_recall",
        "total_fields_checked": total,
        "matched_fields": hit,
        "recall": round(recall, 6),
        "breakdown_by_page": {
            k: {
                "total": v["total"],
                "hit": v["hit"],
                "recall": round((v["hit"] / v["total"]) if v["total"] else 0.0, 6),
            }
            for k, v in sorted(by_page.items(), key=lambda x: (x[0] != "DOC", x[0]))
        },
        "top_misses": [
            {
                "path": m.path,
                "page_number": m.page_number,
                "value": m.value,
            }
            for m in misses[:80]
        ],
    }
    return report

def write_reports(doc_dir: Path, report: Dict[str, Any], merged_path: Path, gt_path: Path) -> None:
    out_json = doc_dir / "eval_report.json"
    out_md = doc_dir / "eval_report.md"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = []
    md_lines.append("# GLM-OCR Simple Overlap Evaluation\n")
    md_lines.append(f"- **Doc dir:** `{doc_dir}`")
    md_lines.append(f"- **Merged file:** `{merged_path}`")
    md_lines.append(f"- **Ground truth:** `{gt_path}`\n")
    md_lines.append("## Overall\n")
    md_lines.append(f"- Total fields checked: **{report['total_fields_checked']}**")
    md_lines.append(f"- Matched fields: **{report['matched_fields']}**")
    md_lines.append(f"- Recall (substring overlap): **{report['recall']}**\n")
    md_lines.append("## Breakdown by page\n")
    md_lines.append("| Scope | Total | Hit | Recall |")
    md_lines.append("|---|---:|---:|---:|")
    for scope, v in report["breakdown_by_page"].items():
        md_lines.append(f"| {scope} | {v['total']} | {v['hit']} | {v['recall']} |")
    md_lines.append("\n## Top misses (first 80)\n")
    for m in report["top_misses"]:
        md_lines.append(f"- `{m['path']}` (page={m['page_number']}): **{m['value']}**")

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


# --------------------------- CLI ---------------------------

DEFAULT_GT = "/home/mtq3kor/aman/GLM/glm-ocr/Golden_folder/schemas/pm_pack/shipping-bill/gt.json"

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate merged GLM-OCR output vs ground truth (simple overlap).")
    ap.add_argument("--doc-dir", required=True, help="PDF output folder (e.g., .../outputs/<pdf_name>)")
    ap.add_argument("--gt", default=DEFAULT_GT, help=f"Path to ground truth JSON (default: {DEFAULT_GT})")
    ap.add_argument("--merged", default=None, help="Path to merged JSON (optional; auto-detect if not given)")
    args = ap.parse_args()

    doc_dir = Path(args.doc_dir).expanduser().resolve()
    if not doc_dir.exists() or not doc_dir.is_dir():
        print(f"[ERROR] Not a directory: {doc_dir}", file=sys.stderr)
        return 2

    gt_path = Path(args.gt).expanduser().resolve()
    if not gt_path.exists():
        print(f"[ERROR] Ground truth not found: {gt_path}", file=sys.stderr)
        return 2

    merged_path = Path(args.merged).expanduser().resolve() if args.merged else pick_merged_file(doc_dir)
    if not merged_path.exists():
        print(f"[ERROR] Merged file not found: {merged_path}", file=sys.stderr)
        return 2

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    merged = json.loads(merged_path.read_text(encoding="utf-8"))

    if not looks_like_merged_glmocr(merged):
        print(f"[ERROR] File doesn't look like merged GLM-OCR JSON: {merged_path}", file=sys.stderr)
        return 2

    report = evaluate_overlap(gt, merged)
    write_reports(doc_dir, report, merged_path, gt_path)

    print(f"[OK] Recall={report['recall']}  ({report['matched_fields']}/{report['total_fields_checked']})")
    print(f"[OK] Saved: {doc_dir / 'eval_report.json'}")
    print(f"[OK] Saved: {doc_dir / 'eval_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())