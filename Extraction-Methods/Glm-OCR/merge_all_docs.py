#!/usr/bin/env python3
"""
merge_all_docs.py

For each document folder inside an outputs root, merge its page_* JSONs into:
  <doc_folder>/merged.pages.json
"""

from __future__ import annotations
import json, re
from datetime import datetime
from pathlib import Path

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def page_num(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 10**9

def merge_one(doc_dir: Path) -> Path | None:
    page_dirs = sorted(
        [p for p in doc_dir.iterdir() if p.is_dir() and p.name.lower().startswith("page")],
        key=lambda p: page_num(p.name),
    )
    if not page_dirs:
        return None

    pages = []
    for pd in page_dirs:
        pnum = page_num(pd.name)
        fullp = pd / "result.full.to_dict.json"
        altp  = pd / "result.json"
        if fullp.exists():
            chosen = fullp
        elif altp.exists():
            chosen = altp
        else:
            continue
        pages.append({"page_number": pnum, "source_file": str(chosen), "data": load_json(chosen)})

    merged = {
        "schema_version": "glmocr_pages_merge_v1",
        "document_dir": str(doc_dir),
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "page_count_merged": len(pages),
        "pages": pages,
    }

    out_path = doc_dir / "merged.pages.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def main():
    outputs_root = Path("/home/mtq3kor/aman/GLM/glm-ocr/outputs").resolve()
    if not outputs_root.exists():
        raise SystemExit(f"[ERROR] outputs root not found: {outputs_root}")

    doc_dirs = [d for d in outputs_root.iterdir() if d.is_dir()]
    if not doc_dirs:
        raise SystemExit(f"[ERROR] no document folders inside: {outputs_root}")

    for d in sorted(doc_dirs):
        outp = merge_one(d)
        if outp:
            print(f"[OK] {d.name} -> {outp}")
        else:
            print(f"[SKIP] {d.name} (no page_* folders)")

if __name__ == "__main__":
    main()