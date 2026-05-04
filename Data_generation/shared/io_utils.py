"""
IO helpers for the pipeline.

All file writes are either append-only (for jsonl) or atomic (tmp + rename,
for json) so a crash mid-write never corrupts state. This is what makes
every stage resumable.
"""

import json
import os
from pathlib import Path
from typing import Any, Iterator


def atomic_write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def count_jsonl(path: Path) -> int:
    if not Path(path).exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
