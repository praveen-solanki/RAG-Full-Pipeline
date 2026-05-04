"""Deterministic text normalization for OCR post-processing.

This module is the SINGLE normalization pass for the entire pipeline (#17).
The assembler calls normalize_text() exactly once on stitched blocks.
result_formatter._clean_content() does only legacy-compat minimal cleaning
(repeated-content, literal \\t) and does NOT call normalize_text().

Also provides:
- AutoSAR terminology protection (#9)
- Block quality scoring (#13)
- Cross-reference extraction (#11)
- Markdown table structure parsing (#12)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# AutoSAR terminology protection (#9)
# ═══════════════════════════════════════════════════════════════════════════

_AUTOSAR_REQ_RE = re.compile(
    r"\[(?:SWS|SRS|ECUC|TPS|RS|CONC)_[A-Za-z]+_\d{4,6}\]"
)
_AUTOSAR_PATH_RE = re.compile(r"(/AUTOSAR(?:/[A-Za-z][A-Za-z0-9_]*)+)")
_AUTOSAR_API_RE = re.compile(r"\b([A-Z][a-z]+(?:_[A-Z][a-zA-Z0-9]*)+)\b")
_AUTOSAR_FUNC_RE = re.compile(r"\b([A-Z][a-z]{1,10}_(?:[A-Z][a-zA-Z0-9]+))\b")
_C_IDENT_RE = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")

_PROTECT_PATTERNS = [
    _AUTOSAR_REQ_RE,
    _AUTOSAR_PATH_RE,
    _AUTOSAR_API_RE,
    _AUTOSAR_FUNC_RE,
    _C_IDENT_RE,
]


def _protect_technical_terms(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace AutoSAR identifiers with placeholders before normalization."""
    restore_map: Dict[str, str] = {}
    reverse: Dict[str, str] = {}
    counter = [0]

    def _repl(match: re.Match) -> str:
        original = match.group(0)
        if original in reverse:
            return reverse[original]
        counter[0] += 1
        ph = f"\x00PROT{counter[0]:04d}\x00"
        restore_map[ph] = original
        reverse[original] = ph
        return ph

    for pattern in _PROTECT_PATTERNS:
        text = pattern.sub(_repl, text)
    return text, restore_map


def _restore_technical_terms(text: str, restore_map: Dict[str, str]) -> str:
    """Restore protected terms from placeholders."""
    for ph, original in restore_map.items():
        text = text.replace(ph, original)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Unicode normalization
# ═══════════════════════════════════════════════════════════════════════════

_CONFUSABLE_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\uff08": "(", "\uff09": ")", "\uff1a": ":", "\uff1b": ";",
    "\uff0c": ",", "\uff0e": ".",
    "\u2013": "-", "\u2014": "--",
    "\u00a0": " ", "\u2002": " ", "\u2003": " ",
    "\u200b": "", "\ufeff": "", "\u00ad": "",
}
_CONFUSABLE_RE = re.compile("|".join(re.escape(k) for k in _CONFUSABLE_MAP))


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return _CONFUSABLE_RE.sub(lambda m: _CONFUSABLE_MAP[m.group()], text)


# ═══════════════════════════════════════════════════════════════════════════
# Whitespace normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_whitespace(text: str) -> str:
    text = text.replace("\\t", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# OCR artifact removal
# ═══════════════════════════════════════════════════════════════════════════

_GARBAGE_PATTERNS = [
    re.compile(r"^[|_\-=]{5,}$"),
    re.compile(r"^[\.\s]{10,}$"),
    re.compile(r"^[^\w\s]{3,}$"),
    re.compile(r"^\s*[|]\s*$"),
]


def remove_ocr_artifacts(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if any(p.match(stripped) for p in _GARBAGE_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ═══════════════════════════════════════════════════════════════════════════
# Repeated content
# ═══════════════════════════════════════════════════════════════════════════

def remove_repeated_lines(text: str, threshold: int = 3) -> str:
    lines = text.split("\n")
    if len(lines) < threshold:
        return text
    result: List[str] = []
    repeat_count = 1
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == lines[i - 1].strip() and line.strip():
            repeat_count += 1
        else:
            repeat_count = 1
        if repeat_count <= 2:
            result.append(line)
    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════════════
# Header / Footer detection (#18 — positional + frequency)
# ═══════════════════════════════════════════════════════════════════════════

_HEADER_FOOTER_TEXT_PATTERNS = [
    re.compile(r"^\s*(?:page|seite)\s+\d+\s*(?:of|von)\s+\d+\s*$", re.I),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*©\s+\d{4}", re.I),
    re.compile(r"^\s*confidential\s*$", re.I),
    re.compile(r"^\s*draft\s*$", re.I),
]


def is_header_footer_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _HEADER_FOOTER_TEXT_PATTERNS)


def is_positional_header_footer(bbox: Optional[List[int]], page_height: int = 1000) -> bool:
    """Check if a block is in the top or bottom 8% of the page by bbox y-coordinate.

    Uses normalized 0-1000 coordinate system (standard in this pipeline).
    """
    if not bbox or len(bbox) < 4 or page_height <= 0:
        return False
    _, y1, _, y2 = bbox
    threshold = int(page_height * 0.08)
    # Top 8%
    if y2 <= threshold:
        return True
    # Bottom 8%
    if y1 >= page_height - threshold:
        return True
    return False


def strip_headers_footers(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(line for line in lines if not is_header_footer_line(line))


# ═══════════════════════════════════════════════════════════════════════════
# Bullet / numbering normalization
# ═══════════════════════════════════════════════════════════════════════════

_BULLET_RE = re.compile(r"^(\s*)([•·◦▪▸►➢➤✓✔☐☑■□▶])(\s*)(.*)$")
_NUMBERED_RE = re.compile(r"^(\s*)(\d{1,3}|[a-zA-Z])([.)\]）])(\s*)(.*)$")


def normalize_bullets(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        m = _BULLET_RE.match(line)
        if m:
            indent, _, _, content = m.groups()
            result.append(f"{indent}- {content.lstrip()}")
        else:
            result.append(line)
    return "\n".join(result)


def normalize_numbered_lists(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        m = _NUMBERED_RE.match(line)
        if m:
            indent, num, sep, _, content = m.groups()
            if sep == "）":
                sep = ")"
            result.append(f"{indent}{num}{sep} {content.lstrip()}")
        else:
            result.append(line)
    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════════════
# Paragraph reconstruction helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_sentence_incomplete(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    if text[-1] in ".!?;:。！？；":
        return False
    if text[-1] in ")]}" and len(text) > 3:
        return False
    if text.startswith("#"):
        return False
    if text.rstrip().endswith("|") or text.rstrip().endswith("```"):
        return False
    if text[-1] == "-":
        return True
    if text[-1] == "," or text[-1].islower():
        return True
    return False


def is_continuation_line(text: str) -> bool:
    text = text.lstrip()
    if not text:
        return False
    if text[0].islower():
        return True
    continuation_words = {
        "and", "or", "but", "the", "a", "an", "in", "of",
        "to", "for", "with", "that", "which", "where",
        "when", "as", "by", "from", "on", "at", "is",
        "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
    }
    first_word = text.split()[0].lower().rstrip(".,;:")
    return first_word in continuation_words


# ═══════════════════════════════════════════════════════════════════════════
# Table text cleanup
# ═══════════════════════════════════════════════════════════════════════════

def clean_table_text(text: str) -> str:
    if not text or "|" not in text:
        return text
    lines = text.split("\n")
    result = []
    for line in lines:
        if "|" in line:
            cells = line.split("|")
            cells = [c.strip() for c in cells]
            result.append(" | ".join(cells))
        else:
            result.append(line)
    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════════════
# Block quality scoring (#13)
# ═══════════════════════════════════════════════════════════════════════════

def compute_block_quality(text: str) -> float:
    """Score text quality 0.0 (garbage) to 1.0 (clean).

    Used to gate cross-page merges and flag bad blocks.
    """
    if not text or not text.strip():
        return 0.0
    stripped = text.strip()
    total = len(stripped)
    if total == 0:
        return 0.0

    # Letter ratio is the strongest signal — no letters = garbage
    letters = sum(1 for c in stripped if c.isalpha())
    letter_ratio = letters / total
    if letter_ratio < 0.1:
        return round(letter_ratio, 3)

    alnum = sum(1 for c in stripped if c.isalnum())
    alnum_ratio = alnum / total

    words = stripped.split()
    if words:
        avg_wl = sum(len(w) for w in words) / len(words)
        word_score = 1.0 if 3 <= avg_wl <= 12 else (0.2 if avg_wl < 2 or avg_wl > 20 else 0.6)
    else:
        word_score = 0.0

    # Real text has spaces between words
    space_score = 1.0 if " " in stripped else 0.3

    return round(min(1.0, max(0.0, 0.35 * letter_ratio + 0.25 * alnum_ratio + 0.2 * word_score + 0.2 * space_score)), 3)


def is_low_quality_block(text: str, threshold: float = 0.3) -> bool:
    return compute_block_quality(text) < threshold


# ═══════════════════════════════════════════════════════════════════════════
# Cross-reference extraction (#11)
# ═══════════════════════════════════════════════════════════════════════════

_XREF_REQ_RE = re.compile(r"\[(?:SWS|SRS|ECUC|TPS|RS|CONC)_[A-Za-z]+_\d{4,6}\]")
_XREF_SECTION_RE = re.compile(r"(?:(?:[Ss]ection|[Ss]ee|[Cc]hapter|[Aa]nnex)\s+)(\d+(?:\.\d+)*)")
_XREF_TABLE_RE = re.compile(r"[Tt]able\s+(\d+(?:[.-]\d+)?)")
_XREF_FIGURE_RE = re.compile(r"(?:[Ff]igure|[Ff]ig\.?)\s+(\d+(?:[.-]\d+)?)")


def extract_cross_references(text: str) -> Dict[str, List[str]]:
    """Extract cross-references from text. Returns deduplicated lists per type."""
    refs: Dict[str, List[str]] = {"requirements": [], "sections": [], "tables": [], "figures": []}
    if not text:
        return refs
    for m in _XREF_REQ_RE.finditer(text):
        r = m.group(0)
        if r not in refs["requirements"]:
            refs["requirements"].append(r)
    for m in _XREF_SECTION_RE.finditer(text):
        r = m.group(1)
        if r not in refs["sections"]:
            refs["sections"].append(r)
    for m in _XREF_TABLE_RE.finditer(text):
        r = f"Table {m.group(1)}"
        if r not in refs["tables"]:
            refs["tables"].append(r)
    for m in _XREF_FIGURE_RE.finditer(text):
        r = f"Figure {m.group(1)}"
        if r not in refs["figures"]:
            refs["figures"].append(r)
    return refs


# ═══════════════════════════════════════════════════════════════════════════
# Markdown table structure parsing (#12)
# ═══════════════════════════════════════════════════════════════════════════

def parse_markdown_table(text: str) -> Optional[Dict]:
    """Parse a Markdown table into {headers, rows, column_count}.
    Returns None if not a valid table. Does NOT hallucinate structure.
    """
    if not text or "|" not in text:
        return None
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    pipe_lines = [l for l in lines if "|" in l]
    if len(pipe_lines) < 2:
        return None

    def _split(line: str) -> List[str]:
        cells = line.split("|")
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        return [c.strip() for c in cells]

    headers = _split(pipe_lines[0])
    col_count = len(headers)
    if col_count == 0:
        return None

    sep_idx = None
    for i, line in enumerate(pipe_lines[1:3], 1):
        if not line.replace("|", "").replace("-", "").replace(":", "").strip():
            sep_idx = i
            break

    data_start = (sep_idx + 1) if sep_idx is not None else 1
    rows = []
    for line in pipe_lines[data_start:]:
        if not line.replace("|", "").replace("-", "").replace(":", "").strip():
            continue
        cells = _split(line)
        while len(cells) < col_count:
            cells.append("")
        rows.append(cells[:col_count])

    return {"headers": headers, "rows": rows, "column_count": col_count}


# ═══════════════════════════════════════════════════════════════════════════
# Composite normalizer (#17 — single pass, protects terms)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """The ONE normalization function the pipeline should call.

    Protects AutoSAR terms → normalizes → restores.
    """
    if not text:
        return text
    text, restore_map = _protect_technical_terms(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = remove_ocr_artifacts(text)
    text = remove_repeated_lines(text)
    text = normalize_bullets(text)
    text = normalize_numbered_lists(text)
    text = _restore_technical_terms(text, restore_map)
    return text.strip()
