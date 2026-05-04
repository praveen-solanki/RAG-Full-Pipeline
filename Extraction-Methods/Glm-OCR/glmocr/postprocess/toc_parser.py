"""TOC detection and parsing — tuned for GLM OCR output.

GLM OCR produces TOC lines like:
    1 Introduction 5
        1.1 Purpose of the Document 5
        5.1.1 Synchronous and Asynchronous Mode 11

Key characteristics:
- NO dotted leaders (OCR strips them)
- Indentation preserved (4 spaces per level)
- Page number at end of line separated by space
- May also handle traditional dotted formats from other OCR engines
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TOC entry
# ═══════════════════════════════════════════════════════════════════════════

class TOCEntry:
    __slots__ = ("number", "title", "page_num", "level")

    def __init__(self, number: str, title: str, page_num: int, level: int):
        self.number = number
        self.title = title
        self.page_num = page_num
        self.level = level

    def full_title(self) -> str:
        return f"{self.number} {self.title}" if self.number else self.title

    def __repr__(self) -> str:
        return f"TOCEntry({self.number!r}, {self.title!r}, p{self.page_num}, L{self.level})"


# ═══════════════════════════════════════════════════════════════════════════
# TOC line patterns
# ═══════════════════════════════════════════════════════════════════════════

# PRIMARY: GLM no-dots format: "    1.1 Purpose of the Document 5"
# Handles indentation, section numbers, title, trailing page number
_TOC_NODOTS_RE = re.compile(
    r"^\s*"                                     # optional indent
    r"(\d+(?:\.\d+)*)"                          # section number: "3.2.1"
    r"\s+"                                       # space after number
    r"([A-Za-z][A-Za-z0-9\s&().,:;/\-_]+?)"    # title (non-greedy)
    r"\s+"                                       # space before page
    r"(\d{1,4})"                                # page number
    r"\s*$"                                      # end of line
)

# SECONDARY: Dotted format: "1.1 Purpose . . . . . 5"
_TOC_DOTS_RE = re.compile(
    r"^\s*"
    r"(\d+(?:\.\d+)*)"
    r"[\s.:]+\s*"
    r"([A-Za-z][^\n]{2,80}?)"
    r"[\s.·…_\-]{2,}"
    r"(\d{1,4})"
    r"\s*$"
)

# TERTIARY: Letter-prefixed: "A Appendix ... 100"
_TOC_LETTER_RE = re.compile(
    r"^\s*"
    r"([A-Z])\s+"
    r"([A-Z][^\n]{2,60}?)"
    r"[\s.·…_\-]*"
    r"\s+(\d{1,4})"
    r"\s*$"
)

# "Table of Contents" heading
_TOC_HEADING_RE = re.compile(
    r"^\s*(?:#{0,3}\s*)?(?:table\s+of\s+contents|contents|"
    r"inhaltsverzeichnis|inhalt|toc)\s*$",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# AutoSAR header/footer patterns (for stripping from content)
# ═══════════════════════════════════════════════════════════════════════════

_AUTOSAR_HEADER_FOOTER_PATTERNS = [
    # "## AUTOSAR" or "AUTOSAR" at start of page
    re.compile(r"^\s*(?:#{1,3}\s+)?AUTOSAR\s*$", re.IGNORECASE),
    # "X of Y Document ID NNN: AUTOSAR_CP_..."
    re.compile(r"^\s*\d+\s+of\s+\d+\s+Document\s+ID\s+\d+", re.IGNORECASE),
    # "Document ID NNN: AUTOSAR_..."
    re.compile(r"^\s*Document\s+ID\s+\d+\s*:", re.IGNORECASE),
    # "AUTOSAR CP R25-11" or "AUTOSAR CP Rxx-xx" (running header)
    re.compile(r"^\s*AUTOSAR\s+(?:CP|AP)\s+R\d{2,4}-\d{2}", re.IGNORECASE),
    # Repeated document title as header (e.g. "Utilization of Crypto Services")
    # We detect this dynamically, not with a static pattern
    # Page number patterns
    re.compile(r"^\s*(?:page|seite)\s+\d+\s*(?:of|von)\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    # Copyright
    re.compile(r"^\s*©\s+\d{4}", re.IGNORECASE),
    # Confidential/Draft
    re.compile(r"^\s*confidential\s*$", re.IGNORECASE),
    re.compile(r"^\s*draft\s*$", re.IGNORECASE),
]


def is_autosar_header_footer(line: str) -> bool:
    """Check if a line is an AutoSAR page header or footer."""
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _AUTOSAR_HEADER_FOOTER_PATTERNS)


def detect_repeated_title_header(
    page_markdowns: Dict[int, str],
    min_pages: int = 3,
) -> Optional[str]:
    """Detect if a document title is repeated as a running header.

    Checks if the same short line appears at the top of >= min_pages pages.
    Returns the repeated title string, or None.
    """
    from collections import Counter

    first_lines: List[str] = []
    for pn in sorted(page_markdowns.keys()):
        md = page_markdowns[pn]
        lines = [l.strip() for l in md.split("\n") if l.strip()]
        # Skip known AUTOSAR header to find the NEXT line
        for line in lines[:3]:
            if not is_autosar_header_footer(line) and 5 <= len(line) <= 80:
                first_lines.append(line)
                break

    if not first_lines:
        return None

    counter = Counter(first_lines)
    most_common, count = counter.most_common(1)[0]
    if count >= min_pages:
        return most_common
    return None


# ═══════════════════════════════════════════════════════════════════════════
# TOC line matching
# ═══════════════════════════════════════════════════════════════════════════

def _match_toc_line(line: str) -> Optional[Tuple[str, str, int]]:
    """Match a TOC line. Returns (number, title, page_num) or None."""
    line_stripped = line.rstrip()
    if not line_stripped or len(line_stripped.strip()) < 3:
        return None
    if _TOC_HEADING_RE.match(line_stripped):
        return None
    # Skip lines that are clearly headers/footers
    if is_autosar_header_footer(line_stripped):
        return None

    # Try no-dots (primary for GLM)
    m = _TOC_NODOTS_RE.match(line_stripped)
    if m:
        return (m.group(1), m.group(2).strip().rstrip(". "), int(m.group(3)))

    # Try dotted
    m = _TOC_DOTS_RE.match(line_stripped)
    if m:
        return (m.group(1), m.group(2).strip().rstrip(". "), int(m.group(3)))

    # Try letter-prefixed
    m = _TOC_LETTER_RE.match(line_stripped)
    if m:
        return (m.group(1), m.group(2).strip().rstrip(". "), int(m.group(3)))

    return None


def _count_toc_lines(text: str) -> int:
    return sum(1 for line in text.split("\n") if _match_toc_line(line) is not None)


# ═══════════════════════════════════════════════════════════════════════════
# TOC page detection
# ═══════════════════════════════════════════════════════════════════════════

def _is_toc_page(text: str, min_toc_lines: int = 3) -> bool:
    if not text or not text.strip():
        return False

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    # Check for TOC heading
    has_toc_heading = False
    for line in lines[:5]:
        if _TOC_HEADING_RE.match(line):
            has_toc_heading = True
            break

    toc_count = _count_toc_lines(text)

    # If TOC heading found, even a few TOC lines confirm it
    if has_toc_heading and toc_count >= 2:
        return True

    # Without heading, need more evidence
    if toc_count < min_toc_lines:
        return False

    # Non-empty non-header lines
    content_lines = [l for l in lines if not is_autosar_header_footer(l)]
    if not content_lines:
        return False

    return toc_count / len(content_lines) >= 0.3


def detect_toc_pages(
    page_markdowns: Dict[int, str],
    max_scan_pages: int = 30,
) -> List[int]:
    toc_pages: List[int] = []
    sorted_pages = sorted(page_markdowns.keys())

    for page_num in sorted_pages[:max_scan_pages]:
        md = page_markdowns.get(page_num, "")
        if _is_toc_page(md):
            toc_pages.append(page_num)
        elif toc_pages:
            break

    if toc_pages:
        logger.info("TOC detected on pages: %s", toc_pages)
    else:
        logger.info("No TOC detected in first %d pages", max_scan_pages)

    return toc_pages


# ═══════════════════════════════════════════════════════════════════════════
# TOC parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_toc_entries(
    page_markdowns: Dict[int, str],
    toc_pages: List[int],
) -> List[TOCEntry]:
    entries: List[TOCEntry] = []
    seen: set = set()

    for page_num in sorted(toc_pages):
        md = page_markdowns.get(page_num, "")
        for line in md.split("\n"):
            result = _match_toc_line(line)
            if result is None:
                continue
            number, title, target_page = result
            level = _number_to_level(number)
            key = f"{number}|{title}"
            if key not in seen:
                entries.append(TOCEntry(number, title, target_page, level))
                seen.add(key)

    # Sort by the section number parsed as a tuple of integers so that the
    # natural document order (1, 1.1, 1.2, 2, 2.1 …) is always preserved.
    # Sorting by (page_num, level) is incorrect because it can place a
    # level-1 section that happens to share a start page with level-2
    # sub-sections of a DIFFERENT earlier chapter before those sub-sections,
    # breaking the hierarchy builder downstream.
    def _sec_sort_key(e: "TOCEntry") -> tuple:
        if not e.number:
            return (float("inf"),)
        try:
            return tuple(int(p) for p in e.number.split("."))
        except ValueError:
            # Non-numeric prefix (e.g. "A", "B" for annexes) — sort after
            # numeric sections using the ordinal of the first character.
            return (float("inf"), ord(e.number[0]))

    entries.sort(key=_sec_sort_key)

    if entries:
        logger.info("Parsed %d TOC entries (pages %d-%d)",
                     len(entries), entries[0].page_num, entries[-1].page_num)
    return entries


def _number_to_level(number: str) -> int:
    if not number:
        return 1
    if number.isalpha():
        return 1
    return min(number.count(".") + 1, 6)


# ═══════════════════════════════════════════════════════════════════════════
# Pre-TOC page title extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_pre_toc_title(text: str) -> str:
    if not text or not text.strip():
        return "blank_page"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return "blank_page"
    # Skip header/footer lines
    content_lines = [l for l in lines if not is_autosar_header_footer(l)]
    if not content_lines:
        return "page_content"
    for line in content_lines[:5]:
        hm = re.match(r"^#{1,3}\s+(.+)$", line)
        if hm:
            return _sanitize_title(hm.group(1))
    for line in content_lines[:5]:
        if 3 <= len(line) <= 60 and line[0].isupper():
            return _sanitize_title(line)
    return _sanitize_title(content_lines[0][:40])


def _sanitize_title(title: str) -> str:
    title = re.sub(r"^#+\s*", "", title).strip()
    title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", title)
    title = re.sub(r"\s+", "_", title)
    title = re.sub(r"_+", "_", title)
    title = title.strip("_").lower()
    return title[:60] or "page_content"