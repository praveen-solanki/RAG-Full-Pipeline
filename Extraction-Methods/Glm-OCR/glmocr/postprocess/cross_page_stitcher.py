"""Cross-page stitching for multi-page documents.

Implements:
- Reading order correction using bounding boxes (#1, #8)
- Positional + frequency header/footer detection (#18)
- Quality-gated cross-page paragraph merging (#13)
- Word-frequency validated hyphenated merges (#19)
- Conservative table continuation
- Figure-caption association (#10)
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from glmocr.postprocess.text_normalizer import (
    is_sentence_incomplete,
    is_continuation_line,
    is_header_footer_line,
    is_positional_header_footer,
    is_low_quality_block,
)
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

class PageBlock:
    """A single content block from a page."""

    __slots__ = (
        "block_type", "content", "label", "native_label",
        "bbox", "page_num", "index", "page_end",
    )

    def __init__(
        self,
        block_type: str = "text",
        content: str = "",
        label: str = "text",
        native_label: str = "text",
        bbox: Optional[List[int]] = None,
        page_num: int = 0,
        index: int = 0,
        page_end: int = 0,
    ):
        self.block_type = block_type
        self.content = content
        self.label = label
        self.native_label = native_label
        self.bbox = bbox
        self.page_num = page_num
        self.index = index
        self.page_end = page_end or page_num

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_type": self.block_type,
            "content": self.content,
            "label": self.label,
            "native_label": self.native_label,
            "bbox": self.bbox,
            "page_num": self.page_num,
            "page_end": self.page_end,
            "index": self.index,
        }


class StitchedDocument:
    """Result of cross-page stitching."""

    def __init__(self):
        self.blocks: List[PageBlock] = []
        self.total_pages: int = 0

    def add_block(self, block: PageBlock) -> None:
        self.blocks.append(block)

    def get_text(self) -> str:
        parts = []
        for b in self.blocks:
            if b.content and b.content.strip():
                parts.append(b.content)
        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Page parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_page_blocks(json_result: Any, page_num: int) -> List[PageBlock]:
    """Convert a page's JSON result into PageBlock objects."""
    blocks: List[PageBlock] = []
    if isinstance(json_result, list):
        if not json_result:
            return blocks
        if isinstance(json_result[0], list):
            for page_regions in json_result:
                for item in page_regions:
                    blocks.append(_item_to_block(item, page_num))
        elif isinstance(json_result[0], dict):
            for item in json_result:
                blocks.append(_item_to_block(item, page_num))
    elif isinstance(json_result, dict):
        blocks.append(_item_to_block(json_result, page_num))
    return blocks


def _item_to_block(item: Dict[str, Any], page_num: int) -> PageBlock:
    label = item.get("label", "text")
    native_label = item.get("native_label", label)
    block_type = "text"
    if label == "table":
        block_type = "table"
    elif label == "image":
        block_type = "image"
    elif label == "formula":
        block_type = "formula"
    elif native_label in ("doc_title", "paragraph_title"):
        block_type = "heading"
    return PageBlock(
        block_type=block_type,
        content=item.get("content", ""),
        label=label,
        native_label=native_label,
        bbox=item.get("bbox_2d"),
        page_num=page_num,
        index=item.get("index", 0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Reading order correction (#1, #8)
# ═══════════════════════════════════════════════════════════════════════════

def _correct_reading_order(blocks: List[PageBlock]) -> List[PageBlock]:
    """Re-sort blocks by spatial position using bbox coordinates.

    Strategy:
    1. Detect columns by x-coordinate clustering
    2. Sort top-to-bottom within each column
    3. Read left column before right column

    Falls back to original index order if bboxes are absent.
    """
    if not blocks:
        return blocks

    # Separate blocks with and without bboxes
    with_bbox = [(i, b) for i, b in enumerate(blocks) if b.bbox and len(b.bbox) >= 4]
    without_bbox = [(i, b) for i, b in enumerate(blocks) if not b.bbox or len(b.bbox) < 4]

    if len(with_bbox) < 2:
        # Not enough bbox data to reorder
        return blocks

    # Detect columns by clustering x-center positions
    x_centers = [(b.bbox[0] + b.bbox[2]) / 2 for _, b in with_bbox]
    columns = _detect_columns(x_centers)

    if len(columns) <= 1:
        # Single column — sort by y-coordinate (top to bottom)
        with_bbox.sort(key=lambda ib: (ib[1].bbox[1], ib[1].bbox[0]))
    else:
        # Multi-column — sort by column (left to right), then y within column
        col_assignments = _assign_columns(x_centers, columns)
        with_bbox_cols = list(zip(with_bbox, col_assignments))
        with_bbox_cols.sort(key=lambda x: (x[1], x[0][1].bbox[1]))
        with_bbox = [wbc[0] for wbc in with_bbox_cols]

    # Reconstruct: bbox-sorted blocks first, then non-bbox blocks at end
    sorted_blocks = [b for _, b in with_bbox] + [b for _, b in without_bbox]
    return sorted_blocks


def _detect_columns(x_centers: List[float], gap_ratio: float = 0.15) -> List[float]:
    """Detect column boundaries from x-center positions.

    Uses a simple gap-based approach: if sorted x-centers have a gap
    larger than gap_ratio * page_width, that's a column boundary.
    """
    if not x_centers:
        return []

    sorted_x = sorted(set(round(x, -1) for x in x_centers))  # round to 10s
    if len(sorted_x) < 2:
        return [sorted_x[0]] if sorted_x else []

    page_width = max(x_centers) - min(x_centers)
    if page_width < 50:
        return [sorted_x[0]]

    threshold = page_width * gap_ratio
    columns = [sorted_x[0]]

    for i in range(1, len(sorted_x)):
        if sorted_x[i] - sorted_x[i - 1] > threshold:
            columns.append(sorted_x[i])

    return columns


def _assign_columns(x_centers: List[float], columns: List[float]) -> List[int]:
    """Assign each x-center to the nearest column."""
    assignments = []
    for x in x_centers:
        min_dist = float("inf")
        best_col = 0
        for ci, col_x in enumerate(columns):
            dist = abs(x - col_x)
            if dist < min_dist:
                min_dist = dist
                best_col = ci
        assignments.append(best_col)
    return assignments


# ═══════════════════════════════════════════════════════════════════════════
# Header / Footer detection (#18 — positional + frequency)
# ═══════════════════════════════════════════════════════════════════════════

def _detect_repeated_headers_footers(
    pages_blocks: List[List[PageBlock]],
    max_check: int = 3,
    min_pages: int = 3,
) -> Tuple[List[str], List[str]]:
    """Detect headers/footers using both text frequency and bbox position."""
    if len(pages_blocks) < min_pages:
        return [], []

    header_cands: List[str] = []
    footer_cands: List[str] = []

    for page_blocks in pages_blocks:
        text_blocks = [b for b in page_blocks if b.content and b.content.strip()]
        if not text_blocks:
            continue

        # Positional detection: top/bottom 8% by bbox (#18)
        for b in text_blocks:
            content = b.content.strip()
            if not content or len(content) > 200:
                continue
            if b.bbox and is_positional_header_footer(b.bbox):
                # Positional match — very likely header/footer
                if b.bbox[1] < 100:  # top region
                    header_cands.append(content)
                else:
                    footer_cands.append(content)
                continue

        # Fallback: first/last N blocks (text-based)
        for b in text_blocks[:max_check]:
            content = b.content.strip()
            if content and len(content) < 200:
                header_cands.append(content)
        for b in text_blocks[-max_check:]:
            content = b.content.strip()
            if content and len(content) < 200:
                footer_cands.append(content)

    header_counter = Counter(header_cands)
    footer_counter = Counter(footer_cands)

    header_patterns = [t for t, c in header_counter.items() if c >= min_pages]
    footer_patterns = [t for t, c in footer_counter.items() if c >= min_pages]

    # Also add regex-detected patterns
    seen = set(header_patterns + footer_patterns)
    for page_blocks in pages_blocks:
        for b in page_blocks:
            if b.content and is_header_footer_line(b.content):
                content = b.content.strip()
                if content not in seen:
                    footer_patterns.append(content)
                    seen.add(content)

    return header_patterns, footer_patterns


def _remove_headers_footers(
    pages_blocks: List[List[PageBlock]],
    header_patterns: List[str],
    footer_patterns: List[str],
) -> List[List[PageBlock]]:
    all_patterns = set(header_patterns + footer_patterns)
    if not all_patterns:
        return pages_blocks
    result = []
    for page_blocks in pages_blocks:
        filtered = [
            b for b in page_blocks
            if b.content.strip() not in all_patterns
            and not is_header_footer_line(b.content)
        ]
        result.append(filtered)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Figure-caption association (#10)
# ═══════════════════════════════════════════════════════════════════════════

_CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.?|Table|Abbildung|Tabelle)\s+\d+",
    re.IGNORECASE,
)


def _associate_figures_captions(blocks: List[PageBlock]) -> List[PageBlock]:
    """Associate figure_caption blocks with preceding image blocks.

    If a caption block immediately follows an image block, merge them
    into a single block with the image ref + caption text.
    Also detects caption-like text blocks that follow images.
    """
    if len(blocks) < 2:
        return blocks

    result: List[PageBlock] = []
    skip_next = False

    for i, block in enumerate(blocks):
        if skip_next:
            skip_next = False
            continue

        if i + 1 < len(blocks):
            next_block = blocks[i + 1]

            # Case 1: image followed by figure_caption label
            if block.block_type == "image" and next_block.native_label == "figure_caption":
                merged = deepcopy(block)
                merged.content = (block.content or "").rstrip() + "\n\n" + (next_block.content or "")
                merged.block_type = "figure"
                result.append(merged)
                skip_next = True
                continue

            # Case 2: image followed by text that looks like a caption
            if block.block_type == "image" and next_block.block_type == "text":
                if next_block.content and _CAPTION_RE.match(next_block.content.strip()):
                    merged = deepcopy(block)
                    merged.content = (block.content or "").rstrip() + "\n\n" + next_block.content
                    merged.block_type = "figure"
                    result.append(merged)
                    skip_next = True
                    continue

        result.append(block)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Cross-page paragraph merging with quality gating (#13, #19)
# ═══════════════════════════════════════════════════════════════════════════

def _should_merge_paragraphs(last: PageBlock, first: PageBlock) -> bool:
    if last.block_type != "text" or first.block_type != "text":
        return False
    if last.native_label in ("doc_title", "paragraph_title"):
        return False
    if first.native_label in ("doc_title", "paragraph_title"):
        return False

    lc = last.content.rstrip()
    fc = first.content.lstrip()
    if not lc or not fc:
        return False

    # Quality gate (#13): don't merge garbage with good content
    if is_low_quality_block(lc) or is_low_quality_block(fc):
        return False

    # Don't merge list items, formulas, tables
    if lc.lstrip().startswith("- ") or fc.startswith("- "):
        return False
    if re.match(r"^\d+[.)]\s", fc):
        return False
    if "$$" in lc or "$$" in fc:
        return False
    if "|" in lc and "|" in fc:
        return False

    if is_sentence_incomplete(lc) and is_continuation_line(fc):
        return True
    if lc.endswith("-") and fc and fc[0].islower():
        return True
    return False


def _merge_paragraph_blocks(last: PageBlock, first: PageBlock) -> PageBlock:
    """Merge paragraphs with word-frequency validation for hyphens (#19)."""
    merged = deepcopy(last)
    lc = last.content.rstrip()
    fc = first.content.lstrip()

    if lc.endswith("-"):
        # Hyphenated word break — try to validate with wordfreq
        word_before = lc.rstrip("-").split()[-1] if lc.rstrip("-").split() else ""
        word_after = fc.split()[0] if fc.split() else ""
        candidate = word_before + word_after

        validated = False
        try:
            from wordfreq import zipf_frequency
            score = zipf_frequency(candidate.lower(), "en")
            if score >= 2.5:
                validated = True
        except ImportError:
            # wordfreq not available — use simple heuristic
            # accept if both fragments are >= 2 chars
            if len(word_before) >= 2 and len(word_after) >= 2:
                validated = True

        if validated:
            merged.content = lc[:-1] + fc
        else:
            # Keep the hyphen, just join with space
            merged.content = lc + " " + fc
    else:
        merged.content = lc + " " + fc

    merged.page_end = first.page_num
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Cross-page table continuation
# ═══════════════════════════════════════════════════════════════════════════

def _is_table_content(content: str) -> bool:
    lines = content.strip().split("\n")
    pipe_lines = sum(1 for l in lines if "|" in l)
    return pipe_lines >= 2 and len(lines) > 0 and pipe_lines / len(lines) > 0.5


def _should_merge_tables(last: PageBlock, first: PageBlock) -> bool:
    if last.block_type != "table" and not _is_table_content(last.content):
        return False
    if first.block_type != "table" and not _is_table_content(first.content):
        return False
    lc = last.content.rstrip()
    fc = first.content.lstrip()
    if not lc or not fc:
        return False

    last_pipe = [l for l in lc.split("\n") if "|" in l]
    first_pipe = [l for l in fc.split("\n") if "|" in l]
    if not last_pipe or not first_pipe:
        return False

    if abs(last_pipe[-1].count("|") - first_pipe[0].count("|")) > 1:
        return False
    if re.match(r"^[\s|:-]+$", first_pipe[0].strip()):
        return False
    return True


def _merge_table_blocks(last: PageBlock, first: PageBlock) -> PageBlock:
    merged = deepcopy(last)
    lc = last.content.rstrip()
    fc = first.content.lstrip()

    lines = fc.split("\n")
    skip = 0
    if len(lines) >= 2 and re.match(r"^[\s|:-]+$", lines[1].strip()):
        # Check if first line matches last table's header (duplicate header)
        last_pipe = [l for l in lc.split("\n") if "|" in l]
        if last_pipe:
            last_header = last_pipe[0].strip()
            first_header = lines[0].strip()
            if last_header == first_header:
                skip = 2
            else:
                skip = 0
    elif lines and re.match(r"^[\s|:-]+$", lines[0].strip()):
        skip = 1

    remaining = "\n".join(lines[skip:])
    merged.content = lc + "\n" + remaining
    merged.page_end = first.page_num
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# List item helpers
# ═══════════════════════════════════════════════════════════════════════════

def _is_list_item(content: str) -> bool:
    stripped = content.lstrip()
    if stripped.startswith("- "):
        return True
    if re.match(r"^\d+[.)]\s", stripped):
        return True
    if re.match(r"^[a-zA-Z][.)]\s", stripped):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Main stitcher
# ═══════════════════════════════════════════════════════════════════════════

class CrossPageStitcher:
    """Stitches page-level OCR results into a continuous document.

    Processing flow:
    1. Correct reading order per page (#1, #8)
    2. Detect and remove repeated headers/footers (#18)
    3. Associate figures with captions (#10)
    4. Merge paragraphs across page boundaries (#13, #19)
    5. Merge tables across page boundaries
    """

    def __init__(self):
        self._pages_blocks: List[List[PageBlock]] = []
        self._pages_markdown: List[str] = []
        self._page_count = 0

    def add_page(self, page_num: int, json_result: Any, markdown_result: str = "") -> None:
        blocks = _parse_page_blocks(json_result, page_num)
        self._pages_blocks.append(blocks)
        self._pages_markdown.append(markdown_result or "")
        self._page_count = max(self._page_count, page_num)

    def stitch(self) -> StitchedDocument:
        doc = StitchedDocument()
        doc.total_pages = self._page_count

        if not self._pages_blocks:
            return doc

        # Step 1: Correct reading order per page (#1, #8)
        reordered_pages = [_correct_reading_order(pb) for pb in self._pages_blocks]

        # Step 2: Detect and remove headers/footers (#18)
        hp, fp = _detect_repeated_headers_footers(reordered_pages)
        if hp or fp:
            logger.debug("Detected %d header, %d footer patterns", len(hp), len(fp))
        cleaned_pages = _remove_headers_footers(reordered_pages, hp, fp)

        # Step 3: Figure-caption association per page (#10)
        cleaned_pages = [_associate_figures_captions(pb) for pb in cleaned_pages]

        # Step 4 & 5: Cross-page stitching
        pending: Optional[PageBlock] = None

        for page_idx, page_blocks in enumerate(cleaned_pages):
            if not page_blocks:
                continue

            for block_idx, block in enumerate(page_blocks):
                if pending is not None:
                    if _should_merge_paragraphs(pending, block):
                        pending = _merge_paragraph_blocks(pending, block)
                        continue
                    elif _should_merge_tables(pending, block):
                        pending = _merge_table_blocks(pending, block)
                        continue
                    else:
                        doc.add_block(pending)
                        pending = None

                is_last = block_idx == len(page_blocks) - 1
                is_not_final_page = page_idx < len(cleaned_pages) - 1
                if is_last and is_not_final_page:
                    pending = deepcopy(block)
                else:
                    doc.add_block(block)

        if pending is not None:
            doc.add_block(pending)

        return doc
