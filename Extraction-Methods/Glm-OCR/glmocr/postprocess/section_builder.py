# """Section builder — hierarchical document structure.

# Implements:
# - Heading detection with layout-label priority (#4)
# - Bbox metadata in section JSON (#15)
# - Cross-reference indexing per section (#11)
# - Structured table parsing in JSON (#12)
# - Content summaries in document index (#16)
# - List block grouping (#6)
# """

# from __future__ import annotations

# import re
# from typing import Any, Dict, List, Optional, Tuple

# from glmocr.postprocess.cross_page_stitcher import PageBlock, StitchedDocument
# from glmocr.postprocess.text_normalizer import (
#     extract_cross_references,
#     parse_markdown_table,
# )
# from glmocr.utils.logging import get_logger

# logger = get_logger(__name__)


# # ═══════════════════════════════════════════════════════════════════════════
# # Heading detection (#4 — layout label takes priority)
# # ═══════════════════════════════════════════════════════════════════════════

# _MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# _NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})$")
# _CHAPTER_RE = re.compile(
#     r"^(?:chapter|section|annex|appendix|teil|kapitel)\s+"
#     r"(\d+|[A-Z])\s*[:.]\s*(.+)$",
#     re.IGNORECASE,
# )
# _ALLCAPS_HEADING_RE = re.compile(r"^([A-Z][A-Z\s\d.&/()-]{2,59})$")


# def detect_heading(content: str, native_label: str = "") -> Optional[Tuple[int, str]]:
#     """Detect if content is a heading and return (level, title).

#     Priority (#4):
#     0. Layout detector labels (doc_title, paragraph_title) — HIGHEST
#     1. Markdown heading markers
#     2. AutoSAR numbered headings
#     3. Chapter/Section patterns
#     4. All-caps short lines — LOWEST
#     """
#     if not content or not content.strip():
#         return None

#     stripped = content.strip()

#     # Priority 0: Trust the layout detector (#4)
#     if native_label == "doc_title":
#         # Remove any existing # markers to get clean title
#         title = re.sub(r"^#+\s*", "", stripped)
#         return (1, title)
#     if native_label == "paragraph_title":
#         title = re.sub(r"^#+\s*", "", stripped)
#         # Try to infer level from numbered prefix
#         num_m = re.match(r"^(\d+(?:\.\d+)*)\s+", title)
#         if num_m:
#             depth = num_m.group(1).count(".") + 1
#             return (min(depth, 6), title)
#         return (2, title)

#     if len(stripped) > 200:
#         return None
#     if "|" in stripped and stripped.count("|") >= 2:
#         return None
#     if stripped.startswith("$$") or stripped.startswith("```"):
#         return None

#     # Priority 1: Markdown markers
#     md_match = _MD_HEADING_RE.match(stripped)
#     if md_match:
#         return (len(md_match.group(1)), md_match.group(2).strip())

#     # Priority 2: Numbered headings
#     num_match = _NUMBERED_HEADING_RE.match(stripped)
#     if num_match:
#         number = num_match.group(1)
#         title = num_match.group(2).strip()
#         depth = number.count(".") + 1
#         return (min(depth, 6), f"{number} {title}")

#     # Priority 3: Chapter/Section
#     chap_match = _CHAPTER_RE.match(stripped)
#     if chap_match:
#         return (1, stripped)

#     # Priority 4: All-caps (very conservative)
#     if (3 <= len(stripped) <= 60
#         and stripped == stripped.upper()
#         and re.search(r"[A-Z]", stripped)
#         and "\n" not in stripped
#         and not stripped.startswith("-")
#         and not stripped.startswith("*")
#         and len(stripped.split()) >= 2
#     ):
#         if _ALLCAPS_HEADING_RE.match(stripped):
#             return (2, stripped.title())

#     return None


# # ═══════════════════════════════════════════════════════════════════════════
# # Block type classification
# # ═══════════════════════════════════════════════════════════════════════════

# def _determine_block_type(block: PageBlock) -> str:
#     content = block.content.strip() if block.content else ""
#     if not content:
#         return "empty"

#     # Trust existing classifications
#     if block.block_type == "heading":
#         return "heading"
#     if block.block_type == "figure":
#         return "figure"
#     if block.block_type in ("table",) or block.label == "table":
#         return "table"
#     if block.block_type in ("image",) or block.label == "image":
#         return "image"
#     if block.block_type in ("formula",) or block.label == "formula":
#         return "formula"

#     # Detect heading from content (with layout label priority)
#     if detect_heading(content, block.native_label) is not None:
#         return "heading"

#     # List detection (#6)
#     lines = content.split("\n")
#     list_lines = sum(1 for l in lines if l.lstrip().startswith("- ") or re.match(r"^\s*\d+[.)]\s", l))
#     if list_lines > 0 and list_lines / len(lines) >= 0.5:
#         return "list"

#     # Table
#     pipe_lines = sum(1 for l in lines if "|" in l)
#     if pipe_lines >= 2 and pipe_lines / len(lines) > 0.5:
#         return "table"

#     if content.startswith("$$") and content.endswith("$$"):
#         return "formula"
#     if content.startswith("!["):
#         return "image"

#     # Caption
#     if block.native_label == "figure_caption":
#         return "caption"

#     # Footnote / bibliography
#     if re.match(r"^\[\d+\]", content):
#         return "bibliography" if len(content) > 100 else "footnote"

#     return "paragraph"


# # ═══════════════════════════════════════════════════════════════════════════
# # Section data structures (#15, #11, #12)
# # ═══════════════════════════════════════════════════════════════════════════

# class SectionNode:
#     """A node in the document section tree."""

#     def __init__(
#         self,
#         section_id: str = "",
#         title: str = "",
#         level: int = 0,
#         node_type: str = "section",
#         page_start: int = 0,
#         page_end: int = 0,
#     ):
#         self.section_id = section_id
#         self.title = title
#         self.level = level
#         self.node_type = node_type
#         self.page_start = page_start
#         self.page_end = page_end
#         self.content: str = ""
#         self.children: List[SectionNode] = []
#         self._blocks: List[PageBlock] = []

#     def add_child(self, child: "SectionNode") -> None:
#         self.children.append(child)
#         if child.page_start > 0:
#             if self.page_start == 0:
#                 self.page_start = child.page_start
#             else:
#                 self.page_start = min(self.page_start, child.page_start)
#         if child.page_end > 0:
#             self.page_end = max(self.page_end, child.page_end)

#     def add_block(self, block: PageBlock) -> None:
#         self._blocks.append(block)
#         pn = block.page_num
#         pe = block.page_end or pn
#         if pn > 0:
#             self.page_start = min(self.page_start, pn) if self.page_start > 0 else pn
#         if pe > 0:
#             self.page_end = max(self.page_end, pe)

#     def get_all_text(self) -> str:
#         """Get all text content including children, for summaries/xrefs."""
#         parts = []
#         for b in self._blocks:
#             if b.content and b.content.strip():
#                 parts.append(b.content)
#         for child in self.children:
#             parts.append(child.get_all_text())
#         return "\n\n".join(parts)

#     def to_markdown(self) -> str:
#         """Full markdown including children (for index/debug)."""
#         parts = []
#         if self.title and self.level > 0:
#             prefix = "#" * min(self.level, 6)
#             parts.append(f"{prefix} {self.title}")
#         for block in self._blocks:
#             if block.content and block.content.strip():
#                 parts.append(block.content)
#         for child in self.children:
#             child_md = child.to_markdown()
#             if child_md.strip():
#                 parts.append(child_md)
#         return "\n\n".join(parts)

#     def to_markdown_own_only(self) -> str:
#         """Markdown with ONLY this section's own blocks — no children.
#         This is what gets written to individual section files to avoid duplication.
#         """
#         parts = []
#         if self.title and self.level > 0:
#             prefix = "#" * min(self.level, 6)
#             parts.append(f"{prefix} {self.title}")
#         for block in self._blocks:
#             if block.content and block.content.strip():
#                 parts.append(block.content)
#         return "\n\n".join(parts)

#     def to_json(self) -> Dict[str, Any]:
#         """Render as JSON with bbox (#15), table structure (#12), xrefs (#11)."""
#         result: Dict[str, Any] = {
#             "type": self.node_type,
#             "title": self.title,
#             "section_id": self.section_id,
#             "page_start": self.page_start,
#             "page_end": self.page_end,
#         }

#         if self._blocks:
#             content_items = []
#             for block in self._blocks:
#                 bt = _determine_block_type(block)
#                 item: Dict[str, Any] = {
#                     "type": bt,
#                     "content": block.content,
#                     "page": block.page_num,
#                 }

#                 # #15: Include bbox metadata
#                 if block.bbox:
#                     item["bbox"] = block.bbox

#                 # #12: Parse table structure
#                 if bt == "table" and block.content:
#                     table_data = parse_markdown_table(block.content)
#                     if table_data:
#                         item["table_structure"] = table_data

#                 content_items.append(item)
#             result["content"] = content_items

#         # #11: Cross-references for this section
#         all_text = self.get_all_text()
#         xrefs = extract_cross_references(all_text)
#         has_xrefs = any(v for v in xrefs.values())
#         if has_xrefs:
#             result["cross_references"] = xrefs

#         if self.children:
#             result["children"] = [child.to_json() for child in self.children]

#         return result

#     def to_json_own_only(self) -> Dict[str, Any]:
#         """JSON with ONLY this section's own blocks — children listed as refs only.
#         This is what gets written to individual section JSON files.
#         """
#         result: Dict[str, Any] = {
#             "type": self.node_type,
#             "title": self.title,
#             "section_id": self.section_id,
#             "page_start": self.page_start,
#             "page_end": self.page_end,
#         }

#         if self._blocks:
#             content_items = []
#             for block in self._blocks:
#                 bt = _determine_block_type(block)
#                 item: Dict[str, Any] = {
#                     "type": bt,
#                     "content": block.content,
#                     "page": block.page_num,
#                 }
#                 if block.bbox:
#                     item["bbox"] = block.bbox
#                 if bt == "table" and block.content:
#                     table_data = parse_markdown_table(block.content)
#                     if table_data:
#                         item["table_structure"] = table_data
#                 content_items.append(item)
#             result["content"] = content_items

#         # Own text only for xrefs
#         own_text = "\n\n".join(b.content for b in self._blocks if b.content and b.content.strip())
#         if own_text:
#             xrefs = extract_cross_references(own_text)
#             if any(v for v in xrefs.values()):
#                 result["cross_references"] = xrefs

#         if self.children:
#             result["child_sections"] = [
#                 {"section_id": c.section_id, "title": c.title}
#                 for c in self.children
#             ]

#         return result


# # ═══════════════════════════════════════════════════════════════════════════
# # Section builder
# # ═══════════════════════════════════════════════════════════════════════════

# class SectionBuilder:
#     def __init__(self):
#         self._section_counter = 0

#     def _next_id(self) -> str:
#         self._section_counter += 1
#         return f"section_{self._section_counter:03d}"

#     def build(self, document: StitchedDocument) -> List[SectionNode]:
#         self._section_counter = 0
#         if not document.blocks:
#             return []

#         # Classify all blocks, using layout labels for heading priority (#4)
#         classified: List[Tuple[PageBlock, str, Optional[Tuple[int, str]]]] = []
#         for block in document.blocks:
#             bt = _determine_block_type(block)
#             heading_info = None
#             if bt == "heading":
#                 heading_info = detect_heading(block.content, block.native_label)
#                 # If detect_heading returns None but block_type is heading (from layout),
#                 # treat as level-2 heading with content as title
#                 if heading_info is None and block.block_type == "heading":
#                     heading_info = (2, block.content.strip()[:80])
#             classified.append((block, bt, heading_info))

#         sections = self._build_sections(classified)

#         if not sections:
#             root = SectionNode(
#                 section_id=self._next_id(), title="Document",
#                 level=0, node_type="section",
#             )
#             for block, _, _ in classified:
#                 root.add_block(block)
#             sections = [root]

#         return sections

#     def _build_sections(
#         self,
#         classified: List[Tuple[PageBlock, str, Optional[Tuple[int, str]]]],
#     ) -> List[SectionNode]:
#         sections: List[SectionNode] = []
#         stack: List[SectionNode] = []

#         # Preamble: blocks before first heading
#         preamble_blocks: List[PageBlock] = []
#         first_heading_idx = None
#         for i, (block, bt, hi) in enumerate(classified):
#             if hi is not None:
#                 first_heading_idx = i
#                 break
#             preamble_blocks.append(block)

#         if preamble_blocks:
#             preamble = SectionNode(
#                 section_id=self._next_id(), title="Preamble",
#                 level=0, node_type="section",
#             )
#             for b in preamble_blocks:
#                 preamble.add_block(b)
#             sections.append(preamble)

#         if first_heading_idx is None:
#             return sections

#         for i in range(first_heading_idx, len(classified)):
#             block, bt, heading_info = classified[i]

#             if heading_info is not None:
#                 level, title = heading_info
#                 new_sec = SectionNode(
#                     section_id=self._next_id(),
#                     title=title, level=level,
#                     node_type="section" if level <= 1 else "subsection",
#                     page_start=block.page_num, page_end=block.page_num,
#                 )
#                 while stack and stack[-1].level >= level:
#                     stack.pop()
#                 if stack:
#                     stack[-1].add_child(new_sec)
#                 else:
#                     sections.append(new_sec)
#                 stack.append(new_sec)
#             else:
#                 if stack:
#                     stack[-1].add_block(block)
#                 elif sections:
#                     sections[-1].add_block(block)
#                 else:
#                     orphan = SectionNode(
#                         section_id=self._next_id(), title="Content",
#                         level=0, node_type="section",
#                     )
#                     orphan.add_block(block)
#                     sections.append(orphan)

#         return sections


# # ═══════════════════════════════════════════════════════════════════════════
# # Document index builder (#16 — content summaries)
# # ═══════════════════════════════════════════════════════════════════════════

# def _make_summary(node: SectionNode, max_len: int = 200) -> str:
#     """Generate a brief content preview for RAG retrieval (#16)."""
#     text = node.get_all_text().strip()
#     if not text:
#         return ""
#     # Take first sentence or max_len chars
#     period_idx = text.find(".")
#     if 0 < period_idx < max_len:
#         return text[:period_idx + 1]
#     if len(text) <= max_len:
#         return text
#     return text[:max_len].rsplit(" ", 1)[0] + "..."


# def build_document_index(
#     sections: List[SectionNode],
#     total_pages: int,
#     source_file: str = "",
# ) -> Dict[str, Any]:
#     """Build document_index.json with content summaries (#16) and xrefs (#11)."""
#     entries = []

#     def _walk(node: SectionNode, depth: int = 0) -> None:
#         # Gather cross-references for this section
#         all_text = node.get_all_text()
#         xrefs = extract_cross_references(all_text)
#         has_xrefs = any(v for v in xrefs.values())

#         entry: Dict[str, Any] = {
#             "section_id": node.section_id,
#             "title": node.title,
#             "level": node.level,
#             "type": node.node_type,
#             "page_start": node.page_start,
#             "page_end": node.page_end,
#             "depth": depth,
#             "num_children": len(node.children),
#             "summary": _make_summary(node),
#         }
#         if has_xrefs:
#             entry["cross_references"] = xrefs

#         entries.append(entry)
#         for child in node.children:
#             _walk(child, depth + 1)

#     for section in sections:
#         _walk(section)

#     return {
#         "source_file": source_file,
#         "total_pages": total_pages,
#         "total_sections": len(entries),
#         "sections": entries,
#     }




"""Section builder — hierarchical document structure.

Implements:
- Heading detection with layout-label priority (#4)
- Bbox metadata in section JSON (#15)
- Cross-reference indexing per section (#11)
- Structured table parsing in JSON (#12)
- Content summaries in document index (#16)
- List block grouping (#6)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from glmocr.postprocess.cross_page_stitcher import PageBlock, StitchedDocument
from glmocr.postprocess.text_normalizer import (
    extract_cross_references,
    parse_markdown_table,
)
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Heading detection (#4 — layout label takes priority)
# ═══════════════════════════════════════════════════════════════════════════

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})$")
_CHAPTER_RE = re.compile(
    r"^(?:chapter|section|annex|appendix|teil|kapitel)\s+"
    r"(\d+|[A-Z])\s*[:.]\s*(.+)$",
    re.IGNORECASE,
)
_ALLCAPS_HEADING_RE = re.compile(r"^([A-Z][A-Z\s\d.&/()-]{2,59})$")


def detect_heading(content: str, native_label: str = "") -> Optional[Tuple[int, str]]:
    """Detect if content is a heading and return (level, title).

    Priority (#4):
    0. Layout detector labels (doc_title, paragraph_title) — HIGHEST
    1. Markdown heading markers
    2. AutoSAR numbered headings
    3. Chapter/Section patterns
    4. All-caps short lines — LOWEST
    """
    if not content or not content.strip():
        return None

    stripped = content.strip()

    # Priority 0: Trust the layout detector (#4)
    if native_label == "doc_title":
        # Remove any existing # markers to get clean title
        title = re.sub(r"^#+\s*", "", stripped)
        return (1, title)
    if native_label == "paragraph_title":
        title = re.sub(r"^#+\s*", "", stripped)
        # Try to infer level from numbered prefix
        num_m = re.match(r"^(\d+(?:\.\d+)*)\s+", title)
        if num_m:
            depth = num_m.group(1).count(".") + 1
            return (min(depth, 6), title)
        return (2, title)

    if len(stripped) > 200:
        return None
    if "|" in stripped and stripped.count("|") >= 2:
        return None
    if stripped.startswith("$$") or stripped.startswith("```"):
        return None

    # Priority 1: Markdown markers
    md_match = _MD_HEADING_RE.match(stripped)
    if md_match:
        return (len(md_match.group(1)), md_match.group(2).strip())

    # Priority 2: Numbered headings
    num_match = _NUMBERED_HEADING_RE.match(stripped)
    if num_match:
        number = num_match.group(1)
        title = num_match.group(2).strip()
        depth = number.count(".") + 1
        return (min(depth, 6), f"{number} {title}")

    # Priority 3: Chapter/Section
    chap_match = _CHAPTER_RE.match(stripped)
    if chap_match:
        return (1, stripped)

    # Priority 4: All-caps (very conservative)
    if (3 <= len(stripped) <= 60
        and stripped == stripped.upper()
        and re.search(r"[A-Z]", stripped)
        and "\n" not in stripped
        and not stripped.startswith("-")
        and not stripped.startswith("*")
        and len(stripped.split()) >= 2
    ):
        if _ALLCAPS_HEADING_RE.match(stripped):
            return (2, stripped.title())

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Block type classification
# ═══════════════════════════════════════════════════════════════════════════

def _determine_block_type(block: PageBlock) -> str:
    content = block.content.strip() if block.content else ""
    if not content:
        return "empty"

    # Trust existing classifications
    if block.block_type == "heading":
        return "heading"
    if block.block_type == "figure":
        return "figure"
    if block.block_type in ("table",) or block.label == "table":
        return "table"
    if block.block_type in ("image",) or block.label == "image":
        return "image"
    if block.block_type in ("formula",) or block.label == "formula":
        return "formula"

    # Detect heading from content (with layout label priority)
    if detect_heading(content, block.native_label) is not None:
        return "heading"

    # List detection (#6)
    lines = content.split("\n")
    list_lines = sum(1 for l in lines if l.lstrip().startswith("- ") or re.match(r"^\s*\d+[.)]\s", l))
    if list_lines > 0 and list_lines / len(lines) >= 0.5:
        return "list"

    # Table
    pipe_lines = sum(1 for l in lines if "|" in l)
    if pipe_lines >= 2 and pipe_lines / len(lines) > 0.5:
        return "table"

    if content.startswith("$$") and content.endswith("$$"):
        return "formula"
    if content.startswith("!["):
        return "image"

    # Caption
    if block.native_label == "figure_caption":
        return "caption"

    # Footnote / bibliography
    if re.match(r"^\[\d+\]", content):
        return "bibliography" if len(content) > 100 else "footnote"

    return "paragraph"


# ═══════════════════════════════════════════════════════════════════════════
# Section data structures (#15, #11, #12)
# ═══════════════════════════════════════════════════════════════════════════

class SectionNode:
    """A node in the document section tree."""

    def __init__(
        self,
        section_id: str = "",
        title: str = "",
        level: int = 0,
        node_type: str = "section",
        page_start: int = 0,
        page_end: int = 0,
    ):
        self.section_id = section_id
        self.title = title
        self.level = level
        self.node_type = node_type
        self.page_start = page_start
        self.page_end = page_end
        self.content: str = ""
        self.children: List[SectionNode] = []
        self._blocks: List[PageBlock] = []

    def add_child(self, child: "SectionNode") -> None:
        self.children.append(child)
        if child.page_start > 0:
            if self.page_start == 0:
                self.page_start = child.page_start
            else:
                self.page_start = min(self.page_start, child.page_start)
        if child.page_end > 0:
            self.page_end = max(self.page_end, child.page_end)

    def add_block(self, block: PageBlock) -> None:
        self._blocks.append(block)
        pn = block.page_num
        pe = block.page_end or pn
        if pn > 0:
            self.page_start = min(self.page_start, pn) if self.page_start > 0 else pn
        if pe > 0:
            self.page_end = max(self.page_end, pe)

    def get_all_text(self) -> str:
        """Get all text content including children, for summaries/xrefs."""
        parts = []
        for b in self._blocks:
            if b.content and b.content.strip():
                parts.append(b.content)
        for child in self.children:
            parts.append(child.get_all_text())
        return "\n\n".join(parts)

    def to_markdown(self) -> str:
        """Full markdown including children — exact content from blocks,
        rendered section-wise instead of page-wise.

        Always emits the ``## Title`` heading prefix so that sections built
        via the fallback path (where the heading block is consumed into
        ``self.title`` but never stored in ``self._blocks``) are not silently
        dropped.  For TOC-driven sections where the heading block *is* also
        stored in ``_blocks``, ``_is_duplicate_heading_block`` prevents
        double-printing it.
        """
        parts = []
        if self.title and self.level > 0:
            prefix = "#" * min(self.level, 6)
            parts.append(f"{prefix} {self.title}")
        for block in self._blocks:
            if not block.content or not block.content.strip():
                continue
            # Skip blocks whose text merely repeats this section's heading
            # (prevents duplication in TOC-driven mode where the heading block
            # is stored in _blocks AND we just emitted "## title" above).
            if self._is_duplicate_heading_block(block.content.strip()):
                continue
            parts.append(block.content)
        for child in self.children:
            child_md = child.to_markdown()
            if child_md.strip():
                parts.append(child_md)
        return "\n\n".join(parts)

    # ── heading-dedup helpers ───────────────────────────────────────────
    @staticmethod
    def _norm_title(text: str) -> str:
        """Normalise a title/heading string for duplicate detection.

        Strips Markdown markers, leading section numbers, trailing
        punctuation, then lowercases and collapses whitespace.
        """
        t = re.sub(r"^#+\s*", "", text.strip())          # remove ## …
        t = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", t)       # remove "5.1.3 "
        t = re.sub(r"\s+", " ", t).lower().strip(".,;: ")
        return t

    def _is_duplicate_heading_block(self, block_content: str) -> bool:
        """Return True when *block_content* is just a restatement of self.title.

        Used to skip OCR heading blocks that duplicate the section title we
        already write via the '## title' prefix in to_markdown_own_only().
        """
        if not self.title:
            return False
        norm_block = self._norm_title(block_content)
        norm_title = self._norm_title(self.title)
        if not norm_block or not norm_title:
            return False
        return norm_block == norm_title
    # ───────────────────────────────────────────────────────────────────

    def to_markdown_own_only(self) -> str:
        """Markdown with ONLY this section's own blocks — no children.
        This is what gets written to individual section files to avoid duplication.
        """
        parts = []
        if self.title and self.level > 0:
            prefix = "#" * min(self.level, 6)
            parts.append(f"{prefix} {self.title}")
        for block in self._blocks:
            if not block.content or not block.content.strip():
                continue
            # Skip blocks whose text merely repeats this section's heading
            if self._is_duplicate_heading_block(block.content.strip()):
                continue
            parts.append(block.content)
        return "\n\n".join(parts)

    def to_json(self) -> Dict[str, Any]:
        """Render as JSON with bbox (#15), table structure (#12), xrefs (#11).
        Includes all children recursively — exact content from blocks,
        rendered section-wise instead of page-wise.
        """
        result: Dict[str, Any] = {
            "type": self.node_type,
            "title": self.title,
            "section_id": self.section_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

        if self._blocks:
            content_items = []
            for block in self._blocks:
                if not block.content or not block.content.strip():
                    continue
                bt = _determine_block_type(block)
                item: Dict[str, Any] = {
                    "type": bt,
                    "content": block.content,
                    "page": block.page_num,
                }
                if block.bbox:
                    item["bbox"] = block.bbox
                if bt == "table" and block.content:
                    table_data = parse_markdown_table(block.content)
                    if table_data:
                        item["table_structure"] = table_data
                content_items.append(item)
            if content_items:
                result["content"] = content_items

        # #11: Cross-references for this section
        all_text = self.get_all_text()
        xrefs = extract_cross_references(all_text)
        if any(v for v in xrefs.values()):
            result["cross_references"] = xrefs

        if self.children:
            result["children"] = [child.to_json() for child in self.children]

        return result

    def to_json_own_only(self) -> Dict[str, Any]:
        """JSON with ONLY this section's own blocks — children listed as refs only.
        This is what gets written to individual section JSON files.
        """
        result: Dict[str, Any] = {
            "type": self.node_type,
            "title": self.title,
            "section_id": self.section_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

        if self._blocks:
            content_items = []
            for block in self._blocks:
                if not block.content or not block.content.strip():
                    continue
                # Skip blocks that merely restate the section heading
                if self._is_duplicate_heading_block(block.content.strip()):
                    continue
                bt = _determine_block_type(block)
                item: Dict[str, Any] = {
                    "type": bt,
                    "content": block.content,
                    "page": block.page_num,
                }
                if block.bbox:
                    item["bbox"] = block.bbox
                if bt == "table" and block.content:
                    table_data = parse_markdown_table(block.content)
                    if table_data:
                        item["table_structure"] = table_data
                content_items.append(item)
            if content_items:
                result["content"] = content_items

        # Own text only for xrefs (excluding duplicate heading blocks)
        own_text = "\n\n".join(
            b.content for b in self._blocks
            if b.content and b.content.strip()
            and not self._is_duplicate_heading_block(b.content.strip())
        )
        if own_text:
            xrefs = extract_cross_references(own_text)
            if any(v for v in xrefs.values()):
                result["cross_references"] = xrefs

        if self.children:
            result["child_sections"] = [
                {"section_id": c.section_id, "title": c.title}
                for c in self.children
            ]

        return result


# ═══════════════════════════════════════════════════════════════════════════
# Section builder
# ═══════════════════════════════════════════════════════════════════════════

class SectionBuilder:
    def __init__(self):
        self._section_counter = 0

    def _next_id(self) -> str:
        self._section_counter += 1
        return f"section_{self._section_counter:03d}"

    def build(self, document: StitchedDocument) -> List[SectionNode]:
        self._section_counter = 0
        if not document.blocks:
            return []

        # Classify all blocks, using layout labels for heading priority (#4)
        classified: List[Tuple[PageBlock, str, Optional[Tuple[int, str]]]] = []
        for block in document.blocks:
            bt = _determine_block_type(block)
            heading_info = None
            if bt == "heading":
                heading_info = detect_heading(block.content, block.native_label)
                # If detect_heading returns None but block_type is heading (from layout),
                # treat as level-2 heading with content as title
                if heading_info is None and block.block_type == "heading":
                    heading_info = (2, block.content.strip()[:80])
            classified.append((block, bt, heading_info))

        sections = self._build_sections(classified)

        if not sections:
            root = SectionNode(
                section_id=self._next_id(), title="Document",
                level=0, node_type="section",
            )
            for block, _, _ in classified:
                root.add_block(block)
            sections = [root]

        return sections

    def _build_sections(
        self,
        classified: List[Tuple[PageBlock, str, Optional[Tuple[int, str]]]],
    ) -> List[SectionNode]:
        sections: List[SectionNode] = []
        stack: List[SectionNode] = []

        # Preamble: blocks before first heading
        preamble_blocks: List[PageBlock] = []
        first_heading_idx = None
        for i, (block, bt, hi) in enumerate(classified):
            if hi is not None:
                first_heading_idx = i
                break
            preamble_blocks.append(block)

        if preamble_blocks:
            preamble = SectionNode(
                section_id=self._next_id(), title="Preamble",
                level=0, node_type="section",
            )
            for b in preamble_blocks:
                preamble.add_block(b)
            sections.append(preamble)

        if first_heading_idx is None:
            return sections

        for i in range(first_heading_idx, len(classified)):
            block, bt, heading_info = classified[i]

            if heading_info is not None:
                level, title = heading_info
                new_sec = SectionNode(
                    section_id=self._next_id(),
                    title=title, level=level,
                    node_type="section" if level <= 1 else "subsection",
                    page_start=block.page_num, page_end=block.page_num,
                )
                while stack and stack[-1].level >= level:
                    stack.pop()
                if stack:
                    stack[-1].add_child(new_sec)
                else:
                    sections.append(new_sec)
                stack.append(new_sec)
            else:
                if stack:
                    stack[-1].add_block(block)
                elif sections:
                    sections[-1].add_block(block)
                else:
                    orphan = SectionNode(
                        section_id=self._next_id(), title="Content",
                        level=0, node_type="section",
                    )
                    orphan.add_block(block)
                    sections.append(orphan)

        return sections


# ═══════════════════════════════════════════════════════════════════════════
# Document index builder (#16 — content summaries)
# ═══════════════════════════════════════════════════════════════════════════

def _make_summary(node: SectionNode, max_len: int = 200) -> str:
    """Generate a brief content preview for RAG retrieval (#16)."""
    text = node.get_all_text().strip()
    if not text:
        return ""
    # Take first sentence or max_len chars
    period_idx = text.find(".")
    if 0 < period_idx < max_len:
        return text[:period_idx + 1]
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def build_document_index(
    sections: List[SectionNode],
    total_pages: int,
    source_file: str = "",
) -> Dict[str, Any]:
    """Build document_index.json with content summaries (#16) and xrefs (#11)."""
    entries = []

    def _walk(node: SectionNode, depth: int = 0) -> None:
        # Gather cross-references for this section
        all_text = node.get_all_text()
        xrefs = extract_cross_references(all_text)
        has_xrefs = any(v for v in xrefs.values())

        entry: Dict[str, Any] = {
            "section_id": node.section_id,
            "title": node.title,
            "level": node.level,
            "type": node.node_type,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "depth": depth,
            "num_children": len(node.children),
            "summary": _make_summary(node),
        }
        if has_xrefs:
            entry["cross_references"] = xrefs

        entries.append(entry)
        for child in node.children:
            _walk(child, depth + 1)

    for section in sections:
        _walk(section)

    return {
        "source_file": source_file,
        "total_pages": total_pages,
        "total_sections": len(entries),
        "sections": entries,
    }