# """Document assembler — TOC-driven section output with header/footer stripping.

# Key improvements:
# - Strips AutoSAR page headers/footers from section content
# - Strips detected repeated document title headers
# - TOC detection handles GLM no-dots format
# - Each section file has ONLY its own content (no duplication)
# - Pre-TOC pages named by heading, TOC pages skipped
# """

# from __future__ import annotations

# import json
# import re
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# from glmocr.postprocess.cross_page_stitcher import (
#     CrossPageStitcher,
#     StitchedDocument,
#     PageBlock,
# )
# from glmocr.postprocess.section_builder import (
#     SectionBuilder,
#     SectionNode,
#     build_document_index,
# )
# from glmocr.postprocess.toc_parser import (
#     TOCEntry,
#     detect_toc_pages,
#     parse_toc_entries,
#     extract_pre_toc_title,
#     is_autosar_header_footer,
#     detect_repeated_title_header,
# )
# from glmocr.postprocess.text_normalizer import normalize_text
# from glmocr.utils.logging import get_logger

# logger = get_logger(__name__)


# class DocumentAssembler:
#     def __init__(self):
#         self._stitcher = CrossPageStitcher()
#         self._section_builder = SectionBuilder()
#         self._pages: List[Dict[str, Any]] = []
#         self._page_count = 0

#     def add_page(self, page_num: int, json_result: Any, markdown_result: str = "") -> None:
#         self._pages.append({
#             "page_num": page_num,
#             "json_result": json_result,
#             "markdown_result": markdown_result,
#         })
#         self._page_count = max(self._page_count, page_num)
#         self._stitcher.add_page(page_num, json_result, markdown_result)

#     @classmethod
#     def from_page_outputs(cls, json_dir: str, markdown_dir: str = "") -> "DocumentAssembler":
#         assembler = cls()
#         jdir = Path(json_dir)
#         if not jdir.exists():
#             raise FileNotFoundError(f"JSON directory not found: {json_dir}")
#         mdir = Path(markdown_dir) if markdown_dir else None
#         json_files = sorted(jdir.glob("page_*.json"))
#         if not json_files:
#             raise FileNotFoundError(f"No page_*.json files found in {json_dir}")
#         for jf in json_files:
#             m = re.search(r"page_(\d+)", jf.stem)
#             if not m:
#                 continue
#             page_num = int(m.group(1))
#             try:
#                 json_result = json.loads(jf.read_text(encoding="utf-8"))
#             except (json.JSONDecodeError, OSError) as e:
#                 logger.warning("Failed to read %s: %s", jf, e)
#                 continue
#             md_result = ""
#             if mdir:
#                 for fmt in (f"page_{page_num:04d}.md", f"page_{page_num:03d}.md"):
#                     md_file = mdir / fmt
#                     if md_file.exists():
#                         try:
#                             md_result = md_file.read_text(encoding="utf-8")
#                         except OSError:
#                             pass
#                         break
#             assembler.add_page(page_num, json_result, md_result)
#         logger.info("Loaded %d pages from %s", assembler._page_count, json_dir)
#         return assembler

#     def assemble(self, output_dir: str, source_file: str = "") -> Dict[str, Any]:
#         out = Path(output_dir)

#         # Step 1: Stitch
#         logger.info("Cross-page stitching %d pages...", self._page_count)
#         stitched = self._stitcher.stitch()

#         # Step 2: Normalize
#         logger.info("Normalizing %d blocks...", len(stitched.blocks))
#         for block in stitched.blocks:
#             if block.content:
#                 block.content = normalize_text(block.content)

#         # Step 3: Build page markdown lookup for TOC detection
#         page_markdowns: Dict[int, str] = {}
#         for p in self._pages:
#             pn = p["page_num"]
#             md = p.get("markdown_result", "")
#             if not md:
#                 page_blocks = [b for b in stitched.blocks if b.page_num == pn]
#                 md = "\n\n".join(b.content for b in page_blocks if b.content and b.content.strip())
#             page_markdowns[pn] = md

#         # Step 4: Detect headers/footers to strip from section content
#         repeated_title = detect_repeated_title_header(page_markdowns)
#         if repeated_title:
#             logger.info("Detected repeated title header: %r", repeated_title)

#         # Step 5: Strip headers/footers from all stitched blocks
#         self._strip_headers_footers(stitched, repeated_title)

#         # Step 6: Detect TOC
#         toc_pages = detect_toc_pages(page_markdowns)
#         toc_entries = parse_toc_entries(page_markdowns, toc_pages) if toc_pages else []

#         # Step 7: Build sections
#         sections_dir = out / "sections"
#         sections_json_dir = out / "sections_json"
#         sections_dir.mkdir(parents=True, exist_ok=True)
#         sections_json_dir.mkdir(parents=True, exist_ok=True)

#         if toc_entries:
#             logger.info("TOC-driven mode: %d entries, TOC pages=%s", len(toc_entries), toc_pages)
#             all_sections = self._build_toc_driven(stitched, toc_pages, toc_entries, page_markdowns)
#         else:
#             logger.info("Fallback: heading-based section building")
#             all_sections = self._build_fallback(stitched)

#         # Step 8: Write section files (own content only, no duplication)
#         for section in all_sections:
#             sid = section.section_id
#             md_content = section.to_markdown_own_only()
#             if md_content.strip():
#                 (sections_dir / f"{sid}.md").write_text(md_content, encoding="utf-8")
#             json_content = section.to_json_own_only()
#             (sections_json_dir / f"{sid}.json").write_text(
#                 json.dumps(json_content, ensure_ascii=False, indent=2),
#                 encoding="utf-8",
#             )

#         # Step 9: Document index
#         top_level = [s for s in all_sections if not getattr(s, '_is_child_of_collected', False)]
#         if not top_level:
#             top_level = all_sections
#         doc_index = build_document_index(top_level, self._page_count, source_file)
#         (out / "document_index.json").write_text(
#             json.dumps(doc_index, ensure_ascii=False, indent=2),
#             encoding="utf-8",
#         )

#         logger.info("Assembly complete: %d section files", len(all_sections))
#         return doc_index

#     # ═══════════════════════════════════════════════════════════════════════
#     # Header/footer stripping
#     # ═══════════════════════════════════════════════════════════════════════

#     def _strip_headers_footers(
#         self,
#         stitched: StitchedDocument,
#         repeated_title: Optional[str],
#     ) -> None:
#         """Remove header/footer content from stitched blocks in-place."""
#         blocks_to_remove = []

#         for i, block in enumerate(stitched.blocks):
#             if not block.content or not block.content.strip():
#                 continue

#             content = block.content.strip()

#             # Check if entire block is a header/footer
#             if is_autosar_header_footer(content):
#                 blocks_to_remove.append(i)
#                 continue

#             # Check if it matches the repeated document title
#             if repeated_title and content == repeated_title:
#                 blocks_to_remove.append(i)
#                 continue

#             # For multi-line blocks, strip header/footer LINES
#             lines = content.split("\n")
#             cleaned = []
#             for line in lines:
#                 if is_autosar_header_footer(line):
#                     continue
#                 if repeated_title and line.strip() == repeated_title:
#                     continue
#                 cleaned.append(line)

#             new_content = "\n".join(cleaned).strip()
#             if new_content != content:
#                 block.content = new_content

#         # Remove blocks that are entirely headers/footers (reverse to preserve indices)
#         for i in reversed(blocks_to_remove):
#             stitched.blocks.pop(i)

#     # ═══════════════════════════════════════════════════════════════════════
#     # TOC-driven
#     # ═══════════════════════════════════════════════════════════════════════

#     def _build_toc_driven(
#         self,
#         stitched: StitchedDocument,
#         toc_pages: List[int],
#         toc_entries: List[TOCEntry],
#         page_markdowns: Dict[int, str],
#     ) -> List[SectionNode]:
#         all_sections: List[SectionNode] = []
#         counter = [0]

#         def _next_id() -> str:
#             counter[0] += 1
#             return f"section_{counter[0]:03d}"

#         toc_set = set(toc_pages)
#         first_toc = min(toc_pages)
#         last_toc = max(toc_pages)
#         sorted_pages = sorted(page_markdowns.keys())

#         # Zone 1: Pre-TOC pages
#         for pn in sorted_pages:
#             if pn >= first_toc:
#                 break
#             page_blocks = [b for b in stitched.blocks if b.page_num == pn]
#             if not page_blocks:
#                 continue
#             page_text = "\n\n".join(b.content for b in page_blocks if b.content and b.content.strip())
#             if not page_text.strip():
#                 continue
#             title = extract_pre_toc_title(page_text)
#             node = SectionNode(
#                 section_id=_next_id(), title=title, level=0,
#                 node_type="pre_toc", page_start=pn, page_end=pn,
#             )
#             for b in page_blocks:
#                 node.add_block(b)
#             all_sections.append(node)

#         # Zone 2: TOC → skipped
#         logger.info("Skipping TOC pages: %s", sorted(toc_set))

#         # Zone 3: Post-TOC → sectioned by TOC entries
#         post_toc_blocks = [b for b in stitched.blocks if b.page_num > last_toc]

#         if not toc_entries or not post_toc_blocks:
#             if post_toc_blocks:
#                 node = SectionNode(
#                     section_id=_next_id(), title="Document Content",
#                     level=1, node_type="section",
#                 )
#                 for b in post_toc_blocks:
#                     node.add_block(b)
#                 all_sections.append(node)
#             return all_sections

#         toc_sections = self._create_toc_sections(toc_entries, post_toc_blocks, counter)
#         all_sections.extend(toc_sections)
#         return all_sections

#     def _create_toc_sections(
#             self,
#             toc_entries: List[TOCEntry],
#             blocks: List[PageBlock],
#             counter: List[int],
#         ) -> List[SectionNode]:
#         def _next_id() -> str:
#             counter[0] += 1
#             return f"section_{counter[0]:03d}"

#         max_page = max(b.page_num for b in blocks) if blocks else 0

#         # Create nodes with page ranges
#         nodes: List[Tuple[TOCEntry, SectionNode]] = []
#         for i, entry in enumerate(toc_entries):
#             if i + 1 < len(toc_entries):
#                 end_page = toc_entries[i + 1].page_num - 1
#                 if end_page < entry.page_num:
#                     end_page = entry.page_num
#             else:
#                 end_page = max_page

#             node = SectionNode(
#                 section_id=_next_id(),
#                 title=entry.full_title(),
#                 level=entry.level,
#                 node_type="section" if entry.level <= 1 else "subsection",
#                 page_start=entry.page_num,
#                 page_end=end_page,
#             )
#             nodes.append((entry, node))

#         # Assign blocks to MOST SPECIFIC (deepest level) matching section
#         for block in blocks:
#             best_node = None
#             best_level = -1
#             for entry, node in nodes:
#                 if entry.page_num <= block.page_num <= node.page_end:
#                     if entry.level > best_level:
#                         best_level = entry.level
#                         best_node = node
#             if best_node is None:
#                 if block.page_num < nodes[0][0].page_num:
#                     nodes[0][1].add_block(block)
#                 else:
#                     nodes[-1][1].add_block(block)
#             else:
#                 best_node.add_block(block)

#         # Build hierarchy
#         top_level: List[SectionNode] = []
#         stack: List[SectionNode] = []
#         for entry, node in nodes:
#             while stack and stack[-1].level >= entry.level:
#                 stack.pop()
#             if stack:
#                 stack[-1].add_child(node)
#                 node._is_child_of_collected = True
#             else:
#                 top_level.append(node)
#                 node._is_child_of_collected = False
#             stack.append(node)

#         # Collect all flat
#         result: List[SectionNode] = []
#         def _collect(ns: List[SectionNode]):
#             for n in ns:
#                 if not hasattr(n, '_is_child_of_collected'):
#                     n._is_child_of_collected = False
#                 result.append(n)
#                 if n.children:
#                     for c in n.children:
#                         c._is_child_of_collected = True
#                     _collect(n.children)
#         _collect(top_level)
#         return result

#     # ═══════════════════════════════════════════════════════════════════════
#     # Fallback
#     # ═══════════════════════════════════════════════════════════════════════

#     def _build_fallback(self, stitched: StitchedDocument) -> List[SectionNode]:
#         sections = self._section_builder.build(stitched)
#         result: List[SectionNode] = []
#         def _collect(ns: List[SectionNode]):
#             for n in ns:
#                 if not hasattr(n, '_is_child_of_collected'):
#                     n._is_child_of_collected = False
#                 result.append(n)
#                 if n.children:
#                     for c in n.children:
#                         c._is_child_of_collected = True
#                     _collect(n.children)
#         _collect(sections)
#         return result

#     def get_stitched_markdown(self) -> str:
#         stitched = self._stitcher.stitch()
#         for block in stitched.blocks:
#             if block.content:
#                 block.content = normalize_text(block.content)
#         return stitched.get_text()

#     def get_page_count(self) -> int:
#         return self._page_count



"""Document assembler — TOC-driven section output with header/footer stripping.

Key design:
- TOC-driven path uses PURE PAGE-LEVEL MARKDOWN CONCATENATION.
  CrossPageStitcher is NOT called for section building.
- Two heading-deduplication passes fix GLM OCR's double-heading artifact:
    Pass A (_deduplicate_page_headings): applied to each raw page markdown
          before any section assignment.  GLM emits every heading twice:
          once as a standalone heading block and once as the first line of
          the following paragraph block.  This pass merges both occurrences
          into a single heading + all associated content.
    Pass B (_strip_leading_heading_from_content): applied to each content
          slice before it is added to a SectionNode.  SectionNode.to_markdown()
          prepends "{hashes} {title}" from node.title; if the content slice
          already starts with the same heading, stripping it here keeps
          exactly one copy in the final output.
- Strips AutoSAR page headers/footers from raw page markdown text.
- Strips detected repeated document-title headers.
- Shared pages (multiple TOC entries on the same page) are split at the
  heading that matches each subsequent TOC entry title.
- Stitcher is kept only for the heading-based fallback path and for
  get_stitched_markdown().
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from glmocr.postprocess.cross_page_stitcher import (
    CrossPageStitcher,
    PageBlock,
    StitchedDocument,
)
from glmocr.postprocess.section_builder import (
    SectionBuilder,
    SectionNode,
    build_document_index,
)
from glmocr.postprocess.toc_parser import (
    TOCEntry,
    detect_repeated_title_header,
    detect_toc_pages,
    extract_pre_toc_title,
    is_autosar_header_footer,
    parse_toc_entries,
)
from glmocr.postprocess.text_normalizer import normalize_text
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentAssembler:
    def __init__(self):
        self._stitcher = CrossPageStitcher()
        self._section_builder = SectionBuilder()
        self._pages: List[Dict[str, Any]] = []
        self._page_count = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Page ingestion
    # ═══════════════════════════════════════════════════════════════════════

    def add_page(self, page_num: int, json_result: Any, markdown_result: str = "") -> None:
        self._pages.append({
            "page_num": page_num,
            "json_result": json_result,
            "markdown_result": markdown_result,
        })
        self._page_count = max(self._page_count, page_num)
        self._stitcher.add_page(page_num, json_result, markdown_result)

    @classmethod
    def from_page_outputs(cls, json_dir: str, markdown_dir: str = "") -> "DocumentAssembler":
        assembler = cls()
        jdir = Path(json_dir)
        if not jdir.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")
        mdir = Path(markdown_dir) if markdown_dir else None
        json_files = sorted(jdir.glob("page_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No page_*.json files found in {json_dir}")
        for jf in json_files:
            m = re.search(r"page_(\d+)", jf.stem)
            if not m:
                continue
            page_num = int(m.group(1))
            try:
                json_result = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", jf, e)
                continue
            md_result = ""
            if mdir:
                for fmt in (f"page_{page_num:04d}.md", f"page_{page_num:03d}.md"):
                    md_file = mdir / fmt
                    if md_file.exists():
                        try:
                            md_result = md_file.read_text(encoding="utf-8")
                        except OSError:
                            pass
                        break
            assembler.add_page(page_num, json_result, md_result)
        logger.info("Loaded %d pages from %s", assembler._page_count, json_dir)
        return assembler

    # ═══════════════════════════════════════════════════════════════════════
    # Main assembly entry point
    # ═══════════════════════════════════════════════════════════════════════

    def assemble(self, output_dir: str, source_file: str = "") -> Dict[str, Any]:
        out = Path(output_dir)

        # Step 1: Build page-level markdown lookup directly from raw OCR pages.
        # CrossPageStitcher.stitch() is NOT called here.
        page_markdowns: Dict[int, str] = {}
        for p in self._pages:
            pn = p["page_num"]
            md = p.get("markdown_result", "")
            page_markdowns[pn] = md if (md and md.strip()) else \
                self._markdown_from_json(p.get("json_result", {}))

        # Step 2: Pass A — deduplicate headings in every page markdown.
        # GLM OCR emits each heading as both a standalone heading block and
        # as the opening line of the following paragraph block.  Collapse
        # all occurrences of the same heading into one before doing anything else.
        page_markdowns = {
            pn: self._deduplicate_page_headings(md)
            for pn, md in page_markdowns.items()
        }

        # Step 3: Detect repeated title header (runs on deduplicated text).
        repeated_title = detect_repeated_title_header(page_markdowns)
        if repeated_title:
            logger.info("Detected repeated title header: %r", repeated_title)

        # Step 4: Detect TOC.
        toc_pages = detect_toc_pages(page_markdowns)
        toc_entries = parse_toc_entries(page_markdowns, toc_pages) if toc_pages else []

        # Step 5: Build section tree.
        sections_dir = out / "sections"
        sections_json_dir = out / "sections_json"
        sections_dir.mkdir(parents=True, exist_ok=True)
        sections_json_dir.mkdir(parents=True, exist_ok=True)

        if toc_entries:
            logger.info(
                "TOC-driven (page-level) mode: %d entries, TOC pages=%s",
                len(toc_entries), toc_pages,
            )
            all_sections = self._build_toc_driven(
                toc_pages, toc_entries, page_markdowns, repeated_title
            )
        else:
            logger.info("Fallback: heading-based section building (stitcher enabled)")
            stitched = self._stitcher.stitch()
            for block in stitched.blocks:
                if block.content:
                    block.content = normalize_text(block.content)
            all_sections = self._build_fallback(stitched)

        # Step 6: Renumber top-level post-TOC sections sequentially.
        seq = 0
        for section in all_sections:
            if getattr(section, "_is_child_of_collected", False):
                continue
            if section.section_id.startswith("section_"):
                seq += 1
                section.section_id = f"section_{seq:03d}"

        # Step 7: Write one file per top-level section.
        written = 0
        for section in all_sections:
            if getattr(section, "_is_child_of_collected", False):
                continue
            sid = section.section_id
            md_content = section.to_markdown()
            if md_content.strip():
                (sections_dir / f"{sid}.md").write_text(md_content, encoding="utf-8")
                written += 1
            json_content = section.to_json()
            (sections_json_dir / f"{sid}.json").write_text(
                json.dumps(json_content, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Step 8: Document index.
        top_level = [s for s in all_sections
                     if not getattr(s, "_is_child_of_collected", False)]
        if not top_level:
            top_level = all_sections
        doc_index = build_document_index(top_level, self._page_count, source_file)
        (out / "document_index.json").write_text(
            json.dumps(doc_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("Assembly complete: %d section files", written)
        return doc_index

    # ═══════════════════════════════════════════════════════════════════════
    # Pass A: Intra-page heading deduplication
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _deduplicate_page_headings(page_md: str) -> str:
        """Collapse duplicate headings that GLM OCR emits within a single page.

        GLM OCR generates each heading TWICE per page:
          1. As a standalone heading block:  "## 5.1 Job Concept"
          2. As the first line of the following paragraph block:
             "## 5.1 Job Concept\\n\\nRequests to the CSM..."

        The two variants may differ in heading level (e.g. "###" vs "##")
        when the standalone block uses the layout-detector level while the
        paragraph block uses the Markdown level in the PDF body.

        Algorithm:
        - Parse the page into (heading_line, body) segments.
        - Group by *normalised* heading text (ignoring "#" count, leading
          section number, case, trailing punctuation).
        - For each group: keep the LAST heading line variant (the
          paragraph-block one has the correct Markdown level) and
          concatenate all body chunks in document order.

        Also handles the "content sandwich" case:
          ## 4.3 heading   <- sidebar/bookmark variant  (no body)
          [stray continuation text from previous section]
          ## 4.3 heading   <- body variant  (has real 4.3 content)
        -> merged: one ## 4.3 heading with both chunks concatenated.
        """
        if not page_md.strip():
            return page_md

        heading_re = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
        matches = list(heading_re.finditer(page_md))
        if not matches:
            return page_md

        preamble = page_md[: matches[0].start()].rstrip()

        segments: List[Tuple[str, str]] = []
        for idx, m in enumerate(matches):
            h_text = m.group(0).rstrip()
            body_start = m.end()
            body_end = (matches[idx + 1].start()
                        if idx + 1 < len(matches) else len(page_md))
            body = page_md[body_start:body_end].strip()
            segments.append((h_text, body))

        def _norm_h(h: str) -> str:
            t = re.sub(r"^#+\s*", "", h.strip())
            t = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", t)
            return re.sub(r"\s+", " ", t).lower().strip(".,;: ")

        merged_order: List[str] = []
        merged_heading: Dict[str, str] = {}
        merged_bodies: Dict[str, List[str]] = {}

        for h_text, body in segments:
            n = _norm_h(h_text)
            if n not in merged_heading:
                merged_heading[n] = h_text
                merged_bodies[n] = []
                merged_order.append(n)
            else:
                # Keep the LAST variant — the paragraph-block heading has the
                # correct Markdown level (e.g. "##" not "###").
                merged_heading[n] = h_text
            if body:
                merged_bodies[n].append(body)

        parts: List[str] = []
        if preamble:
            parts.append(preamble)
        for n in merged_order:
            parts.append(merged_heading[n])
            combined = "\n\n".join(merged_bodies[n])
            if combined:
                parts.append(combined)

        return "\n\n".join(parts)

    # ═══════════════════════════════════════════════════════════════════════
    # Pass B: Strip leading heading that duplicates SectionNode title prefix
    # ═══════════════════════════════════════════════════════════════════════

    @classmethod
    def _strip_leading_heading_from_content(cls, content: str, section_title: str) -> str:
        """Remove the heading at the start of *content* if it matches *section_title*.

        SectionNode.to_markdown() always prepends "{hashes} {title}" from
        node.title.  After Pass A, each page has at most one copy of every
        heading.  But when a content slice starts exactly at the heading
        position (which is how _assign_shared_page cuts its slices), that
        single copy collides with the node.title prefix, producing a duplicate.
        Stripping the first line here resolves that collision.

        Only the FIRST line of the content is examined; headings deeper in
        the content (sub-section headings) are intentionally preserved.
        """
        stripped = content.lstrip()
        if not stripped:
            return content

        first_nl = stripped.find("\n")
        first_line = (stripped[:first_nl].strip()
                      if first_nl != -1 else stripped.strip())

        if (re.match(r"^#{1,6}\s+", first_line)
                and cls._heading_matches_entry(first_line, section_title)):
            remainder = stripped[first_nl:].lstrip() if first_nl != -1 else ""
            return remainder

        return content

    # ═══════════════════════════════════════════════════════════════════════
    # TOC-driven section building  (page-level concatenation)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_toc_driven(
        self,
        toc_pages: List[int],
        toc_entries: List[TOCEntry],
        page_markdowns: Dict[int, str],
        repeated_title: Optional[str],
    ) -> List[SectionNode]:
        all_sections: List[SectionNode] = []
        counter = [0]

        def _next_id() -> str:
            counter[0] += 1
            return f"section_{counter[0]:03d}"

        toc_set = set(toc_pages)
        first_toc = min(toc_pages)
        last_toc = max(toc_pages)
        sorted_pages = sorted(page_markdowns.keys())
        max_page = sorted_pages[-1] if sorted_pages else 0

        # Zone 1: Pre-TOC pages
        used_pre_toc_names: set = set()
        for pn in sorted_pages:
            if pn >= first_toc:
                break
            page_md = page_markdowns.get(pn, "")
            clean_md = self._strip_page_headers_footers(page_md, repeated_title)
            if not clean_md.strip():
                continue
            title = extract_pre_toc_title(clean_md)
            file_stem = title
            if file_stem in used_pre_toc_names:
                file_stem = f"{title}_p{pn:03d}"
            used_pre_toc_names.add(file_stem)
            node = SectionNode(
                section_id=file_stem, title=title, level=0,
                node_type="pre_toc", page_start=pn, page_end=pn,
            )
            # Pass B: strip leading heading before creating block
            content = self._strip_leading_heading_from_content(clean_md, title)
            node.add_block(PageBlock(
                block_type="text", content=content, page_num=pn, page_end=pn,
            ))
            all_sections.append(node)

        logger.info("Skipping TOC pages: %s", sorted(toc_set))

        # Zone 3: Post-TOC pages
        post_toc_pages = [pn for pn in sorted_pages if pn > last_toc]
        if not toc_entries or not post_toc_pages:
            if post_toc_pages:
                node = SectionNode(
                    section_id=_next_id(), title="Document Content",
                    level=1, node_type="section",
                    page_start=post_toc_pages[0], page_end=post_toc_pages[-1],
                )
                for pn in post_toc_pages:
                    md = page_markdowns.get(pn, "")
                    clean = self._strip_page_headers_footers(md, repeated_title)
                    if clean.strip():
                        node.add_block(PageBlock(
                            block_type="text", content=clean,
                            page_num=pn, page_end=pn,
                        ))
                all_sections.append(node)
            return all_sections

        # Compute page ranges
        entry_ranges: List[Tuple[int, int]] = []
        for i, entry in enumerate(toc_entries):
            start_page = entry.page_num
            if i + 1 < len(toc_entries):
                end_page = toc_entries[i + 1].page_num - 1
                if end_page < start_page:
                    end_page = start_page
            else:
                end_page = max_page
            entry_ranges.append((start_page, end_page))

        # Create SectionNodes
        nodes: List[Tuple[TOCEntry, SectionNode]] = []
        for i, entry in enumerate(toc_entries):
            start_page, end_page = entry_ranges[i]
            node = SectionNode(
                section_id=_next_id(),
                title=entry.full_title(),
                level=entry.level,
                node_type="section" if entry.level <= 1 else "subsection",
                page_start=start_page,
                page_end=end_page,
            )
            nodes.append((entry, node))

        # Assign page content
        for pn in post_toc_pages:
            page_md = page_markdowns.get(pn, "")
            clean_md = self._strip_page_headers_footers(page_md, repeated_title)
            if not clean_md.strip():
                continue

            starting_here = [
                (i, entry, node)
                for i, (entry, node) in enumerate(nodes)
                if toc_entries[i].page_num == pn
            ]

            if len(starting_here) > 1:
                self._assign_shared_page(
                    clean_md, pn, starting_here, nodes, toc_entries
                )
            elif len(starting_here) == 1:
                i_new, entry_new, node_new = starting_here[0]
                split_pos = self._find_heading_in_markdown(
                    clean_md, entry_new.full_title()
                )
                if split_pos > 0:
                    before_md = clean_md[:split_pos].strip()
                    if before_md:
                        preceding = self._find_covering_section(
                            pn, i_new, nodes, toc_entries
                        )
                        if preceding is not None:
                            preceding.add_block(PageBlock(
                                block_type="text", content=before_md,
                                page_num=pn, page_end=pn,
                            ))
                    after_raw = clean_md[split_pos:].strip()
                    if after_raw:
                        # Pass B
                        content = self._strip_leading_heading_from_content(
                            after_raw, entry_new.full_title()
                        )
                        node_new.add_block(PageBlock(
                            block_type="text", content=content,
                            page_num=pn, page_end=pn,
                        ))
                else:
                    # Heading not found — whole page to new section (Pass B)
                    content = self._strip_leading_heading_from_content(
                        clean_md, entry_new.full_title()
                    )
                    node_new.add_block(PageBlock(
                        block_type="text", content=content,
                        page_num=pn, page_end=pn,
                    ))
            else:
                covering = self._find_deepest_covering(pn, nodes, toc_entries)
                if covering is not None:
                    covering.add_block(PageBlock(
                        block_type="text", content=clean_md,
                        page_num=pn, page_end=pn,
                    ))

        return self._build_hierarchy(nodes)

    # ═══════════════════════════════════════════════════════════════════════
    # Shared-page content splitter
    # ═══════════════════════════════════════════════════════════════════════

    def _assign_shared_page(
        self,
        page_md: str,
        pn: int,
        starting_here: List[Tuple[int, "TOCEntry", SectionNode]],
        nodes: List[Tuple["TOCEntry", SectionNode]],
        toc_entries: List["TOCEntry"],
    ) -> None:
        """Split *page_md* at each section heading and assign slices.

        After Pass A each heading appears exactly once.  Locate each entry's
        heading, sort by position, and assign each slice.  Pass B is applied
        to every slice.
        """
        found: List[Tuple[int, int, "TOCEntry", SectionNode]] = []
        for i, entry, node in starting_here:
            pos = self._find_heading_in_markdown(page_md, entry.full_title())
            if pos >= 0:
                found.append((pos, i, entry, node))

        if not found:
            if starting_here:
                starting_here[0][2].add_block(PageBlock(
                    block_type="text", content=page_md,
                    page_num=pn, page_end=pn,
                ))
            return

        found.sort(key=lambda x: x[0])

        # Content before the first heading → prior covering section
        first_pos = found[0][0]
        if first_pos > 0:
            before_md = page_md[:first_pos].strip()
            if before_md:
                prior = self._find_covering_section(
                    pn, found[0][1], nodes, toc_entries
                )
                if prior is not None:
                    prior.add_block(PageBlock(
                        block_type="text", content=before_md,
                        page_num=pn, page_end=pn,
                    ))

        # Slice between consecutive heading positions; Pass B on each slice
        for k, (pos, i, entry, node) in enumerate(found):
            end_pos = found[k + 1][0] if k + 1 < len(found) else len(page_md)
            raw_slice = page_md[pos:end_pos].strip()
            if not raw_slice:
                continue
            content = self._strip_leading_heading_from_content(
                raw_slice, entry.full_title()
            )
            if content.strip():
                node.add_block(PageBlock(
                    block_type="text", content=content,
                    page_num=pn, page_end=pn,
                ))

    # ═══════════════════════════════════════════════════════════════════════
    # Heading matching helpers
    # ═══════════════════════════════════════════════════════════════════════

    _HASH_RE = re.compile(r"^#+\s*")
    _SEC_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+")
    _WS_RE = re.compile(r"\s+")

    @classmethod
    def _normalize_heading(cls, text: str) -> str:
        t = cls._HASH_RE.sub("", text.strip(), count=1)
        t = cls._SEC_NUM_RE.sub("", t, count=1)
        return cls._WS_RE.sub(" ", t).lower().strip(".,;: ")

    @classmethod
    def _heading_matches_entry(cls, block_text: str, entry_title: str) -> bool:
        norm_block = cls._normalize_heading(block_text)
        norm_entry = cls._normalize_heading(entry_title)
        if not norm_block or not norm_entry:
            return False
        if norm_block == norm_entry:
            return True

        block_raw = re.sub(r"^#+\s*", "", block_text.strip())
        entry_raw = re.sub(r"^#+\s*", "", entry_title.strip())
        block_num_m = cls._SEC_NUM_RE.match(block_raw)
        entry_num_m = cls._SEC_NUM_RE.match(entry_raw)

        if block_num_m or entry_num_m:
            if not (block_num_m and entry_num_m):
                return False
            if block_num_m.group(1) != entry_num_m.group(1):
                return False
            return True  # same section number is a strong disambiguator

        min_len = min(len(norm_block), len(norm_entry))
        max_len = max(len(norm_block), len(norm_entry))
        if min_len >= 6 and max_len > 0 and min_len / max_len >= 0.75:
            if norm_block in norm_entry or norm_entry in norm_block:
                return True
        return False

    def _find_heading_in_markdown(self, page_md: str, entry_title: str) -> int:
        """Return the char offset of the line matching *entry_title*, or -1."""
        offset = 0
        for line in page_md.splitlines(keepends=True):
            stripped = line.strip()
            if stripped and re.match(r"^#{1,6}\s+", stripped):
                if self._heading_matches_entry(stripped, entry_title):
                    return offset
            offset += len(line)
        return -1

    # ═══════════════════════════════════════════════════════════════════════
    # Section-range helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _find_covering_section(
        pn: int,
        exclude_idx: int,
        nodes: List[Tuple["TOCEntry", SectionNode]],
        toc_entries: List["TOCEntry"],
    ) -> Optional[SectionNode]:
        best_node: Optional[SectionNode] = None
        best_level = -1
        for i, (entry, node) in enumerate(nodes):
            if i == exclude_idx:
                continue
            start = toc_entries[i].page_num
            end = (max(toc_entries[i + 1].page_num - 1, start)
                   if i + 1 < len(toc_entries) else pn)
            if start <= pn <= end and entry.level > best_level:
                best_level = entry.level
                best_node = node
        return best_node

    @staticmethod
    def _find_deepest_covering(
        pn: int,
        nodes: List[Tuple["TOCEntry", SectionNode]],
        toc_entries: List["TOCEntry"],
    ) -> Optional[SectionNode]:
        best_node: Optional[SectionNode] = None
        best_level = -1
        for i, (entry, node) in enumerate(nodes):
            start = toc_entries[i].page_num
            end = (max(toc_entries[i + 1].page_num - 1, start)
                   if i + 1 < len(toc_entries) else pn)
            if start <= pn <= end and entry.level > best_level:
                best_level = entry.level
                best_node = node
        return best_node

    # ═══════════════════════════════════════════════════════════════════════
    # Hierarchy builder
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_hierarchy(
        nodes: List[Tuple["TOCEntry", SectionNode]],
    ) -> List[SectionNode]:
        top_level: List[SectionNode] = []
        stack: List[SectionNode] = []
        for entry, node in nodes:
            while stack and stack[-1].level >= entry.level:
                stack.pop()
            if stack:
                stack[-1].add_child(node)
                node._is_child_of_collected = True
            else:
                top_level.append(node)
                node._is_child_of_collected = False
            stack.append(node)

        result: List[SectionNode] = []

        def _collect(ns: List[SectionNode]) -> None:
            for n in ns:
                if not hasattr(n, "_is_child_of_collected"):
                    n._is_child_of_collected = False
                result.append(n)
                if n.children:
                    for c in n.children:
                        c._is_child_of_collected = True
                    _collect(n.children)

        _collect(top_level)
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Header / footer stripping  (text-level, not block-level)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _strip_page_headers_footers(
        page_md: str, repeated_title: Optional[str]
    ) -> str:
        if not page_md:
            return page_md
        cleaned = []
        for line in page_md.splitlines():
            if is_autosar_header_footer(line):
                continue
            if repeated_title and line.strip() == repeated_title:
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    # ═══════════════════════════════════════════════════════════════════════
    # JSON-to-markdown (fallback when no raw markdown stored)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _markdown_from_json(json_result: Any) -> str:
        if not json_result or not isinstance(json_result, dict):
            return ""
        blocks = json_result.get("blocks") or json_result.get("content") or []
        parts = []
        for b in blocks:
            if isinstance(b, dict):
                text = (b.get("markdown") or b.get("text")
                        or b.get("content") or "")
            elif isinstance(b, str):
                text = b
            else:
                continue
            if text and text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)

    # ═══════════════════════════════════════════════════════════════════════
    # Fallback: heading-based section building (uses stitcher)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_fallback(self, stitched: StitchedDocument) -> List[SectionNode]:
        sections = self._section_builder.build(stitched)
        result: List[SectionNode] = []

        def _collect(ns: List[SectionNode]) -> None:
            for n in ns:
                if not hasattr(n, "_is_child_of_collected"):
                    n._is_child_of_collected = False
                result.append(n)
                if n.children:
                    for c in n.children:
                        c._is_child_of_collected = True
                    _collect(n.children)

        _collect(sections)
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Public helpers
    # ═══════════════════════════════════════════════════════════════════════

    def get_stitched_markdown(self) -> str:
        """Full document as a single merged markdown string.
        Stitcher IS used here intentionally — cross-page merging is desirable
        for a human-readable export; it is not involved in section building.
        """
        stitched = self._stitcher.stitch()
        for block in stitched.blocks:
            if block.content:
                block.content = normalize_text(block.content)
        return stitched.get_text()

    def get_page_count(self) -> int:
        return self._page_count