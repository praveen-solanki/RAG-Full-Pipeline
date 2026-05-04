"""Post-processing module.

Provides:
- ResultFormatter: per-page OCR result formatting (existing, minimal cleaning)
- text_normalizer: deterministic text cleanup, quality scoring, xref extraction
- CrossPageStitcher: reading order correction, header/footer removal, merging
- SectionBuilder: hierarchical section construction with structured output
- DocumentAssembler: full pipeline orchestrator (single normalization pass)
"""

from .base_post_processor import BasePostProcessor
from .result_formatter import ResultFormatter
from .text_normalizer import (
    normalize_text,
    compute_block_quality,
    is_low_quality_block,
    extract_cross_references,
    parse_markdown_table,
)
from .cross_page_stitcher import CrossPageStitcher, StitchedDocument, PageBlock
from .section_builder import SectionBuilder, SectionNode, build_document_index
from .document_assembler import DocumentAssembler

__all__ = [
    "BasePostProcessor",
    "ResultFormatter",
    "normalize_text",
    "compute_block_quality",
    "is_low_quality_block",
    "extract_cross_references",
    "parse_markdown_table",
    "CrossPageStitcher",
    "StitchedDocument",
    "PageBlock",
    "SectionBuilder",
    "SectionNode",
    "build_document_index",
    "DocumentAssembler",
]
