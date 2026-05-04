"""
Shared data contracts for the RAG comparative-evaluation pipeline.

These Pydantic models are the single source of truth for what a retrieval
output, a normalized context, and a generated answer look like. Both
generate.py and evaluate.py import from this file so the two scripts can
never disagree about field names or types.

Stage A (retrieval.py — owned by the user) writes two JSONL files:
  - pageindex_retrieval.jsonl  -> each line matches PageIndexRow
  - bgem3_retrieval.jsonl      -> each line matches BGEM3Row

Stage B (generate.py) reads both files, normalizes them to Context, runs
the frozen generator, and writes results.jsonl (each line matches ResultRow).

Stage C (evaluate.py) reads results.jsonl only.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Retrieval-side rows (inputs to generate.py)                                 #
# --------------------------------------------------------------------------- #

class PageIndexRow(BaseModel):
    """
    One line of pageindex_retrieval.jsonl.

    PageIndex is a vectorless / reasoning-based retriever that returns one
    plain-text passage per query, so we store it as a single string.
    """
    query: str
    ground_truth: str
    retriever: Literal["pageindex"] = "pageindex"
    context_text: str
    source_id: str = "pageindex_tree"


class BGEM3Chunk(BaseModel):
    """One chunk returned by the BGE-M3 vector retriever."""
    text: str
    source_id: str
    score: Optional[float] = None


class BGEM3Row(BaseModel):
    """
    One line of bgem3_retrieval.jsonl.

    BGE-M3 returns a ranked list of text chunks (already converted from
    embeddings back to their source text by the user's retriever).
    """
    query: str
    ground_truth: str
    retriever: Literal["bge_m3"] = "bge_m3"
    context_chunks: List[BGEM3Chunk]


# --------------------------------------------------------------------------- #
# Normalized context used by the frozen generator                             #
# --------------------------------------------------------------------------- #

class ContextChunk(BaseModel):
    """Unified chunk representation used after normalization."""
    text: str
    source_id: str
    score: Optional[float] = None


class Context(BaseModel):
    """
    Normalized context container fed into the generator.

    Both PageIndex (single big chunk) and BGE-M3 (multiple ranked chunks)
    converge to this shape, so the generator sees exactly one data structure.
    """
    chunks: List[ContextChunk]
    retriever: Literal["pageindex", "bge_m3"]

    def as_text(self, sep: str = "\n\n---\n\n") -> str:
        """Flatten chunks into a single prompt-ready string."""
        return sep.join(chunk.text for chunk in self.chunks)

    def total_chars(self) -> int:
        return sum(len(chunk.text) for chunk in self.chunks)


# --------------------------------------------------------------------------- #
# Generation-side row (output of generate.py, input to evaluate.py)           #
# --------------------------------------------------------------------------- #

class ResultRow(BaseModel):
    """
    One line of results.jsonl.

    This is the only file evaluate.py reads. It contains everything the
    metric tiers need: the query, the ground truth, the retrieved context
    that was actually used, the generated answer, and provenance fields.
    """
    query: str
    ground_truth: str
    retriever: Literal["pageindex", "bge_m3"]
    context_chunks: List[ContextChunk]
    answer: str
    model_id: str
    latency_ms: float
    prompt_chars: int = Field(
        default=0,
        description="Character length of the full prompt (for auditability).",
    )