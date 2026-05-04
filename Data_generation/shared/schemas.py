"""
Schemas for the gold-dataset pipeline.

One record flows through the stages with fields added at each stage:

    build_kg          -> (no record; produces knowledge_graph.json)
    generate_candidates -> candidate record (user_input, reference, reference_contexts, ...)
    validate_candidates -> + scores, judge_rationales
    finalize_dataset   -> split into gold/rejected/human_review

Everything is a plain dict so jsonl files are trivially readable.
"""

from typing import Any
from datetime import datetime, timezone


# ── Synthesizer names (we define these ourselves; not RAGAS's names) ──────────
SYNTH_SINGLE_HOP_SPECIFIC = "single_hop_specific"
SYNTH_SINGLE_HOP_ABSTRACT = "single_hop_abstract"
SYNTH_MULTI_HOP_SPECIFIC  = "multi_hop_specific"
SYNTH_MULTI_HOP_ABSTRACT  = "multi_hop_abstract"

ALL_SYNTHESIZERS = [
    SYNTH_SINGLE_HOP_SPECIFIC,
    SYNTH_SINGLE_HOP_ABSTRACT,
    SYNTH_MULTI_HOP_SPECIFIC,
    SYNTH_MULTI_HOP_ABSTRACT,
]

# Distribution: how candidates should be split across synthesizers.
# Tuned per Zoloev's production RAGAS pipeline and RAGalyst's single+multi mix.
DEFAULT_SYNTH_DISTRIBUTION = {
    SYNTH_SINGLE_HOP_SPECIFIC: 0.35,
    SYNTH_SINGLE_HOP_ABSTRACT: 0.15,
    SYNTH_MULTI_HOP_SPECIFIC:  0.25,
    SYNTH_MULTI_HOP_ABSTRACT:  0.25,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_candidate(
    candidate_id: str,
    user_input: str,
    reference: str,
    reference_contexts: list[str],
    synthesizer_name: str,
    persona_name: str,
    source_node_ids: list[str],
    source_documents: list[str],
    generator_model: str,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    """A candidate before any validation."""
    return {
        "candidate_id": candidate_id,
        "user_input": user_input,
        "reference": reference,
        "reference_contexts": reference_contexts,
        "synthesizer_name": synthesizer_name,
        "persona_name": persona_name,
        "source_node_ids": source_node_ids,
        "source_documents": source_documents,
        "generator_model": generator_model,
        "generator_config": generator_config,
        "generated_at": now_iso(),
    }


def attach_scores(
    candidate: dict[str, Any],
    scores: dict[str, Any],
    judge_rationales: dict[str, str],
    judge_model: str,
) -> dict[str, Any]:
    """Return a new dict with validation fields appended (does not mutate)."""
    scored = dict(candidate)
    scored["scores"] = scores
    scored["judge_rationales"] = judge_rationales
    scored["judge_model"] = judge_model
    scored["scored_at"] = now_iso()
    return scored
