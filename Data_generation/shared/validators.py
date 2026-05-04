"""
Cheap rule-based validators that do not need an LLM.

These catch the specific defect classes we identified in the v0 dataset:
  - Noisy/informal queries (wat, whot, iz, lowercase-start)
  - TOC-only contexts (dotted leaders, page-number-ending lines)
  - Boilerplate-only contexts (AUTOSAR confidential, doc headers)
  - Synthesizer vs context-count mismatches
  - Duplicate queries
  - Echo-style references (answer is question reworded)
  - Ungrounded references (content words not in context)

Each function returns (passed: bool, reason: str|None).
"""

import re
from typing import Iterable


# ── Noise / typo detection ────────────────────────────────────────────────────

_NOISE_PATTERNS = [
    r"\bwat\b", r"\bwot\b", r"\bwhot\b", r"\bwhat\s+iz\b",
    r"\bwher\b", r"\bwer\b(?!e)", r"\biz\b",
    r"\bfigur\b(?!e)", r"\bdescriptin\b",
    r"\bstuff\b", r"\bthingy\b",
    r"\bwut\b", r"\bsumthing\b", r"\bprolly\b",
    r"\bWha\b(?!\s*t\b)",  # "Wha is"
    r"\bWha\s+t\b",         # "Wha t"
    r"\bWhaet\b",
]

_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)


def check_query_not_noisy(question: str) -> tuple[bool, str | None]:
    """Reject questions with informal/slang/typo patterns."""
    hit = _NOISE_RE.search(question)
    if hit:
        return False, f"noise_pattern:{hit.group()!r}"
    if question and question[0].islower():
        return False, "starts_with_lowercase"
    if re.search(r"[?!]{2,}", question):
        return False, "repeated_punctuation"
    return True, None


# ── Context quality ───────────────────────────────────────────────────────────

def _toc_score(context: str) -> int:
    """Rough score for TOC-ness. >= 3 → almost certainly a TOC chunk."""
    if not context.strip():
        return 0
    score = 0
    # Dotted leaders like ". . . . ." or "........"
    if re.search(r"(?:\. ){5,}", context) or re.search(r"\.{5,}", context):
        score += 2
    # Lines ending in a page number
    lines = [l for l in context.split("\n") if l.strip()]
    if lines:
        page_end = sum(1 for l in lines if re.search(r"\s+\d{1,4}\s*$", l.strip()))
        if page_end / len(lines) > 0.5:
            score += 2
    # Section numbering "3.3.6 Foo . . . 39"
    if re.search(r"\b\d+\.\d+(?:\.\d+)*\s+\S+", context) and "..." in context:
        score += 1
    return score


_BOILERPLATE_MARKERS = [
    "AUTOSAR confidential",
    "This document is published by AUTOSAR",
    "Copyright",
    "www.autosar.org",
]


def strip_boilerplate(context: str) -> str:
    """Remove known footer/header boilerplate LINES (preserving other content)."""
    keep = []
    for line in context.split("\n"):
        if any(m.lower() in line.lower() for m in _BOILERPLATE_MARKERS):
            continue
        # Standard doc-header pattern: "Document ID 123: AUTOSAR_FOO"
        if re.match(r"\s*Document ID\s+\d+\s*:", line):
            continue
        # "X of Y" page marker on its own
        if re.match(r"\s*\d+\s+of\s+\d+\s*$", line):
            continue
        keep.append(line)
    return "\n".join(keep)


def check_context_not_toc(context: str) -> tuple[bool, str | None]:
    """Reject contexts that are mostly TOC / index / page-list."""
    if _toc_score(context) >= 3:
        return False, "toc_only_context"
    return True, None


def check_context_has_substance(context: str, min_chars: int = 200) -> tuple[bool, str | None]:
    stripped = strip_boilerplate(context).strip()
    if len(stripped) < min_chars:
        return False, f"too_short_after_boilerplate:{len(stripped)}_chars"
    # At least one sentence-ending punctuation
    if not re.search(r"[.!?]", stripped):
        return False, "no_sentence_punctuation"
    return True, None


# ── Synthesizer / context-count consistency ───────────────────────────────────

def check_synth_context_count(synth_name: str, n_contexts: int) -> tuple[bool, str | None]:
    if synth_name.startswith("multi_hop") and n_contexts < 2:
        return False, f"multi_hop_but_{n_contexts}_contexts"
    if synth_name.startswith("single_hop") and n_contexts != 1:
        return False, f"single_hop_but_{n_contexts}_contexts"
    return True, None


# ── Reference quality ─────────────────────────────────────────────────────────

_VAGUE_REF_PATTERNS = [
    r"\bas described in section\b",
    r"\bsee section\b",
    r"\brefer to section\b",
    r"\bmentioned under section\b",
    r"\bdiscussed under section\b",
    r"\bdiscussed in section\b",
    r"\bstarting at page\b",
    r"\bcovered in section\b",
    r"\boutlined in section\b",
]

_VAGUE_RE = re.compile("|".join(_VAGUE_REF_PATTERNS), re.IGNORECASE)


def check_reference_not_vague(reference: str) -> tuple[bool, str | None]:
    hit = _VAGUE_RE.search(reference)
    if hit:
        return False, f"vague_reference:{hit.group()!r}"
    return True, None


# ── Echo detection ────────────────────────────────────────────────────────────

_STOPWORDS = set(
    "the a an of in for to and or is are was were be been being this that "
    "these those it its with on at by from as what how can could would "
    "should may might must have has had do does did about between into "
    "through during before after above below up down".split()
)


def _content_words(text: str) -> set[str]:
    return {
        w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }


def check_not_echo(question: str, reference: str, max_overlap: float = 0.7) -> tuple[bool, str | None]:
    """Echo: the reference is mostly question-words rearranged (trivial answer)."""
    q_words = _content_words(question)
    r_words = _content_words(reference)
    if not r_words or len(q_words) < 5:
        return True, None  # too short to judge
    overlap = len(q_words & r_words) / len(r_words)
    if overlap > max_overlap and len(reference) < 200:
        return False, f"echo_overlap={overlap:.2f}"
    return True, None


# ── Grounding overlap (reference vs contexts) ─────────────────────────────────

def check_grounding_overlap(
    reference: str,
    contexts: Iterable[str],
    min_overlap: float = 0.5,
) -> tuple[bool, str | None]:
    """Reject if less than min_overlap of reference content-words appear in contexts."""
    r_words = _content_words(reference)
    if not r_words:
        return True, None
    ctx_words: set[str] = set()
    for c in contexts:
        ctx_words |= _content_words(c)
    if not ctx_words:
        return False, "no_context_content"
    overlap = len(r_words & ctx_words) / len(r_words)
    if overlap < min_overlap:
        return False, f"grounding_overlap={overlap:.2f}"
    return True, None


# ── Duplicate detection helpers ───────────────────────────────────────────────

def normalize_question(question: str) -> str:
    q = question.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(
        r"\b(the|a|an|of|in|for|to|and|or|is|are|what|how|can|you|please|"
        r"could|provide|explain|describe|detail|elaborate|specifically)\b",
        " ",
        q,
    )
    q = re.sub(r"\s+", " ", q).strip()
    return q


# ── One-shot validator that runs all cheap checks ─────────────────────────────

def run_all_structural_checks(candidate: dict) -> list[str]:
    """
    Return a list of failure reasons (empty = passed).
    These are the CHEAP checks — no LLM needed.
    """
    reasons: list[str] = []
    q = candidate["user_input"]
    ref = candidate["reference"]
    ctxs = candidate.get("reference_contexts", []) or []
    synth = candidate.get("synthesizer_name", "")

    for check_fn, args in [
        (check_query_not_noisy,       (q,)),
        (check_reference_not_vague,   (ref,)),
        (check_not_echo,              (q, ref)),
        (check_synth_context_count,   (synth, len(ctxs))),
        (check_grounding_overlap,     (ref, ctxs)),
    ]:
        ok, reason = check_fn(*args)
        if not ok:
            reasons.append(reason)

    # Per-context checks: EVERY context must pass
    for i, c in enumerate(ctxs):
        ok, reason = check_context_not_toc(c)
        if not ok:
            reasons.append(f"ctx[{i}]:{reason}")
        ok, reason = check_context_has_substance(c)
        if not ok:
            reasons.append(f"ctx[{i}]:{reason}")

    return reasons
