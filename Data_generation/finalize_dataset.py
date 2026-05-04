"""
Stage C: Finalize the gold dataset.

No LLM calls here — this stage is pure data filtering. It:

  1. Reads scored.jsonl from Stage B.
  2. Applies threshold filters (configurable via CLI):
         - answerability == 1
         - question_specificity == 1
         - faithfulness >= --min-faithfulness
         - answer_relevance >= --min-answer-relevance
         - zero structural failures
  3. Deduplicates (normalized question text).
  4. Enforces a diversity check: no single source document >= --max-source-share
     of the gold set.
  5. Downsamples (stratified by synthesizer) to --target size.
  6. Splits into train/dev/test (60/20/20 by default, seeded).
  7. Writes:
         gold_v1.0.json           — all accepted samples
         gold_v1.0_train.json
         gold_v1.0_dev.json
         gold_v1.0_test.json
         rejected.json            — with rejection reason per sample
         human_review_queue.csv   — 10% stratified sample for SME review
         dataset_card.md          — the datasheet
         summary.json             — stats

Because this script is pure filtering, you can tune thresholds and re-run
without ever re-calling the LLM.

Usage:
    python finalize_dataset.py --output-dir ./output --target 500
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from shared.io_utils import atomic_write_json, load_jsonl
from shared.validators import normalize_question


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", type=int, default=1000,
                   help="Target final dataset size (will downsample to hit this)")
    p.add_argument("--version", default="1.0")
    # Thresholds
    p.add_argument("--min-faithfulness",    type=float, default=0.85)
    p.add_argument("--min-answer-relevance", type=float, default=0.80)
    p.add_argument("--max-source-share",    type=float, default=0.25,
                   help="Max fraction of gold samples from a single PDF")
    # Splits
    p.add_argument("--train-frac", type=float, default=0.60)
    p.add_argument("--dev-frac",   type=float, default=0.20)
    # test is the remainder
    p.add_argument("--human-review-frac", type=float, default=0.10,
                   help="Fraction of gold set to send to SME review queue")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def apply_thresholds(rec: dict, args: argparse.Namespace) -> list[str]:
    """Return a list of threshold failure reasons; [] = passed."""
    reasons: list[str] = []
    s = rec.get("scores", {})
    if s.get("answerability") != 1:
        reasons.append("answerability_0")
    if s.get("question_specificity") != 1:
        reasons.append("question_specificity_0")
    if (s.get("faithfulness") or 0) < args.min_faithfulness:
        reasons.append(f"faithfulness_below_{args.min_faithfulness}")
    if (s.get("answer_relevance") or 0) < args.min_answer_relevance:
        reasons.append(f"answer_relevance_below_{args.min_answer_relevance}")
    for r in s.get("structural_fail_reasons", []) or []:
        reasons.append(f"structural:{r}")
    return reasons


def deduplicate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the highest-scoring record for each normalized question."""
    by_norm: dict[str, dict] = {}
    discarded: list[dict] = []

    def score_quality(r: dict) -> tuple:
        # Higher = better
        s = r.get("scores", {})
        return (
            s.get("answerability", 0),
            s.get("faithfulness", 0),
            s.get("answer_relevance", 0),
            len(r.get("reference_contexts", []) or []),
        )

    for r in records:
        key = normalize_question(r["user_input"])
        if key in by_norm:
            current = by_norm[key]
            if score_quality(r) > score_quality(current):
                discarded.append({**current, "rejection_reason": "duplicate_dominated"})
                by_norm[key] = r
            else:
                discarded.append({**r, "rejection_reason": "duplicate_dominated"})
        else:
            by_norm[key] = r

    return list(by_norm.values()), discarded


def enforce_source_diversity(
    records: list[dict],
    max_share: float,
    target: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """
    If any source document contributes more than max_share of the records,
    randomly downsample it. We cap each source at floor(max_share * target).
    """
    cap = max(1, int(max_share * target))
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        docs = r.get("source_documents") or ["_unknown_"]
        # Use the primary source doc as the grouping key
        key = docs[0]
        by_source[key].append(r)

    kept: list[dict] = []
    capped: list[dict] = []
    for src, items in by_source.items():
        rng.shuffle(items)
        kept.extend(items[:cap])
        for x in items[cap:]:
            capped.append({**x, "rejection_reason": f"source_cap_{src}"})

    return kept, capped


def stratified_downsample(
    records: list[dict],
    target: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Downsample to `target` keeping the synthesizer distribution stable."""
    if len(records) <= target:
        return records, []

    by_synth: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_synth[r.get("synthesizer_name", "_unknown_")].append(r)

    # Compute proportional quota per synthesizer
    n_total = len(records)
    quotas: dict[str, int] = {}
    for synth, items in by_synth.items():
        quotas[synth] = max(1, int(round(target * len(items) / n_total)))
    # Fix rounding so the sum equals target
    diff = target - sum(quotas.values())
    if diff != 0:
        synth_sorted = sorted(quotas.keys(), key=lambda s: -len(by_synth[s]))
        i = 0
        while diff != 0:
            step = 1 if diff > 0 else -1
            quotas[synth_sorted[i % len(synth_sorted)]] += step
            diff -= step
            i += 1

    kept: list[dict] = []
    dropped: list[dict] = []
    for synth, items in by_synth.items():
        rng.shuffle(items)
        kept.extend(items[:quotas[synth]])
        for x in items[quotas[synth]:]:
            dropped.append({**x, "rejection_reason": "downsampled_to_target"})

    return kept, dropped


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATS
# ══════════════════════════════════════════════════════════════════════════════

def to_public_record(rec: dict) -> dict:
    """
    The 'public' shape of a gold record — scores and rationales included
    so downstream evaluators can filter further if they want.
    """
    return {
        "id":                    rec["candidate_id"],
        "user_input":            rec["user_input"],
        "reference":             rec["reference"],
        "reference_contexts":    rec["reference_contexts"],
        "synthesizer_name":      rec["synthesizer_name"],
        "persona":               rec.get("persona_name"),
        "source_documents":      rec.get("source_documents", []),
        "scores":                rec.get("scores", {}),
        "metadata": {
            "generator_model":   rec.get("generator_model"),
            "judge_model":       rec.get("judge_model"),
            "generated_at":      rec.get("generated_at"),
            "scored_at":         rec.get("scored_at"),
        },
    }


def write_human_review_csv(
    records: list[dict],
    path: Path,
    frac: float,
    rng: random.Random,
) -> int:
    """
    Stratified sample of the gold set for SME spot-check.
    Columns designed so a non-technical reviewer can annotate in a spreadsheet.
    """
    by_synth: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_synth[r["synthesizer_name"]].append(r)

    sample: list[dict] = []
    for synth, items in by_synth.items():
        n = max(1, int(round(frac * len(items))))
        rng.shuffle(items)
        sample.extend(items[:n])

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "synthesizer", "persona",
        "question", "reference_answer",
        "judge_answerability", "judge_faithfulness",
        "judge_answer_relevance", "judge_question_specificity",
        # SME fields — blank, to be filled in
        "sme_answerability_0_1", "sme_faithfulness_0_5_1",
        "sme_answer_correct_0_5_1", "sme_notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sample:
            s = r.get("scores", {})
            w.writerow({
                "id":                             r["candidate_id"],
                "synthesizer":                    r["synthesizer_name"],
                "persona":                        r.get("persona_name"),
                "question":                       r["user_input"],
                "reference_answer":               r["reference"],
                "judge_answerability":            s.get("answerability"),
                "judge_faithfulness":             s.get("faithfulness"),
                "judge_answer_relevance":         s.get("answer_relevance"),
                "judge_question_specificity":     s.get("question_specificity"),
                "sme_answerability_0_1":          "",
                "sme_faithfulness_0_5_1":         "",
                "sme_answer_correct_0_5_1":       "",
                "sme_notes":                      "",
            })
    return len(sample)


def write_dataset_card(
    stage_c_dir: Path,
    version: str,
    gold: list[dict],
    args: argparse.Namespace,
    rejected_count: int,
    source_distribution: Counter,
) -> None:
    synth_dist = Counter(r["synthesizer_name"] for r in gold)
    persona_dist = Counter(r.get("persona_name") for r in gold)

    faith = [r["scores"]["faithfulness"] for r in gold]
    relev = [r["scores"]["answer_relevance"] for r in gold]

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0

    lines = [
        f"# AUTOSAR RAG Eval Dataset v{version}",
        "",
        "## Motivation",
        "",
        "This dataset is a gold-standard RAG evaluation benchmark for",
        "AUTOSAR technical documentation. It was built with a three-stage",
        "pipeline (over-generate → validate → finalize) using open-weight",
        "models only, following the methodology from:",
        "- RAGalyst (Gao et al., Nov 2025)",
        "- Judge's Verdict (NVIDIA, Oct 2025)",
        "- TREC 2024 RAG Track nugget methodology",
        "",
        "## Composition",
        "",
        f"- Total samples: **{len(gold)}**",
        f"- Train / Dev / Test: **{int(args.train_frac * len(gold))} / "
        f"{int(args.dev_frac * len(gold))} / "
        f"{len(gold) - int(args.train_frac * len(gold)) - int(args.dev_frac * len(gold))}**",
        f"- Rejected during finalization: **{rejected_count}**",
        "",
        "### Synthesizer distribution",
        "",
    ]
    total = sum(synth_dist.values())
    for k, v in synth_dist.most_common():
        lines.append(f"- {k}: {v} ({100*v/total:.1f}%)")
    lines += [
        "",
        "### Persona distribution",
        "",
    ]
    for k, v in persona_dist.most_common():
        lines.append(f"- {k}: {v} ({100*v/total:.1f}%)")
    lines += [
        "",
        "### Source document distribution",
        "",
    ]
    for k, v in source_distribution.most_common():
        lines.append(f"- `{k}`: {v} ({100*v/total:.1f}%)")
    lines += [
        "",
        "### Score statistics",
        "",
        f"- Mean faithfulness: **{mean(faith):.3f}**",
        f"- Mean answer relevance: **{mean(relev):.3f}**",
        "",
        "## Collection process",
        "",
        "Candidates were generated from the source PDFs using",
        f"Qwen/Qwen2.5-72B-Instruct-AWQ with controlled personas and a",
        "split question/answer prompt (answer prompt returns NOT_ANSWERABLE",
        "when the context is insufficient).",
        "",
        "Each candidate was scored by Qwen/Qwen3-30B-A3B-Instruct-2507",
        "on four metrics: answerability, faithfulness, answer relevance,",
        "and question specificity. Thresholds applied in this dataset:",
        "",
        f"- answerability == 1 (binary)",
        f"- question_specificity == 1 (binary)",
        f"- faithfulness >= {args.min_faithfulness}",
        f"- answer_relevance >= {args.min_answer_relevance}",
        "- no structural rule failures (see `shared/validators.py`)",
        "",
        "## Known limitations",
        "",
        "- Judge agreement with human experts has NOT yet been calibrated",
        "  on this domain. Run the `human_review_queue.csv` spot-check to",
        "  compute Cohen's κ between the judge and 2-3 domain SMEs before",
        "  using scores for publication.",
        "- Synthetic questions, so they may not match the distribution of",
        "  real user queries. Augment with real queries in production use.",
        "- No decontamination check was performed against common RAG",
        "  benchmarks (MS MARCO, TREC, HotpotQA). If the source PDFs are",
        "  public, overlap with pretraining corpora is possible.",
        "",
        "## Recommended use",
        "",
        "- **train** split: for prompt engineering / RAG tuning",
        "- **dev** split: for iteration during development",
        "- **test** split: held out, used rarely for final reporting",
    ]
    card_path = stage_c_dir / "dataset_card.md"
    card_path.write_text("\n".join(lines), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    scored_path = output_dir / "stage_b_validation" / "scored.jsonl"
    stage_dir   = output_dir / "stage_c_finalization"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if not scored_path.exists():
        raise SystemExit(f"scored.jsonl not found: {scored_path}. "
                         f"Run validate_candidates.py first.")

    print("=" * 70)
    print(" Stage C :: Finalize Gold Dataset")
    print("=" * 70)

    rng = random.Random(args.seed)

    # ── 1. Load scored candidates ────────────────────────────────────────────
    print("\n[1/6] Loading scored candidates ...")
    records = load_jsonl(scored_path)
    print(f"  Loaded {len(records)} scored records")

    # ── 2. Apply thresholds ──────────────────────────────────────────────────
    print("\n[2/6] Applying thresholds ...")
    passed: list[dict] = []
    rejected: list[dict] = []
    reason_counts: Counter = Counter()
    for r in records:
        fail_reasons = apply_thresholds(r, args)
        if fail_reasons:
            for fr in fail_reasons:
                reason_counts[fr] += 1
            rejected.append({**r, "rejection_reason": ",".join(fail_reasons)})
        else:
            passed.append(r)
    print(f"  Passed: {len(passed)} / {len(records)}")
    print(f"  Top rejection reasons:")
    for reason, n in reason_counts.most_common(10):
        print(f"     {n:>4} :: {reason}")

    # ── 3. Deduplicate ───────────────────────────────────────────────────────
    print("\n[3/6] Deduplicating ...")
    passed, dup_dropped = deduplicate(passed)
    rejected.extend(dup_dropped)
    print(f"  After dedup: {len(passed)}  (removed {len(dup_dropped)} duplicates)")

    # ── 4. Source-diversity cap ──────────────────────────────────────────────
    print("\n[4/6] Enforcing source diversity (max "
          f"{args.max_source_share:.0%} per source doc) ...")
    passed, cap_dropped = enforce_source_diversity(
        passed, args.max_source_share, target=args.target, rng=rng,
    )
    rejected.extend(cap_dropped)
    print(f"  After cap:   {len(passed)}  (capped {len(cap_dropped)})")

    # ── 5. Downsample to target ──────────────────────────────────────────────
    print(f"\n[5/6] Downsampling to target={args.target} ...")
    if len(passed) < args.target:
        print(f"  WARNING: only {len(passed)} passed, below target {args.target}. "
              f"Consider loosening thresholds or regenerating more candidates.")
        gold = passed
    else:
        gold, excess = stratified_downsample(passed, args.target, rng=rng)
        rejected.extend(excess)
        print(f"  Gold set size: {len(gold)}")

    # ── 6. Splits, writes, datasheet, human-review queue ─────────────────────
    print(f"\n[6/6] Writing gold set + splits + human review queue ...")

    # Sort by id for stable output
    gold.sort(key=lambda r: r["candidate_id"])
    public = [to_public_record(r) for r in gold]

    atomic_write_json(public, stage_dir / f"gold_v{args.version}.json")

    # Deterministic train/dev/test split
    rng_split = random.Random(args.seed + 1)
    shuffled = list(public)
    rng_split.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(args.train_frac * n)
    n_dev   = int(args.dev_frac * n)
    train = shuffled[:n_train]
    dev   = shuffled[n_train : n_train + n_dev]
    test  = shuffled[n_train + n_dev:]

    atomic_write_json(train, stage_dir / f"gold_v{args.version}_train.json")
    atomic_write_json(dev,   stage_dir / f"gold_v{args.version}_dev.json")
    atomic_write_json(test,  stage_dir / f"gold_v{args.version}_test.json")
    atomic_write_json(rejected, stage_dir / "rejected.json")

    # Human review queue
    n_review = write_human_review_csv(
        gold, stage_dir / "human_review_queue.csv",
        args.human_review_frac, rng=rng_split,
    )

    # Source distribution stats
    src_counter: Counter = Counter()
    for r in gold:
        for d in r.get("source_documents") or ["_unknown_"]:
            src_counter[d] += 1

    # Summary
    summary = {
        "version":          args.version,
        "total_scored":     len(records),
        "total_rejected":   len(rejected),
        "gold_size":        len(gold),
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "human_review_queue_size": n_review,
        "thresholds": {
            "min_faithfulness":      args.min_faithfulness,
            "min_answer_relevance":  args.min_answer_relevance,
            "max_source_share":      args.max_source_share,
        },
        "rejection_reason_counts": dict(reason_counts),
    }
    atomic_write_json(summary, stage_dir / "summary.json")

    write_dataset_card(stage_dir, args.version, gold, args,
                       rejected_count=len(rejected),
                       source_distribution=src_counter)

    # Content hash for versioning
    all_ids = ",".join(sorted(r["id"] for r in public))
    dataset_hash = hashlib.sha256(all_ids.encode()).hexdigest()[:16]
    print(f"\n  Dataset hash: {dataset_hash}")

    print("\n" + "=" * 70)
    print(f" Stage C complete. Gold dataset v{args.version} saved to:")
    print(f"   {stage_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
