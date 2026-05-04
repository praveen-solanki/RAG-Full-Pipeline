"""
Stage B: Validate candidates with the judge model.

For each candidate we compute four scores:
  1. Answerability (0/1)       — can Q be answered from the contexts alone?
  2. Faithfulness (0-1)         — is every claim in A supported by contexts?
  3. Answer relevance (0-1)    — does A address Q?
  4. Question specificity (0/1) — is Q self-contained?

Plus the cheap rule-based structural checks (no LLM).

Why a separate script:
  - It loads a different model (Qwen3-30B-A3B-Instruct-2507 vs the 72B
    generator), so keeping them in separate processes means you only ever
    have one model in VRAM at a time.
  - It's idempotent at the record level: re-running appends scores only
    for candidates that don't already have them.
  - You can change thresholds and re-run Stage C without ever touching
    this script.

Model: Qwen/Qwen3-30B-A3B-Instruct-2507 (bf16, TP=2 on your 2x48GB).
Per the Judge's Verdict benchmark (NVIDIA, Oct 2025) this is the single
most human-like open judge: |z| = 0.04 from human-to-human agreement.

Usage:
    python validate_candidates.py --output-dir ./output
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

from shared.io_utils import (
    atomic_write_json, append_jsonl, iter_jsonl, load_jsonl, count_jsonl,
)
from shared.prompts import (
    ANSWERABILITY_SYSTEM, ANSWERABILITY_USER,
    FAITHFULNESS_SYSTEM,  FAITHFULNESS_USER,
    ANSWER_RELEVANCE_SYSTEM, ANSWER_RELEVANCE_USER,
    QUESTION_SPECIFICITY_SYSTEM, QUESTION_SPECIFICITY_USER,
)
from shared.schemas import attach_scores
from shared.validators import run_all_structural_checks
# from shared.llm_batch import VLLMBatchClient, messages
from shared.llm_batch import messages


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--vllm-url",    default="http://localhost:8011/v1",
                   help="URL of the running vLLM server")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()

def build_client(args: argparse.Namespace):
    from openai import OpenAI
    return OpenAI(api_key="dummy", base_url=args.vllm_url)

def _judge_chat_json(
    client,
    conversations: list[list[dict]],
    model: str,
    seed: int,
    max_tokens: int,
) -> list[dict | None]:
    """
    Send each conversation to the vLLM server, parse JSON response.
    Returns a list aligned to conversations — None on any failure.
    """
    import json as _json
    results = []
    for conv in conversations:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=conv,
                temperature=0.0,
                max_tokens=max_tokens,
                seed=seed,
            )
            text = resp.choices[0].message.content.strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results.append(_json.loads(text))
        except Exception:
            results.append(None)
    return results
# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _concat_contexts(contexts: list[str]) -> str:
    if len(contexts) == 1:
        return contexts[0]
    return "\n\n".join(
        f"--- CONTEXT {i+1} ---\n{c}" for i, c in enumerate(contexts)
    )


def _clip(x: Any, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


def build_judge_conversations(candidates: list[dict]) -> dict[str, list[list[dict]]]:
    """
    For a batch of candidates, build 4 separate conversation lists.
    Each returned list is aligned to `candidates` by index.
    """
    answerability   = []
    faithfulness    = []
    answer_rel      = []
    question_spec   = []

    for c in candidates:
        q   = c["user_input"]
        a   = c["reference"]
        ctx = _concat_contexts(c.get("reference_contexts") or [])

        answerability.append(
            messages(ANSWERABILITY_SYSTEM,
                     ANSWERABILITY_USER.format(context=ctx, question=q))
        )
        faithfulness.append(
            messages(FAITHFULNESS_SYSTEM,
                     FAITHFULNESS_USER.format(context=ctx, answer=a))
        )
        answer_rel.append(
            messages(ANSWER_RELEVANCE_SYSTEM,
                     ANSWER_RELEVANCE_USER.format(question=q, answer=a))
        )
        question_spec.append(
            messages(QUESTION_SPECIFICITY_SYSTEM,
                     QUESTION_SPECIFICITY_USER.format(question=q))
        )

    return {
        "answerability":         answerability,
        "faithfulness":          faithfulness,
        "answer_relevance":      answer_rel,
        "question_specificity":  question_spec,
    }


def parse_judge_results(
    results: dict[str, list[Optional[dict]]],
    n: int,
) -> list[dict]:
    """
    For each of n candidates, merge the 4 judge outputs into one
    {scores, rationales} dict. Missing values become safe defaults (0 / 0.0).
    """
    out = []
    for i in range(n):
        ans = results["answerability"][i]         or {}
        fai = results["faithfulness"][i]          or {}
        rel = results["answer_relevance"][i]      or {}
        spe = results["question_specificity"][i]  or {}

        scores = {
            "answerability":         int(_clip(ans.get("answerability", 0), 0, 1)),
            "faithfulness":          round(_clip(fai.get("faithfulness", 0), 0, 1), 3),
            "answer_relevance":      round(_clip(rel.get("answer_relevance", 0), 0, 1), 3),
            "question_specificity":  int(_clip(spe.get("question_specificity", 0), 0, 1)),
        }
        rationales = {
            "answerability":         str(ans.get("rationale", ""))[:400],
            "faithfulness":          str(fai.get("rationale", ""))[:400],
            "answer_relevance":      str(rel.get("rationale", ""))[:400],
            "question_specificity":  str(spe.get("rationale", ""))[:400],
        }
        out.append({"scores": scores, "rationales": rationales})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def validate(
    candidates: list[dict],
    client,                        # was: VLLMBatchClient
    args: argparse.Namespace,
    output_path: Path,
    already_scored_ids: set[str],
) -> None:
    to_score = [c for c in candidates if c["candidate_id"] not in already_scored_ids]
    total = len(candidates)
    done = total - len(to_score)
    print(f"\n  To score: {len(to_score)} (of {total}, already done: {done})")

    if not to_score:
        return

    for i in range(0, len(to_score), args.batch_size):
        batch = to_score[i : i + args.batch_size]
        t0 = time.time()

        structural_fails_per_cand = [run_all_structural_checks(c) for c in batch]

        convs = build_judge_conversations(batch)
        answerability_out  = _judge_chat_json(client, convs["answerability"],
                                              model=args.judge_model, seed=args.seed, max_tokens=150)
        faithfulness_out   = _judge_chat_json(client, convs["faithfulness"],
                                              model=args.judge_model, seed=args.seed, max_tokens=200)
        answer_rel_out     = _judge_chat_json(client, convs["answer_relevance"],
                                              model=args.judge_model, seed=args.seed, max_tokens=150)
        question_spec_out  = _judge_chat_json(client, convs["question_specificity"],
                                              model=args.judge_model, seed=args.seed, max_tokens=150)

        parsed = parse_judge_results(
            {
                "answerability":       answerability_out,
                "faithfulness":        faithfulness_out,
                "answer_relevance":    answer_rel_out,
                "question_specificity": question_spec_out,
            },
            n=len(batch),
        )

        scored_batch: list[dict] = []
        for cand, pr, struct_fails in zip(batch, parsed, structural_fails_per_cand):
            scores = dict(pr["scores"])
            scores["structural_fail_reasons"] = struct_fails
            rec = attach_scores(
                candidate=cand,
                scores=scores,
                judge_rationales=pr["rationales"],
                judge_model=args.judge_model,
            )
            scored_batch.append(rec)

        append_jsonl(scored_batch, output_path)
        dt = time.time() - t0
        done += len(scored_batch)

        pass_all = sum(
            1 for r in scored_batch
            if not r["scores"]["structural_fail_reasons"]
            and r["scores"]["answerability"] == 1
            and r["scores"]["question_specificity"] == 1
            and r["scores"]["faithfulness"] >= 0.80
            and r["scores"]["answer_relevance"] >= 0.80
        )
        print(
            f"     Batch {i // args.batch_size + 1:>3}: "
            f"scored {len(scored_batch)}, "
            f"pass_all={pass_all}/{len(scored_batch)}  "
            f"[{dt:.1f}s]  total_scored={done}"
        )

def load_already_scored(path: Path) -> set[str]:
    """Scan scored.jsonl and return candidate_ids already processed."""
    ids: set[str] = set()
    for rec in iter_jsonl(path):
        cid = rec.get("candidate_id")
        if cid:
            ids.add(cid)
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    candidates_path = output_dir / "stage_a_generation" / "candidates.jsonl"
    stage_dir       = output_dir / "stage_b_validation"
    scored_path     = stage_dir / "scored.jsonl"
    config_path     = stage_dir / "validation_config.json"

    if not candidates_path.exists():
        raise SystemExit(f"candidates.jsonl not found: {candidates_path}. "
                         f"Run generate_candidates.py first.")

    stage_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Stage B :: Validate Candidates")
    print("=" * 70)
    print(f" Input     : {candidates_path}")
    print(f" Output    : {scored_path}")
    print(f" Judge     : {args.judge_model}")
    print(f" vLLM URL  : {args.vllm_url}")
    print("=" * 70)

    print("\n[1/3] Loading candidates ...")
    candidates = load_jsonl(candidates_path)
    print(f"  Loaded {len(candidates)} candidates")

    already_scored_ids = load_already_scored(scored_path)
    if already_scored_ids:
        print(f"  Found {len(already_scored_ids)} already-scored; will resume")

    print("\n[2/3] Connecting to vLLM server ...")
    print(f"  URL   : {args.vllm_url}")
    print(f"  Model : {args.judge_model}")
    client = build_client(args)

    atomic_write_json(
        {
            "judge_model": args.judge_model,
            "vllm_url":    args.vllm_url,
            "batch_size":  args.batch_size,
            "seed":        args.seed,
        },
        config_path,
    )

    print("\n[3/3] Scoring ...")
    validate(candidates, client, args, scored_path, already_scored_ids)

    n_scored = count_jsonl(scored_path)
    print("\n" + "=" * 70)
    print(f" Stage B complete.")
    print(f" Scored records: {n_scored} / {len(candidates)}")
    print(f" Next: python finalize_dataset.py --output-dir {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
