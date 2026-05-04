"""
evaluate.py — Stage C of the comparative RAG pipeline.

Reads results.jsonl (the only file produced by Stage B) and runs three
tiers of evaluation on every (query, retriever, answer) row:

  Tier 1  surface metrics (no LLM):     BLEU-4, ROUGE-L, BERTScore
  Tier 2  LLM-as-judge:                 RAGAS quartet + noise / factual
  Tier 3  diagnostic (optional):        RAGChecker claim-level metrics

After scoring, a paired bootstrap confidence interval is computed per
metric on the per-query deltas  metric(bge_m3) - metric(pageindex)  so
you can tell which retriever wins on which axis with statistical support.

Usage
-----
    # start a judge LLM on a DIFFERENT port than the generator, e.g.:
    #   vllm serve prometheus-eval/prometheus-7b-v2.0 --port 8001 \
    #       --guided-decoding-backend outlines
    #
    # then (pick any subset of tiers):
    python evaluate.py \
        --results-file   ./results.jsonl \
        --output-dir     ./eval_out      \
        --judge-model-id prometheus-eval/prometheus-7b-v2.0 \
        --judge-base-url http://localhost:8001/v1 \
        --run-tier1 --run-tier2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError

from schemas import ResultRow


# --------------------------------------------------------------------------- #
# Load results                                                                #
# --------------------------------------------------------------------------- #

def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_no}: {exc}"
                ) from exc


def load_results(path: Path) -> List[ResultRow]:
    rows: List[ResultRow] = []
    for obj in _iter_jsonl(path):
        try:
            rows.append(ResultRow(**obj))
        except ValidationError as exc:
            raise ValueError(
                f"{path}: row does not match ResultRow schema:\n{exc}"
            ) from exc
    return rows


# --------------------------------------------------------------------------- #
# Tier 1 — surface metrics (BLEU, ROUGE, BERTScore)                           #
# --------------------------------------------------------------------------- #

def run_tier1(rows: List[ResultRow], use_bertscore: bool = True) -> pd.DataFrame:
    """
    Compute deterministic surface metrics. No LLM calls. Safe to run as
    often as you like. BERTScore loads a fairly large model the first time
    — disable it with --no-bertscore if you want a quick pass.
    """
    import evaluate as hf_eval  # imported here to keep import cost on demand

    sacrebleu_m = hf_eval.load("sacrebleu")
    rouge_m = hf_eval.load("rouge")
    bertscore_m = hf_eval.load("bertscore") if use_bertscore else None

    records: List[dict] = []
    for i, row in enumerate(rows):
        pred = row.answer
        ref = row.ground_truth

        # sacrebleu expects references as list-of-lists.
        bleu = sacrebleu_m.compute(
            predictions=[pred], references=[[ref]]
        )["score"]
        rouge_scores = rouge_m.compute(
            predictions=[pred], references=[ref], use_stemmer=True
        )

        record: Dict[str, object] = {
            "idx": i,
            "query": row.query,
            "retriever": row.retriever,
            "bleu4": round(float(bleu), 4),
            "rougeL": round(float(rouge_scores["rougeL"]), 4),
            "rouge1": round(float(rouge_scores["rouge1"]), 4),
        }

        if bertscore_m is not None:
            # DeBERTa-xlarge-mnli is the most human-correlated model per
            # the BERTScore authors; rescale_with_baseline gives scores in
            # a useful [0,1] range.
            bs = bertscore_m.compute(
                predictions=[pred],
                references=[ref],
                model_type="microsoft/deberta-xlarge-mnli",
                rescale_with_baseline=True,
                lang="en",
            )
            record["bertscore_f1"] = round(float(bs["f1"][0]), 4)

        records.append(record)

    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Tier 2 — RAGAS (LLM-as-judge + embedding similarity)                        #
# --------------------------------------------------------------------------- #

def _build_ragas_judge(
    judge_model_id: str,
    judge_base_url: str,
    judge_api_key: str,
    emb_model_name: str,
):
    """
    Build the (judge_llm, judge_embeddings) pair used by every RAGAS metric.

    Uses an OpenAI-compatible endpoint (vLLM/TGI) for the judge LLM and
    local BGE-M3 for the embeddings. BGE-M3 embeddings MUST be normalized
    — without normalize_embeddings=True, SemanticSimilarity is meaningless.
    """
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=judge_model_id,
            base_url=judge_base_url,
            api_key=judge_api_key,
            temperature=0.0,
            max_tokens=2048,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=emb_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return judge_llm, judge_emb


def run_tier2(
    rows: List[ResultRow],
    judge_model_id: str,
    judge_base_url: str,
    judge_api_key: str = "EMPTY",
    emb_model_name: str = "BAAI/bge-m3",
) -> pd.DataFrame:
    """
    Run the RAGAS metric suite on every row. Returns a long-format
    DataFrame with one row per (query, retriever) and one column per metric.

    raise_exceptions=False means judge-side failures produce NaN rather
    than crashing the whole run; the NaN rate itself is a quality signal.
    """
    from ragas import EvaluationDataset, SingleTurnSample
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        NoiseSensitivity,
        ResponseRelevancy,
        SemanticSimilarity,
    )

    judge_llm, judge_emb = _build_ragas_judge(
        judge_model_id=judge_model_id,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        emb_model_name=emb_model_name,
    )

    samples = [
        SingleTurnSample(
            user_input=row.query,
            retrieved_contexts=[c.text for c in row.context_chunks],
            response=row.answer,
            reference=row.ground_truth,
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_emb),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
        NoiseSensitivity(llm=judge_llm),
        FactualCorrectness(llm=judge_llm),
        SemanticSimilarity(embeddings=judge_emb),
    ]

    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    # Attach the retriever tag so the comparison step can split rows.
    df.insert(0, "retriever", [row.retriever for row in rows])
    df.insert(0, "query", [row.query for row in rows])
    return df


# --------------------------------------------------------------------------- #
# Tier 3 — RAGChecker (optional diagnostic)                                   #
# --------------------------------------------------------------------------- #

def run_tier3(
    rows: List[ResultRow],
    judge_model_id: str,
    judge_base_url: str,
    judge_api_key: str = "EMPTY",
) -> pd.DataFrame:
    """
    Run RAGChecker's fine-grained claim-level metrics (Amazon Science,
    NeurIPS 2024). Splits quality into retriever-side vs generator-side
    causes, which is the cleanest way to explain *why* one retriever wins.

    RAGChecker is heavier than RAGAS (more judge calls per row), so treat
    this as a periodic diagnostic, not something you run on every commit.
    """
    try:
        from ragchecker import RAGChecker, RAGResults
        from ragchecker.metrics import all_metrics
    except ImportError as exc:
        raise RuntimeError(
            "ragchecker is not installed. Run: pip install ragchecker"
        ) from exc

    payload = {
        "results": [
            {
                "query_id": f"q{i}",
                "query": row.query,
                "gt_answer": row.ground_truth,
                "response": row.answer,
                "retrieved_context": [
                    {"doc_id": c.source_id, "text": c.text}
                    for c in row.context_chunks
                ],
                # Extra field we use for grouping; RAGChecker ignores unknowns.
                "retriever": row.retriever,
            }
            for i, row in enumerate(rows)
        ]
    }

    rag_results = RAGResults.from_dict(payload)
    checker = RAGChecker(
        extractor_name=judge_model_id,
        checker_name=judge_model_id,
        batch_size_extractor=16,
        batch_size_checker=16,
        openai_api_base=judge_base_url,
        openai_api_key=judge_api_key,
    )
    checker.evaluate(rag_results, all_metrics)

    records: List[dict] = []
    for i, item in enumerate(rag_results.results):
        record: Dict[str, object] = {
            "idx": i,
            "query": rows[i].query,
            "retriever": rows[i].retriever,
        }
        record.update(item.metrics or {})
        records.append(record)
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Comparison report — paired bootstrap CIs                                    #
# --------------------------------------------------------------------------- #

def _paired_bootstrap_ci(
    deltas: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float, float]:
    """Return (mean_delta, ci_low, ci_high) from a paired bootstrap."""
    rng = np.random.default_rng(seed)
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size == 0:
        return float("nan"), float("nan"), float("nan")

    idx = rng.integers(0, deltas.size, size=(n_resamples, deltas.size))
    means = deltas[idx].mean(axis=1)
    ci_low, ci_high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(deltas.mean()), float(ci_low), float(ci_high)


def build_comparison_report(
    df: pd.DataFrame,
    metric_cols: List[str],
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    """
    Given a long-format DataFrame with columns [query, retriever, <metrics>],
    pivot to wide form, compute per-query deltas (bge_m3 - pageindex),
    bootstrap CIs, and report who wins each metric.
    """
    needed = {"query", "retriever"}
    if not needed.issubset(df.columns):
        raise ValueError(f"DataFrame is missing required columns: {needed}")

    records: List[dict] = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(
            index="query", columns="retriever", values=metric, aggfunc="mean"
        )
        if "pageindex" not in pivot.columns or "bge_m3" not in pivot.columns:
            continue
        pairs = pivot.dropna(subset=["pageindex", "bge_m3"])
        if pairs.empty:
            continue

        deltas = (pairs["bge_m3"] - pairs["pageindex"]).to_numpy()
        mean_delta, ci_low, ci_high = _paired_bootstrap_ci(
            deltas, n_resamples=n_resamples
        )

        if ci_low > 0:
            winner = "bge_m3"
        elif ci_high < 0:
            winner = "pageindex"
        else:
            winner = "tie (CI crosses 0)"

        records.append(
            {
                "metric": metric,
                "n_paired": int(len(pairs)),
                "pageindex_mean": round(float(pairs["pageindex"].mean()), 4),
                "bgem3_mean": round(float(pairs["bge_m3"].mean()), 4),
                "mean_delta(bge_m3 - pageindex)": round(mean_delta, 4),
                "ci95_low": round(ci_low, 4),
                "ci95_high": round(ci_high, 4),
                "winner": winner,
            }
        )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("eval_out"))

    p.add_argument("--run-tier1", action="store_true", help="Run BLEU/ROUGE/BERTScore.")
    p.add_argument("--run-tier2", action="store_true", help="Run RAGAS.")
    p.add_argument("--run-tier3", action="store_true", help="Run RAGChecker.")
    p.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore in Tier 1.")

    p.add_argument(
        "--judge-model-id",
        default="prometheus-eval/prometheus-7b-v2.0",
        help="Judge LLM. MUST be a different model family than the generator.",
    )
    p.add_argument(
        "--judge-base-url",
        default="http://localhost:8001/v1",
        help="OpenAI-compatible endpoint for the judge LLM.",
    )
    p.add_argument("--judge-api-key", default="EMPTY")
    p.add_argument("--emb-model-name", default="BAAI/bge-m3")
    p.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not any([args.run_tier1, args.run_tier2, args.run_tier3]):
        sys.exit(
            "error: pick at least one of --run-tier1 / --run-tier2 / --run-tier3"
        )

    if not args.results_file.exists():
        sys.exit(f"error: {args.results_file} does not exist")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(args.results_file)
    print(f"Loaded {len(rows)} rows from {args.results_file}", file=sys.stderr)

    combined_long: Optional[pd.DataFrame] = None

    # ---- Tier 1 ------------------------------------------------------- #
    if args.run_tier1:
        print("Running Tier 1 (surface metrics)...", file=sys.stderr)
        df1 = run_tier1(rows, use_bertscore=not args.no_bertscore)
        df1.to_csv(args.output_dir / "tier1_surface.csv", index=False)
        combined_long = df1.copy()

    # ---- Tier 2 ------------------------------------------------------- #
    if args.run_tier2:
        print("Running Tier 2 (RAGAS)...", file=sys.stderr)
        df2 = run_tier2(
            rows,
            judge_model_id=args.judge_model_id,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
            emb_model_name=args.emb_model_name,
        )
        df2.to_csv(args.output_dir / "tier2_ragas.csv", index=False)
        combined_long = (
            df2 if combined_long is None
            else combined_long.merge(df2, on=["query", "retriever"], how="outer")
        )

    # ---- Tier 3 ------------------------------------------------------- #
    if args.run_tier3:
        print("Running Tier 3 (RAGChecker)...", file=sys.stderr)
        df3 = run_tier3(
            rows,
            judge_model_id=args.judge_model_id,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
        )
        df3.to_csv(args.output_dir / "tier3_ragchecker.csv", index=False)
        combined_long = (
            df3 if combined_long is None
            else combined_long.merge(df3, on=["query", "retriever"], how="outer")
        )

    # ---- Comparison report ------------------------------------------- #
    if combined_long is not None:
        non_metric = {"idx", "query", "retriever"}
        metric_cols = [
            c for c in combined_long.columns
            if c not in non_metric
            and pd.api.types.is_numeric_dtype(combined_long[c])
        ]
        report = build_comparison_report(
            combined_long,
            metric_cols=metric_cols,
            n_resamples=args.bootstrap_resamples,
        )
        report_path = args.output_dir / "comparison_report.csv"
        report.to_csv(report_path, index=False)
        combined_long.to_csv(args.output_dir / "all_metrics_long.csv", index=False)

        print("\n=== Comparison report (paired bootstrap, 95% CI) ===\n", file=sys.stderr)
        if report.empty:
            print("(no paired rows available)", file=sys.stderr)
        else:
            print(report.to_string(index=False), file=sys.stderr)
        print(f"\nWrote {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()