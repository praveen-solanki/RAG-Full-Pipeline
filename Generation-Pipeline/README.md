# Generation Pipeline — Comparative RAG Evaluation Framework

A three-stage pipeline for rigorously comparing two retrieval strategies — **PageIndex** (reasoning-based) and **BGE-M3** (vector-based / hybrid) — under a fully frozen generator. Because the generator is held constant across both retrievers, any quality difference in the final answers is attributable solely to retrieval.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Data Schemas](#data-schemas)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Stage A — Retrieval (bring your own)](#stage-a--retrieval-bring-your-own)
  - [Stage B — Generation (`generate.py`)](#stage-b--generation-generatepy)
  - [Stage C — Evaluation (`evaluate.py`)](#stage-c--evaluation-evaluatepy)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Configuration Reference](#configuration-reference)
- [Design Decisions](#design-decisions)
- [Known Limitations & Caveats](#known-limitations--caveats)
- [Contributing](#contributing)

---

## Overview

This repository implements **Stages B and C** of a comparative RAG (Retrieval-Augmented Generation) pipeline. Given pre-computed retrieval outputs from two different retrieval systems, it:

1. **Generates** answers using a single, deterministic (frozen) LLM for every `(query, retriever)` pair.
2. **Evaluates** those answers across three tiers of metrics — from fast lexical surface scores up to expensive claim-level LLM-as-judge diagnostics.
3. **Reports** per-metric winners with statistically rigorous paired bootstrap 95% confidence intervals.

The key design principle is isolation: the generator is identical for both retrievers (same model, same prompt template, same temperature, same random seed), so evaluation results cleanly attribute quality differences to retrieval rather than generation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage A  (not in this repo — bring your own retriever)         │
│                                                                 │
│  pageindex_retrieval.jsonl   bgem3_retrieval.jsonl              │
└────────────────────┬────────────────────┬───────────────────────┘
                     │                    │
                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage B — generate.py                                          │
│                                                                 │
│  1. Load & normalize both retrieval files → unified Context     │
│  2. Run FrozenRAGGenerator (vLLM / TGI / Ollama endpoint)       │
│     - temperature=0, seed=42 (fixed for reproducibility)        │
│  3. Stream results to results.jsonl (crash-safe, resumable)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage C — evaluate.py                                          │
│                                                                 │
│  Tier 1  BLEU-4 · ROUGE-L · BERTScore (no LLM)                 │
│  Tier 2  RAGAS: Faithfulness · ContextPrecision · Recall ·      │
│          ResponseRelevancy · NoiseSensitivity ·                 │
│          FactualCorrectness · SemanticSimilarity                │
│  Tier 3  RAGChecker claim-level diagnostics (optional)          │
│                                                                 │
│  → paired bootstrap CI per metric → comparison_report.csv       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Generation-Pipeline/
├── generate.py              # Stage B — frozen generation over retrieved contexts
├── evaluate.py              # Stage C — three-tier metric evaluation + comparison
├── schemas.py               # Pydantic data contracts shared by both scripts
├── data/
│   ├── results.json         # Sample generation outputs (JSON format)
│   └── retrieval_results.jsonl  # Sample retrieval outputs (JSONL format)
└── eval_out/
    ├── tier1_surface.csv        # Per-row BLEU / ROUGE / BERTScore results
    ├── all_metrics_long.csv     # All tiers merged in long format
    └── comparison_report.csv   # Paired bootstrap summary — who wins per metric
```

---

## Data Schemas

All data contracts are defined in `schemas.py` using Pydantic. Both `generate.py` and `evaluate.py` import from this single source of truth.

### Input to Stage B

**`PageIndexRow`** — one line of `pageindex_retrieval.jsonl`

| Field | Type | Notes |
|---|---|---|
| `query` | `str` | Also accepted as `"question"` (alias) |
| `ground_truth` | `str` | Reference answer for evaluation |
| `retriever` | `Literal["pageindex"]` | Always `"pageindex"` |
| `evidence_snippets` | `List[str]` | Raw text evidence returned by PageIndex |
| `source_id` | `str` | Defaults to `"pageindex_tree"` |

**`BGEM3Row`** — one line of `bgem3_retrieval.jsonl`

| Field | Type | Notes |
|---|---|---|
| `query` | `str` | |
| `ground_truth` | `str` | |
| `retriever` | `Literal["bge_m3", "hybrid_qdrr"]` | |
| `context_chunks` | `List[BGEM3Chunk]` | Ranked list of retrieved chunks |

**`BGEM3Chunk`** fields: `text`, `source_id`, `score`, `rerank_score`, `rrf_score`, `dense_score`, `sparse_score`, `hop_index` (all optional except `text` and `source_id`).

### Internal normalized format (Stage B → generator)

**`Context`** — unified view fed into `FrozenRAGGenerator`

| Field | Type | Notes |
|---|---|---|
| `chunks` | `List[ContextChunk]` | One or more context chunks |
| `retriever` | `Literal["pageindex", "bge_m3", "hybrid_qdrr"]` | |

Both PageIndex (single concatenated chunk) and BGE-M3 (multiple ranked chunks) converge to this shape, ensuring the generator sees exactly one data structure regardless of retriever type.

### Output of Stage B / Input to Stage C

**`ResultRow`** — one line of `results.jsonl`

| Field | Type | Notes |
|---|---|---|
| `query` | `str` | |
| `ground_truth` | `str` | |
| `retriever` | `str` | Which retriever produced this row |
| `context_chunks` | `List[ContextChunk]` | The exact context that was used |
| `answer` | `str` | LLM-generated answer |
| `model_id` | `str` | Generator model identifier |
| `latency_ms` | `float` | End-to-end generation latency |
| `prompt_chars` | `int` | Total prompt character length (for auditing) |

---

## Prerequisites

- **Python 3.10+**
- A running **OpenAI-compatible LLM server** for generation (vLLM, TGI, or Ollama with the OpenAI-compat layer)
- For **Stage C** (evaluation):
  - A **separate** judge LLM server on a different port (e.g., `prometheus-eval/prometheus-7b-v2.0`)
  - `BAAI/bge-m3` model downloaded locally (for Tier 2 semantic embeddings)

### Recommended models

| Role | Example model | Notes |
|---|---|---|
| Generator | `mistralai/Mistral-7B-Instruct-v0.3` | Any instruction-tuned model works |
| Generator | `Qwen/Qwen2.5-32B-Instruct-AWQ` | Default in CLI args |
| Judge (Tier 2/3) | `prometheus-eval/prometheus-7b-v2.0` | Must differ from generator family |

---

## Installation

```bash
git clone https://github.com/<your-org>/Generation-Pipeline.git
cd Generation-Pipeline

pip install pydantic openai numpy pandas \
            evaluate sacrebleu rouge-score bert-score \
            langchain-openai langchain-huggingface ragas
```

For Tier 3 (RAGChecker), install separately:

```bash
pip install ragchecker
```

> **Note:** `bert_score` with DeBERTa-xlarge-mnli requires significant GPU/CPU memory on first run. Use `--no-bertscore` to skip it during development.

---

## Usage

### Stage A — Retrieval (bring your own)

Stage A is not included in this repository. You are expected to produce two JSONL files:

- `pageindex_retrieval.jsonl` — one JSON object per line matching `PageIndexRow`
- `bgem3_retrieval.jsonl` — one JSON object per line matching `BGEM3Row`

The pipeline is entirely agnostic to how you produce these files. Any retriever that outputs the correct schema will work.

---

### Stage B — Generation (`generate.py`)

First, start your generator LLM server:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8011
```

Then run generation:

```bash
python generate.py \
    --pageindex-file ./pageindex_retrieval.jsonl \
    --bgem3-file     ./bgem3_retrieval.jsonl    \
    --output-file    ./results.jsonl            \
    --model-id       mistralai/Mistral-7B-Instruct-v0.3 \
    --base-url       http://localhost:8011/v1
```

**Resume a crashed run** (skips already-completed `(query, retriever)` pairs):

```bash
python generate.py \
    --pageindex-file ./pageindex_retrieval.jsonl \
    --bgem3-file     ./bgem3_retrieval.jsonl    \
    --output-file    ./results.jsonl            \
    --model-id       mistralai/Mistral-7B-Instruct-v0.3 \
    --base-url       http://localhost:8011/v1   \
    --resume
```

Results are written **incrementally** line-by-line so a mid-run crash loses nothing already written.

#### CLI Reference — `generate.py`

| Argument | Default | Description |
|---|---|---|
| `--pageindex-file` | *(required)* | Path to PageIndex retrieval JSONL |
| `--bgem3-file` | *(required)* | Path to BGE-M3 retrieval JSONL |
| `--output-file` | `./results.jsonl` | Where to write generation outputs |
| `--model-id` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | Model name as served on the endpoint |
| `--base-url` | `http://localhost:8011/v1` | OpenAI-compatible endpoint URL |
| `--api-key` | `EMPTY` | API key (use `EMPTY` for local servers) |
| `--temperature` | `0.0` | Sampling temperature (keep at 0 for reproducibility) |
| `--max-tokens` | `1024` | Maximum tokens to generate per answer |
| `--seed` | `42` | Random seed (frozen for determinism) |
| `--resume` | `False` | Append to existing output, skipping done rows |

---

### Stage C — Evaluation (`evaluate.py`)

Start a **separate** judge LLM server on a different port:

```bash
vllm serve prometheus-eval/prometheus-7b-v2.0 --port 8012 \
    --guided-decoding-backend outlines
```

Run any combination of tiers:

```bash
# Tier 1 only (fast, no LLM needed)
python evaluate.py \
    --results-file ./results.jsonl \
    --output-dir   ./eval_out      \
    --run-tier1

# Tier 1 + Tier 2 (recommended for full evaluation)
python evaluate.py \
    --results-file   ./results.jsonl \
    --output-dir     ./eval_out      \
    --judge-model-id prometheus-eval/prometheus-7b-v2.0 \
    --judge-base-url http://localhost:8012/v1 \
    --run-tier1 --run-tier2

# All three tiers
python evaluate.py \
    --results-file   ./results.jsonl \
    --output-dir     ./eval_out      \
    --judge-model-id prometheus-eval/prometheus-7b-v2.0 \
    --judge-base-url http://localhost:8012/v1 \
    --run-tier1 --run-tier2 --run-tier3

# Skip BERTScore (faster Tier 1, useful during development)
python evaluate.py \
    --results-file ./results.jsonl \
    --output-dir   ./eval_out      \
    --run-tier1    --no-bertscore
```

#### CLI Reference — `evaluate.py`

| Argument | Default | Description |
|---|---|---|
| `--results-file` | *(required)* | Path to `results.jsonl` from Stage B |
| `--output-dir` | `./eval_out` | Directory for all output CSVs |
| `--run-tier1` | `False` | Run BLEU / ROUGE / BERTScore |
| `--run-tier2` | `False` | Run RAGAS (LLM-as-judge) |
| `--run-tier3` | `False` | Run RAGChecker (claim-level diagnostics) |
| `--no-bertscore` | `False` | Skip BERTScore in Tier 1 |
| `--judge-model-id` | `prometheus-eval/prometheus-7b-v2.0` | Judge LLM (must differ from generator) |
| `--judge-base-url` | `http://10.47.39.43:8011/v1` | OpenAI-compatible endpoint for judge |
| `--judge-api-key` | `EMPTY` | Judge API key |
| `--emb-model-name` | `BAAI/bge-m3` | Embedding model for RAGAS SemanticSimilarity |
| `--bootstrap-resamples` | `10000` | Number of bootstrap resamples for CIs |

---

## Evaluation Metrics

### Tier 1 — Surface Metrics (no LLM required)

Fast, deterministic, and safe to run repeatedly. Uses [HuggingFace `evaluate`](https://huggingface.co/docs/evaluate/index).

| Metric | Library | Notes |
|---|---|---|
| **BLEU-4** | `sacrebleu` | 4-gram precision with brevity penalty |
| **ROUGE-1** | `rouge` | Unigram overlap (recall-oriented) |
| **ROUGE-L** | `rouge` | Longest common subsequence F1 |
| **BERTScore F1** | `bert_score` | Contextual embedding similarity using `microsoft/deberta-xlarge-mnli` with baseline rescaling |

> A monkey-patch is applied to `bert_score` at import time to fix a tokenizer overflow bug in DeBERTa-xlarge (`model_max_length = 1e30` causes a `usize` overflow in the Rust tokenizer backend). Sequences are capped at 512 tokens.

### Tier 2 — LLM-as-Judge via RAGAS

Uses [RAGAS](https://docs.ragas.io) with a local judge LLM and local BGE-M3 embeddings. Judge failures produce `NaN` (rather than crashing), and the NaN rate is itself a diagnostic signal.

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **ResponseRelevancy** | Is the answer relevant to the question? |
| **LLMContextPrecisionWithReference** | Are the retrieved chunks actually useful? |
| **LLMContextRecall** | Does the context cover the ground truth? |
| **NoiseSensitivity** | How much does irrelevant context degrade the answer? |
| **FactualCorrectness** | Does the answer agree with the ground truth factually? |
| **SemanticSimilarity** | Embedding-based similarity to the reference answer |

> BGE-M3 embeddings **must** be normalized (`normalize_embeddings=True`) for SemanticSimilarity to be meaningful.

### Tier 3 — RAGChecker (optional, periodic)

Uses [RAGChecker](https://github.com/amazon-science/RAGChecker) (Amazon Science, NeurIPS 2024) for claim-level decomposition. This is heavier than RAGAS (more judge calls per row) and is best treated as a periodic diagnostic rather than a per-commit check. Its main value is splitting observed quality differences into **retriever-side** vs **generator-side** causes.

### Comparison Report — Paired Bootstrap CIs

After any tier completes, a comparison report is generated automatically. For each metric:

1. Rows are matched by `query` across the two retrievers (only paired rows are used).
2. Per-query deltas are computed: `metric(bge_m3) − metric(pageindex)`.
3. A **paired bootstrap** with 10,000 resamples produces a 95% CI on the mean delta.
4. The winner is declared only if the CI excludes zero; otherwise the result is a tie.

```
metric           n_paired  pageindex_mean  bge_m3_mean  mean_delta  ci95_low  ci95_high  winner
bleu4              245          0.1823         0.2114      0.0291     0.0187     0.0395   bge_m3
rougeL             245          0.3201         0.3089     -0.0112    -0.0249     0.0025   tie (CI crosses 0)
...
```

---

## Output Files

| File | Produced by | Contents |
|---|---|---|
| `results.jsonl` | `generate.py` | One `ResultRow` per line — the single handoff between Stage B and C |
| `eval_out/tier1_surface.csv` | `evaluate.py --run-tier1` | Per-row BLEU, ROUGE, BERTScore |
| `eval_out/tier2_ragas.csv` | `evaluate.py --run-tier2` | Per-row RAGAS metric suite |
| `eval_out/tier3_ragchecker.csv` | `evaluate.py --run-tier3` | Per-row RAGChecker claim-level metrics |
| `eval_out/all_metrics_long.csv` | `evaluate.py` (any tier) | All completed tiers merged in long format |
| `eval_out/comparison_report.csv` | `evaluate.py` (any tier) | Per-metric winner with bootstrap CIs |

---

## Configuration Reference

### Generator system prompt

The frozen generator uses a strict context-only prompt that prevents hallucination:

```
You are a careful question-answering assistant.
Answer the user's question using ONLY the information in the provided context.
If the answer is not contained in the context, reply exactly: I don't know.
Do not add facts that are not in the context.
```

The user turn follows the template:

```
Context:
{context}

Question: {query}

Answer:
```

To change the prompts, edit the `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` class attributes in `FrozenRAGGenerator` in `generate.py`. Note that changing the prompt invalidates any existing `results.jsonl` for comparison purposes.

### Context normalization

- **PageIndex** returns a flat list of `evidence_snippets` which are joined with `\n\n---\n\n` into a single `ContextChunk`.
- **BGE-M3 / Hybrid QDRR** returns multiple ranked chunks which are preserved in retriever order (already sorted by similarity score).

Both are flattened to a single prompt-ready string via `Context.as_text()` using `\n\n---\n\n` as a separator between chunks.

---

## Design Decisions

**Why a frozen generator?**
Any difference in sampling (temperature > 0, different seeds, different prompts) would introduce confounds that make it impossible to attribute answer quality differences to retrieval. The generator is frozen so the experiment has a single independent variable.

**Why write results incrementally?**
Generation over hundreds of queries against a remote LLM endpoint can take hours. Incremental writes mean a crash or network interruption doesn't require starting over — use `--resume` to continue.

**Why paired bootstrap instead of a t-test?**
The distribution of metric deltas is typically non-normal and often bimodal (one retriever wins badly on some queries, the other wins on others). Bootstrap makes no parametric assumptions and handles this correctly.

**Why a separate judge model?**
Using the same model family for both generation and judging creates evaluation bias — models tend to rate outputs from their own family higher. The judge must come from a different model family.

**Why is `eval.py` partially commented out?**
The file contains a complete earlier draft of the evaluation code (commented out) followed by the final active version. The commented section is preserved for reference and shows the incremental development history.

---

## Known Limitations & Caveats

- **Stage A is not provided.** You must supply the retrieval JSONL files in the correct schema. The pipeline validates them strictly with Pydantic and will fail fast on schema mismatches.
- **`--resume` deduplicates by `(query, retriever)` exact match.** If your queries contain minor formatting differences between runs, duplicate rows may be generated.
- **The hardcoded BGE-M3 path** in `_build_ragas_judge` (in `evaluate.py`) points to a local HuggingFace cache snapshot. Change this to your own local path or use `emb_model_name` to specify the model name and let HuggingFace download it.
- **BERTScore requires a GPU** for practical performance. On CPU it is very slow. Use `--no-bertscore` for fast iteration.
- **Tier 3 / RAGChecker** is computationally expensive and not recommended as part of a regular CI/CD loop.
- The `results.json` in `data/` is in JSON (not JSONL) format and represents PageIndex-style output; it is not directly consumed by `generate.py` which expects JSONL for BGE-M3 and JSON with a `results` key for PageIndex.

---

## Contributing

1. Fork the repository and create a feature branch.
2. All data contracts must go through `schemas.py` — do not introduce ad-hoc field parsing in `generate.py` or `evaluate.py`.
3. If you add a new retriever type, add a corresponding `Literal` value to the `retriever` fields in `schemas.py` and a new loader function in `generate.py`.
4. Run Tier 1 evaluation as a smoke test before opening a PR.
5. Do not commit retrieval JSONL files or `results.jsonl` to the repository — they can be very large and contain sensitive query data.

---

## License

Specify your license here (e.g., MIT, Apache 2.0).
