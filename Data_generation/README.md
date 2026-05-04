# Gold-Standard RAG Evaluation Dataset Pipeline

A three-stage, file-based pipeline to build a calibrated RAG evaluation
dataset from PDFs using **only open-weight vLLM models** on 2x48GB GPUs.

Each stage reads from disk and writes to disk, so they can be run
independently, resumed after crashes, and re-tuned without re-running
upstream work.

## Pipeline architecture

```
build_kg.py                 generate_candidates.py       validate_candidates.py      finalize_dataset.py
(uses vLLM server)          (loads Qwen2.5-72B-AWQ)      (loads Qwen3-30B-A3B)       (CPU only)
      |                             |                             |                           |
      v                             v                             v                           v
  output/kg/              stage_a_generation/            stage_b_validation/       stage_c_finalization/
  knowledge_graph.json    candidates.jsonl                scored.jsonl              gold_v1.0.json
                                                                                    gold_v1.0_{train,dev,test}.json
                                                                                    rejected.json
                                                                                    human_review_queue.csv
                                                                                    dataset_card.md
                                                                                    summary.json
```

## Hardware

Designed for 2 x 48GB GPUs (e.g. 2x RTX 6000 Ada, 2x A6000, 2x L40S).
Nothing in the pipeline needs more than this.

At any given moment only **one** model is in GPU memory:
- Stage 0 (build_kg): Qwen2.5-72B-AWQ (~40GB weights + KV) via vLLM **server**
- Stage A (generate): Qwen2.5-72B-AWQ via vLLM **offline batch**
- Stage B (validate): Qwen3-30B-A3B-Instruct-2507 (~60GB bf16) via vLLM offline batch
- Stage C (finalize): CPU only

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Build the knowledge graph (once per corpus)

Start the vLLM server in a separate terminal:

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --quantization awq \
    --port 8011
```

Then in another terminal:

```bash
python build_kg.py \
    --pdf-dir ./pdfs \
    --output-dir ./output
```

Kill the vLLM server when done. Typical runtime: 20-60 min for ~100 pages.

### 2. Generate candidates (over-generate 2x target)

```bash
python generate_candidates.py \
    --kg-file ./output/kg/knowledge_graph.json \
    --output-dir ./output \
    --target 500
```

This loads Qwen2.5-72B-AWQ via **offline batch** (no server needed — it
loads the model directly into the Python process), generates 1000
candidates (2x target), and writes them to
`output/stage_a_generation/candidates.jsonl`.

Typical runtime: 90-120 min. Crash-resumable — rerunning picks up from
the last persisted candidate.

### 3. Validate candidates

```bash
python validate_candidates.py --output-dir ./output
```

This loads Qwen3-30B-A3B-Instruct-2507 and scores every candidate on
4 metrics. Output: `output/stage_b_validation/scored.jsonl`.

Typical runtime: 45-60 min. Crash-resumable.

### 4. Finalize

```bash
python finalize_dataset.py --output-dir ./output --target 500
```

No model needed. Applies thresholds, dedupes, splits, writes everything.
Typical runtime: seconds. Safe to re-run with different thresholds
without redoing any LLM work.

## Tuning

After the first run, inspect `output/stage_c_finalization/summary.json`:
- If too few candidates pass, loosen `--min-faithfulness` or
  `--min-answer-relevance` in `finalize_dataset.py`.
- If too many pass, tighten them.
- You can also lower `--overgen-ratio` from 2.0 to save time on future
  runs once you know your filter survival rate.

## Human calibration (the step that makes it "gold")

1. Open `human_review_queue.csv` in Excel/Sheets.
2. Have 2-3 AUTOSAR SMEs fill in the `sme_*` columns.
3. Compute Cohen's κ between the SMEs and the judge columns.
4. If κ >= 0.7, the judge is calibrated; the gold set stands as-is.
5. If κ < 0.7, either:
   - Manually fix the disagreeing samples in `gold_v1.0.json`, or
   - Lower/raise judge thresholds and re-run `finalize_dataset.py`.

## Files you'll share

The final dataset consumers only need:
- `output/stage_c_finalization/gold_v1.0.json` (or the train/dev/test splits)
- `output/stage_c_finalization/dataset_card.md`

The intermediate files are for auditing and regeneration.
