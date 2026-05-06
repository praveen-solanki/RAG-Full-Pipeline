# AUTOSAR RAG — BGE-M3 Hybrid Vector Search Pipeline

A production-grade **Retrieval-Augmented Generation (RAG)** system purpose-built for AUTOSAR technical documentation. The pipeline combines **contextual parent-child chunking**, **BGE-M3 three-signal embeddings** (dense + sparse + ColBERT), and **multi-hop query decomposition** to achieve state-of-the-art retrieval quality over large AUTOSAR specification corpora.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Ingestion](#1-ingestion)
  - [2. Retrieval](#2-retrieval)
  - [3. Evaluation](#3-evaluation)
  - [4. Chunking Benchmark](#4-chunking-benchmark)
  - [5. Utility Scripts](#5-utility-scripts)
- [Evaluation Results](#evaluation-results)
- [Design Decisions & Problem Log](#design-decisions--problem-log)
- [Failure Taxonomy](#failure-taxonomy)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

AUTOSAR specifications are dense, cross-referential technical documents that break standard RAG pipelines. This project addresses 21 specific failure modes (tracked as I1–I21) identified during development and rebuilds the entire stack from scratch:

| Stage | Approach |
|---|---|
| **Document Parsing** | `pypdfium2` + `pdfplumber` for text and tables |
| **Chunking** | Contextual Parent-Child (child=400 tok, parent=1600 tok) |
| **Embeddings** | BGE-M3: dense (1024-dim) + native sparse + ColBERT multi-vector |
| **Vector Store** | Qdrant with `MultiVectorConfig(MAX_SIM)` for ColBERT |
| **Retrieval** | Hybrid dense+sparse → RRF fusion → ColBERT/cross-encoder rerank |
| **Multi-hop** | IRCoT-style iterative sub-query decomposition via local LLM |
| **Evaluation** | Ragas-style chunk-level metrics: Recall@k, Precision@k, NDCG@k, MRR |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      INGESTION                          │
│                                                         │
│  PDF/DOCX ──► extract text + tables (pdfplumber)        │
│            ──► structural split (bookmarks / headings)  │
│            ──► parent chunks (1600 tokens)              │
│            ──► child chunks  (400 tokens, 10% overlap)  │
│            ──► contextual enrichment per child (LLM)    │
│            ──► BGEM3FlagModel.encode(enriched_child)    │
│                   ├── dense_vecs   (1024-dim, L2-norm)  │
│                   ├── lexical_weights  (native sparse)  │
│                   └── colbert_vecs    (N × 1024)        │
│            ──► Qdrant upsert {dense, sparse, colbert}   │
│                payload: child_text, parent_text,        │
│                         section, page, filename         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      RETRIEVAL                          │
│                                                         │
│  Single-hop                                             │
│    Query ──► BGE-M3 encode                              │
│           ──► Qdrant prefetch (dense + sparse)          │
│           ──► RRF fusion                                │
│           ──► Cross-encoder / ColBERT rerank            │
│           ──► Fetch parent_text from payload ──► LLM    │
│                                                         │
│  Multi-hop  (QD+RR, IRCoT pattern)                      │
│    Query ──► LLM decompose into N sub-queries           │
│    for each sub-query (iterative):                      │
│       ──► retrieve candidate pool (dense+sparse+RRF)    │
│       ──► feed top chunks as context for next sub-query │
│    ──► merge all per-hop pools (dedup)                  │
│    ──► single cross-encoder rerank vs ORIGINAL query    │
│    ──► enforce min-coverage per hop                     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

**Embedding & Indexing**
- BGE-M3 three-signal encoding: dense vectors, native learned sparse weights (replaces external BM25), and ColBERT multi-vectors stored in Qdrant
- Token-count overlap via `tiktoken` (not character-count) for precise chunk boundaries
- Jaccard deduplication using AUTOSAR-aware tokenization that preserves identifiers
- Tables converted to natural-language sentences before embedding
- Streaming batch upload — chunks never all held in RAM simultaneously
- Embedding model fingerprint written to collection metadata with mismatch guard
- L2-norm guard: degenerate near-zero vectors are logged and skipped

**Retrieval**
- Hybrid dense + sparse retrieval with RRF fusion (`k=60`)
- Asymmetric candidate pool sizes: hop-1 broad (120) for recall, hop-N targeted (80) for precision
- Two rerank modes: `cross_encoder` (bge-reranker-v2-m3, higher quality) or `colbert` (no extra VRAM)
- Parent text served to LLM — child chunks indexed for precision, parent text for LLM context quality
- PYTORCH_CUDA_ALLOC_CONF expandable segments to prevent OOM on large rerank pools

**Evaluation**
- Per-question metrics: Recall@k, Precision@k, NDCG@k, All-found@k (strict multi-hop), MRR
- Two context-matching modes: `fuzzy` (fast, text-based) or `embedding` (BGE-M3 cosine, handles OCR artefacts)
- Failure taxonomy: wrong_doc / wrong_chunk / below_cutoff / unknown
- Resume support (`--resume`) for long evaluation runs

---

## Project Structure

```
Vector_DB/
│
├── Ingestion_BGE_M3.py              # Document ingestion pipeline (PDF/DOCX → Qdrant)
├── HybridRetriever_BGE_M3.py        # Hybrid retrieval + multi-hop query decomposition
├── Retrival_BGE_M3.py               # Retrieval evaluation against ground-truth dataset
├── subset_queries.py                # Utility: filter a subset from a large GT JSON
│
├── chunking-methods/
│   ├── runner.py                    # Parallel benchmark: 9 docs × 3 chunk sizes × 3 top-k
│   ├── split_by_source.py           # Split multi-source eval JSON into per-document files
│   ├── queries_format_converter.py  # Convert evidence_snippets → rag-chunk format
│   ├── combine_results.py           # Merge all benchmark result .txt files into one
│   └── Q_2_ragchunk.json            # Sample evaluation question set (rag-chunk format)
│
├── chunk_level_results/
│   ├── Autosar_test_chunck.json     # Chunk-level evaluation output (full)
│   └── Autosar_test_chunck_summary.txt
│
├── doc_level_results/
│   ├── Autosar_pdf_doc.json         # Document-level evaluation output (full)
│   └── Autosar_pdf_doc_summary.txt
│
├── output/                          # v1 evaluation: 1000-question doc-level run
│   ├── complete_evaluation_results.json
│   └── complete_evaluation_results_summary.txt
│
├── output_v2/                       # v2 evaluation: 365-question QD+RR chunk-level run
│   ├── evaluation_results.log
│   ├── evaluation_results.zip
│   ├── evaluation_results_summary.txt
│   └── retrieval_results.jsonl
│
└── rag_pipeline_full_visualization.svg   # End-to-end pipeline diagram
```

---

## Requirements

**Python:** 3.9+

**Services:**
- [Qdrant](https://qdrant.tech/) vector database (default: `http://localhost:7333`)
- A locally-hosted LLM for contextual enrichment and query decomposition (vLLM or similar, default: `http://localhost:8011/v1`)

**Python packages:**

```bash
pip install FlagEmbedding sentence-transformers qdrant-client \
            langchain-text-splitters tiktoken nltk pypdfium2 pdfplumber \
            python-docx requests numpy
```

Full list with pinned versions (recommended):

| Package | Purpose |
|---|---|
| `FlagEmbedding` | BGE-M3 dense + sparse + ColBERT encoding |
| `sentence-transformers` | Cross-encoder reranking (`bge-reranker-v2-m3`) |
| `qdrant-client` | Vector store interface |
| `tiktoken` | Token-count-based chunking and overlap |
| `pdfplumber` | Table extraction from PDFs |
| `pypdfium2` | PDF text extraction and bookmark parsing |
| `python-docx` | DOCX document support |
| `langchain-text-splitters` | Sentence-boundary-aware text splitting |
| `nltk` | Sentence tokenization (`punkt_tab`) |
| `numpy` | Vector math and L2-norm guard |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/autosar-rag-vector-db.git
cd autosar-rag-vector-db

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install FlagEmbedding sentence-transformers qdrant-client \
            langchain-text-splitters tiktoken nltk pypdfium2 pdfplumber \
            python-docx requests numpy

# 4. Download NLTK punkt tokenizer
python -c "import nltk; nltk.download('punkt_tab')"

# 5. Start Qdrant (Docker)
docker run -d -p 7333:6333 qdrant/qdrant
```

> **GPU note:** BGE-M3 encoding and cross-encoder reranking run significantly faster on GPU. The pipeline automatically detects CUDA via PyTorch.

---

## Configuration

All key constants are defined at the top of each script. Edit them before running.

### `Ingestion_BGE_M3.py`

```python
DATA_DIR         = "/path/to/your/autosar_docs"   # folder containing PDF/DOCX files
COLLECTION       = "Dear_autosar"                  # Qdrant collection name
QDRANT_URL       = "http://localhost:7333"
EMBEDDING_MODEL  = "BAAI/bge-m3"
EMBED_BATCH_SIZE = 8          # chunks per GPU forward pass (lower if OOM)
```

### `HybridRetriever_BGE_M3.py`

```python
COLLECTION       = "Dear_autosar"
QDRANT_URL       = "http://localhost:7333"
EMBEDDING_MODEL  = "BAAI/bge-m3"
RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"
RERANK_MODE      = "cross_encoder"  # or "colbert" to save VRAM
DECOMP_BASE_URL  = "http://localhost:8011/v1"
DECOMP_MODEL     = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TOP_K    = 10
```

### `Retrival_BGE_M3.py`

```python
MATCH_MODE                 = "fuzzy"   # or "embedding"
EMBEDDING_MATCH_THRESHOLD  = 0.70      # cosine threshold for embedding match mode
STORE_FULL_POOL            = False     # True: retrieve top_k * 3 to detect below_cutoff
```

---

## Usage

### 1. Ingestion

Ingest all PDF and DOCX files from your documents folder into Qdrant:

```bash
python Ingestion_BGE_M3.py
```

What this does:
1. Reads every PDF/DOCX from `DATA_DIR`
2. Extracts text (with pdfplumber for tables) and splits by structural bookmarks/headings
3. Creates **parent chunks** (1600 tokens) and **child chunks** (400 tokens, 10% overlap)
4. Calls the local LLM to generate an 80-token contextual prefix for each child chunk
5. Encodes enriched child chunks with BGE-M3 (dense + sparse + ColBERT)
6. Streams batches into Qdrant; child text and parent text both stored in payload

### 2. Retrieval

Use the `HybridRetriever` in your own code:

```python
from HybridRetriever_BGE_M3 import HybridRetriever

retriever = HybridRetriever(rerank_mode="cross_encoder")

# Single-hop search
results = retriever.search("What is the AUTOSAR Adaptive Platform memory model?", top_k=5)

for r in results:
    print(r.score, r.payload["parent_text"][:300])
```

Multi-hop queries are handled automatically — the retriever detects complex questions and applies IRCoT-style decomposition internally.

**CLI (evaluation mode):**

```bash
python Retrival_BGE_M3.py \
    --questions path/to/evaluation_questions.json \
    --top-k 10 \
    --match-mode fuzzy
```

With resume support for long runs:

```bash
python Retrival_BGE_M3.py \
    --questions evaluation_questions.json \
    --resume    output_v2/retrieval_results.jsonl \
    --top-k     10 \
    --store-full-pool
```

### 3. Evaluation

The evaluation script (`Retrival_BGE_M3.py`) reads a ground-truth JSON file and reports:

- `recall@k`, `precision@k`, `ndcg@k`, `all_found@k`, `mrr`
- Per-question failure taxonomy
- Latency statistics (mean, median, P95, P99)

**Ground-truth format** (rag-chunk style):

```json
{
  "questions": [
    {
      "id": "q001",
      "question": "What is the purpose of the AUTOSAR BSW scheduler?",
      "relevant": [
        "The BSW Scheduler is responsible for ...",
        "Each BSW module registers its main function ..."
      ]
    }
  ]
}
```

### 4. Chunking Benchmark

The `chunking-methods/runner.py` script benchmarks the pipeline across all combinations of documents, chunk sizes, and top-k values (9 × 3 × 3 = 81 runs) using parallel workers:

```bash
# Edit runner.py to set COMMANDS, CHUNK_SIZES, TOP_K_VALUES, OUTPUT_DIR, MAX_WORKERS
python chunking-methods/runner.py
```

After the run, combine all result `.txt` files:

```bash
python chunking-methods/combine_results.py
# Output: chunking-methods/result/combined_all_results.txt
```

### 5. Utility Scripts

**Split a multi-source evaluation JSON into per-document files:**

```bash
python chunking-methods/split_by_source.py Q_2.json ./ragchunk_questions/
```

**Convert `evidence_snippets` format to `relevant` format:**

```bash
python chunking-methods/queries_format_converter.py Q_2.json output.json
```

**Extract a subset of questions not present in an existing split:**

```bash
# Edit file paths inside subset_queries.py, then:
python subset_queries.py
```

---

## Evaluation Results

### v2 — QD+RR Chunk-Level (365 questions)

Full hybrid pipeline with multi-hop decomposition and cross-encoder reranking.

| Metric | Value |
|---|---|
| **Strict success rate** | **92.33%** (337 / 365) |
| **MRR** | **0.9378** |
| P@1 | 0.9123 ± 0.28 |
| P@3 | 0.8584 ± 0.27 |
| P@5 | 0.8263 ± 0.26 |
| P@10 | 0.7570 ± 0.27 |
| R@1 | 0.7082 |
| R@3 | 0.8589 |
| R@5 | 0.8986 |
| R@10 | 0.9315 |
| All@5 (strict multi-hop) | 82.74% |
| All@10 (strict multi-hop) | 87.95% |
| NDCG@10 | 2.2032 |

**Latency** (per question): mean 5363 ms · median 3566 ms · P95 11793 ms · P99 12829 ms

### v1 — Doc-Level Baseline (1000 questions)

Simpler doc-level pipeline (no multi-hop, no cross-encoder rerank).

| Metric | Value |
|---|---|
| **Strict success rate** | **100.0%** (1000 / 1000) |
| **MRR** | **0.8786** |
| R@10 | 1.0000 |
| NDCG@10 | 0.9081 |

**Latency** (per question): mean 190 ms · P95 216 ms · P99 232 ms

> The v1 baseline shows that doc-level retrieval is fast and achieves 100% recall at k=10, but chunk-level precision (NDCG, P@1) is lower. The v2 pipeline trades latency for dramatically better precision and handles complex multi-hop questions correctly.

---

## Design Decisions & Problem Log

The following 21 problems were identified and fixed during development (referenced as I1–I21 in code comments):

| ID | Problem | Fix |
|---|---|---|
| I1 | Character-count overlap caused splits mid-token | Token-count overlap via `tiktoken` |
| I2 | Chunk sizes in characters (inconsistent) | Chunk sizes in tokens (child=400, parent=1600) |
| I3 | Flat single-level chunking | Parent-Child: small child for retrieval, large parent for LLM |
| I4 | No context injection | LLM-generated 80-token prefix prepended before embedding |
| I5 | `[Page N]` tags polluted chunk text | Stripped before embedding; page stored in metadata only |
| I6 | Regex-only section splitting (fragile) | PDF bookmark tree as primary splitter, regex as fallback |
| I7 | Jaccard dedup split on spaces (breaks AUTOSAR identifiers) | AUTOSAR-aware tokenization in Jaccard set |
| I8 | External BM25 JSON (IDF drift, versioning) | BGE-M3 native sparse vectors replace external BM25 entirely |
| I9 | Hand-built TF formula for sparse weights | BGE-M3 learned sparse weights (`lexical_weights`) |
| I10 | Length-normalization bug in custom TF | Eliminated — BGE-M3 sparse is trained, not formula-based |
| I11 | BM25 JSON versioning mismatches | No BM25 JSON; problem category eliminated |
| I12 | Single-threaded embedding | `BGEM3FlagModel` with GPU-parallel batch encoding |
| I13 | Only dense vectors stored | Dense + sparse + ColBERT all stored and used |
| I14 | Dense vectors not normalized | `normalize_embeddings=True` equivalent (L2-normalised) |
| I15 | ColBERT multi-vectors not in Qdrant | `MultiVectorConfig(MAX_SIM)` collection config |
| I16 | Tables skipped or embedded as raw text | Tables converted to natural-language sentences |
| I17 | `pdfplumber` used only for counting tables | Used for full table content extraction |
| I18 | All chunks held in RAM before upload | Streaming batch upload |
| I19 | No embedding model version tracking | Model fingerprint written to collection metadata |
| I20 | Page markers in BM25 tokenization | Stripped before sparse encoding and dense embedding |
| I21 | Silent near-zero vector uploads | L2-norm guard: vectors below threshold logged and skipped |

---

## Failure Taxonomy

When a question fails, each missed context is classified into one of four buckets:

| Category | Meaning | v2 Share |
|---|---|---|
| `wrong_doc` | The correct document was never retrieved into the candidate pool | 90.6% |
| `wrong_chunk` | Correct document retrieved but not the specific chunk | 9.4% |
| `below_cutoff` | Chunk was in the extended pool but ranked below top-k | 0.0% |
| `unknown` | Cannot be determined | 0.0% |

The dominant failure mode (`wrong_doc`, 90.6%) indicates that the remaining errors are primarily a **cross-document routing** problem, not a reranking or cutoff problem. Future work: query expansion, document-level pre-filtering.

---

## Contributing

1. Fork the repository and create a feature branch
2. Keep the I1–I21 fix log convention — new bugs get the next available ID
3. Run the evaluation pipeline on at least a 100-question subset before opening a PR
4. Include updated `evaluation_results_summary.txt` with your changes

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built for AUTOSAR Adaptive Platform and Classic Platform specification retrieval. Tested against 1000+ questions spanning SOME/IP, Crypto, OS, COM, and diagnostics specifications.*
