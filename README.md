# RAG Full Pipeline — AUTOSAR Specification Intelligence System

> **A complete, end-to-end Retrieval-Augmented Generation research system for AUTOSAR Classic Platform documentation.** Built at Bosch Global Software Technologies (BGSW), Bengaluru. Covers everything from PDF extraction and dataset construction to dual-paradigm RAG evaluation and answer generation.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![vLLM](https://img.shields.io/badge/backend-vLLM-blueviolet)
![GPU](https://img.shields.io/badge/hardware-2×48GB%20GPU-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents

- [Project Overview](#project-overview)
- [The Problem](#the-problem)
- [What This Repository Solves](#what-this-repository-solves)
- [Repository Structure](#repository-structure)
- [Full Pipeline Architecture](#full-pipeline-architecture)
- [Module 1 — Extraction Methods](#module-1--extraction-methods)
- [Module 2 — Data Generation](#module-2--data-generation)
- [Module 3 — Vector DB (Hybrid RAG)](#module-3--vector-db-hybrid-rag)
- [Module 4 — VectorLess DB (Vectorless RAG)](#module-4--vectorless-db-vectorless-rag)
- [Module 5 — Generation Pipeline](#module-5--generation-pipeline)
- [System Comparison](#system-comparison)
- [Hardware & Infrastructure](#hardware--infrastructure)
- [Getting Started](#getting-started)
- [Key Research Findings](#key-research-findings)
- [Technologies Used](#technologies-used)
- [Authors](#authors)

---

## Project Overview

This repository is the complete research codebase for evaluating Retrieval-Augmented Generation (RAG) on the **AUTOSAR Classic Platform specification corpus** — 103 official automotive ECU specification PDFs spanning 4,500+ pages of safety-critical engineering content.

The project was conducted at **Bosch Global Software Technologies (BGSW), Bengaluru** as an applied AI research initiative. It compares two fundamentally different RAG paradigms:

- **Hybrid Vector RAG** — embedding-based retrieval using BGE-M3 dense+sparse vectors, Qdrant, and Reciprocal Rank Fusion
- **Vectorless RAG** — LLM-native document navigation using hierarchical JSON tree indices and beam search, with no embedding model, no vector database, and no chunking pipeline

Both systems are evaluated against a purpose-built gold-standard benchmark of **validated AUTOSAR question-answer pairs**, built from scratch using a custom RAGAS-based pipeline with LLM-as-judge quality validation.

---

## The Problem

AUTOSAR engineers spend significant time manually navigating documentation to answer questions like:

- *"What are the configuration parameters for the ComM module that affect bus wake-up behaviour?"*
- *"Which SWS requirement IDs govern the DET error reporting interface?"*
- *"How does the NvM manager handle write-once blocks across different memory partitions?"*

Standard RAG systems fail on these queries for three core reasons:

**Structural complexity.** AUTOSAR documents have multi-level section hierarchies, requirement tables with specific identifiers, and cross-module references that fixed-size text chunking cannot preserve.

**Lexical precision requirements.** Requirement identifiers like `SWS_ComM_00107`, `ECUC_ComConfig`, and `ASIL-D` demand exact lexical matching that dense-only embeddings cannot provide.

**Multi-hop reasoning.** Many engineering questions require connecting information from multiple sections or modules simultaneously — a capability requiring explicit multi-hop retrieval strategies.

---

## What This Repository Solves

This codebase delivers three production-scale artefacts:

1. **AUTOSAR RAG Benchmark** — The first domain-specific RAG evaluation dataset for automotive ECU specifications: 1,000+ validated QA pairs with 60/20/20 train/dev/test split, covering 103 AUTOSAR PDFs.

2. **Hybrid Vector RAG System** — A production-quality RAG pipeline with section-wise PDF parsing, BGE-M3 hybrid embeddings, Qdrant vector storage, RRF fusion, and multi-hop query decomposition.

3. **Vectorless RAG System** — A novel LLM-native document navigation system using hierarchical tree indices. No embedding model. No vector database. No chunking. Full reasoning traces and exact page provenance.

---

## Repository Structure

```
RAG-Full-Pipeline/
│
├── Extraction-Methods/          # PDF parsing and document extraction
│   ├── Docling/                 # IBM Docling extraction pipeline
│   ├── Glm-OCR/                 # GLM-OCR multimodal extraction (0.9B VLM)
│   └── Nemotron-Parse/          # NVIDIA NIM Nemotron Parse pipeline
│
├── Data_generation/             # Gold-standard benchmark construction
│   ├── build_kg.py              # Stage 0: Knowledge graph from PDFs
│   ├── generate_candidates.py   # Stage A: QA candidate generation
│   ├── validate_candidates.py   # Stage B: LLM-as-judge validation
│   ├── finalize_dataset.py      # Stage C: Filtering, splits, dataset card
│   ├── llm_generation/          # Earlier RAGAS-based generation pipeline
│   └── output/                  # Generated dataset artefacts
│       └── stage_c_finalization/
│           └── dataset_card.md  # Final dataset statistics and metadata
│
├── Vector_DB/                   # Hybrid Vector RAG system
│   ├── Ingestion_BGE_M3.py      # PDF ingestion → Qdrant
│   ├── HybridRetriever_BGE_M3.py # Hybrid retrieval + multi-hop decomposition
│   ├── Retrival_BGE_M3.py       # Retrieval evaluation
│   └── chunking-methods/        # Chunking strategy benchmark
│
├── VectorLess_DB/               # Vectorless RAG system (PageIndex-based)
│   ├── pageindex/               # Core hierarchical tree library
│   ├── pageindex_RAG_simple*.py # Simple single-pass RAG variants
│   ├── level_rag.py             # LevelRAG: 4-stage query decomposition
│   ├── run_rag_v3_best.py         # Recommended evaluation runner
│   └── cookbook/                # Jupyter notebook tutorials
│
└── Generation-Pipeline/         # Comparative generation and evaluation
    ├── generate.py              # Stage B: Frozen generator over retrieved contexts
    ├── evaluate.py              # Stage C: Three-tier metric evaluation
    └── schemas.py               # Pydantic data contracts
```

---

## Full Pipeline Architecture

The system operates as a three-stage pipeline where both RAG systems share identical Stage B and Stage C implementations, ensuring that measured quality differences are attributable solely to retrieval strategy.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SOURCE DOCUMENTS                                                       │
│  103 AUTOSAR Classic Platform PDFs (4,500+ pages)                      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  EXTRACTION LAYER  (Extraction-Methods/)                                │
│  Docling / GLM-OCR / NVIDIA Nemotron Parse                              │
│  → Structured Markdown + Hierarchical JSON per document                │
└──────────────┬───────────────────────────────┬──────────────────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────────────────────┐
│  BENCHMARK CONSTRUCTION  │    │  DOCUMENT INDEXING                       │
│  (Data_generation/)      │    │                                          │
│                          │    │  Vector_DB/          VectorLess_DB/      │
│  Stage 0: KG Build       │    │  BGE-M3 hybrid       Hierarchical JSON   │
│  Stage A: QA Generation  │    │  embeddings +        tree per doc        │
│  Stage B: LLM Validation │    │  Qdrant storage      (TOC + summaries)   │
│  Stage C: Finalization   │    │                                          │
│                          │    └──────────────────────────────────────────┘
│  → 1,000+ QA pairs       │                   │
└──────────────────────────┘                   ▼
               │                ┌──────────────────────────────────────────┐
               │                │  STAGE A — RETRIEVAL                     │
               │                │                                          │
               └───────────────►│  BGE-M3 Hybrid Vector         PageIndex  │
                                │  dense+sparse → RRF → rerank  beam search│
                                │  → bgem3_retrieval.jsonl  → pageindex_   │
                                │                               retrieval. │
                                │                               jsonl      │
                                └──────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────────┐
                                │  STAGE B — GENERATION (generate.py)      │
                                │  FrozenRAGGenerator (temp=0, seed=42)    │
                                │  Qwen2.5-72B-Instruct-AWQ (vLLM)         │
                                │  → results.jsonl                         │
                                └──────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────────┐
                                │  STAGE C — EVALUATION (evaluate.py)      │
                                │  Tier 1: BLEU / ROUGE / BERTScore        │
                                │  Tier 2: RAGAS LLM-as-judge              │
                                │  Tier 3: RAGChecker claim diagnostics    │
                                │  → comparison_report.csv                 │
                                └──────────────────────────────────────────┘
```

---

## Module 1 — Extraction Methods

**Path:** `Extraction-Methods/`

This module evaluates and compares three state-of-the-art PDF parsing approaches for converting AUTOSAR PDFs into machine-readable structured formats suitable for RAG pipelines. The key insight from this evaluation: **the parser choice determines chunking quality, and chunking quality is the single largest performance lever in the entire RAG pipeline** — contributing a 20–25% retrieval accuracy gain over fixed-size chunking.

### Docling (`Extraction-Methods/Docling/`)

IBM Docling uses the RT-DETR layout model (DocLayNet) to extract full table structure from PDFs and convert them into hierarchical JSON and clean Markdown. The `flat_to_tree.py` utility reconstructs parent-child document hierarchies for semantic chunking.

Key scripts:

| Script | Purpose |
|---|---|
| `granite_docling_pipeline.py` | Main PDF → structured output pipeline |
| `flat_to_tree.py` | Flat extraction → semantic tree hierarchy |
| `fix_folder_rag-chunk.py` | Post-processing for vector DB ingestion |

### GLM-OCR (`Extraction-Methods/Glm-OCR/`)

GLM-OCR is a 0.9B-parameter multimodal OCR model (CogViT + GLM architecture) achieving 94.62 on OmniDocBench V1.5. It provides explicit heading-level classification (`heading1/2/3`) which maps directly to document hierarchy construction without heuristics. Deployable via vLLM, SGLang, or Ollama.

The `apps/` subdirectory contains a full-stack web application: a FastAPI backend with async task queuing and a React 19 frontend with virtual scrolling for large documents.

### NVIDIA Nemotron Parse (`Extraction-Methods/Nemotron-Parse/`)

NVIDIA NIM Nemotron Parse uses a VLM (C-RADIO + mBart, 885M parameters) to classify 13 semantic region types and detect section-wise boundaries from page images rendered at 300 DPI. **This parser produced the best chunking results in the evaluation** — section-wise chunking via Nemotron Parse achieved a Hit Rate @10 of 90.00% versus 69.47% for fixed-size 256-token chunking.

```
PDF → page images (300 DPI) → Nemotron Parse → 13-class region detection
     → section boundary detection → Markdown with structural metadata
     → build_hierarchy_llm.py → hierarchical JSON tree
```

### Extraction Comparison

| Parser | Architecture | Best Strength |
|---|---|---|
| NVIDIA Nemotron Parse | VLM (C-RADIO + mBart, 885M) | Section-wise boundary detection — best chunking performance |
| GLM-OCR | VLM (CogViT + GLM, 0.9B) | Explicit heading level classification |
| IBM Docling | RT-DETR (DocLayNet) | Full table structure reconstruction |

---

## Module 2 — Data Generation

**Path:** `Data_generation/`

No publicly available benchmark covers AUTOSAR or comparable automotive safety specifications. This module builds the first domain-specific RAG evaluation dataset for AUTOSAR from scratch: 1,000+ validated QA pairs across 103 PDFs with a rigorous four-stage pipeline.

### Pipeline Overview

```
Stage 0 (build_kg.py)
  103 AUTOSAR PDFs → boilerplate stripping → substance filtering
  → RAGAS Knowledge Graph (Qwen2.5-72B-AWQ)
  → knowledge_graph.json

Stage A (generate_candidates.py)
  KG nodes → four synthesizer types × four domain-expert personas
  → 3,000+ candidate QA pairs → candidates.jsonl

Stage B (validate_candidates.py)
  Candidates → Qwen3-30B-A3B-Instruct-2507 (LLM-as-judge)
  → scored on Answerability / Faithfulness / Relevance / Specificity
  → scored.jsonl

Stage C (finalize_dataset.py)
  Score filtering → deduplication → diversity cap (25% max per PDF)
  → 60/20/20 train/dev/test split → gold_v1.0.json + dataset_card.md
```

### Question Types

| Type | Target Share | Description |
|---|---|---|
| Single-Hop Specific | 35% | Factual question from a single document section |
| Single-Hop Abstract | 15% | Conceptual synthesis from a single section |
| Multi-Hop Specific | 25% | Specific fact requiring two document sections |
| Multi-Hop Abstract | 25% | Cross-section conceptual reasoning |

### Domain-Expert Personas

Generic RAGAS personas produced 35% low-quality queries in initial testing. Four AUTOSAR-specific personas replaced them:

- **AUTOSAR Integration Engineer** — BSW module integration, COM/CAN/Mode Management interfaces
- **Tier-1 Supplier Developer** — API contracts, runtime behaviour, SWS requirement satisfaction
- **Functional Safety Engineer** — ISO 26262 ASIL decomposition, fault detection, safety mechanisms
- **Application SW Engineer** — SWC/RTE interfaces, port configurations, mode switching

### Quality Thresholds (Stage B)

| Criterion | Type | Threshold |
|---|---|---|
| Answerability | Binary | == 1 |
| Question Specificity | Binary | == 1 |
| Faithfulness | Float | ≥ 0.85 |
| Answer Relevance | Float | ≥ 0.80 |

### Final Dataset Statistics

| Attribute | Value |
|---|---|
| Total validated QA pairs | 1,000+ (365 used for retrieval evaluation) |
| Train / Dev / Test | 60% / 20% / 20% |
| Source documents | 103 AUTOSAR PDFs |
| Max contribution per PDF | 25% (diversity cap) |
| Generator LLM | Qwen2.5-72B-Instruct-AWQ (local, on-premise) |
| Judge LLM | Qwen3-30B-A3B-Instruct-2507 (local, on-premise) |

---

## Module 3 — Vector DB (Hybrid RAG)

**Path:** `Vector_DB/`

A production-grade Hybrid Vector RAG system purpose-built for AUTOSAR documentation. The pipeline combines contextual parent-child chunking, BGE-M3 three-signal embeddings, and multi-hop query decomposition.

### Ingestion Pipeline

```
PDF/DOCX
  → pypdfium2 + pdfplumber (text + table extraction)
  → structural split by PDF bookmark tree (regex fallback)
  → parent chunks (1600 tokens) + child chunks (400 tokens, 10% overlap)
  → LLM contextual prefix per child chunk (80 tokens)
  → BGE-M3 encode → dense (1024-dim) + sparse (SPLADE-style) + ColBERT
  → Qdrant upsert: child indexed for precision, parent stored for LLM context
```

### BGE-M3 Three-Signal Embeddings

BGE-M3 generates all three vector types in a single model pass:

- **Dense vectors (1024-dim)** — semantic similarity for conceptual queries about module purpose and architecture
- **Sparse vectors (SPLADE-style)** — exact keyword matching, critical for AUTOSAR requirement ID queries (`SWS_ComM_00107`, `ECUC_ComConfig`)
- **ColBERT multi-vectors** — token-level matching for fine-grained retrieval over long chunks

### Retrieval

At query time:

1. BGE-M3 encodes the query to dense + sparse vectors in a single pass
2. Qdrant executes parallel ANN (cosine similarity) and BM25-style sparse search
3. Reciprocal Rank Fusion merges ranked lists: `score(d) = Σ 1/(k+rank(d))`, `k=60`
4. Optional cross-encoder reranker (`bge-reranker-v2-m3`) for higher top-1 precision
5. Multi-hop queries are decomposed by LLM into sub-queries using IRCoT-style iteration

### Chunking Benchmark Results

Seven chunking strategies were evaluated. This comparison drove the finding that chunking strategy dominates all other optimizations:

| Chunking Method | Hit Rate @10 | vs. Baseline |
|---|---|---|
| Fixed-size (256 tokens) | 69.47% | Baseline |
| Fixed-size (512 tokens) | 74.74% | +5–8% |
| **Section-wise (NIM Nemotron Parse)** | **90.00%** | **+20–25% ✓ Best** |

### 21 Tracked Engineering Issues (I1–I21)

Development tracked and resolved 21 specific failure modes in the pipeline, including: character-count vs. token-count overlap (I1–I2), flat vs. parent-child chunking (I3), lack of contextual enrichment (I4), external BM25 IDF drift eliminated by BGE-M3 native sparse (I8–I11), ColBERT multi-vectors added to Qdrant (I13, I15), table extraction with pdfplumber (I16–I17), and streaming batch upload to prevent RAM overflow (I18).

---

## Module 4 — VectorLess DB (Vectorless RAG)

**Path:** `VectorLess_DB/`

A novel LLM-native document navigation system built on top of PageIndex. Instead of embedding documents into float vectors and ranking by cosine similarity, it builds a structured table-of-contents tree from each document and lets an LLM navigate that tree to find relevant pages — the same way a human expert consults a specification.

**Design principle:** No embedding model. No vector database. No chunking. The LLM navigates the document structure hierarchically.

### Two-Phase Architecture

**Phase 1 — Index Construction (offline, once per document)**

Each AUTOSAR PDF is processed through five LLM calls to produce a hierarchical JSON tree — a deeply annotated table of contents:

- LLM-1: TOC detection (first 25 pages)
- LLM-2: TOC extraction (section titles and page numbers)
- LLM-3: TOC completeness verification
- LLM-4: Node summarization (1–3 sentences per node, run asynchronously)
- LLM-5: Document description (single sentence for multi-document routing)

Each tree node stores: `node_id`, `title`, `start_index`, `end_index`, `summary`, `children`.

**Phase 2 — Hierarchical Beam Search (per query)**

```
User Query
  → AUTOSAR domain injection (12 keyword-to-routing-hint rules)
  → Level 0: LLM selects relevant root nodes (title + summary only)
  → Conditional descent: expand children of selected nodes
  → Repeat until leaves or max_depth=4
  → O(1) flat index lookup → physical page ranges
  → Page text extraction → answer generation with citations
```

This design solves two concrete failure modes: token limit overflow from sending all 200+ nodes of a large document at once, and attention dilution over irrelevant nodes at deep tree levels.

### LevelRAG (4-Stage Decomposition)

For complex multi-hop queries, `level_rag.py` implements a four-stage pipeline:

| Stage | What Happens |
|---|---|
| Query Decomposition | LLM splits complex query into atomic sub-queries |
| Per-Sub-Query Retrieval | Independent tree search + page extraction per sub-query |
| Partial Answer Generation | LLM answers each sub-query from its own retrieved context |
| Final Synthesis | LLM combines partial answers into a final grounded response |

### Four Index Construction Methods

| Method | Input | Key Strength |
|---|---|---|
| LLM-Direct (primary) | PDF text (PyPDF2/PyMuPDF) | Highest accuracy; handles irregular formatting |
| NVIDIA Nemotron Parse | Page images (300 DPI) | Strong multi-column and complex layout handling |
| GLM-OCR | Page images | Direct heading-level mapping; no heuristics |
| IBM Docling | PDF native | Best table structure reconstruction |

---

## Module 5 — Generation Pipeline

**Path:** `Generation-Pipeline/`

A three-stage comparative evaluation framework that holds the generator constant across both RAG systems, ensuring any quality difference in final answers is attributable solely to retrieval strategy.

### Stage B — Generation (`generate.py`)

Both retrieval JSONL files are loaded and normalized into a unified `Context` format via `FrozenRAGGenerator`. The generation LLM uses `temperature=0.0` and `seed=42` for fully deterministic, reproducible outputs across all queries.

```
pageindex_retrieval.jsonl ──┐
                             ├──► FrozenRAGGenerator ──► results.jsonl
bgem3_retrieval.jsonl ──────┘    (Qwen2.5-72B, temp=0)
```

Results are written incrementally (line-by-line) so crashes lose nothing, and `--resume` skips already-completed pairs.

### Stage C — Evaluation (`evaluate.py`)

Three evaluation tiers of increasing cost and depth:

| Tier | Metrics | LLM Required |
|---|---|---|
| **Tier 1 — Surface** | BLEU-4, ROUGE-1, ROUGE-L, BERTScore F1 | No |
| **Tier 2 — RAGAS** | Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall, NoiseSensitivity, FactualCorrectness, SemanticSimilarity | Yes (judge LLM) |
| **Tier 3 — RAGChecker** | Claim-level diagnostics: retriever-side vs. generator-side failure attribution | Yes (judge LLM) |

A paired bootstrap comparison (10,000 resamples, 95% CI) identifies the winning system per metric. The winner is declared only if the confidence interval excludes zero; otherwise the result is a tie.

### Output Artefacts

| File | Contents |
|---|---|
| `results.jsonl` | Generated answers + retrieval metadata for both systems |
| `eval_out/tier1_surface.csv` | Per-query BLEU / ROUGE / BERTScore |
| `eval_out/tier2_ragas.csv` | Per-query RAGAS metric suite |
| `eval_out/comparison_report.csv` | Per-metric winner with 95% bootstrap CIs |

---

## System Comparison

### Retrieval Metrics

| Metric | Hybrid Vector RAG | Vectorless RAG |
|---|---|---|
| Mean Reciprocal Rank (MRR) | 0.8141 | 0.757 (LLM judge) |
| Recall@10 | 0.8479 | 0.751 (context recall) |
| NDCG@10 | 0.9488 | 0.736 (context precision) |
| Queries evaluated | 365 (294 successful) | 1,000 (993 successful) |
| Mean latency | 8,671.8 ms | 2–5 seconds |
| Infrastructure | Qdrant + GPU (BGE-M3) | LLM API only |
| Interpretability | Low (similarity scores) | High (full reasoning trace) |
| Hallucination-free rate | ~80% | 79.8% (798/1,000) |

### Multi-Hop Generation Results (Tier 1, n=352 paired queries)

On multi-hop queries, PageIndex (Vectorless) outperforms BGE-M3 (Hybrid) across all four surface metrics with statistical significance:

| Metric | PageIndex Advantage | 95% CI |
|---|---|---|
| BLEU-4 | +3.39 points | [+1.76, +5.02] ✓ Significant |
| ROUGE-L | +0.039 | [+0.016, +0.063] ✓ Significant |
| ROUGE-1 | +0.036 | [+0.005, +0.067] ✓ Significant |
| BERTScore F1 | +0.051 | [+0.011, +0.092] ✓ Significant |

### Trade-off Matrix

| Dimension | Hybrid Vector RAG | Vectorless RAG | Recommendation |
|---|---|---|---|
| Single-hop exact lookup | ✓ Higher MRR (0.8141) | Moderate | Hybrid preferred |
| Multi-hop cross-section reasoning | Requires decomposition | ✓ Natural via tree | Vectorless preferred |
| Exact requirement ID matching | ✓ Strong (BM25 sparse) | Summary-dependent | Hybrid preferred |
| Infrastructure cost | Qdrant + GPU required | ✓ LLM API only | Vectorless for low-infra |
| Interpretability / auditability | Opaque (similarity score) | ✓ Full reasoning trace | Vectorless for compliance |
| Corpus update cost | Re-chunk + re-embed all | ✓ Re-index changed docs only | Vectorless for dynamic corpora |
| Page provenance | Chunk metadata | ✓ Exact physical page # | Vectorless for citation |

### Recommended Production Strategy

Route queries by complexity:

- **Single-hop queries** (specific requirement lookup, exact ID search) → Hybrid Vector RAG
- **Multi-hop queries** (cross-section reasoning, system integration questions) → Vectorless RAG

---

## Hardware & Infrastructure

All models run on-premise with no external data exposure.

| Component | Specification |
|---|---|
| Hardware | 2 × 48 GB GPU nodes |
| LLM Inference | vLLM (OpenAI-compatible) |
| Primary LLM | Qwen2.5-72B-Instruct-AWQ |
| Judge LLM | Qwen3-30B-A3B-Instruct-2507 |
| Embedding Model | BAAI/bge-m3 |
| Vector Database | Qdrant (open-source) |
| Primary Parser | NVIDIA NIM Nemotron Parse |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A running vLLM server (or compatible OpenAI-compatible endpoint)
- Qdrant (for Vector DB module only)
- 2× 48 GB GPU nodes (recommended for full pipeline; smaller setups can use API endpoints)

### Installation

```bash
git clone https://github.com/<your-org>/RAG-Full-Pipeline.git
cd RAG-Full-Pipeline

python -m venv .venv
source .venv/bin/activate

# Install module-specific dependencies (see each module's README)
# Minimum for Vector DB:
pip install FlagEmbedding sentence-transformers qdrant-client \
            langchain-text-splitters tiktoken nltk pypdfium2 pdfplumber \
            python-docx requests numpy

# Minimum for VectorLess DB:
pip install openai litellm PyPDF2 pymupdf python-dotenv pyyaml

# For Data Generation:
pip install -r Data_generation/requirements.txt

# For Generation Pipeline:
pip install pydantic openai numpy pandas evaluate sacrebleu \
            rouge-score bert-score langchain-openai langchain-huggingface ragas
```

### Quickstart — Vectorless RAG (no GPU required for inference)

```bash
# 1. Index a PDF
python VectorLess_DB/run_pageindex.py --pdf path/to/autosar_doc.pdf --output_dir ./indexed/

# 2. Run a query
python VectorLess_DB/run_rag_v3_best.py \
    --mode infer \
    --query "What are the configuration parameters for ComM bus wake-up?" \
    --tree_dir ./indexed/ \
    --pdf_dir ./pdfs/

# 3. Or run the full evaluation pipeline
python VectorLess_DB/run_rag_v3_best.py \
    --mode eval \
    --query ./Data_generation/output/stage_c_finalization/gold_v1.0_test.json \
    --tree_dir ./indexed/ \
    --pdf_dir ./pdfs/ \
    --output_dir ./results/
```

### Quickstart — Hybrid Vector RAG

```bash
# 1. Start Qdrant and vLLM
docker run -d -p 7333:6333 qdrant/qdrant
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --port 8011 --tensor-parallel-size 2

# 2. Ingest documents
python Vector_DB/Ingestion_BGE_M3.py  # configure DATA_DIR at top of file

# 3. Run retrieval evaluation
python Vector_DB/Retrival_BGE_M3.py \
    --questions ./Data_generation/output/stage_c_finalization/gold_v1.0_test.json \
    --top-k 10
```

### Build the Benchmark Dataset

```bash
# Start vLLM with the generator model
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 2 --port 8011

# Stage 0: Build knowledge graph
python Data_generation/build_kg.py --pdf-dir ./autosar_pdfs --output-dir ./output

# Stage A: Generate candidates
python Data_generation/generate_candidates.py \
    --kg-file ./output/kg/knowledge_graph.json --target 1000

# Stage B: Validate with judge
python Data_generation/validate_candidates.py --output-dir ./output

# Stage C: Finalize dataset
python Data_generation/finalize_dataset.py --output-dir ./output --target 1000
```

---

## Key Research Findings

**Chunking strategy is the dominant performance factor.** Section-wise chunking via NVIDIA NIM Nemotron Parse delivered a 20–25% improvement in retrieval accuracy over fixed-size chunking (Hit Rate @10: 90.00% vs. 69.47%). This gain exceeds that of any embedding model or retrieval strategy optimization.

**Hybrid retrieval is a functional requirement for technical domains.** The combination of dense (semantic) and sparse (BM25-style exact-match) retrieval is mandatory for AUTOSAR, where queries frequently contain requirement identifiers that dense-only retrievers cannot match.

**Vectorless RAG is a viable infrastructure-light alternative.** Achieving 75.7% accuracy with no vector database or embedding infrastructure, the Vectorless system offers interpretable, auditable retrieval traces with minimal ML infrastructure overhead.

**PageIndex outperforms Hybrid Vector on multi-hop queries.** The architectural advantage of hierarchical tree navigation — retrieving entire page ranges that preserve surrounding section context — makes cross-section dependencies explicit for the generator. All four Tier 1 generation metrics favour PageIndex on multi-hop queries with statistical significance (95% CI entirely below zero).

**Domain-expert personas are critical for benchmark quality.** Replacing generic auto-generated personas with four AUTOSAR domain-expert personas eliminated 35% of low-quality single-hop noise in the initial evaluation dataset.

---

## Technologies Used

| Component | Technology |
|---|---|
| LLM Inference Server | vLLM (local on-premise) |
| Primary LLM | Qwen2.5-72B-Instruct-AWQ |
| Judge LLM | Qwen3-30B-A3B-Instruct-2507 |
| Embedding Model | BAAI/bge-m3 (dense + sparse + ColBERT) |
| Vector Database | Qdrant (open-source) |
| Primary PDF Parser | NVIDIA NIM Nemotron Parse |
| Alternative Parser 1 | GLM-OCR (0.9B VLM) |
| Alternative Parser 2 | IBM Docling / MinerU |
| RAG Evaluation Framework | RAGAS (modified) |
| Chunking Evaluation | rag-chunk library |
| Tree Navigation Library | PageIndex (VectifyAI, extended) |
| LevelRAG | 4-stage decomposition-synthesis pipeline |
| Generation Metrics | BLEU, ROUGE, BERTScore, RAGAS, RAGChecker |
| Statistical Analysis | Paired bootstrap (10,000 resamples, 95% CI) |

---

## Authors

**Praveen Solanki (OLJ3KOR)**  
**Priya Kumari (MPQ3KOR)**  

Organization: Bosch Global Software Technologies Pvt Ltd, Bengaluru  
Domain: Automotive Software — AUTOSAR Classic Platform  
Report Date: April 2026  

---

## Module READMEs

Each module contains its own detailed README with CLI reference, configuration options, and usage examples:

| Module | Documentation |
|---|---|
| Extraction Methods (overview) | `Extraction-Methods/README.md` |
| Docling | `Extraction-Methods/Docling/README.md` |
| GLM-OCR | `Extraction-Methods/Glm-OCR/README.md` |
| Nemotron Parse | `Extraction-Methods/Nemotron-Parse/README.md` |
| Data Generation | `Data_generation/README.md` |
| LLM Generation (earlier pipeline) | `Data_generation/llm_generation/README.md` |
| Vector DB (Hybrid RAG) | `Vector_DB/README.md` |
| VectorLess DB | `VectorLess_DB/README.md` |
| Generation Pipeline | `Generation-Pipeline/README.md` |

---

<div align="center">
  <sub>Built on AUTOSAR Classic Platform specs · No vectors harmed in the Vectorless RAG · Internal — Confidential — BGSW Bengaluru 2026</sub>
</div>
