# VectorLess_DB

> **Vector-free, reasoning-native RAG for structured technical documents — no embeddings, no chunking, no vector database.**

VectorLess_DB is a research and production pipeline built on top of **[PageIndex](https://github.com/VectifyAI/PageIndex)** that replaces traditional embedding-based retrieval with LLM-powered hierarchical tree search. Instead of converting documents into float vectors and measuring cosine similarity, it builds a structured table-of-contents tree from each document and lets a language model *navigate* that tree to find relevant pages — the same way a human expert would.

The project was evaluated on a large corpus of **AUTOSAR** specification documents (50+ PDFs) and includes two complete RAG pipeline variants: a standard single-pass pipeline and a faithful **LevelRAG** four-stage implementation.

---

## ✨ Key Features

- **Zero vectors** — no embedding model, no FAISS/Chroma/Pinecone required
- **No chunking** — retrieval is page-granular, not chunk-granular
- **Hierarchical tree search** — LLM navigates a document outline to pick relevant nodes, then resolves exact pages programmatically
- **LevelRAG support** — decomposes complex queries into atomic sub-queries, retrieves independently, synthesises a final answer
- **Multi-backend** — supports OpenAI, NVIDIA NIM, vLLM (local), and Ollama
- **Structured evaluation** — context recall, context precision, LLM-as-judge correctness/completeness, hallucination classification
- **Batch runner** — `run_all.sh` loops over an entire folder of pre-indexed trees
- **Metrics aggregation** — `combine_results_structred_output.py` merges `metrics_summary.json` files into a single comparison CSV

---

## 📁 Repository Structure

```
VectorLess_DB/
│
├── pageindex/                        # Core PageIndex library (forked/extended)
│   ├── __init__.py                   # Public API exports
│   ├── client.py                     # PageIndexClient: index() + retrieve()
│   ├── config.yaml                   # Provider & model config (vllm / nvidia / openai)
│   ├── page_index.py                 # PDF → hierarchical tree builder
│   ├── page_index_md.py              # Markdown → tree builder
│   ├── retrieve.py                   # get_document / get_document_structure / get_page_content
│   └── utils.py                      # LiteLLM wrapper, ConfigLoader, helpers
│
├── level_rag.py                      # LevelRAG: 4-stage query decomposition pipeline
├── pageindex_RAG_simple.py           # Simple single-pass RAG (main/reference version)
├── pageindex_RAG_simple_vllm.py      # vLLM-optimised simple RAG
├── pageindex_RAG_simple_vllm_v2.py   # vLLM v2 (async concurrency improvements)
├── pageindex_RAG_simple_vllm_v3.py   # vLLM v3 (latest, best concurrency)
├── Pageindex_retrival.py             # CLI entrypoint used by run_all.sh
├── run_rag.py                        # RAG pipeline v1
├── run_rag_v2.py                     # RAG pipeline v2 (structure-key bug fixed)
├── run_rag_v3_best.py                  # RAG pipeline v3 (PageIndex-aligned, recommended)
├── run_pageindex.py                  # Standalone indexing runner
├── combine_results_structred_output.py # Metrics aggregator → CSV
├── run_all.sh                        # Batch evaluation script
│
├── cookbook/                         # Jupyter notebook tutorials
│   ├── README.md
│   ├── pageIndex_chat_quickstart.ipynb
│   ├── pageindex_RAG_simple.ipynb    # Minimal hands-on RAG example
│   ├── agentic_retrieval.ipynb       # Agent-based vectorless RAG
│   └── vision_RAG_pageindex.ipynb    # Vision RAG (no OCR, page images)
│
├── logs/                             # Per-run LLM call logs (JSON, one file per PDF per run)
├── Retrival_result/                  # Evaluation results — run 1 (365 questions)
│   ├── results.json
│   └── metrics_summary.json
├── Retrival_result_2/                # Evaluation results — run 2 (300 questions)
│   ├── results.json
│   └── metrics_summary.json
│
└── .env                              # API keys (never commit — add to .gitignore)
```

---

## 🏗 Architecture

### How It Works

```
PDF Document
    │
    ▼
[page_index.py]  — LLM reads the table of contents and builds a JSON tree
    │               Each node: {title, summary, page_range, children}
    ▼
Hierarchical Tree (stored as _structure.json)
    │
    ▼  User Query
[retrieve.py / tree_search()]  — LLM selects relevant node IDs from the tree
    │
    ▼
[resolve_nodes()]  — Page ranges looked up programmatically (no hallucination)
    │
    ▼
[get_page_content()]  — Raw page text extracted (PDF or Markdown)
    │
    ▼
[Answer Generation]  — LLM answers the question with grounded page context
    │
    ▼
[LLM Judge]  — Evaluates correctness, completeness, hallucination
```

### LevelRAG Pipeline (4 Stages)

| Stage | Name | What happens |
|-------|------|--------------|
| 1 | **Query Decomposition** | LLM splits complex query → atomic sub-queries |
| 2 | **Per-Sub-Query Retrieval** | Independent tree search + page extraction for each sub-query |
| 3 | **Partial Answer Generation** | LLM answers each sub-query using only its own retrieved context |
| 4 | **Final Synthesis** | LLM combines all partial answers → one final answer for the original query |

---

## 📊 Evaluation Results

Evaluated on 50+ AUTOSAR specification PDFs with domain-specific QA pairs.

### Run 1 — `Retrival_result/`

| Metric | Value |
|--------|-------|
| Total questions | 365 |
| Successful | 352 (96.4%) |
| **Avg Context Recall** | **97.02%** |
| **Avg Context Precision** | **92.80%** |
| Accuracy (binary) | 64.38% |
| Avg Correctness Score | 77.16% |
| Avg Completeness Score | 75.27% |
| Hallucination — None | 73 |
| Hallucination — Minor | 220 |
| Hallucination — Major | 59 |

### Run 2 — `Retrival_result_2/`

| Metric | Value |
|--------|-------|
| Total questions | 300 |
| Successful | 269 (89.7%) |
| **Avg Context Recall** | **84.76%** |
| **Avg Context Precision** | **83.19%** |
| Accuracy (binary) | 53.33% |
| Avg Correctness Score | 70.80% |
| Avg Completeness Score | 65.50% |
| Hallucination — None | 131 |
| Hallucination — Minor | 79 |
| Hallucination — Major | 59 |

> Context recall and precision are computed without LLM involvement (page-overlap vs gold `page_reference`). Correctness and completeness use an LLM judge.

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
git clone https://github.com/<your-username>/VectorLess_DB.git
cd VectorLess_DB

pip install -r requirements.txt   # or install manually (see Dependencies below)
```

### Dependencies

```bash
pip install openai litellm PyPDF2 pymupdf python-dotenv pyyaml
```

For Jupyter notebooks:
```bash
pip install jupyter ipykernel
```

### Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```ini
# NVIDIA NIM (used by default in pageindex/config.yaml)
NVIDIA_API_KEY=nvapi-...
NVIDIA_API_KEY_LARGE=nvapi-...
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1

# Or standard OpenAI
# OPENAI_API_KEY=sk-...
```

Edit `pageindex/config.yaml` to select your provider:

```yaml
provider: "nvidia"          # openai | nvidia | vllm
api_base: ""                # leave empty for openai/nvidia defaults
model: ""                   # empty = provider default
async_concurrency: 10       # parallel LLM calls
max_output_tokens: 10000
```

---

## 🔧 Usage

### 1. Index a PDF

```python
from pageindex import PageIndexClient

client = PageIndexClient(
    api_key="your-openai-key",
    model="gpt-4o",
    workspace="./my_index"
)

doc_id = client.index("path/to/document.pdf")
print(f"Indexed as: {doc_id}")
```

### 2. Retrieve relevant pages

```python
structure = client.get_document_structure(doc_id, query="What is the SOME/IP protocol?")
pages     = client.get_page_content(doc_id, pages="5-7")
```

### 3. Run the standard RAG pipeline (eval mode)

```bash
python run_rag_v3_best.py \
    --mode eval \
    --query /path/to/questions.json \
    --pdf_dir /path/to/pdfs/ \
    --tree_dir /path/to/indexed_trees/ \
    --output_dir ./results/
```

### 4. Run LevelRAG (4-stage decomposition)

```bash
python level_rag.py \
    --mode eval \
    --query /path/to/questions.json \
    --pdf_dir /path/to/pdfs/ \
    --tree_dir /path/to/indexed_trees/ \
    --provider nvidia \
    --model meta/llama-3.1-70b-instruct \
    --use_level_rag \
    --level_rag_max_subqueries 4 \
    --output_dir ./results_levelrag/
```

### 5. Batch evaluation over multiple indexed folders

```bash
# Edit STRUCTURED_DIR, PDF_DIR, QUERY, PROVIDER, MODEL in run_all.sh first
bash run_all.sh
```

### 6. Aggregate metrics from multiple runs

```bash
python combine_results_structred_output.py \
    --root /path/to/Structured_files_outputs
# → combined_metrics.json + metrics_comparison.csv
```

---

## 📋 Question File Format

The pipeline expects a JSON file with this schema:

```json
[
  {
    "id": "doc001_q001",
    "question": "What is the SOME/IP service discovery protocol?",
    "source_document": "AUTOSAR_PRS_SOMEIPServiceDiscoveryProtocol.pdf",
    "answer": "Ground truth answer ...",
    "evidence_snippets": ["snippet 1", "snippet 2"],
    "page_reference": "Pages 5-6",
    "difficulty": "medium",
    "question_type": "factual"
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✅ | Unique question identifier |
| `question` | ✅ | Natural language query |
| `source_document` | ✅ | PDF filename (must exist in `pdf_dir`) |
| `answer` | Eval only | Ground truth reference answer |
| `evidence_snippets` | Eval only | Gold evidence passages for judge |
| `page_reference` | Eval only | Gold page range, e.g. `"Pages 5-7"` |
| `difficulty` | ❌ | Metadata — passed through to output |
| `question_type` | ❌ | Metadata — passed through to output |

---

## 🧪 Cookbooks

Interactive Jupyter notebooks in the `cookbook/` folder:

| Notebook | Description |
|----------|-------------|
| `pageindex_RAG_simple.ipynb` | Minimal hands-on RAG — zero vectors, zero chunking |
| `agentic_retrieval.ipynb` | Agent-based vectorless RAG with tool-use |
| `vision_RAG_pageindex.ipynb` | Vision RAG — reasons over page images, no OCR needed |
| `pageIndex_chat_quickstart.ipynb` | Interactive chat over a single document |

---

## 🔌 Supported Backends

| Provider | Config value | Notes |
|----------|-------------|-------|
| OpenAI | `openai` | GPT-4o, GPT-4.1, etc. |
| NVIDIA NIM | `nvidia` | Llama 3.1 70B, Kimi-K2, Qwen, etc. — via `integrate.api.nvidia.com` |
| vLLM (local) | `vllm` | Any model served with `vllm serve`; set `api_base: http://localhost:8000/v1` |
| Ollama (local) | `ollama` | Set `OLLAMA_BASE_URL` and model name in the script |

---

## 📂 Output Files

After a pipeline run, the output directory contains:

```
output_dir/
├── step1_decomposition.json      # LevelRAG: sub-queries per question
├── step2_retrieval.json          # LevelRAG: nodes + pages per sub-query
├── step3_partial_answers.json    # LevelRAG: partial answers per sub-query
├── step4_final_answers.json      # LevelRAG: synthesised final answers
├── results.json                  # Full pipeline output (eval mode)
├── metrics_summary.json          # Aggregated evaluation metrics
├── infer_results.json            # Full output (infer mode)
│
│   Live scratch files (one JSON-lines entry per question):
├── step1_decomposition.jsonl
├── step2_retrieval.jsonl
├── step3_partial_answers.jsonl
└── step4_final_answers.jsonl
```

---

## ⚙️ Configuration Reference (`pageindex/config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `provider` | `vllm` | LLM backend (`openai` / `nvidia` / `vllm`) |
| `api_base` | `""` | Override API base URL |
| `api_key` | `""` | Override API key (falls back to env var) |
| `model` | `""` | Model name (empty = provider default) |
| `retrieve_model` | `""` | Separate model for tree-search step (defaults to `model`) |
| `toc_check_page_num` | `25` | Pages scanned when building the table-of-contents tree |
| `max_page_num_each_node` | `10` | Max pages per tree node |
| `max_token_num_each_node` | `12000` | Token budget per node context window |
| `if_add_node_id` | `yes` | Include node IDs in tree output |
| `if_add_node_summary` | `yes` | Include per-node LLM-generated summaries |
| `if_add_doc_description` | `no` | Include whole-document description |
| `if_add_node_text` | `no` | Include raw text in tree nodes |
| `max_output_tokens` | `10000` | Max tokens for LLM responses |
| `async_concurrency` | `20` | Max concurrent async LLM calls |

---

## 🔒 Security Note

The `.env` file contains real API keys. **Add it to `.gitignore` before pushing:**

```bash
echo ".env" >> .gitignore
```

Use environment variables or a secrets manager in production.

---

## 🗺 Roadmap

- [ ] `requirements.txt` with pinned dependencies
- [ ] Docker / devcontainer setup
- [ ] REST API wrapper (`FastAPI`)
- [ ] Web UI for interactive document QA
- [ ] Support for `.docx` and `.pptx` indexing
- [ ] Automated `.gitignore` and CI workflow

---

## 📚 References

- [PageIndex — VectifyAI](https://github.com/VectifyAI/PageIndex): the upstream library this project extends
- [LevelRAG paper](https://arxiv.org/abs/2505.01283): the 4-stage decomposition-synthesis RAG method implemented in `level_rag.py`
- [AUTOSAR Specification Documents](https://www.autosar.org/standards): the document corpus used in evaluation

---

## 📄 License

This project is released for research and educational purposes. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ on top of <a href="https://github.com/VectifyAI/PageIndex">PageIndex</a> · Evaluated on AUTOSAR specs · No vectors harmed in this RAG</sub>
</div>
