# 📄 Docling PDF Processing Pipeline

> A standalone document-processing pipeline for converting technical PDF documents into structured Markdown and JSON outputs — designed for RAG and LLM ingestion workflows.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project provides an end-to-end pipeline for parsing large, structure-heavy PDFs — technical specs, research papers, enterprise documents — and transforming them into formats ready for downstream AI workflows.

**What it does:**

- Extracts text, metadata, and structure from PDFs using Docling
- Generates clean Markdown and hierarchical JSON representations
- Reconstructs parent-child document hierarchies for semantic chunking
- Organizes outputs into folder structures compatible with vector databases and RAG pipelines

---

## Project Structure

```
Docling/
├── granite_docling_pipeline.py    # Core PDF conversion pipeline
├── flat_to_tree.py                # Transforms flat extractions into tree hierarchies
├── fix_folder_rag-chunk.py        # Cleans and reorganizes chunk outputs
├── conversion_log.txt             # Conversion logs
└── results/
    └── Utilization of Crypto Services/
```

---

## Components

### `granite_docling_pipeline.py`
The main entry point. Reads PDF files, parses document structure, extracts text and metadata, and exports structured outputs ready for RAG pipelines.

### `flat_to_tree.py`
Converts flat extracted document structures into nested parent-child hierarchies. Useful for semantic chunking, context-aware retrieval, and hierarchical navigation.

### `fix_folder_rag-chunk.py`
A post-processing utility that cleans generated folders, reorganizes chunk outputs, and prepares directory structures for vector databases.

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/docling-pipeline.git
cd docling-pipeline

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Run the main pipeline

```bash
python granite_docling_pipeline.py
```

### 2. Convert flat structure to tree hierarchy

```bash
python flat_to_tree.py
```

### 3. Fix and organize chunk folders

```bash
python fix_folder_rag-chunk.py
```

---

## Workflow

```
PDF Documents
      │
      ▼
Docling Extraction Pipeline
      │
      ▼
Structured Markdown / JSON
      │
      ▼
Hierarchy Reconstruction
      │
      ▼
Chunk Organization
      │
      ▼
Vector Database / RAG Pipeline
```

---

## Output

Each run generates:

| Output | Description |
|--------|-------------|
| `.md` files | Clean Markdown per document/section |
| `.json` files | Structured, hierarchical document representation |
| Chunk folders | Organized chunks ready for vector DB ingestion |
| `conversion_log.txt` | Per-run processing logs |

---

## Use Cases

- **RAG pipelines** — feed chunked, structured content into retrieval systems
- **Technical document parsing** — handle spec-heavy or deeply nested PDFs
- **Knowledge base construction** — build structured corpora for LLM ingestion
- **Semantic chunking** — produce context-aware chunks preserving document hierarchy

---

## Roadmap

- [ ] Parallel document processing
- [ ] Native vector DB integration (Pinecone, Weaviate, Chroma)
- [ ] Improved metadata extraction
- [ ] Automatic semantic chunking strategies
- [ ] GPU acceleration support
- [ ] OCR enhancement pipeline

---

## Author

**Praveen Solanki**  
M.Tech CSE — IIT Mandi

---

## License

This project is licensed under the [MIT License](LICENSE).
