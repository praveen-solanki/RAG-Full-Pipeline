# 📚 Extraction-Methods

<div align="center">

# Advanced Document Extraction, OCR, Parsing, and RAG Preparation Framework

### A Comprehensive Research Repository for Structured Document Understanding

</div>

---

# 📖 Introduction

`Extraction-Methods` is a large-scale research and experimentation repository focused on evaluating, comparing, and developing multiple document extraction pipelines for modern AI systems, especially Retrieval-Augmented Generation (RAG), semantic search, enterprise knowledge systems, OCR-heavy workflows, and structured document understanding.

The repository contains multiple independent yet complementary extraction frameworks that transform complex documents such as PDFs, scanned files, technical manuals, research papers, AUTOSAR documentation, and enterprise reports into machine-readable structured formats.

The primary objective of this repository is to solve major challenges in:

- PDF parsing
- OCR extraction
- Layout understanding
- Semantic hierarchy reconstruction
- Markdown generation
- Structured JSON generation
- Chunking for RAG pipelines
- Enterprise-scale ingestion systems
- Vector database preparation
- Knowledge graph preprocessing
- Technical document understanding

The repository includes multiple state-of-the-art approaches and frameworks including:

- Docling-based extraction pipelines
- GLM-OCR multimodal extraction
- NVIDIA Nemotron Parse pipelines
- Hierarchical document reconstruction systems
- OCR evaluation frameworks
- Markdown cleanup systems
- Chunk organization pipelines
- LLM-assisted hierarchy generation

---

# 🎯 Core Goals of the Repository

The repository was designed to investigate and solve several important problems in document intelligence systems.

## Main Research Objectives

### 1. Accurate PDF Parsing

Extract high-quality structured content from:

- Research papers
- Technical manuals
- AUTOSAR specifications
- Enterprise documentation
- Multi-column PDFs
- Scanned documents
- Image-heavy documents
- Tables and structured layouts

---

### 2. Hierarchical Document Reconstruction

Rebuild semantic document structure such as:

- Headings
- Subheadings
- Nested sections
- TOC structures
- Lists
- Tables
- Semantic parent-child relationships

---

### 3. OCR and Layout Understanding

Improve extraction quality from scanned and complex documents by using:

- OCR models
- Layout-aware parsing
- Multimodal extraction
- Vision-language models

---

### 4. RAG-Oriented Preprocessing

Prepare extracted content for:

- Vector databases
- Dense retrieval
- Sparse retrieval
- Hybrid retrieval
- Semantic chunking
- Long-context retrieval systems

---

### 5. Benchmarking and Evaluation

Provide utilities for evaluating:

- OCR overlap quality
- Markdown cleanliness
- Structural preservation
- Semantic consistency
- Hierarchical correctness
- Extraction completeness

---

# 🧠 Repository Philosophy

This repository follows a modular experimentation-oriented design.

Each extraction method is largely independent and can be:

- Used standalone
- Benchmarked independently
- Integrated into larger RAG systems
- Combined with custom chunking pipelines
- Extended with additional parsers or OCR systems

The repository is intentionally designed for research flexibility rather than a single monolithic production application.

---

# 🗂 Repository Structure

```text
Extraction-Methods/
│
├── Docling/
│   ├── granite_docling_pipeline.py
│   ├── granite_docling_pipeline_with_llm_summary.py
│   ├── pdf_to_md.py
│   ├── pdf_to_md_parallel.py
│   ├── flat_to_tree.py
│   ├── fix_folder_rag-chunk.py
│   ├── only_llm.py
│   ├── setup_MinerU.sh
│   ├── README.md
│   └── additional utilities...
│
├── Glm-OCR/
│   ├── run_glmocr_images.py
│   ├── run_glmocr_pdf_pages.py
│   ├── eval_backend.py
│   ├── eval_glmocr_overlap.py
│   ├── merge_all_docs.py
│   ├── pyproject.toml
│   ├── setup.py
│   ├── README.md
│   ├── README_zh.md
│   └── additional utilities...
│
├── Nemotron-Parse/
│   ├── nemotron_parse_pipeline.py
│   ├── nemotron_parse_pipeline_skelaton_only.py
│   ├── build_hierarchy.py
│   ├── build_hierarchy_llm.py
│   ├── PyMuPdf_structre_extraction.py
│   ├── folder_copy.py
│   ├── rename.py
│   ├── README.md
│   └── additional utilities...
│
├── Structured_files_outputs/
│   ├── combined_metrics.json
│   ├── metrics_comparison.csv
│   ├── combined_output_table
│   └── evaluation outputs...
│
└── Additional supporting files...
```

---

# 🔍 Major Components

---

# 1️⃣ Docling Extraction Framework

📂 Folder: `Docling/`

The Docling pipeline is one of the central document extraction systems in this repository.

It focuses on converting PDFs into structured machine-readable representations suitable for downstream AI systems.

---

## 🔹 Main Features

### PDF to Markdown Conversion

Converts PDF documents into:

- Clean markdown
- Structured markdown
- RAG-ready text
- Hierarchical content

---

### Hierarchy Reconstruction

Rebuilds:

- Parent-child section relationships
- TOC structures
- Nested headings
- Structured semantic trees

---

### LLM-Assisted Summarization

Some pipelines integrate LLMs for:

- Content summarization
- Section summarization
- Context reduction
- Semantic enrichment

---

### Parallel Processing

Supports scalable processing using:

- Multiprocessing
- Batch document conversion
- Parallel extraction

---

## 🔹 Important Scripts

| Script | Description |
|---|---|
| `granite_docling_pipeline.py` | Main document extraction pipeline |
| `granite_docling_pipeline_with_llm_summary.py` | Extraction + LLM summarization |
| `pdf_to_md.py` | Basic PDF → Markdown conversion |
| `pdf_to_md_parallel.py` | Parallelized extraction |
| `flat_to_tree.py` | Converts flat outputs into semantic trees |
| `fix_folder_rag-chunk.py` | Organizes chunk outputs |
| `only_llm.py` | LLM-only semantic processing |
| `setup_MinerU.sh` | Environment/setup utility |

---

## 🔹 Typical Workflow

```text
PDF
 ↓
Docling Parsing
 ↓
Markdown Generation
 ↓
Hierarchy Reconstruction
 ↓
Chunk Structuring
 ↓
RAG Ingestion
```

---

## 🔹 Main Use Cases

- Technical PDF parsing
- Research paper extraction
- Semantic chunk generation
- Markdown generation
- RAG preprocessing
- Enterprise document ingestion

---

## 🔹 Documentation

Detailed documentation already exists inside:

```text
Docling/README.md
```

---

# 2️⃣ GLM-OCR Framework

📂 Folder: `Glm-OCR/`

GLM-OCR is a multimodal OCR framework optimized for complex document understanding.

It is particularly useful for:

- Scanned PDFs
- Technical documents
- Layout-heavy files
- Tables
- Formula extraction
- OCR-intensive workflows

---

# 🔹 Key Features

## Layout-Aware OCR

The system preserves:

- Layout information
- Reading order
- Structural organization
- Text positioning

---

## Formula Recognition

Supports extraction of:

- Mathematical expressions
- Scientific notation
- Technical equations

---

## Table Understanding

Improves extraction quality for:

- Tables
- Grid structures
- Technical tabular data

---

## Multi-Page PDF Processing

Supports:

- Batch OCR
- Page-by-page processing
- Large document extraction

---

# 🔹 Important Scripts

| Script | Description |
|---|---|
| `run_glmocr_images.py` | OCR extraction from images |
| `run_glmocr_pdf_pages.py` | OCR extraction from PDF pages |
| `eval_backend.py` | Evaluation backend |
| `eval_glmocr_overlap.py` | OCR overlap evaluation |
| `merge_all_docs.py` | Merge extracted outputs |

---

# 🔹 Evaluation Capabilities

The framework evaluates:

- OCR overlap
- Structural consistency
- Extraction completeness
- Content preservation

---

# 🔹 Typical Workflow

```text
PDF / Images
 ↓
Page Rendering
 ↓
OCR Extraction
 ↓
Layout Understanding
 ↓
Structured Output Generation
 ↓
Evaluation
```

---

# 🔹 Documentation

Detailed documentation already exists inside:

```text
Glm-OCR/README.md
```

Chinese documentation:

```text
Glm-OCR/README_zh.md
```

---

# 3️⃣ NVIDIA Nemotron Parse Framework

📂 Folder: `Nemotron-Parse/`

Nemotron Parse is a structured parsing pipeline focused heavily on semantic hierarchy extraction and document organization.

This framework is especially useful for:

- TOC reconstruction
- Semantic structure extraction
- JSON hierarchy generation
- Structured markdown generation

---

# 🔹 Main Features

## Structured Extraction

Extracts:

- Headings
- Sections
- Subsections
- Document skeletons
- Structured text blocks

---

## Hierarchy Reconstruction

Builds:

- Semantic trees
- Nested document structures
- Parent-child relationships

---

## LLM-Assisted Structure Generation

Some utilities leverage LLMs for:

- Hierarchy correction
- Semantic grouping
- Structure refinement

---

# 🔹 Important Scripts

| Script | Description |
|---|---|
| `nemotron_parse_pipeline.py` | Main extraction pipeline |
| `nemotron_parse_pipeline_skelaton_only.py` | Skeleton-only extraction |
| `build_hierarchy.py` | Local hierarchy reconstruction |
| `build_hierarchy_llm.py` | LLM-based hierarchy reconstruction |
| `PyMuPdf_structre_extraction.py` | PyMuPDF-based extraction |
| `folder_copy.py` | Folder organization utility |
| `rename.py` | Batch renaming utility |

---

# 🔹 Typical Workflow

```text
PDF
 ↓
Structural Parsing
 ↓
Skeleton Extraction
 ↓
Hierarchy Reconstruction
 ↓
Markdown / JSON Generation
 ↓
RAG Preparation
```

---

# 🔹 Documentation

Detailed documentation already exists inside:

```text
Nemotron-Parse/README.md
```

---

# 📊 Evaluation & Benchmark Outputs

📂 Folder: `Structured_files_outputs/`

This directory contains evaluation outputs and benchmark results for comparing different extraction systems.

---

# 🔹 Available Outputs

| File | Description |
|---|---|
| `combined_metrics.json` | Aggregated evaluation metrics |
| `metrics_comparison.csv` | CSV comparison of methods |
| `combined_output_table` | Consolidated structured outputs |

---

# 🔹 Evaluation Metrics

The repository supports evaluation of:

- OCR quality
- Markdown cleanliness
- Structural consistency
- Hierarchy preservation
- Semantic continuity
- Chunk quality
- Extraction completeness

---

# ⚙️ Installation

---

# 🔹 Clone Repository

```bash
git clone <your-repository-url>
cd Extraction-Methods
```

---

# 🔹 Create Virtual Environment

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

# 🔹 Install Dependencies

Some frameworks may require independent dependency installation.

General installation:

```bash
pip install -r requirements.txt
```

Some modules may use:

```bash
pip install -e .
```

depending on the framework.

---

# 🚀 Running Pipelines

---

# 🔹 Run Docling Pipeline

```bash
cd Docling
python granite_docling_pipeline.py
```

---

# 🔹 Run GLM-OCR Pipeline

```bash
cd Glm-OCR
python run_glmocr_pdf_pages.py
```

---

# 🔹 Run Nemotron Parse Pipeline

```bash
cd Nemotron-Parse
python nemotron_parse_pipeline.py
```

---

# 🧩 Overall System Architecture

```text
Input Documents
(PDFs / Images / Technical Manuals)
                │
                ▼
Document Parsing Layer
                │
                ▼
OCR + Layout Understanding
                │
                ▼
Markdown / Structured Text Generation
                │
                ▼
Hierarchy Reconstruction
                │
                ▼
Semantic Chunking
                │
                ▼
RAG Preprocessing
                │
                ▼
Vector Database Ingestion
                │
                ▼
Retrieval-Augmented Generation Systems
```

---

# 🧠 RAG Integration Purpose

The extracted outputs generated by these pipelines are specifically designed for downstream RAG systems.

---

# 🔹 Supported RAG Tasks

- Semantic search
- Hybrid retrieval
- Dense retrieval
- Sparse retrieval
- Chunk-based retrieval
- Long-context retrieval
- Technical document QA
- Enterprise assistant systems

---

# 🔹 Compatible Vector Databases

The outputs can easily integrate with:

- Qdrant
- FAISS
- ChromaDB
- Milvus
- Pinecone
- Elasticsearch
- Weaviate

---

# 🔬 Research Applications

This repository is useful for:

- M.Tech research projects
- OCR benchmarking
- Enterprise AI systems
- RAG experimentation
- Document intelligence systems
- Semantic extraction research
- Technical document understanding
- AI ingestion systems

---

# 🧪 Benchmarking Capabilities

| Capability | Supported |
|---|---|
| OCR Evaluation | ✅ |
| Markdown Quality Analysis | ✅ |
| Layout Understanding | ✅ |
| Hierarchy Reconstruction | ✅ |
| Table Extraction | ✅ |
| Formula Recognition | ✅ |
| Chunk Evaluation | ✅ |
| RAG Preparation | ✅ |
| Structured JSON Generation | ✅ |
| Semantic Parsing | ✅ |

---

# 🔧 Technologies Used

---

# 🔹 Programming Languages

- Python
- Bash

---

# 🔹 Core Libraries & Frameworks

- PyMuPDF
- Docling
- GLM-OCR
- NVIDIA Nemotron Parse
- Ollama
- Markdown Processing Utilities
- JSON Processing Utilities
- OCR Frameworks
- PDF Processing Libraries

---

# 🔹 AI/LLM Components

- LLM-assisted hierarchy generation
- Semantic summarization
- OCR-enhanced parsing
- Vision-language extraction
- Layout-aware document understanding

---

# 📌 Important Notes

- Many submodules are independent.
- Some folders contain their own setup instructions.
- Existing README files inside subfolders should be referred to for module-specific details.
- The repository is designed primarily for research and experimentation.
- Several pipelines are intended for integration into larger RAG systems.

---

# 📄 Existing Module Documentation

| Module | Documentation |
|---|---|
| Docling | `Docling/README.md` |
| GLM-OCR | `Glm-OCR/README.md` |
| Nemotron Parse | `Nemotron-Parse/README.md` |

---

# 🤝 Contribution Guidelines

Contributions are welcome for:

- New extraction frameworks
- OCR improvements
- Better hierarchy reconstruction
- Benchmarking utilities
- Chunking strategies
- RAG optimization
- Evaluation pipelines
- Layout understanding improvements

---

# 🛣 Future Improvements

Potential future additions:

- Better multimodal extraction
- Improved table reconstruction
- Advanced semantic chunking
- Better OCR evaluation metrics
- GPU acceleration
- Distributed extraction pipelines
- Hybrid parser ensembles
- Document graph generation
- Knowledge graph integration

---

# 📜 License

Please refer to individual submodules for licensing information.

Some external models and frameworks may follow their own licenses.

---

# 👨‍💻 Author

Developed as part of advanced document understanding and RAG experimentation research.

Focused on:

- OCR systems
- Document intelligence
- Retrieval-Augmented Generation
- Semantic parsing
- Enterprise document extraction
- Structured document understanding

---

# ⭐ If You Found This Useful

Consider starring the repository to support the research and future development.

