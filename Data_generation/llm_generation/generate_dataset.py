# # """
# # RAGAS v0.2+ — PDF -> Q&A Evaluation Dataset Generator
# # ======================================================
# # Uses EXACTLY what the RAGAS source library does internally.

# # Key design decisions sourced directly from RAGAS source:
# #   - default_query_distribution() from ragas.testset.synthesizers
# #     produces exactly 3 synthesizers:
# #       SingleHopSpecificQuerySynthesizer  weight=0.5
# #       MultiHopAbstractQuerySynthesizer   weight=0.25
# #       MultiHopSpecificQuerySynthesizer   weight=0.25
# #   - SingleHopSpecificQuerySynthesizer uses property_name="entities" (source default)
# #   - MultiHopSpecificQuerySynthesizer   uses property_name="entities" (source default)
# #   - apply_transforms() accepts run_config as 3rd positional arg (confirmed from source)
# #   - RunConfig lives at ragas.run_config (not ragas.runners)
# #   - KnowledgeGraph.save() / KnowledgeGraph.load() are the correct persistence methods

# # Usage
# # -----
# #   # Both from NVIDIA (default)
# #   python generate_dataset.py --pdf-dir ./my_pdfs

# #   # Both from Ollama
# #   python generate_dataset.py --pdf-dir ./my_pdfs \
# #       --llm-provider ollama --embed-provider ollama

# #   # LLM from NVIDIA, embeddings from Ollama
# #   python generate_dataset.py --pdf-dir ./my_pdfs \
# #       --llm-provider nvidia --embed-provider ollama

# #   # LLM from Ollama, embeddings from NVIDIA
# #   python generate_dataset.py --pdf-dir ./my_pdfs \
# #       --llm-provider ollama --embed-provider nvidia

# #   # Custom question count
# #   python generate_dataset.py --pdf-dir ./my_pdfs -n 100

# #   # Reuse existing KG (skip the expensive transform step)
# #   python generate_dataset.py --pdf-dir ./my_pdfs -n 50 --load-kg

# #   # Custom output directory
# #   python generate_dataset.py --pdf-dir ./my_pdfs -n 20 --output-dir ./results

# # Required env vars
# # -----------------
# #   NVIDIA_API_KEY   — required when --llm-provider nvidia OR --embed-provider nvidia
# #                      not needed if both providers are ollama

# # Install
# # -------
# #   # For NVIDIA:
# #   pip install ragas langchain langchain-nvidia-ai-endpoints \
# #               langchain-community pypdf nest_asyncio rapidfuzz

# #   # For Ollama (additional):
# #   pip install langchain-ollama
# #   # and make sure Ollama is running: ollama serve
# # """

# # import os
# # import sys
# # import argparse
# # import time
# # import json
# # import nest_asyncio
# # from pathlib import Path

# # # RAGAS uses async internally. This patches the running event loop
# # # so it works correctly from scripts (not just Jupyter notebooks).
# # nest_asyncio.apply()


# # # ══════════════════════════════════════════════════════════════════════════════
# # # CLI
# # # ══════════════════════════════════════════════════════════════════════════════

# # def parse_args() -> argparse.Namespace:
# #     parser = argparse.ArgumentParser(
# #         description="Generate a RAG evaluation dataset from PDFs using RAGAS v0.2+",
# #         formatter_class=argparse.RawDescriptionHelpFormatter,
# #         epilog=__doc__,
# #     )
# #     parser.add_argument(
# #         "--pdf-dir", "-d",
# #         required=True,
# #         help="Path to the directory containing PDF files.",
# #     )
# #     parser.add_argument(
# #         "--num-questions", "-n",
# #         type=int,
# #         default=50,
# #         help="Number of Q&A samples to generate (default: 50).",
# #     )
# #     parser.add_argument(
# #         "--output-dir", "-o",
# #         default="./ragas_output",
# #         help="Directory to save outputs (default: ./ragas_output).",
# #     )
# #     parser.add_argument(
# #         "--load-kg",
# #         action="store_true",
# #         default=False,
# #         help="Load existing knowledge_graph.json instead of rebuilding it.",
# #     )
# #     parser.add_argument(
# #         "--kg-file",
# #         default="knowledge_graph.json",
# #         help="KG filename inside --output-dir (default: knowledge_graph.json).",
# #     )
# #     parser.add_argument(
# #         "--dataset-file",
# #         default="dataset.csv",
# #         help="Output CSV filename (default: dataset.csv).",
# #     )
# #     parser.add_argument(
# #         "--llm-provider",
# #         choices=["nvidia", "ollama"],
# #         default="nvidia",
# #         help="Provider for the LLM: 'nvidia' (default) or 'ollama'.",
# #     )
# #     parser.add_argument(
# #         "--embed-provider",
# #         choices=["nvidia", "ollama"],
# #         default="nvidia",
# #         help="Provider for embeddings: 'nvidia' (default) or 'ollama'.",
# #     )
# #     return parser.parse_args()


# # # ══════════════════════════════════════════════════════════════════════════════
# # # HELPERS
# # # ══════════════════════════════════════════════════════════════════════════════

# # def banner(text: str) -> None:
# #     bar = "─" * 60
# #     print(f"\n{bar}\n  {text}\n{bar}")


# # def check_env(llm_provider: str, embed_provider: str) -> None:
# #     # Only require NVIDIA_API_KEY if at least one component uses nvidia
# #     if (llm_provider == "nvidia" or embed_provider == "nvidia") and not os.environ.get("NVIDIA_API_KEY"):
# #         print(
# #             "\n❌  NVIDIA_API_KEY is not set.\n"
# #             "   export NVIDIA_API_KEY='nvapi-...'\n"
# #         )
# #         sys.exit(1)


# # def make_output_dir(path: str) -> Path:
# #     p = Path(path)
# #     p.mkdir(parents=True, exist_ok=True)
# #     return p


# # # ══════════════════════════════════════════════════════════════════════════════
# # # STEP 1 — LOAD PDFs
# # # ══════════════════════════════════════════════════════════════════════════════

# # def load_pdfs(pdf_dir: str):
# #     """
# #     Load PDFs using LangChain DirectoryLoader + PyPDFLoader.
# #     Sets 'filename' metadata — required by RAGAS for multi-hop generation.
# #     Source ref: RAGAS Node constructor expects document_metadata with filename.
# #     """
# #     from langchain_community.document_loaders import DirectoryLoader
# #     from langchain_community.document_loaders import PyPDFLoader

# #     pdf_path = Path(pdf_dir)
# #     if not pdf_path.exists():
# #         print(f"\n❌  PDF directory not found: {pdf_dir}")
# #         sys.exit(1)

# #     pdf_files = list(pdf_path.glob("**/*.pdf"))
# #     if not pdf_files:
# #         print(f"\n❌  No PDF files found in: {pdf_dir}")
# #         sys.exit(1)

# #     print(f"  Found {len(pdf_files)} PDF file(s)")

# #     loader = DirectoryLoader(
# #         str(pdf_path),
# #         glob="**/*.pdf",
# #         loader_cls=PyPDFLoader,
# #         show_progress=True,
# #     )
# #     docs = loader.load()

# #     # RAGAS multi-hop synthesizers need 'filename' in metadata
# #     # to identify which chunks come from the same document
# #     for doc in docs:
# #         if "filename" not in doc.metadata:
# #             doc.metadata["filename"] = doc.metadata.get("source", "unknown")

# #     print(f"  Loaded {len(docs)} page(s)")
# #     return docs


# # # ══════════════════════════════════════════════════════════════════════════════
# # # STEP 2 — MODEL CONFIGURATION
# # # ══════════════════════════════════════════════════════════════════════════════

# # # ┌─────────────────────────────────────────────────────────────────────────┐
# # # │  NVIDIA NIM settings (used when --provider nvidia)                      │
# # # │                                                                         │
# # # │  LLM options:                                                           │
# # # │    "meta/llama-3.1-8b-instruct"       <- fast, free tier friendly      │
# # # │    "meta/llama-3.1-70b-instruct"      <- better quality                │
# # # │    "mistralai/mixtral-8x7b-instruct-v0.1"                               │
# # # │    "nvidia/nemotron-4-340b-instruct"  <- highest quality                │
# # # │                                                                         │
# # # │  Embedding options:                                                     │
# # # │    "nvidia/nv-embedqa-e5-v5"          <- recommended                   │
# # # │    "nvidia/nv-embed-v1"                                                 │
# # # │    "baai/bge-m3"                                                        │
# # # └─────────────────────────────────────────────────────────────────────────┘

# # NVIDIA_LLM_MODEL       = "meta/llama-3.1-70b-instruct"   # <- change me
# # NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"       # <- change me
# # NVIDIA_BASE_URL        = "https://integrate.api.nvidia.com/v1"

# # # ┌─────────────────────────────────────────────────────────────────────────┐
# # # │  Ollama settings (used when --provider ollama)                          │
# # # │                                                                         │
# # # │  Requires:  ollama serve  (running locally on port 11434)               │
# # # │  Install:   pip install langchain-ollama                                │
# # # │                                                                         │
# # # │  LLM options (must be pulled first: ollama pull <model>):               │
# # # │    "llama3.1"          <- good general purpose                          │
# # # │    "llama3.2"          <- smaller, faster                               │
# # # │    "mistral"           <- good quality                                  │
# # # │    "gemma2"            <- lightweight alternative                       │
# # # │                                                                         │
# # # │  Embedding options (must be pulled first: ollama pull <model>):         │
# # # │    "nomic-embed-text"  <- recommended, small and fast                   │
# # # │    "mxbai-embed-large" <- higher quality                                │
# # # │    "all-minilm"        <- very lightweight                              │
# # # └─────────────────────────────────────────────────────────────────────────┘

# # OLLAMA_LLM_MODEL       = "qwen3:latest"          # <- change me
# # OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # <- change me
# # OLLAMA_BASE_URL        = "http://localhost:11434"


# # def build_llm(llm_provider: str):
# #     """
# #     Build LLM for the selected provider, wrapped in LangchainLLMWrapper.
# #     Source ref: ragas.llms.LangchainLLMWrapper
# #     """
# #     from ragas.llms import LangchainLLMWrapper

# #     if llm_provider == "nvidia":
# #         from langchain_nvidia_ai_endpoints import ChatNVIDIA
# #         print(f"  LLM provider : NVIDIA NIM")
# #         print(f"  LLM model    : {NVIDIA_LLM_MODEL}")
# #         llm = ChatNVIDIA(
# #             model=NVIDIA_LLM_MODEL,
# #             nvidia_api_key=os.environ["NVIDIA_API_KEY"],
# #             base_url=NVIDIA_BASE_URL,
# #             temperature=0.1,
# #             max_tokens=1024,
# #         )

# #     else:  # ollama
# #         from langchain_ollama import ChatOllama
# #         print(f"  LLM provider : Ollama (local)")
# #         print(f"  LLM model    : {OLLAMA_LLM_MODEL}")
# #         print(f"  LLM base URL : {OLLAMA_BASE_URL}")
# #         llm = ChatOllama(
# #             model=OLLAMA_LLM_MODEL,
# #             base_url=OLLAMA_BASE_URL,
# #             temperature=0.1,
# #         )

# #     return LangchainLLMWrapper(llm)


# # def build_embeddings(embed_provider: str):
# #     """
# #     Build embeddings for the selected provider, wrapped in LangchainEmbeddingsWrapper.
# #     Source ref: ragas.embeddings.LangchainEmbeddingsWrapper
# #     """
# #     from ragas.embeddings import LangchainEmbeddingsWrapper

# #     if embed_provider == "nvidia":
# #         from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# #         print(f"  Embed provider : NVIDIA NIM")
# #         print(f"  Embed model    : {NVIDIA_EMBEDDING_MODEL}")
# #         emb = NVIDIAEmbeddings(
# #             model=NVIDIA_EMBEDDING_MODEL,
# #             nvidia_api_key=os.environ["NVIDIA_API_KEY"],
# #             base_url=NVIDIA_BASE_URL,
# #         )

# #     else:  # ollama
# #         from langchain_ollama import OllamaEmbeddings
# #         print(f"  Embed provider : Ollama (local)")
# #         print(f"  Embed model    : {OLLAMA_EMBEDDING_MODEL}")
# #         emb = OllamaEmbeddings(
# #             model=OLLAMA_EMBEDDING_MODEL,
# #             base_url=OLLAMA_BASE_URL,
# #         )

# #     return LangchainEmbeddingsWrapper(emb)


# # # ══════════════════════════════════════════════════════════════════════════════
# # # STEP 3 — KNOWLEDGE GRAPH
# # # ══════════════════════════════════════════════════════════════════════════════

# # def build_run_config(timeout: int = 180):
# #     """
# #     RunConfig tuned for NVIDIA free tier (~40 LLM req/min).
# #     Source ref: ragas.run_config.RunConfig
# #     max_workers=1 keeps requests fully sequential — zero burst.
# #     max_retries=10 + max_wait=120 handles 429s via exponential backoff.
# #     """
# #     from ragas.run_config import RunConfig
# #     return RunConfig(
# #         timeout=timeout,
# #         max_retries=10,
# #         max_wait=120,
# #         max_workers=4,   # sequential — critical for free tier
# #         seed=42,
# #     )


# # def build_knowledge_graph(docs, llm, embeddings):
# #     """
# #     Build KnowledgeGraph exactly as RAGAS source and docs show:

# #       1. Create empty KnowledgeGraph()
# #       2. Append Node(type=NodeType.DOCUMENT, ...) for each doc
# #       3. Run default_transforms(documents, llm, embedding_model)
# #       4. apply_transforms(kg, trans, run_config)
# #       5. kg.save(path)

# #     Source ref:
# #       ragas.testset.graph       -> KnowledgeGraph, Node, NodeType
# #       ragas.testset.transforms  -> default_transforms, apply_transforms
# #     """
# #     from ragas.testset.graph import KnowledgeGraph, Node, NodeType
# #     from ragas.testset.transforms import default_transforms, apply_transforms

# #     print("  Creating empty KnowledgeGraph ...")
# #     kg = KnowledgeGraph()

# #     # Exact pattern from RAGAS official source and docs:
# #     for doc in docs:
# #         kg.nodes.append(
# #             Node(
# #                 type=NodeType.DOCUMENT,
# #                 properties={
# #                     "page_content": doc.page_content,
# #                     "document_metadata": doc.metadata,
# #                 },
# #             )
# #         )
# #     print(f"  Added {len(kg.nodes)} DOCUMENT node(s)")

# #     print("\n  Running default_transforms ...")
# #     print("  Enrichments: NERExtractor, KeyphrasesExtractor, HeadlinesExtractor,")
# #     print("               SummaryExtractor, CosineSimilarityBuilder, OverlapScoreBuilder")
# #     print(f"  Est. LLM calls : ~{len(docs) * 4}  (4 extractors x {len(docs)} chunks)")
# #     print("  max_workers=1  : sequential, safe for NVIDIA free tier\n")

# #     # default_transforms signature (source): default_transforms(documents, llm, embedding_model)
# #     trans = default_transforms(
# #         documents=docs,
# #         llm=llm,
# #         embedding_model=embeddings,
# #     )

# #     # apply_transforms signature (source):
# #     # apply_transforms(kg, transforms, run_config=RunConfig(), callbacks=None)
# #     run_config = build_run_config(timeout=180)
# #     apply_transforms(kg, trans, run_config)

# #     print(f"\n  KG complete — nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
# #     return kg


# # def save_kg(kg, path: Path) -> None:
# #     # Source: KnowledgeGraph.save(path: str)
# #     kg.save(str(path))
# #     print(f"  KG saved -> {path}")


# # def load_kg(path: Path):
# #     # Source: KnowledgeGraph.load(path: str) -> KnowledgeGraph
# #     from ragas.testset.graph import KnowledgeGraph
# #     print(f"  Loading KG from {path} ...")
# #     kg = KnowledgeGraph.load(str(path))
# #     print(f"  KG loaded — nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
# #     return kg


# # # ══════════════════════════════════════════════════════════════════════════════
# # # STEP 4 — QUERY DISTRIBUTION
# # # ══════════════════════════════════════════════════════════════════════════════

# # def get_query_distribution(llm):
# #     """
# #     Use default_query_distribution() directly — this is what RAGAS itself uses
# #     internally when no distribution is provided.

# #     Source: ragas/src/ragas/testset/synthesizers/__init__.py
# #     The function produces exactly:
# #         [
# #             (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
# #             (MultiHopAbstractQuerySynthesizer(llm=llm),  0.25),
# #             (MultiHopSpecificQuerySynthesizer(llm=llm),  0.25),
# #         ]

# #     SingleHopSpecificQuerySynthesizer source default:
# #         property_name = "entities"   <- populated by NERExtractor in default_transforms
# #     MultiHopSpecificQuerySynthesizer source default:
# #         property_name = "entities"
# #         relation_type = "entities_overlap"  <- populated by OverlapScoreBuilder

# #     These defaults are guaranteed to work with default_transforms output.
# #     This is the ONLY officially supported distribution in RAGAS v0.2+.
# #     """
# #     from ragas.testset.synthesizers import default_query_distribution

# #     distribution = default_query_distribution(llm)

# #     print("  Source: ragas.testset.synthesizers.default_query_distribution()")
# #     print()
# #     print("  ┌──────────────────────────────────────────────────────┬────────┐")
# #     print("  │ Synthesizer                                          │ Weight │")
# #     print("  ├──────────────────────────────────────────────────────┼────────┤")
# #     for synth, weight in distribution:
# #         name = type(synth).__name__
# #         print(f"  │  {name:<51} │  {weight:.2f}  │")
# #     print("  └──────────────────────────────────────────────────────┴────────┘")
# #     print()
# #     print("  Question types covered:")
# #     print("    SingleHopSpecific (50%) — factual, from single chunk entities")
# #     print("    MultiHopAbstract  (25%) — reasoning across multiple chunks")
# #     print("    MultiHopSpecific  (25%) — factual across multiple chunks")

# #     return distribution


# # # ══════════════════════════════════════════════════════════════════════════════
# # # STEP 5 — GENERATE TESTSET
# # # ══════════════════════════════════════════════════════════════════════════════

# # def generate_testset(kg, llm, embeddings, num_questions: int, query_distribution):
# #     """
# #     Run TestsetGenerator.generate() exactly as RAGAS source and docs show.

# #     Source ref: ragas.testset.TestsetGenerator
# #     Constructor: TestsetGenerator(llm, embedding_model, knowledge_graph,
# #                                   persona_list=None, llm_context=None)
# #     generate() sig: testset_size, query_distribution, num_personas,
# #                     run_config, with_debugging_logs, raise_exceptions
# #     """
# #     from ragas.testset import TestsetGenerator

# #     # Source: TestsetGenerator(llm, embedding_model, knowledge_graph)
# #     generator = TestsetGenerator(
# #         llm=llm,
# #         embedding_model=embeddings,
# #         knowledge_graph=kg,
# #         # persona_list=None -> RAGAS auto-generates personas from KG summaries
# #     )

# #     # Synthesis prompts are longer than extraction prompts -> larger timeout
# #     run_config = build_run_config(timeout=300)

# #     print(f"  Generating {num_questions} samples ...")
# #     print("  max_workers=1 (sequential) — safe for NVIDIA free tier\n")

# #     # Source: generate(testset_size, query_distribution, num_personas,
# #     #                  run_config, with_debugging_logs, raise_exceptions)
# #     testset = generator.generate(
# #         testset_size=num_questions,
# #         query_distribution=query_distribution,
# #         num_personas=3,           # auto-generate 3 personas from KG
# #         run_config=run_config,
# #         with_debugging_logs=False,
# #         raise_exceptions=False,   # log failures, don't abort the whole run
# #     )

# #     return testset


# # # ══════════════════════════════════════════════════════════════════════════════
# # # STEP 6 — EXPORT
# # # ══════════════════════════════════════════════════════════════════════════════

# # def export_testset(testset, output_dir: Path, csv_filename: str) -> None:
# #     """
# #     Export to CSV and JSON.
# #     Source ref: Testset.to_pandas() returns DataFrame with columns:
# #       user_input, reference_contexts, reference, synthesizer_name
# #     """
# #     import ast

# #     df = testset.to_pandas()

# #     if df.empty:
# #         print("\n  ⚠️  Testset is empty — no questions were generated.")
# #         print("  Possible causes:")
# #         print("    - PDFs too short or too few (need substantial text)")
# #         print("    - NER extraction failed (no 'entities' nodes in KG)")
# #         print("    - All generation attempts raised exceptions (raise_exceptions=False)")
# #         print("  Try: re-run with more/longer PDFs, or check logs above.\n")
# #         return

# #     # ── CSV ──────────────────────────────────────────────────────────────────
# #     csv_path = output_dir / csv_filename
# #     df.to_csv(csv_path, index=False)
# #     print(f"  CSV  -> {csv_path}")

# #     # ── JSON with metadata ────────────────────────────────────────────────────
# #     json_path = output_dir / csv_filename.replace(".csv", ".json")
# #     records = []
# #     for _, row in df.iterrows():
# #         ref_ctx = row.get("reference_contexts", [])
# #         # Pandas may serialize lists as strings — parse them back
# #         if isinstance(ref_ctx, str):
# #             try:
# #                 ref_ctx = ast.literal_eval(ref_ctx)
# #             except Exception:
# #                 ref_ctx = [ref_ctx]

# #         records.append({
# #             "user_input":         row.get("user_input", ""),
# #             "reference":          row.get("reference", ""),
# #             "reference_contexts": ref_ctx if isinstance(ref_ctx, list) else [],
# #             "synthesizer_name":   row.get("synthesizer_name", ""),
# #             "metadata": {
# #                 "synthesizer_name": row.get("synthesizer_name", ""),
# #                 "num_contexts":     len(ref_ctx) if isinstance(ref_ctx, list) else 0,
# #             },
# #         })

# #     with open(json_path, "w", encoding="utf-8") as f:
# #         json.dump(records, f, indent=2, ensure_ascii=False)
# #     print(f"  JSON -> {json_path}")

# #     # ── Distribution summary ──────────────────────────────────────────────────
# #     print("\n  Question-type distribution:")
# #     dist  = df["synthesizer_name"].value_counts()
# #     total = len(df)
# #     for name, count in dist.items():
# #         pct = count / total * 100
# #         bar = "█" * int(pct / 2)
# #         print(f"    {name:<52} {count:>3}  ({pct:5.1f}%)  {bar}")
# #     print(f"    {'TOTAL':<52} {total:>3}\n")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # MAIN
# # # ══════════════════════════════════════════════════════════════════════════════

# # def main() -> None:
# #     args       = parse_args()
# #     check_env(args.llm_provider, args.embed_provider)
# #     output_dir = make_output_dir(args.output_dir)
# #     kg_path    = output_dir / args.kg_file

# #     print(
# #         f"\n  LLM        : {args.llm_provider.upper()} "
# #         f"({'NVIDIA NIM' if args.llm_provider == 'nvidia' else 'Ollama local'})\n"
# #         f"  Embeddings : {args.embed_provider.upper()} "
# #         f"({'NVIDIA NIM' if args.embed_provider == 'nvidia' else 'Ollama local'})\n"
# #         f"  Mode       : max_workers=1 (sequential), retries=10, max_wait=120s\n"
# #         f"  Target     : {args.num_questions} question(s)\n"
# #     )

# #     # Step 1
# #     banner("Step 1 · Load PDFs")
# #     docs = load_pdfs(args.pdf_dir)

# #     # Step 2
# #     banner("Step 2 · Models")
# #     llm        = build_llm(args.llm_provider)
# #     embeddings = build_embeddings(args.embed_provider)

# #     # Step 3
# #     banner("Step 3 · Knowledge Graph")
# #     if args.load_kg and kg_path.exists():
# #         kg = load_kg(kg_path)
# #     else:
# #         if args.load_kg:
# #             print(f"  ⚠  --load-kg set but '{kg_path}' not found — building from scratch")
# #         kg = build_knowledge_graph(docs, llm, embeddings)
# #         save_kg(kg, kg_path)

# #     # Step 4
# #     banner("Step 4 · Query Distribution")
# #     query_distribution = get_query_distribution(llm)

# #     # Step 5
# #     banner(f"Step 5 · Generate {args.num_questions} Questions")
# #     t0      = time.time()
# #     testset = generate_testset(kg, llm, embeddings, args.num_questions, query_distribution)
# #     elapsed = time.time() - t0
# #     print(f"\n  Completed in {elapsed:.1f}s")

# #     # Step 6
# #     banner("Step 6 · Export")
# #     export_testset(testset, output_dir, args.dataset_file)

# #     banner("Done")
# #     print(f"  All outputs saved to: {output_dir.resolve()}\n")


# # if __name__ == "__main__":
# #     main()







"""
RAGAS v0.2+ — PDF -> Q&A Evaluation Dataset Generator
======================================================
Uses EXACTLY what the RAGAS source library does internally.

Key design decisions sourced directly from RAGAS source:
  - default_query_distribution() from ragas.testset.synthesizers
    produces exactly 3 synthesizers:
      SingleHopSpecificQuerySynthesizer  weight=0.5
      MultiHopAbstractQuerySynthesizer   weight=0.25
      MultiHopSpecificQuerySynthesizer   weight=0.25
  - SingleHopSpecificQuerySynthesizer uses property_name="entities" (source default)
  - MultiHopSpecificQuerySynthesizer   uses property_name="entities" (source default)
  - apply_transforms() accepts run_config as 3rd positional arg (confirmed from source)
  - RunConfig lives at ragas.run_config (not ragas.runners)
  - KnowledgeGraph.save() / KnowledgeGraph.load() are the correct persistence methods

Usage
-----
  # Both from NVIDIA (default)
  python generate_dataset.py --pdf-dir ./my_pdfs

  # Both from Ollama
  python generate_dataset.py --pdf-dir ./my_pdfs \
      --llm-provider ollama --embed-provider ollama

  # LLM from NVIDIA, embeddings from Ollama
  python generate_dataset.py --pdf-dir ./my_pdfs \
      --llm-provider nvidia --embed-provider ollama

  # LLM from Ollama, embeddings from NVIDIA
  python generate_dataset.py --pdf-dir ./my_pdfs \
      --llm-provider ollama --embed-provider nvidia

  # Custom question count
  python generate_dataset.py --pdf-dir ./my_pdfs -n 100

  # Reuse existing KG (skip the expensive transform step)
  python generate_dataset.py --pdf-dir ./my_pdfs -n 50 --load-kg

  # Custom output directory
  python generate_dataset.py --pdf-dir ./my_pdfs -n 20 --output-dir ./results

Required env vars
-----------------
  NVIDIA_API_KEY   — required when --llm-provider nvidia OR --embed-provider nvidia
                     not needed if both providers are ollama

Install
-------
  # For NVIDIA:
  pip install ragas langchain langchain-nvidia-ai-endpoints \
              langchain-community pypdf nest_asyncio rapidfuzz

  # For Ollama (additional):
  pip install langchain-ollama
  # and make sure Ollama is running: ollama serve
"""

import os
import sys
import argparse
import time
import json
import nest_asyncio
from pathlib import Path

# RAGAS uses async internally. This patches the running event loop
# so it works correctly from scripts (not just Jupyter notebooks).
nest_asyncio.apply()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RAG evaluation dataset from PDFs using RAGAS v0.2+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdf-dir", "-d",
        required=True,
        help="Path to the directory containing PDF files.",
    )
    parser.add_argument(
        "--num-questions", "-n",
        type=int,
        default=50,
        help="Number of Q&A samples to generate (default: 50).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./ragas_output",
        help="Directory to save outputs (default: ./ragas_output).",
    )
    parser.add_argument(
        "--load-kg",
        action="store_true",
        default=False,
        help="Load existing knowledge_graph.json instead of rebuilding it.",
    )
    parser.add_argument(
        "--kg-file",
        default="knowledge_graph.json",
        help="KG filename inside --output-dir (default: knowledge_graph.json).",
    )
    parser.add_argument(
        "--dataset-file",
        default="dataset.csv",
        help="Output CSV filename (default: dataset.csv).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["nvidia", "ollama"],
        default="nvidia",
        help="Provider for the LLM: 'nvidia' (default) or 'ollama'.",
    )
    parser.add_argument(
        "--embed-provider",
        choices=["nvidia", "ollama", "sentence-transformers"],
        default="nvidia",
        help="Provider for embeddings: 'nvidia' (default), 'ollama', or 'sentence-transformers' (BGE-M3, fully local).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for KG transforms and generation (default: 4).",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def banner(text: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def check_env(llm_provider: str, embed_provider: str) -> None:
    # Only require NVIDIA_API_KEY if at least one component uses nvidia
    if (llm_provider == "nvidia" or embed_provider == "nvidia") and not os.environ.get("NVIDIA_API_KEY"):
        print(
            "\n❌  NVIDIA_API_KEY is not set.\n"
            "   export NVIDIA_API_KEY='nvapi-...'\n"
        )
        sys.exit(1)


def make_output_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD PDFs
# ══════════════════════════════════════════════════════════════════════════════

def load_pdfs(pdf_dir: str):
    """
    Load PDFs using LangChain DirectoryLoader + PyPDFLoader.
    Sets 'filename' metadata — required by RAGAS for multi-hop generation.
    Source ref: RAGAS Node constructor expects document_metadata with filename.
    """
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import PyPDFLoader

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"\n❌  PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_files = list(pdf_path.glob("**/*.pdf"))
    if not pdf_files:
        print(f"\n❌  No PDF files found in: {pdf_dir}")
        sys.exit(1)

    print(f"  Found {len(pdf_files)} PDF file(s)")

    loader = DirectoryLoader(
        str(pdf_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()

    # RAGAS multi-hop synthesizers need 'filename' in metadata
    # to identify which chunks come from the same document
    for doc in docs:
        if "filename" not in doc.metadata:
            doc.metadata["filename"] = doc.metadata.get("source", "unknown")

    print(f"  Loaded {len(docs)} page(s)")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  NVIDIA NIM settings (used when --provider nvidia)                      │
# │                                                                         │
# │  LLM options:                                                           │
# │    "meta/llama-3.1-8b-instruct"       <- fast, free tier friendly      │
# │    "meta/llama-3.1-70b-instruct"      <- better quality                │
# │    "mistralai/mixtral-8x7b-instruct-v0.1"                               │
# │    "nvidia/nemotron-4-340b-instruct"  <- highest quality                │
# │                                                                         │
# │  Embedding options:                                                     │
# │    "nvidia/nv-embedqa-e5-v5"          <- recommended                   │
# │    "nvidia/nv-embed-v1"                                                 │
# │    "baai/bge-m3"                                                        │
# └─────────────────────────────────────────────────────────────────────────┘

NVIDIA_LLM_MODEL       = "meta/llama-3.1-70b-instruct"   # <- change me
NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"       # <- change me
SENTENCE_TRANSFORMER_EMBEDDING_MODEL = "BAAI/bge-m3"     # HuggingFace BGE-M3
NVIDIA_BASE_URL        = "https://integrate.api.nvidia.com/v1"

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  Ollama settings (used when --provider ollama)                          │
# │                                                                         │
# │  Requires:  ollama serve  (running locally on port 11434)               │
# │  Install:   pip install langchain-ollama                                │
# │                                                                         │
# │  LLM options (must be pulled first: ollama pull <model>):               │
# │    "llama3.1"          <- good general purpose                          │
# │    "llama3.2"          <- smaller, faster                               │
# │    "mistral"           <- good quality                                  │
# │    "gemma2"            <- lightweight alternative                       │
# │                                                                         │
# │  Embedding options (must be pulled first: ollama pull <model>):         │
# │    "nomic-embed-text"  <- recommended, small and fast                   │
# │    "mxbai-embed-large" <- higher quality                                │
# │    "all-minilm"        <- very lightweight                              │
# └─────────────────────────────────────────────────────────────────────────┘


OLLAMA_LLM_MODEL       = "Qwen/Qwen2.5-32B-Instruct-AWQ"          # <- change me
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # <- change me
OLLAMA_BASE_URL        = "http://localhost:8011/v1"



def build_llm(llm_provider: str):
    """
    Build LLM for the selected provider, wrapped in LangchainLLMWrapper.
    Source ref: ragas.llms.LangchainLLMWrapper
    """
    from ragas.llms import LangchainLLMWrapper

    if llm_provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        print(f"  LLM provider : NVIDIA NIM")
        print(f"  LLM model    : {NVIDIA_LLM_MODEL}")
        llm = ChatNVIDIA(
            model=NVIDIA_LLM_MODEL,
            nvidia_api_key=os.environ["NVIDIA_API_KEY"],
            base_url=NVIDIA_BASE_URL,
            temperature=0.1,
            max_tokens=1024,
        )

    # else:  # ollama
    #     from langchain_ollama import ChatOllama
    #     print(f"  LLM provider : Ollama (local)")
    #     print(f"  LLM model    : {OLLAMA_LLM_MODEL}")
    #     print(f"  LLM base URL : {OLLAMA_BASE_URL}")
    #     llm = ChatOllama(
    #         model=OLLAMA_LLM_MODEL,
    #         base_url=OLLAMA_BASE_URL,
    #         temperature=0.1,
    #     )
    else:  # ollama → now points to vLLM
        import httpx
        from langchain_openai import ChatOpenAI
        print(f"  LLM provider : vLLM (local)")
        print(f"  LLM model    : {OLLAMA_LLM_MODEL}")
        llm = ChatOpenAI(
            model=OLLAMA_LLM_MODEL,
            openai_api_key="dummy",
            openai_api_base=OLLAMA_BASE_URL,
            temperature=0.1,
            max_tokens=1024,
            http_client=httpx.Client(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=60,
                ),
            ),
            http_async_client=httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=60,
                ),
            ),
        )

    return LangchainLLMWrapper(llm)


def build_embeddings(embed_provider: str):
    """
    Build embeddings for the selected provider, wrapped in LangchainEmbeddingsWrapper.
    Source ref: ragas.embeddings.LangchainEmbeddingsWrapper
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if embed_provider == "nvidia":
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        print(f"  Embed provider : NVIDIA NIM")
        print(f"  Embed model    : {NVIDIA_EMBEDDING_MODEL}")
        emb = NVIDIAEmbeddings(
            model=NVIDIA_EMBEDDING_MODEL,
            nvidia_api_key=os.environ["NVIDIA_API_KEY"],
            base_url=NVIDIA_BASE_URL,
        )

    elif embed_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        print(f"  Embed provider : Ollama (local)")
        print(f"  Embed model    : {OLLAMA_EMBEDDING_MODEL}")
        emb = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

    else:  # sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"  Embed provider : Sentence Transformers (local)")
        print(f"  Embed model    : {SENTENCE_TRANSFORMER_EMBEDDING_MODEL}")
        emb = HuggingFaceEmbeddings(
            model_name=SENTENCE_TRANSFORMER_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},        # change to "cuda" if you have a GPU
            encode_kwargs={"normalize_embeddings": True},  # BGE-M3 benefits from normalization
        )

    return LangchainEmbeddingsWrapper(emb)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_run_config(timeout: int = 300, max_workers: int = 4):
    """
    RunConfig for the transform and generation steps.
    Source ref: ragas.run_config.RunConfig
    max_retries=10 + max_wait=120 handles 429s via exponential backoff.
    """
    from ragas.run_config import RunConfig
    return RunConfig(
        timeout=600,
        max_retries=15,
        max_wait=180,
        max_workers=max_workers,
        seed=42,
    )


def make_safe_headline_splitter():
    """
    Returns a HeadlineSplitter subclass that silently skips nodes where
    HeadlinesExtractor found no headlines (instead of crashing with ValueError).

    Nodes without headlines are kept as-is (full-page nodes) in the KG.
    Nodes with headlines are split into sub-chunks as normal.

    This makes the pipeline robust for 100+ PDFs where some pages have no
    headings (tables, references, figure captions, short paragraphs, etc).
    """
    from ragas.testset.transforms.splitters.headline import HeadlineSplitter

    class SafeHeadlineSplitter(HeadlineSplitter):
        async def split(self, node):
            headlines = node.properties.get("headlines")
            if not headlines:
                # No headlines found — skip silently, keep node as-is in KG
                return [], []
            return await super().split(node)

    return SafeHeadlineSplitter()



def _checkpoint_path(output_dir, step_name):
    """Return path for a per-step KG checkpoint file."""
    return output_dir / f"kg_checkpoint_{step_name}.json"


def _already_applied(kg, step_name):
    """
    Check if a transform step was already applied by looking for its
    expected output property on any KG node.
    Relationship builders have no node properties so always return False
    (they are fast, no LLM calls, safe to re-run).
    """
    STEP_PROPERTY_MAP = {
        "NERExtractor":        "entities",
        "KeyphrasesExtractor": "keyphrases",
        "HeadlinesExtractor":  "headlines",
        "SummaryExtractor":    "summary",
    }
    prop = STEP_PROPERTY_MAP.get(step_name)
    if prop is None:
        return False
    return any(prop in node.properties for node in kg.nodes)


def build_knowledge_graph(docs, llm, embeddings, max_workers=4, output_dir=None):
    """
    Build KnowledgeGraph with per-step checkpointing.

    Each transform step runs individually. After each step the KG is saved:
      kg_checkpoint_NERExtractor.json
      kg_checkpoint_KeyphrasesExtractor.json
      kg_checkpoint_HeadlinesExtractor.json
      kg_checkpoint_HeadlineSplitter.json
      kg_checkpoint_SummaryExtractor.json
      kg_checkpoint_CosineSimilarityBuilder.json
      kg_checkpoint_OverlapScoreBuilder.json

    On re-run after a crash:
      - Latest checkpoint is loaded automatically
      - Completed steps are detected and skipped (no redundant LLM calls)
      - Only remaining steps execute

    Source ref:
      ragas.testset.graph       -> KnowledgeGraph, Node, NodeType
      ragas.testset.transforms  -> default_transforms, apply_transforms
    """
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.testset.transforms.splitters.headline import HeadlineSplitter

    ckpt_dir = output_dir if output_dir else Path(".")

    # Try to load the most recently completed checkpoint (search in reverse order)
    STEP_ORDER_REVERSED = [
        "OverlapScoreBuilder",
        "CosineSimilarityBuilder",
        "SummaryExtractor",
        "HeadlineSplitter",
        "HeadlinesExtractor",
        "KeyphrasesExtractor",
        "NERExtractor",
    ]
    kg = None
    for step_name in STEP_ORDER_REVERSED:
        ckpt = _checkpoint_path(ckpt_dir, step_name)
        if ckpt.exists():
            print(f"  Resuming from checkpoint: {ckpt.name}")
            kg = KnowledgeGraph.load(str(ckpt))
            print(f"     nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
            break

    # No checkpoint found -- build fresh KG from documents
    if kg is None:
        print("  Creating empty KnowledgeGraph ...")
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
            )
        print(f"  Added {len(kg.nodes)} DOCUMENT node(s)")
    else:
        print(f"  Using {len(kg.nodes)} node(s) from checkpoint -- skipping doc reload")

    # Build the full transform pipeline
    print("\n  Building transform pipeline ...")
    trans = default_transforms(documents=docs, llm=llm, embedding_model=embeddings)

    # Swap HeadlineSplitter -> SafeHeadlineSplitter
    safe_splitter = make_safe_headline_splitter()
    for i, t in enumerate(trans):
        if isinstance(t, HeadlineSplitter):
            trans[i] = safe_splitter
            break

    run_config = build_run_config(timeout=600, max_workers=max_workers)

    # Apply each step individually, saving a checkpoint after each one
    print("\n  Applying transforms (checkpoint saved after each step) ...\n")
    print(f"  {'Step':<38} Status")
    print(f"  {'-'*38} {'-'*22}")

    for transform in trans:
        step_name = type(transform).__name__

        if _already_applied(kg, step_name):
            print(f"  {'  ' + step_name:<38} [skipped - already in KG]")
            continue

        print(f"  {'  ' + step_name:<38} [running ...]")
        t_start = time.time()
        try:
            apply_transforms(kg, [transform], run_config)
            elapsed = time.time() - t_start
            ckpt = _checkpoint_path(ckpt_dir, step_name)
            kg.save(str(ckpt))
            print(f"  {'  ' + step_name:<38} [done {elapsed:.1f}s -> {ckpt.name}]")
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  {'  ' + step_name:<38} [FAILED {elapsed:.1f}s]")
            print(f"     {type(e).__name__}: {e}")
            print(f"\n  WARNING: Interrupted at step: {step_name}")
            print(f"  WARNING: Re-run to resume from last checkpoint automatically.")
            raise

    print(f"\n  KG complete -- nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
    return kg


# def build_knowledge_graph(docs, llm, embeddings, max_workers: int = 4):
#     """
#     Build KnowledgeGraph exactly as RAGAS source and docs show:

#       1. Create empty KnowledgeGraph()
#       2. Append Node(type=NodeType.DOCUMENT, ...) for each doc
#       3. Run default_transforms(documents, llm, embedding_model)
#       4. apply_transforms(kg, trans, run_config)
#       5. kg.save(path)

#     Source ref:
#       ragas.testset.graph       -> KnowledgeGraph, Node, NodeType
#       ragas.testset.transforms  -> default_transforms, apply_transforms
#     """
#     from ragas.testset.graph import KnowledgeGraph, Node, NodeType
#     from ragas.testset.transforms import default_transforms, apply_transforms
#     from ragas.testset.transforms.splitters.headline import HeadlineSplitter

#     print("  Creating empty KnowledgeGraph ...")
#     kg = KnowledgeGraph()

#     # Exact pattern from RAGAS official source and docs:
#     for doc in docs:
#         kg.nodes.append(
#             Node(
#                 type=NodeType.DOCUMENT,
#                 properties={
#                     "page_content": doc.page_content,
#                     "document_metadata": doc.metadata,
#                 },
#             )
#         )
#     print(f"  Added {len(kg.nodes)} DOCUMENT node(s)")

#     print("\n  Running default_transforms ...")
#     print("  Enrichments: NERExtractor, KeyphrasesExtractor, HeadlinesExtractor,")
#     print("               SummaryExtractor, CosineSimilarityBuilder, OverlapScoreBuilder")
#     print(f"  Est. LLM calls : ~{len(docs) * 4}  (4 extractors x {len(docs)} chunks)\n")

#     # default_transforms signature (source): default_transforms(documents, llm, embedding_model)
#     trans = default_transforms(
#         documents=docs,
#         llm=llm,
#         embedding_model=embeddings,
#     )

#     # Swap out the default HeadlineSplitter with SafeHeadlineSplitter so that
#     # nodes missing the 'headlines' property are skipped instead of crashing.
#     safe_splitter = make_safe_headline_splitter()
#     for i, t in enumerate(trans):
#         if isinstance(t, HeadlineSplitter):
#             trans[i] = safe_splitter
#             print("  ✔  Swapped HeadlineSplitter -> SafeHeadlineSplitter")
#             break

#     # apply_transforms signature (source):
#     # apply_transforms(kg, transforms, run_config=RunConfig(), callbacks=None)
#     run_config = build_run_config(timeout=300, max_workers=max_workers)
#     apply_transforms(kg, trans, run_config)

#     print(f"\n  KG complete — nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
#     return kg


def save_kg(kg, path: Path) -> None:
    # Source: KnowledgeGraph.save(path: str)
    kg.save(str(path))
    print(f"  KG saved -> {path}")


def load_kg(path: Path):
    # Source: KnowledgeGraph.load(path: str) -> KnowledgeGraph
    from ragas.testset.graph import KnowledgeGraph
    print(f"  Loading KG from {path} ...")
    kg = KnowledgeGraph.load(str(path))
    print(f"  KG loaded — nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
    return kg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — QUERY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def get_query_distribution(llm):
    """
    Use default_query_distribution() directly — this is what RAGAS itself uses
    internally when no distribution is provided.

    Source: ragas/src/ragas/testset/synthesizers/__init__.py
    The function produces exactly:
        [
            (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
            (MultiHopAbstractQuerySynthesizer(llm=llm),  0.25),
            (MultiHopSpecificQuerySynthesizer(llm=llm),  0.25),
        ]

    SingleHopSpecificQuerySynthesizer source default:
        property_name = "entities"   <- populated by NERExtractor in default_transforms
    MultiHopSpecificQuerySynthesizer source default:
        property_name = "entities"
        relation_type = "entities_overlap"  <- populated by OverlapScoreBuilder

    These defaults are guaranteed to work with default_transforms output.
    This is the ONLY officially supported distribution in RAGAS v0.2+.
    """
    from ragas.testset.synthesizers import default_query_distribution

    distribution = default_query_distribution(llm)

    # from ragas.testset.synthesizers import (
    #     SingleHopSpecificQuerySynthesizer,
    #     MultiHopAbstractQuerySynthesizer,
    #     MultiHopSpecificQuerySynthesizer,
    # )

    # distribution = [
    #     (SingleHopSpecificQuerySynthesizer(llm=llm), 0.40),
    #     (MultiHopAbstractQuerySynthesizer(llm=llm),  0.30),
    #     (MultiHopSpecificQuerySynthesizer(llm=llm),  0.30),
    # ]

    print("  Source: ragas.testset.synthesizers.default_query_distribution()")
    print()
    print("  ┌──────────────────────────────────────────────────────┬────────┐")
    print("  │ Synthesizer                                          │ Weight │")
    print("  ├──────────────────────────────────────────────────────┼────────┤")
    for synth, weight in distribution:
        name = type(synth).__name__
        print(f"  │  {name:<51} │  {weight:.2f}  │")
    print("  └──────────────────────────────────────────────────────┴────────┘")
    print()
    print("  Question types covered:")
    print("    SingleHopSpecific (50%) — factual, from single chunk entities")
    print("    MultiHopAbstract  (25%) — reasoning across multiple chunks")
    print("    MultiHopSpecific  (25%) — factual across multiple chunks")

    return distribution


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — GENERATE TESTSET
# ══════════════════════════════════════════════════════════════════════════════

def generate_testset(kg, llm, embeddings, num_questions: int, query_distribution, max_workers: int = 4):
    """
    Run TestsetGenerator.generate() exactly as RAGAS source and docs show.

    Source ref: ragas.testset.TestsetGenerator
    Constructor: TestsetGenerator(llm, embedding_model, knowledge_graph,
                                  persona_list=None, llm_context=None)
    generate() sig: testset_size, query_distribution, num_personas,
                    run_config, with_debugging_logs, raise_exceptions
    """
    from ragas.testset import TestsetGenerator

    # Source: TestsetGenerator(llm, embedding_model, knowledge_graph)
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        # persona_list=None -> RAGAS auto-generates personas from KG summaries
    )

    # Synthesis prompts are longer than extraction prompts -> larger timeout
    run_config = build_run_config(timeout=300, max_workers=max_workers)

    print(f"  Generating {num_questions} samples ...")
    print(f"  max_workers={max_workers}\n")

    # Source: generate(testset_size, query_distribution, num_personas,
    #                  run_config, with_debugging_logs, raise_exceptions)
    testset = generator.generate(
        testset_size=num_questions,
        query_distribution=query_distribution,
        num_personas=3,           # auto-generate 3 personas from KG
        run_config=run_config,
        with_debugging_logs=False,
        raise_exceptions=False,   # log failures, don't abort the whole run
    )

    return testset


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_testset(testset, output_dir: Path, csv_filename: str) -> None:
    """
    Export to CSV and JSON.
    Source ref: Testset.to_pandas() returns DataFrame with columns:
      user_input, reference_contexts, reference, synthesizer_name
    """
    import ast

    df = testset.to_pandas()

    if df.empty:
        print("\n  ⚠️  Testset is empty — no questions were generated.")
        print("  Possible causes:")
        print("    - PDFs too short or too few (need substantial text)")
        print("    - NER extraction failed (no 'entities' nodes in KG)")
        print("    - All generation attempts raised exceptions (raise_exceptions=False)")
        print("  Try: re-run with more/longer PDFs, or check logs above.\n")
        return

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"  CSV  -> {csv_path}")

    # ── JSON with metadata ────────────────────────────────────────────────────
    json_path = output_dir / csv_filename.replace(".csv", ".json")
    records = []
    for _, row in df.iterrows():
        ref_ctx = row.get("reference_contexts", [])
        # Pandas may serialize lists as strings — parse them back
        if isinstance(ref_ctx, str):
            try:
                ref_ctx = ast.literal_eval(ref_ctx)
            except Exception:
                ref_ctx = [ref_ctx]

        records.append({
            "user_input":         row.get("user_input", ""),
            "reference":          row.get("reference", ""),
            "reference_contexts": ref_ctx if isinstance(ref_ctx, list) else [],
            "synthesizer_name":   row.get("synthesizer_name", ""),
            "metadata": {
                "synthesizer_name": row.get("synthesizer_name", ""),
                "num_contexts":     len(ref_ctx) if isinstance(ref_ctx, list) else 0,
            },
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  JSON -> {json_path}")

    # ── Distribution summary ──────────────────────────────────────────────────
    print("\n  Question-type distribution:")
    dist  = df["synthesizer_name"].value_counts()
    total = len(df)
    for name, count in dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:<52} {count:>3}  ({pct:5.1f}%)  {bar}")
    print(f"    {'TOTAL':<52} {total:>3}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args       = parse_args()
    check_env(args.llm_provider, args.embed_provider)
    output_dir = make_output_dir(args.output_dir)
    kg_path    = output_dir / args.kg_file

    print(
        f"\n  LLM        : {args.llm_provider.upper()} "
        f"({'NVIDIA NIM' if args.llm_provider == 'nvidia' else 'Ollama local'})\n"
        f"  Embeddings : {args.embed_provider.upper()} "
        f"({'NVIDIA NIM' if args.embed_provider == 'nvidia' else 'Ollama local' if args.embed_provider == 'ollama' else 'Sentence Transformers local (BGE-M3)'})\n"
        f"  Mode       : max_workers={args.max_workers}, retries=10, max_wait=120s\n"
        f"  Target     : {args.num_questions} question(s)\n"
    )

    # Step 1
    banner("Step 1 · Load PDFs")
    docs = load_pdfs(args.pdf_dir)

    # Step 2
    banner("Step 2 · Models")
    llm        = build_llm(args.llm_provider)
    embeddings = build_embeddings(args.embed_provider)

    # Step 3
    banner("Step 3 · Knowledge Graph")
    if args.load_kg and kg_path.exists():
        kg = load_kg(kg_path)
    else:
        if args.load_kg:
            print(f"  ⚠  --load-kg set but '{kg_path}' not found — building from scratch")
        # kg = build_knowledge_graph(docs, llm, embeddings, args.max_workers)
        kg = build_knowledge_graph(docs, llm, embeddings, args.max_workers, output_dir)
        save_kg(kg, kg_path)

    # Step 4
    banner("Step 4 · Query Distribution")
    query_distribution = get_query_distribution(llm)

    # Step 5
    banner(f"Step 5 · Generate {args.num_questions} Questions")
    t0      = time.time()
    testset = generate_testset(kg, llm, embeddings, args.num_questions, query_distribution, args.max_workers)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    # Step 6
    banner("Step 6 · Export")
    export_testset(testset, output_dir, args.dataset_file)

    banner("Done")
    print(f"  All outputs saved to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()





# """
# RAGAS v0.2+ — PDF -> Q&A Evaluation Dataset Generator
# ======================================================
# Uses EXACTLY what the RAGAS source library does internally.

# Key design decisions sourced directly from RAGAS source:
#   - default_query_distribution() from ragas.testset.synthesizers
#     produces exactly 3 synthesizers:
#       SingleHopSpecificQuerySynthesizer  weight=0.5
#       MultiHopAbstractQuerySynthesizer   weight=0.25
#       MultiHopSpecificQuerySynthesizer   weight=0.25
#   - SingleHopSpecificQuerySynthesizer uses property_name="entities" (source default)
#   - MultiHopSpecificQuerySynthesizer   uses property_name="entities" (source default)
#   - apply_transforms() accepts run_config as 3rd positional arg (confirmed from source)
#   - RunConfig lives at ragas.run_config (not ragas.runners)
#   - KnowledgeGraph.save() / KnowledgeGraph.load() are the correct persistence methods

# Usage
# -----
#   # Both from NVIDIA (default)
#   python generate_dataset.py --pdf-dir ./my_pdfs

#   # Both from Ollama
#   python generate_dataset.py --pdf-dir ./my_pdfs \
#       --llm-provider ollama --embed-provider ollama

#   # LLM from NVIDIA, embeddings from Ollama
#   python generate_dataset.py --pdf-dir ./my_pdfs \
#       --llm-provider nvidia --embed-provider ollama

#   # LLM from Ollama, embeddings from NVIDIA
#   python generate_dataset.py --pdf-dir ./my_pdfs \
#       --llm-provider ollama --embed-provider nvidia

#   # Custom question count
#   python generate_dataset.py --pdf-dir ./my_pdfs -n 100

#   # Reuse existing KG (skip the expensive transform step)
#   python generate_dataset.py --pdf-dir ./my_pdfs -n 50 --load-kg

#   # Custom output directory
#   python generate_dataset.py --pdf-dir ./my_pdfs -n 20 --output-dir ./results

# Required env vars
# -----------------
#   NVIDIA_API_KEY   — required when --llm-provider nvidia OR --embed-provider nvidia
#                      not needed if both providers are ollama

# Install
# -------
#   # For NVIDIA:
#   pip install ragas langchain langchain-nvidia-ai-endpoints \
#               langchain-community pypdf nest_asyncio rapidfuzz

#   # For Ollama (additional):
#   pip install langchain-ollama
#   # and make sure Ollama is running: ollama serve
# """

# import os
# import sys
# import argparse
# import time
# import json
# import nest_asyncio
# from pathlib import Path

# # RAGAS uses async internally. This patches the running event loop
# # so it works correctly from scripts (not just Jupyter notebooks).
# nest_asyncio.apply()


# # ══════════════════════════════════════════════════════════════════════════════
# # CLI
# # ══════════════════════════════════════════════════════════════════════════════

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Generate a RAG evaluation dataset from PDFs using RAGAS v0.2+",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog=__doc__,
#     )
#     parser.add_argument(
#         "--pdf-dir", "-d",
#         required=True,
#         help="Path to the directory containing PDF files.",
#     )
#     parser.add_argument(
#         "--num-questions", "-n",
#         type=int,
#         default=50,
#         help="Number of Q&A samples to generate (default: 50).",
#     )
#     parser.add_argument(
#         "--output-dir", "-o",
#         default="./ragas_output",
#         help="Directory to save outputs (default: ./ragas_output).",
#     )
#     parser.add_argument(
#         "--load-kg",
#         action="store_true",
#         default=False,
#         help="Load existing knowledge_graph.json instead of rebuilding it.",
#     )
#     parser.add_argument(
#         "--kg-file",
#         default="knowledge_graph.json",
#         help="KG filename inside --output-dir (default: knowledge_graph.json).",
#     )
#     parser.add_argument(
#         "--dataset-file",
#         default="dataset.csv",
#         help="Output CSV filename (default: dataset.csv).",
#     )
#     parser.add_argument(
#         "--llm-provider",
#         choices=["nvidia", "ollama"],
#         default="nvidia",
#         help="Provider for the LLM: 'nvidia' (default) or 'ollama'.",
#     )
#     parser.add_argument(
#         "--embed-provider",
#         choices=["nvidia", "ollama", "sentence-transformers"],
#         default="nvidia",
#         help="Provider for embeddings: 'nvidia' (default), 'ollama', or 'sentence-transformers' (BGE-M3, fully local).",
#     )
#     parser.add_argument(
#         "--max-workers",
#         type=int,
#         default=4,
#         help="Number of parallel workers for KG transforms and generation (default: 4).",
#     )
#     return parser.parse_args()


# # ══════════════════════════════════════════════════════════════════════════════
# # HELPERS
# # ══════════════════════════════════════════════════════════════════════════════

# def banner(text: str) -> None:
#     bar = "─" * 60
#     print(f"\n{bar}\n  {text}\n{bar}")


# def check_env(llm_provider: str, embed_provider: str) -> None:
#     # Only require NVIDIA_API_KEY if at least one component uses nvidia
#     if (llm_provider == "nvidia" or embed_provider == "nvidia") and not os.environ.get("NVIDIA_API_KEY"):
#         print(
#             "\n❌  NVIDIA_API_KEY is not set.\n"
#             "   export NVIDIA_API_KEY='nvapi-...'\n"
#         )
#         sys.exit(1)


# def make_output_dir(path: str) -> Path:
#     p = Path(path)
#     p.mkdir(parents=True, exist_ok=True)
#     return p


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 1 — LOAD PDFs
# # ══════════════════════════════════════════════════════════════════════════════

# def load_pdfs(pdf_dir: str):
#     """
#     Load PDFs using LangChain DirectoryLoader + PyPDFLoader.
#     Sets 'filename' metadata — required by RAGAS for multi-hop generation.
#     Source ref: RAGAS Node constructor expects document_metadata with filename.

#     Returns:
#         docs         — list of LangChain Document objects
#         context_meta — dict mapping chunk text prefix -> {source, page, filename}
#                        Used in export to attach source metadata to each Q&A row.
#                        PyPDFLoader sets source (full path) and page (0-indexed).
#     """
#     from langchain_community.document_loaders import DirectoryLoader
#     from langchain_community.document_loaders import PyPDFLoader

#     pdf_path = Path(pdf_dir)
#     if not pdf_path.exists():
#         print(f"\n❌  PDF directory not found: {pdf_dir}")
#         sys.exit(1)

#     pdf_files = list(pdf_path.glob("**/*.pdf"))
#     if not pdf_files:
#         print(f"\n❌  No PDF files found in: {pdf_dir}")
#         sys.exit(1)

#     print(f"  Found {len(pdf_files)} PDF file(s)")

#     loader = DirectoryLoader(
#         str(pdf_path),
#         glob="**/*.pdf",
#         loader_cls=PyPDFLoader,
#         show_progress=True,
#     )
#     docs = loader.load()

#     # RAGAS multi-hop synthesizers need 'filename' in metadata
#     # to identify which chunks come from the same document
#     for doc in docs:
#         if "filename" not in doc.metadata:
#             doc.metadata["filename"] = doc.metadata.get("source", "unknown")

#     # Build text -> metadata lookup map.
#     # Key: first 300 chars of page_content (enough to uniquely identify a chunk).
#     # Value: source (full path), page (0-indexed), filename.
#     # Used later in export_testset() to attach source info to each Q&A record.
#     context_meta = {}
#     for doc in docs:
#         key = doc.page_content[:300]
#         context_meta[key] = {
#             "source":   doc.metadata.get("source", ""),
#             "page":     doc.metadata.get("page", ""),
#             "filename": doc.metadata.get("filename", ""),
#         }

#     print(f"  Loaded {len(docs)} page(s)")
#     return docs, context_meta


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 2 — MODEL CONFIGURATION
# # ══════════════════════════════════════════════════════════════════════════════

# # ┌─────────────────────────────────────────────────────────────────────────┐
# # │  NVIDIA NIM settings (used when --provider nvidia)                      │
# # │                                                                         │
# # │  LLM options:                                                           │
# # │    "meta/llama-3.1-8b-instruct"       <- fast, free tier friendly      │
# # │    "meta/llama-3.1-70b-instruct"      <- better quality                │
# # │    "mistralai/mixtral-8x7b-instruct-v0.1"                               │
# # │    "nvidia/nemotron-4-340b-instruct"  <- highest quality                │
# # │                                                                         │
# # │  Embedding options:                                                     │
# # │    "nvidia/nv-embedqa-e5-v5"          <- recommended                   │
# # │    "nvidia/nv-embed-v1"                                                 │
# # │    "baai/bge-m3"                                                        │
# # └─────────────────────────────────────────────────────────────────────────┘

# NVIDIA_LLM_MODEL       = "meta/llama-3.1-70b-instruct"   # <- change me
# NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"       # <- change me
# SENTENCE_TRANSFORMER_EMBEDDING_MODEL = "BAAI/bge-m3"     # HuggingFace BGE-M3
# NVIDIA_BASE_URL        = "https://integrate.api.nvidia.com/v1"

# # ┌─────────────────────────────────────────────────────────────────────────┐
# # │  Ollama settings (used when --provider ollama)                          │
# # │                                                                         │
# # │  Requires:  ollama serve  (running locally on port 11434)               │
# # │  Install:   pip install langchain-ollama                                │
# # │                                                                         │
# # │  LLM options (must be pulled first: ollama pull <model>):               │
# # │    "llama3.1"          <- good general purpose                          │
# # │    "llama3.2"          <- smaller, faster                               │
# # │    "mistral"           <- good quality                                  │
# # │    "gemma2"            <- lightweight alternative                       │
# # │                                                                         │
# # │  Embedding options (must be pulled first: ollama pull <model>):         │
# # │    "nomic-embed-text"  <- recommended, small and fast                   │
# # │    "mxbai-embed-large" <- higher quality                                │
# # │    "all-minilm"        <- very lightweight                              │
# # └─────────────────────────────────────────────────────────────────────────┘


# OLLAMA_LLM_MODEL       = "Qwen/Qwen2.5-32B-Instruct-AWQ"          # <- change me
# OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # <- change me
# OLLAMA_BASE_URL        = "http://localhost:8011/v1"



# def build_llm(llm_provider: str):
#     """
#     Build LLM for the selected provider, wrapped in LangchainLLMWrapper.
#     Source ref: ragas.llms.LangchainLLMWrapper
#     """
#     from ragas.llms import LangchainLLMWrapper

#     if llm_provider == "nvidia":
#         from langchain_nvidia_ai_endpoints import ChatNVIDIA
#         print(f"  LLM provider : NVIDIA NIM")
#         print(f"  LLM model    : {NVIDIA_LLM_MODEL}")
#         llm = ChatNVIDIA(
#             model=NVIDIA_LLM_MODEL,
#             nvidia_api_key=os.environ["NVIDIA_API_KEY"],
#             base_url=NVIDIA_BASE_URL,
#             temperature=0.1,
#             max_tokens=1024,
#         )

#     # else:  # ollama
#     #     from langchain_ollama import ChatOllama
#     #     print(f"  LLM provider : Ollama (local)")
#     #     print(f"  LLM model    : {OLLAMA_LLM_MODEL}")
#     #     print(f"  LLM base URL : {OLLAMA_BASE_URL}")
#     #     llm = ChatOllama(
#     #         model=OLLAMA_LLM_MODEL,
#     #         base_url=OLLAMA_BASE_URL,
#     #         temperature=0.1,
#     #     )
#     else:  # ollama → now points to vLLM
#         import httpx
#         from langchain_openai import ChatOpenAI
#         print(f"  LLM provider : vLLM (local)")
#         print(f"  LLM model    : {OLLAMA_LLM_MODEL}")
#         llm = ChatOpenAI(
#             model=OLLAMA_LLM_MODEL,
#             openai_api_key="dummy",
#             openai_api_base=OLLAMA_BASE_URL,
#             temperature=0.1,
#             max_tokens=1024,
#             http_client=httpx.Client(
#                 timeout=httpx.Timeout(120.0),
#                 limits=httpx.Limits(
#                     max_keepalive_connections=5,
#                     max_connections=10,
#                     keepalive_expiry=60,
#                 ),
#             ),
#             http_async_client=httpx.AsyncClient(
#                 timeout=httpx.Timeout(120.0),
#                 limits=httpx.Limits(
#                     max_keepalive_connections=5,
#                     max_connections=10,
#                     keepalive_expiry=60,
#                 ),
#             ),
#         )

#     return LangchainLLMWrapper(llm)


# def build_embeddings(embed_provider: str):
#     """
#     Build embeddings for the selected provider, wrapped in LangchainEmbeddingsWrapper.
#     Source ref: ragas.embeddings.LangchainEmbeddingsWrapper
#     """
#     from ragas.embeddings import LangchainEmbeddingsWrapper

#     if embed_provider == "nvidia":
#         from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
#         print(f"  Embed provider : NVIDIA NIM")
#         print(f"  Embed model    : {NVIDIA_EMBEDDING_MODEL}")
#         emb = NVIDIAEmbeddings(
#             model=NVIDIA_EMBEDDING_MODEL,
#             nvidia_api_key=os.environ["NVIDIA_API_KEY"],
#             base_url=NVIDIA_BASE_URL,
#         )

#     elif embed_provider == "ollama":
#         from langchain_ollama import OllamaEmbeddings
#         print(f"  Embed provider : Ollama (local)")
#         print(f"  Embed model    : {OLLAMA_EMBEDDING_MODEL}")
#         emb = OllamaEmbeddings(
#             model=OLLAMA_EMBEDDING_MODEL,
#             base_url=OLLAMA_BASE_URL,
#         )

#     else:  # sentence-transformers
#         from langchain_huggingface import HuggingFaceEmbeddings
#         print(f"  Embed provider : Sentence Transformers (local)")
#         print(f"  Embed model    : {SENTENCE_TRANSFORMER_EMBEDDING_MODEL}")
#         emb = HuggingFaceEmbeddings(
#             model_name=SENTENCE_TRANSFORMER_EMBEDDING_MODEL,
#             model_kwargs={"device": "cpu"},        # change to "cuda" if you have a GPU
#             encode_kwargs={"normalize_embeddings": True},  # BGE-M3 benefits from normalization
#         )

#     return LangchainEmbeddingsWrapper(emb)


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 3 — KNOWLEDGE GRAPH
# # ══════════════════════════════════════════════════════════════════════════════

# def build_run_config(timeout: int = 300, max_workers: int = 4):
#     """
#     RunConfig for the transform and generation steps.
#     Source ref: ragas.run_config.RunConfig
#     max_retries=10 + max_wait=120 handles 429s via exponential backoff.
#     """
#     from ragas.run_config import RunConfig
#     return RunConfig(
#         timeout=600,
#         max_retries=15,
#         max_wait=180,
#         max_workers=max_workers,
#         seed=42,
#     )


# def make_safe_headline_splitter():
#     """
#     Returns a HeadlineSplitter subclass that silently skips nodes where
#     HeadlinesExtractor found no headlines (instead of crashing with ValueError).

#     Nodes without headlines are kept as-is (full-page nodes) in the KG.
#     Nodes with headlines are split into sub-chunks as normal.

#     This makes the pipeline robust for 100+ PDFs where some pages have no
#     headings (tables, references, figure captions, short paragraphs, etc).
#     """
#     from ragas.testset.transforms.splitters.headline import HeadlineSplitter

#     class SafeHeadlineSplitter(HeadlineSplitter):
#         async def split(self, node):
#             headlines = node.properties.get("headlines")
#             if not headlines:
#                 # No headlines found — skip silently, keep node as-is in KG
#                 return [], []
#             return await super().split(node)

#     return SafeHeadlineSplitter()



# def _checkpoint_path(output_dir, step_name):
#     """Return path for a per-step KG checkpoint file."""
#     return output_dir / f"kg_checkpoint_{step_name}.json"


# def _already_applied(kg, step_name):
#     """
#     Check if a transform step was already applied by looking for its
#     expected output property on any KG node.
#     Relationship builders have no node properties so always return False
#     (they are fast, no LLM calls, safe to re-run).
#     """
#     STEP_PROPERTY_MAP = {
#         "NERExtractor":        "entities",
#         "KeyphrasesExtractor": "keyphrases",
#         "HeadlinesExtractor":  "headlines",
#         "SummaryExtractor":    "summary",
#     }
#     prop = STEP_PROPERTY_MAP.get(step_name)
#     if prop is None:
#         return False
#     return any(prop in node.properties for node in kg.nodes)


# def build_knowledge_graph(docs, llm, embeddings, max_workers=4, output_dir=None):
#     """
#     Build KnowledgeGraph with per-step checkpointing.

#     Each transform step runs individually. After each step the KG is saved:
#       kg_checkpoint_NERExtractor.json
#       kg_checkpoint_KeyphrasesExtractor.json
#       kg_checkpoint_HeadlinesExtractor.json
#       kg_checkpoint_HeadlineSplitter.json
#       kg_checkpoint_SummaryExtractor.json
#       kg_checkpoint_CosineSimilarityBuilder.json
#       kg_checkpoint_OverlapScoreBuilder.json

#     On re-run after a crash:
#       - Latest checkpoint is loaded automatically
#       - Completed steps are detected and skipped (no redundant LLM calls)
#       - Only remaining steps execute

#     Source ref:
#       ragas.testset.graph       -> KnowledgeGraph, Node, NodeType
#       ragas.testset.transforms  -> default_transforms, apply_transforms
#     """
#     from ragas.testset.graph import KnowledgeGraph, Node, NodeType
#     from ragas.testset.transforms import default_transforms, apply_transforms
#     from ragas.testset.transforms.splitters.headline import HeadlineSplitter

#     ckpt_dir = output_dir if output_dir else Path(".")

#     # Try to load the most recently completed checkpoint (search in reverse order)
#     STEP_ORDER_REVERSED = [
#         "OverlapScoreBuilder",
#         "CosineSimilarityBuilder",
#         "SummaryExtractor",
#         "HeadlineSplitter",
#         "HeadlinesExtractor",
#         "KeyphrasesExtractor",
#         "NERExtractor",
#     ]
#     kg = None
#     for step_name in STEP_ORDER_REVERSED:
#         ckpt = _checkpoint_path(ckpt_dir, step_name)
#         if ckpt.exists():
#             print(f"  Resuming from checkpoint: {ckpt.name}")
#             kg = KnowledgeGraph.load(str(ckpt))
#             print(f"     nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
#             break

#     # No checkpoint found -- build fresh KG from documents
#     if kg is None:
#         print("  Creating empty KnowledgeGraph ...")
#         kg = KnowledgeGraph()
#         for doc in docs:
#             kg.nodes.append(
#                 Node(
#                     type=NodeType.DOCUMENT,
#                     properties={
#                         "page_content": doc.page_content,
#                         "document_metadata": doc.metadata,
#                     },
#                 )
#             )
#         print(f"  Added {len(kg.nodes)} DOCUMENT node(s)")
#     else:
#         print(f"  Using {len(kg.nodes)} node(s) from checkpoint -- skipping doc reload")

#     # Build the full transform pipeline
#     print("\n  Building transform pipeline ...")
#     trans = default_transforms(documents=docs, llm=llm, embedding_model=embeddings)

#     # Swap HeadlineSplitter -> SafeHeadlineSplitter
#     safe_splitter = make_safe_headline_splitter()
#     for i, t in enumerate(trans):
#         if isinstance(t, HeadlineSplitter):
#             trans[i] = safe_splitter
#             break

#     run_config = build_run_config(timeout=600, max_workers=max_workers)

#     # Apply each step individually, saving a checkpoint after each one
#     print("\n  Applying transforms (checkpoint saved after each step) ...\n")
#     print(f"  {'Step':<38} Status")
#     print(f"  {'-'*38} {'-'*22}")

#     for transform in trans:
#         step_name = type(transform).__name__

#         if _already_applied(kg, step_name):
#             print(f"  {'  ' + step_name:<38} [skipped - already in KG]")
#             continue

#         print(f"  {'  ' + step_name:<38} [running ...]")
#         t_start = time.time()
#         try:
#             apply_transforms(kg, [transform], run_config)
#             elapsed = time.time() - t_start
#             ckpt = _checkpoint_path(ckpt_dir, step_name)
#             kg.save(str(ckpt))
#             print(f"  {'  ' + step_name:<38} [done {elapsed:.1f}s -> {ckpt.name}]")
#         except Exception as e:
#             elapsed = time.time() - t_start
#             print(f"  {'  ' + step_name:<38} [FAILED {elapsed:.1f}s]")
#             print(f"     {type(e).__name__}: {e}")
#             print(f"\n  WARNING: Interrupted at step: {step_name}")
#             print(f"  WARNING: Re-run to resume from last checkpoint automatically.")
#             raise

#     print(f"\n  KG complete -- nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
#     return kg


# # def build_knowledge_graph(docs, llm, embeddings, max_workers: int = 4):
# #     """
# #     Build KnowledgeGraph exactly as RAGAS source and docs show:

# #       1. Create empty KnowledgeGraph()
# #       2. Append Node(type=NodeType.DOCUMENT, ...) for each doc
# #       3. Run default_transforms(documents, llm, embedding_model)
# #       4. apply_transforms(kg, trans, run_config)
# #       5. kg.save(path)

# #     Source ref:
# #       ragas.testset.graph       -> KnowledgeGraph, Node, NodeType
# #       ragas.testset.transforms  -> default_transforms, apply_transforms
# #     """
# #     from ragas.testset.graph import KnowledgeGraph, Node, NodeType
# #     from ragas.testset.transforms import default_transforms, apply_transforms
# #     from ragas.testset.transforms.splitters.headline import HeadlineSplitter

# #     print("  Creating empty KnowledgeGraph ...")
# #     kg = KnowledgeGraph()

# #     # Exact pattern from RAGAS official source and docs:
# #     for doc in docs:
# #         kg.nodes.append(
# #             Node(
# #                 type=NodeType.DOCUMENT,
# #                 properties={
# #                     "page_content": doc.page_content,
# #                     "document_metadata": doc.metadata,
# #                 },
# #             )
# #         )
# #     print(f"  Added {len(kg.nodes)} DOCUMENT node(s)")

# #     print("\n  Running default_transforms ...")
# #     print("  Enrichments: NERExtractor, KeyphrasesExtractor, HeadlinesExtractor,")
# #     print("               SummaryExtractor, CosineSimilarityBuilder, OverlapScoreBuilder")
# #     print(f"  Est. LLM calls : ~{len(docs) * 4}  (4 extractors x {len(docs)} chunks)\n")

# #     # default_transforms signature (source): default_transforms(documents, llm, embedding_model)
# #     trans = default_transforms(
# #         documents=docs,
# #         llm=llm,
# #         embedding_model=embeddings,
# #     )

# #     # Swap out the default HeadlineSplitter with SafeHeadlineSplitter so that
# #     # nodes missing the 'headlines' property are skipped instead of crashing.
# #     safe_splitter = make_safe_headline_splitter()
# #     for i, t in enumerate(trans):
# #         if isinstance(t, HeadlineSplitter):
# #             trans[i] = safe_splitter
# #             print("  ✔  Swapped HeadlineSplitter -> SafeHeadlineSplitter")
# #             break

# #     # apply_transforms signature (source):
# #     # apply_transforms(kg, transforms, run_config=RunConfig(), callbacks=None)
# #     run_config = build_run_config(timeout=300, max_workers=max_workers)
# #     apply_transforms(kg, trans, run_config)

# #     print(f"\n  KG complete — nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
# #     return kg


# def save_kg(kg, path: Path) -> None:
#     # Source: KnowledgeGraph.save(path: str)
#     kg.save(str(path))
#     print(f"  KG saved -> {path}")


# def load_kg(path: Path):
#     # Source: KnowledgeGraph.load(path: str) -> KnowledgeGraph
#     from ragas.testset.graph import KnowledgeGraph
#     print(f"  Loading KG from {path} ...")
#     kg = KnowledgeGraph.load(str(path))
#     print(f"  KG loaded — nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
#     return kg


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 4 — QUERY DISTRIBUTION
# # ══════════════════════════════════════════════════════════════════════════════

# def get_query_distribution(llm):
#     """
#     Use default_query_distribution() directly — this is what RAGAS itself uses
#     internally when no distribution is provided.

#     Source: ragas/src/ragas/testset/synthesizers/__init__.py
#     The function produces exactly:
#         [
#             (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
#             (MultiHopAbstractQuerySynthesizer(llm=llm),  0.25),
#             (MultiHopSpecificQuerySynthesizer(llm=llm),  0.25),
#         ]

#     SingleHopSpecificQuerySynthesizer source default:
#         property_name = "entities"   <- populated by NERExtractor in default_transforms
#     MultiHopSpecificQuerySynthesizer source default:
#         property_name = "entities"
#         relation_type = "entities_overlap"  <- populated by OverlapScoreBuilder

#     These defaults are guaranteed to work with default_transforms output.
#     This is the ONLY officially supported distribution in RAGAS v0.2+.
#     """
#     # from ragas.testset.synthesizers import default_query_distribution

#     # distribution = default_query_distribution(llm)

#     from ragas.testset.synthesizers import (
#         SingleHopSpecificQuerySynthesizer,
#         MultiHopAbstractQuerySynthesizer,
#         MultiHopSpecificQuerySynthesizer,
#     )

#     distribution = [
#         (SingleHopSpecificQuerySynthesizer(llm=llm), 0.40),
#         (MultiHopAbstractQuerySynthesizer(llm=llm),  0.30),
#         (MultiHopSpecificQuerySynthesizer(llm=llm),  0.30),
#     ]

#     print("  Source: ragas.testset.synthesizers.default_query_distribution()")
#     print()
#     print("  ┌──────────────────────────────────────────────────────┬────────┐")
#     print("  │ Synthesizer                                          │ Weight │")
#     print("  ├──────────────────────────────────────────────────────┼────────┤")
#     for synth, weight in distribution:
#         name = type(synth).__name__
#         print(f"  │  {name:<51} │  {weight:.2f}  │")
#     print("  └──────────────────────────────────────────────────────┴────────┘")
#     print()
#     print("  Question types covered:")
#     print("    SingleHopSpecific (50%) — factual, from single chunk entities")
#     print("    MultiHopAbstract  (25%) — reasoning across multiple chunks")
#     print("    MultiHopSpecific  (25%) — factual across multiple chunks")

#     return distribution


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 5 — GENERATE TESTSET
# # ══════════════════════════════════════════════════════════════════════════════

# def generate_testset(kg, llm, embeddings, num_questions: int, query_distribution, max_workers: int = 4):
#     """
#     Run TestsetGenerator.generate() exactly as RAGAS source and docs show.

#     Source ref: ragas.testset.TestsetGenerator
#     Constructor: TestsetGenerator(llm, embedding_model, knowledge_graph,
#                                   persona_list=None, llm_context=None)
#     generate() sig: testset_size, query_distribution, num_personas,
#                     run_config, with_debugging_logs, raise_exceptions
#     """
#     from ragas.testset import TestsetGenerator

#     # Source: TestsetGenerator(llm, embedding_model, knowledge_graph)
#     generator = TestsetGenerator(
#         llm=llm,
#         embedding_model=embeddings,
#         knowledge_graph=kg,
#         # persona_list=None -> RAGAS auto-generates personas from KG summaries
#     )

#     # Synthesis prompts are longer than extraction prompts -> larger timeout
#     run_config = build_run_config(timeout=300, max_workers=max_workers)

#     print(f"  Generating {num_questions} samples ...")
#     print(f"  max_workers={max_workers}\n")

#     # Source: generate(testset_size, query_distribution, num_personas,
#     #                  run_config, with_debugging_logs, raise_exceptions)
#     try:
#         testset = generator.generate(
#             testset_size=num_questions,
#             query_distribution=query_distribution,
#             num_personas=3,           # auto-generate 3 personas from KG
#             run_config=run_config,
#             with_debugging_logs=False,
#             raise_exceptions=False,   # log failures, don't abort the whole run
#         )
#     except TypeError as e:
#         # Known RAGAS bug: when ALL scenario generation attempts fail (e.g. due to
#         # connection errors), RAGAS returns a float instead of a list and crashes
#         # with "TypeError: float object is not iterable".
#         # Root cause: APIConnectionError during scenario generation — check vLLM
#         # server is healthy and reachable, then re-run.
#         print(f"\n  ERROR: Testset generation failed — {e}")
#         print("  This usually means ALL scenario generation attempts failed.")
#         print("  Check: is your vLLM server running and reachable?")
#         print("  Check: APIConnectionError in the logs above.")
#         print("  Fix:   ensure the server is healthy then re-run.\n")
#         sys.exit(1)

#     return testset


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 6 — EXPORT
# # ══════════════════════════════════════════════════════════════════════════════

# def _lookup_context_meta(context_text: str, context_meta: dict) -> dict:
#     """
#     Look up source metadata for a context chunk using its first 300 chars as key.
#     Strips RAGAS multi-hop hop-labels (<1-hop>, <2-hop> etc.) before matching.
#     Returns dict with source, page, filename — empty strings if not found.
#     """
#     import re
#     # Strip RAGAS multi-hop labels added to context strings
#     clean = re.sub(r"<\d+-hop>\s*", "", context_text).strip()
#     key = clean[:300]
#     return context_meta.get(key, {"source": "", "page": "", "filename": ""})


# def export_testset(testset, output_dir: Path, csv_filename: str, context_meta: dict = None) -> None:
#     """
#     Export to CSV and JSON.
#     Source ref: Testset.to_pandas() returns DataFrame with columns:
#       user_input, reference_contexts, reference, synthesizer_name

#     If context_meta is provided (text->metadata lookup built in load_pdfs),
#     each record is enriched with source metadata per context chunk:
#       source   — full file path (set by PyPDFLoader)
#       page     — page number 0-indexed (set by PyPDFLoader)
#       filename — PDF filename (set by load_pdfs)
#     For MultiHop rows with multiple contexts, these are lists.
#     For SingleHop rows with one context, these are single values.
#     """
#     import ast

#     if context_meta is None:
#         context_meta = {}

#     df = testset.to_pandas()

#     if df.empty:
#         print("\n  ⚠️  Testset is empty — no questions were generated.")
#         print("  Possible causes:")
#         print("    - PDFs too short or too few (need substantial text)")
#         print("    - NER extraction failed (no 'entities' nodes in KG)")
#         print("    - All generation attempts raised exceptions (raise_exceptions=False)")
#         print("  Try: re-run with more/longer PDFs, or check logs above.\n")
#         return

#     # ── CSV ──────────────────────────────────────────────────────────────────
#     csv_path = output_dir / csv_filename
#     df.to_csv(csv_path, index=False)
#     print(f"  CSV  -> {csv_path}")

#     # ── JSON with metadata ────────────────────────────────────────────────────
#     json_path = output_dir / csv_filename.replace(".csv", ".json")
#     records = []
#     for _, row in df.iterrows():
#         ref_ctx = row.get("reference_contexts", [])
#         # Pandas may serialize lists as strings — parse them back
#         if isinstance(ref_ctx, str):
#             try:
#                 ref_ctx = ast.literal_eval(ref_ctx)
#             except Exception:
#                 ref_ctx = [ref_ctx]

#         ref_ctx = ref_ctx if isinstance(ref_ctx, list) else []

#         # Resolve source metadata for each context chunk
#         sources   = [_lookup_context_meta(c, context_meta).get("source",   "") for c in ref_ctx]
#         pages     = [_lookup_context_meta(c, context_meta).get("page",     "") for c in ref_ctx]
#         filenames = [_lookup_context_meta(c, context_meta).get("filename", "") for c in ref_ctx]

#         # For SingleHop (1 context), unwrap to scalar for cleaner output
#         source_out   = sources[0]   if len(sources)   == 1 else sources
#         page_out     = pages[0]     if len(pages)      == 1 else pages
#         filename_out = filenames[0] if len(filenames)  == 1 else filenames

#         records.append({
#             "user_input":         row.get("user_input", ""),
#             "reference":          row.get("reference", ""),
#             "reference_contexts": ref_ctx,
#             "synthesizer_name":   row.get("synthesizer_name", ""),
#             "metadata": {
#                 "synthesizer_name": row.get("synthesizer_name", ""),
#                 "num_contexts":     len(ref_ctx),
#                 "source":           source_out,
#                 "page":             page_out,
#                 "filename":         filename_out,
#             },
#         })

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(records, f, indent=2, ensure_ascii=False)
#     print(f"  JSON -> {json_path}")

#     # ── Distribution summary ──────────────────────────────────────────────────
#     print("\n  Question-type distribution:")
#     dist  = df["synthesizer_name"].value_counts()
#     total = len(df)
#     for name, count in dist.items():
#         pct = count / total * 100
#         bar = "█" * int(pct / 2)
#         print(f"    {name:<52} {count:>3}  ({pct:5.1f}%)  {bar}")
#     print(f"    {'TOTAL':<52} {total:>3}\n")


# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def main() -> None:
#     args       = parse_args()
#     check_env(args.llm_provider, args.embed_provider)
#     output_dir = make_output_dir(args.output_dir)
#     kg_path    = output_dir / args.kg_file

#     print(
#         f"\n  LLM        : {args.llm_provider.upper()} "
#         f"({'NVIDIA NIM' if args.llm_provider == 'nvidia' else 'Ollama local'})\n"
#         f"  Embeddings : {args.embed_provider.upper()} "
#         f"({'NVIDIA NIM' if args.embed_provider == 'nvidia' else 'Ollama local' if args.embed_provider == 'ollama' else 'Sentence Transformers local (BGE-M3)'})\n"
#         f"  Mode       : max_workers={args.max_workers}, retries=10, max_wait=120s\n"
#         f"  Target     : {args.num_questions} question(s)\n"
#     )

#     # Step 1
#     banner("Step 1 · Load PDFs")
#     docs, context_meta = load_pdfs(args.pdf_dir)

#     # Step 2
#     banner("Step 2 · Models")
#     llm        = build_llm(args.llm_provider)
#     embeddings = build_embeddings(args.embed_provider)

#     # Step 3
#     banner("Step 3 · Knowledge Graph")
#     if args.load_kg and kg_path.exists():
#         kg = load_kg(kg_path)
#     else:
#         if args.load_kg:
#             print(f"  ⚠  --load-kg set but '{kg_path}' not found — building from scratch")
#         # kg = build_knowledge_graph(docs, llm, embeddings, args.max_workers)
#         kg = build_knowledge_graph(docs, llm, embeddings, args.max_workers, output_dir)
#         save_kg(kg, kg_path)

#     # Step 4
#     banner("Step 4 · Query Distribution")
#     query_distribution = get_query_distribution(llm)

#     # Step 5
#     banner(f"Step 5 · Generate {args.num_questions} Questions")
#     t0      = time.time()
#     testset = generate_testset(kg, llm, embeddings, args.num_questions, query_distribution, args.max_workers)
#     elapsed = time.time() - t0
#     print(f"\n  Completed in {elapsed:.1f}s")

#     # Step 6
#     banner("Step 6 · Export")
#     export_testset(testset, output_dir, args.dataset_file, context_meta)

#     banner("Done")
#     print(f"  All outputs saved to: {output_dir.resolve()}\n")


# if __name__ == "__main__":
#     main()
