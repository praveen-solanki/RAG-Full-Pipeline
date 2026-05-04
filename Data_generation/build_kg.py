# """
# Stage 0: Build the knowledge graph from PDFs.

# This script does four things:
#   1. Loads PDFs via PyPDFLoader (one Document per page).
#   2. Applies boilerplate stripping and a node-quality pre-filter to drop
#      TOCs, copyright pages, and pages that are too short to be useful.
#   3. Builds a RAGAS KnowledgeGraph using RAGAS's own transforms pipeline
#      (headline splitting, NER, keyphrases, summaries, embeddings, cosine
#      similarity relationships). We keep this part of RAGAS because the
#      transforms are solid — the quality issues come from the generator,
#      not the KG builder.
#   4. Saves knowledge_graph.json atomically.

# This script uses the SAME model you will use for generation (Qwen2.5-72B-
# Instruct-AWQ by default) because the transform extractors are LLM-based.

# Typical runtime: 20-60 minutes for ~100 pages, depending on PDF density.
# The KG is reusable: once built, you can run generate_candidates.py multiple
# times against it (different personas, different distributions, different
# target sizes) without rebuilding.

# Usage:
#     python build_kg.py --pdf-dir ./pdfs --output-dir ./output

# Resumability:
#     - Final output: output/kg/knowledge_graph.json
#     - If it already exists, this script does nothing (unless --force).
# """

# from __future__ import annotations

# import argparse
# import sys
# import time
# from pathlib import Path

# from shared.io_utils import atomic_write_json, read_json
# from shared.validators import strip_boilerplate, check_context_has_substance

# # ══════════════════════════════════════════════════════════════════════════════
# # DEFAULTS
# # ══════════════════════════════════════════════════════════════════════════════

# DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
# DEFAULT_EMBED_MODEL     = "BAAI/bge-m3"  # Strong open-source multilingual embedder


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     p.add_argument("--pdf-dir",     required=True, help="Directory containing .pdf files")
#     p.add_argument("--output-dir",  required=True, help="Output directory")
#     p.add_argument("--llm-model",   default=DEFAULT_GENERATOR_MODEL)
#     p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
#     p.add_argument("--tensor-parallel-size", type=int, default=2)
#     p.add_argument("--max-model-len",        type=int, default=8192)
#     p.add_argument("--quantization",         default="awq",
#                    help="vLLM quantization ('awq', 'fp8', 'gptq', or '' for none)")
#     p.add_argument("--max-workers", type=int, default=8,
#                    help="RAGAS transform parallelism")
#     p.add_argument("--force", action="store_true", help="Rebuild even if KG exists")
#     return p.parse_args()


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 1 — LOAD PDFs
# # ══════════════════════════════════════════════════════════════════════════════

# def load_pdfs(pdf_dir: Path) -> list:
#     from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

#     if not pdf_dir.exists():
#         sys.exit(f"PDF directory not found: {pdf_dir}")

#     pdfs = list(pdf_dir.glob("**/*.pdf"))
#     if not pdfs:
#         sys.exit(f"No PDFs found in: {pdf_dir}")
#     print(f"  Found {len(pdfs)} PDF(s)")

#     loader = DirectoryLoader(
#         str(pdf_dir), glob="**/*.pdf",
#         loader_cls=PyPDFLoader, show_progress=True,
#     )
#     docs = loader.load()

#     for d in docs:
#         # Strip boilerplate lines up-front so they don't pollute the KG
#         d.page_content = strip_boilerplate(d.page_content)
#         # Normalize metadata
#         if "filename" not in d.metadata:
#             d.metadata["filename"] = d.metadata.get("source", "unknown")

#     print(f"  Loaded {len(docs)} pages")
#     return docs


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 2 — FILTER LOW-QUALITY PAGES BEFORE KG CONSTRUCTION
# # ══════════════════════════════════════════════════════════════════════════════

# def filter_pages(docs: list) -> list:
#     """
#     Drop pages that are boilerplate-only, TOC-only, or too short.
#     This is the cheapest, most effective quality fix in the whole pipeline.
#     """
#     from shared.validators import check_context_not_toc, check_context_has_substance

#     kept = []
#     dropped_by_reason: dict[str, int] = {}

#     for d in docs:
#         text = d.page_content
#         ok_toc,  reason_toc  = check_context_not_toc(text)
#         ok_subs, reason_subs = check_context_has_substance(text, min_chars=300)
#         if not ok_toc:
#             dropped_by_reason[reason_toc] = dropped_by_reason.get(reason_toc, 0) + 1
#             continue
#         if not ok_subs:
#             dropped_by_reason[reason_subs] = dropped_by_reason.get(reason_subs, 0) + 1
#             continue
#         kept.append(d)

#     print(f"  Kept {len(kept)} / {len(docs)} pages after quality filter")
#     for reason, count in sorted(dropped_by_reason.items(), key=lambda x: -x[1]):
#         print(f"     dropped {count:>4} :: {reason}")
#     return kept


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 3 — BUILD LLM + EMBEDDINGS (vLLM offline + BGE-M3)
# # ══════════════════════════════════════════════════════════════════════════════

# def build_llm(args: argparse.Namespace):
#     """
#     Build a RAGAS-compatible LLM wrapper that calls vLLM offline-batch.
    
#     Note: RAGAS's transforms expect a LangchainLLMWrapper. We use a local
#     vLLM server proxied through LangChain's OpenAI-compatible client because
#     the RAGAS transforms issue many small LLM calls and LLM.chat() offline-
#     batch is awkward to plug into their async executor.
    
#     So: YOU MUST START THE vLLM SERVER BEFORE RUNNING THIS SCRIPT.
    
#     Example launch (in a separate terminal):
#         vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \\
#             --tensor-parallel-size 2 \\
#             --max-model-len 8192 \\
#             --gpu-memory-utilization 0.90 \\
#             --quantization awq \\
#             --port 8011
    
#     For generate_candidates.py and validate_candidates.py we use offline-batch
#     directly (no server needed) because those are throughput-bound batch jobs.
#     """
#     from langchain_openai import ChatOpenAI
#     from ragas.llms import LangchainLLMWrapper

#     llm = ChatOpenAI(
#         model=args.llm_model,
#         openai_api_key="dummy",
#         openai_api_base="http://localhost:8011/v1",
#         temperature=0.1,
#         max_tokens=1024,
#         timeout=300,
#     )
#     return LangchainLLMWrapper(llm)


# def build_embeddings(args: argparse.Namespace):
#     from langchain_huggingface import HuggingFaceEmbeddings
#     from ragas.embeddings import LangchainEmbeddingsWrapper

#     emb = HuggingFaceEmbeddings(
#         model_name=args.embed_model,
#         model_kwargs={"device": "cuda"},  # one GPU is enough for embeddings
#         encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
#     )
#     return LangchainEmbeddingsWrapper(emb)


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 4 — BUILD KNOWLEDGE GRAPH WITH RAGAS TRANSFORMS
# # ══════════════════════════════════════════════════════════════════════════════

# def build_knowledge_graph(docs, llm, embeddings, max_workers: int):
#     from ragas.testset.graph import KnowledgeGraph, Node, NodeType
#     from ragas.testset.transforms import default_transforms, apply_transforms
#     from ragas.testset.transforms.splitters.headline import HeadlineSplitter
#     from ragas.run_config import RunConfig

#     # --- Create initial graph from docs ---
#     kg = KnowledgeGraph()
#     for d in docs:
#         kg.nodes.append(Node(
#             type=NodeType.DOCUMENT,
#             properties={
#                 "page_content": d.page_content,
#                 "document_metadata": d.metadata,
#             },
#         ))
#     print(f"  Created KG with {len(kg.nodes)} DOCUMENT node(s)")

#     # --- Safe HeadlineSplitter: don't crash on nodes without headlines ---
#     class SafeHeadlineSplitter(HeadlineSplitter):
#         async def split(self, node):
#             if not node.properties.get("headlines"):
#                 return [], []
#             return await super().split(node)

#     # --- Get default transforms and swap HeadlineSplitter ---
#     trans = default_transforms(documents=docs, llm=llm, embedding_model=embeddings)

#     def _swap(node_or_list):
#         from ragas.testset.transforms.engine import Parallel
#         if isinstance(node_or_list, Parallel):
#             for i, sub in enumerate(node_or_list.transformations):
#                 if type(sub).__name__ == "HeadlineSplitter":
#                     node_or_list.transformations[i] = SafeHeadlineSplitter()
#                 else:
#                     _swap(sub)
#         elif isinstance(node_or_list, list):
#             for i, sub in enumerate(node_or_list):
#                 if type(sub).__name__ == "HeadlineSplitter":
#                     node_or_list[i] = SafeHeadlineSplitter()
#                 else:
#                     _swap(sub)

#     _swap(trans)

#     # --- Apply transforms ---
#     run_config = RunConfig(
#         timeout=600, max_retries=15, max_wait=180,
#         max_workers=max_workers, seed=42,
#     )
#     print(f"\n  Applying RAGAS transforms (workers={max_workers}) ...")
#     t0 = time.time()
#     apply_transforms(kg, trans, run_config=run_config)
#     print(f"  Transforms done in {time.time() - t0:.0f}s")
#     print(f"  Final KG: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")

#     return kg


# def save_kg(kg, path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     kg.save(str(tmp))
#     tmp.replace(path)
#     print(f"  Saved KG to {path}")


# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def main() -> None:
#     args = parse_args()
#     output_dir = Path(args.output_dir)
#     kg_dir = output_dir / "kg"
#     kg_path = kg_dir / "knowledge_graph.json"

#     if kg_path.exists() and not args.force:
#         print(f"  KG already exists at {kg_path}")
#         print(f"  Use --force to rebuild.")
#         return

#     print("=" * 70)
#     print(" Stage 0 :: Build Knowledge Graph")
#     print("=" * 70)

#     # 1. Load
#     print("\n[1/4] Loading PDFs")
#     docs = load_pdfs(Path(args.pdf_dir))

#     # 2. Filter
#     print("\n[2/4] Filtering low-quality pages")
#     docs = filter_pages(docs)
#     if not docs:
#         sys.exit("All pages were filtered out! Check your PDFs and filter thresholds.")

#     # 3. Models
#     print("\n[3/4] Loading models")
#     print(f"  LLM (remote vLLM server): {args.llm_model}")
#     print(f"  Embeddings (local):       {args.embed_model}")
#     print(f"  Ensure vLLM server is running at http://localhost:8011")
#     llm = build_llm(args)
#     embeddings = build_embeddings(args)

#     # 4. Build KG
#     print("\n[4/4] Building KG with RAGAS transforms")
#     kg = build_knowledge_graph(docs, llm, embeddings, args.max_workers)
#     save_kg(kg, kg_path)

#     # Config dump
#     atomic_write_json(
#         {
#             "llm_model":         args.llm_model,
#             "embed_model":       args.embed_model,
#             "tensor_parallel":   args.tensor_parallel_size,
#             "max_model_len":     args.max_model_len,
#             "quantization":      args.quantization,
#             "n_pages_input":     len(list(Path(args.pdf_dir).glob("**/*.pdf"))),
#             "n_pages_kept":      len(docs),
#         },
#         kg_dir / "kg_build_config.json",
#     )

#     print("\n" + "=" * 70)
#     print(" KG build complete.")
#     print(f" Next: python generate_candidates.py --kg-file {kg_path} ...")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()




"""
Stage 0: Build the knowledge graph from PDFs. (ROBUST VERSION)

This version includes fixes for real-world RAGAS failures:
  1. Per-node exception isolation: if a single node's LLM output can't be
     parsed (common with PDFs containing code, regex, MISRA rules, etc.),
     that node is skipped with property=None instead of killing the whole run.
  2. Per-step checkpointing: after each transform step completes, the KG
     is saved to disk. A crash in step N doesn't lose work from steps 1..N-1.
  3. Resume-from-checkpoint: on rerun, the latest checkpoint is loaded and
     completed steps are skipped.

Usage:
    # Terminal 1: start vLLM server
    vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 2 \\
        --max-model-len 8192 --quantization awq --port 8011

    # Terminal 2:
    python build_kg.py --pdf-dir ./pdfs --output-dir ./output
"""

from __future__ import annotations

import argparse
import sys
import time
import functools
import traceback
from pathlib import Path

from shared.io_utils import atomic_write_json, read_json
from shared.validators import strip_boilerplate, check_context_has_substance


DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
DEFAULT_EMBED_MODEL     = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf-dir",     required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--llm-model",   default=DEFAULT_GENERATOR_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--vllm-url",    default="http://localhost:8011/v1",
                   help="URL of the running vLLM server")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--force", action="store_true",
                   help="Rebuild KG even if final knowledge_graph.json exists")
    p.add_argument("--fresh", action="store_true",
                   help="Discard all checkpoints and restart from scratch")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD PDFs
# ══════════════════════════════════════════════════════════════════════════════

def load_pdfs(pdf_dir: Path) -> list:
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

    if not pdf_dir.exists():
        sys.exit(f"PDF directory not found: {pdf_dir}")
    pdfs = list(pdf_dir.glob("**/*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in: {pdf_dir}")
    print(f"  Found {len(pdfs)} PDF(s)")

    loader = DirectoryLoader(
        str(pdf_dir), glob="**/*.pdf",
        loader_cls=PyPDFLoader, show_progress=True,
    )
    docs = loader.load()
    for d in docs:
        d.page_content = strip_boilerplate(d.page_content)
        if "filename" not in d.metadata:
            d.metadata["filename"] = d.metadata.get("source", "unknown")
    print(f"  Loaded {len(docs)} pages")
    return docs


def filter_pages(docs: list) -> list:
    from shared.validators import check_context_not_toc
    kept = []
    dropped: dict[str, int] = {}
    for d in docs:
        ok_toc, r_toc  = check_context_not_toc(d.page_content)
        ok_sub, r_sub  = check_context_has_substance(d.page_content, min_chars=300)
        if not ok_toc:
            dropped[r_toc] = dropped.get(r_toc, 0) + 1
            continue
        if not ok_sub:
            dropped[r_sub] = dropped.get(r_sub, 0) + 1
            continue
        kept.append(d)
    print(f"  Kept {len(kept)} / {len(docs)} pages after quality filter")
    for r, n in sorted(dropped.items(), key=lambda x: -x[1]):
        print(f"     dropped {n:>5} :: {r}")
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MODELS
# ══════════════════════════════════════════════════════════════════════════════

def build_llm(args: argparse.Namespace):
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model=args.llm_model,
        openai_api_key="dummy",
        openai_api_base=args.vllm_url,
        temperature=0.1,
        max_tokens=1024,
        timeout=300,
    )
    return LangchainLLMWrapper(llm)


def build_embeddings(args: argparse.Namespace):
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = HuggingFaceEmbeddings(
        model_name=args.embed_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    return LangchainEmbeddingsWrapper(emb)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ROBUST EXTRACTOR WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def make_llm_extractor_robust(extractor, step_name: str, fail_counter: dict):
    """
    Wrap the extractor's .extract() method so a parse failure on a single
    node returns (property_name, None) instead of raising and killing the
    whole pipeline.

    RAGAS's raise_exceptions=False flag does NOT reach transform extractors
    cleanly — their fix_output_format retry loop raises RagasOutputParserException
    unconditionally. We intercept it here.
    """
    try:
        from ragas.exceptions import RagasOutputParserException
    except ImportError:
        # older RAGAS versions
        class RagasOutputParserException(Exception):
            pass

    original_extract = extractor.extract

    @functools.wraps(original_extract)
    async def robust_extract(node):
        try:
            return await original_extract(node)
        except RagasOutputParserException:
            fail_counter[step_name] = fail_counter.get(step_name, 0) + 1
            prop = (
                getattr(extractor, "property_name", None)
                or getattr(extractor, "name", None)
                or step_name.lower()
            )
            return (prop, None)
        except Exception as e:
            fail_counter[step_name] = fail_counter.get(step_name, 0) + 1
            if fail_counter[step_name] < 3:
                print(f"     [{step_name}] non-parser exception: "
                      f"{type(e).__name__}: {str(e)[:120]}")
            prop = (
                getattr(extractor, "property_name", None)
                or getattr(extractor, "name", None)
                or step_name.lower()
            )
            return (prop, None)

    extractor.extract = robust_extract
    return extractor


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FLATTEN + CHECKPOINT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def flatten_pipeline(transforms):
    """Walk Parallel/list wrappers, return a flat list of leaf transforms in order."""
    from ragas.testset.transforms.engine import Parallel
    flat = []

    def _walk(node):
        if isinstance(node, Parallel):
            for sub in node.transformations:
                _walk(sub)
            return
        if isinstance(node, (list, tuple)):
            for sub in node:
                _walk(sub)
            return
        flat.append(node)

    _walk(transforms)
    return flat


def _ckpt_path(out_dir: Path, step_name: str) -> Path:
    return out_dir / f"kg_ckpt_{step_name}.json"


def _done_path(out_dir: Path, step_name: str) -> Path:
    return out_dir / f"kg_ckpt_{step_name}.done"


def save_kg_atomic(kg, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    kg.save(str(tmp))
    tmp.replace(path)


def load_latest_checkpoint(kg_dir: Path):
    """Return (kg, last_step_name) or (None, None) if no checkpoint found."""
    from ragas.testset.graph import KnowledgeGraph

    candidates = [p for p in kg_dir.glob("kg_ckpt_*.json")
                  if not p.name.endswith(".tmp")]
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    step_name = latest.stem.replace("kg_ckpt_", "")
    kg = KnowledgeGraph.load(str(latest))
    return kg, step_name


def is_step_done(kg_dir: Path, step_name: str) -> bool:
    return _done_path(kg_dir, step_name).exists()


def mark_step_done(kg_dir: Path, step_name: str, stats: dict) -> None:
    atomic_write_json(stats, _done_path(kg_dir, step_name))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD KG WITH ROBUST PER-STEP EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(docs, llm, embeddings, max_workers: int, kg_dir: Path):
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.testset.transforms.splitters.headline import HeadlineSplitter
    from ragas.testset.transforms.extractors.llm_based import LLMBasedExtractor
    from ragas.run_config import RunConfig

    # --- Resume from latest checkpoint, or start fresh ---
    kg, last_step = load_latest_checkpoint(kg_dir)
    if kg is None:
        kg = KnowledgeGraph()
        for d in docs:
            kg.nodes.append(Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": d.page_content,
                    "document_metadata": d.metadata,
                },
            ))
        print(f"  Fresh KG: {len(kg.nodes)} DOCUMENT node(s)")
    else:
        print(f"  Resumed from checkpoint after step '{last_step}': "
              f"{len(kg.nodes)} nodes, {len(kg.relationships)} relationships")

    # --- Safe HeadlineSplitter ---
    class SafeHeadlineSplitter(HeadlineSplitter):
        async def split(self, node):
            if not node.properties.get("headlines"):
                return [], []
            return await super().split(node)

    # --- Build transforms pipeline ---
    trans = default_transforms(documents=docs, llm=llm, embedding_model=embeddings)

    def _swap(node_or_list):
        from ragas.testset.transforms.engine import Parallel
        if isinstance(node_or_list, Parallel):
            for i, sub in enumerate(node_or_list.transformations):
                if type(sub).__name__ == "HeadlineSplitter":
                    node_or_list.transformations[i] = SafeHeadlineSplitter()
                else:
                    _swap(sub)
        elif isinstance(node_or_list, list):
            for i, sub in enumerate(node_or_list):
                if type(sub).__name__ == "HeadlineSplitter":
                    node_or_list[i] = SafeHeadlineSplitter()
                else:
                    _swap(sub)
    _swap(trans)

    # --- Flatten into leaf transforms ---
    leaf_transforms = flatten_pipeline(trans)
    print(f"\n  Pipeline has {len(leaf_transforms)} leaf transform(s):")
    for i, t in enumerate(leaf_transforms):
        print(f"     [{i+1}] {type(t).__name__}")

    # --- Wrap every LLM-based extractor in robust error handling ---
    fail_counter: dict = {}
    for t in leaf_transforms:
        if isinstance(t, LLMBasedExtractor):
            make_llm_extractor_robust(t, type(t).__name__, fail_counter)

    # --- Run each leaf separately, checkpoint after each success ---
    run_config = RunConfig(
        timeout=600, max_retries=5, max_wait=60,
        max_workers=max_workers, seed=42,
    )

    print("\n  Executing pipeline (with per-step checkpointing) ...\n")
    for idx, transform in enumerate(leaf_transforms, start=1):
        step_name = type(transform).__name__

        if is_step_done(kg_dir, step_name):
            print(f"  [{idx}/{len(leaf_transforms)}] {step_name} "
                  f"-- already done, skipping")
            continue

        print(f"\n  [{idx}/{len(leaf_transforms)}] Running {step_name} ...")
        t0 = time.time()
        try:
            apply_transforms(kg, transform, run_config=run_config)
        except Exception as e:
            print(f"\n  Step {step_name} hit an unrecoverable error: "
                  f"{type(e).__name__}: {str(e)[:200]}")
            print(f"  Progress through previous steps is saved.")
            print(f"  You can rerun this script to resume.")
            traceback.print_exc()
            raise SystemExit(1)

        dt = time.time() - t0
        failed_nodes = fail_counter.get(step_name, 0)
        print(f"     done in {dt:.0f}s "
              f"(nodes: {len(kg.nodes)}, rels: {len(kg.relationships)}, "
              f"per-node failures: {failed_nodes})")

        # Checkpoint after every step
        ckpt = _ckpt_path(kg_dir, step_name)
        save_kg_atomic(kg, ckpt)
        mark_step_done(kg_dir, step_name, {
            "step": step_name,
            "elapsed_s": dt,
            "n_nodes": len(kg.nodes),
            "n_relationships": len(kg.relationships),
            "per_node_failures": failed_nodes,
        })
        print(f"     checkpoint -> {ckpt.name}")

    return kg, fail_counter


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_checkpoints(kg_dir: Path) -> None:
    """After successful run, remove intermediate checkpoints."""
    n = 0
    for pat in ("kg_ckpt_*.json", "kg_ckpt_*.done", "kg_ckpt_*.json.tmp"):
        for p in kg_dir.glob(pat):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    if n:
        print(f"  Cleaned up {n} intermediate checkpoint file(s)")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    kg_dir = output_dir / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)
    kg_path = kg_dir / "knowledge_graph.json"

    if args.fresh:
        print("  --fresh: discarding all checkpoints")
        for p in kg_dir.glob("kg_ckpt_*"):
            p.unlink()

    if kg_path.exists() and not args.force:
        print(f"  KG already exists at {kg_path}  (use --force to rebuild)")
        return

    print("=" * 70)
    print(" Stage 0 :: Build Knowledge Graph (ROBUST)")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading PDFs")
    docs = load_pdfs(Path(args.pdf_dir))

    # 2. Filter
    print("\n[2/4] Filtering low-quality pages")
    docs = filter_pages(docs)
    if not docs:
        sys.exit("All pages filtered out!")

    # 3. Models
    print("\n[3/4] Loading models")
    print(f"  LLM (vLLM server): {args.llm_model}")
    print(f"  Embeddings:        {args.embed_model}")
    print(f"  vLLM URL:          {args.vllm_url}")
    llm = build_llm(args)
    embeddings = build_embeddings(args)

    # 4. Build KG
    print("\n[4/4] Building KG")
    kg, fail_counter = build_knowledge_graph(
        docs, llm, embeddings, args.max_workers, kg_dir,
    )

    # Save final KG
    save_kg_atomic(kg, kg_path)
    atomic_write_json(
        {
            "llm_model":             args.llm_model,
            "embed_model":           args.embed_model,
            "n_pages_input":         len(list(Path(args.pdf_dir).glob("**/*.pdf"))),
            "n_pages_kept":          len(docs),
            "n_nodes":               len(kg.nodes),
            "n_relationships":       len(kg.relationships),
            "per_step_failures":     fail_counter,
        },
        kg_dir / "kg_build_config.json",
    )
    print(f"\n  Saved final KG to {kg_path}")

    # Cleanup intermediate checkpoints
    cleanup_checkpoints(kg_dir)

    if fail_counter:
        print(f"\n  Per-node failures summary (non-fatal, these nodes simply "
              f"don't have that property):")
        for step, n in fail_counter.items():
            print(f"     {step}: {n}")

    print("\n" + "=" * 70)
    print(" KG build complete.")
    print(f" Next: python generate_candidates.py --kg-file {kg_path} ...")
    print("=" * 70)


if __name__ == "__main__":
    main()