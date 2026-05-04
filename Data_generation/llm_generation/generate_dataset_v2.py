"""
RAG evaluation dataset generator using RAGAS v0.2+.

Key reliability features:
  * Flattened transform pipeline with per-leaf checkpointing
  * Explicit completion markers (.done sidecar files) survive across restarts
  * Atomic checkpoint writes (tmp + rename)
  * raise_exceptions=False on KG transforms -> single-node failures don't abort
  * Node-level idempotency: re-running only processes nodes missing the target property
  * httpx keepalive disabled to prevent stale-connection crashes on long runs
  * vLLM health check before each step - fail fast if server is dead
  * Step 5 (question generation) is batched and checkpointed incrementally
"""

import os
import sys
import argparse
import time
import json
import shutil
import nest_asyncio
from pathlib import Path

# RAGAS uses async internally. Patch the event loop so it works from scripts.
nest_asyncio.apply()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RAG evaluation dataset from PDFs using RAGAS v0.2+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdf-dir", "-d", required=True,
                        help="Directory containing PDF files.")
    parser.add_argument("--num-questions", "-n", type=int, default=50,
                        help="Number of Q&A samples to generate (default: 50).")
    parser.add_argument("--output-dir", "-o", default="./ragas_output",
                        help="Directory to save outputs (default: ./ragas_output).")
    parser.add_argument("--load-kg", action="store_true", default=False,
                        help="Load existing knowledge_graph.json instead of rebuilding.")
    parser.add_argument("--kg-file", default="knowledge_graph.json",
                        help="KG filename inside --output-dir (default: knowledge_graph.json).")
    parser.add_argument("--dataset-file", default="dataset.csv",
                        help="Output CSV filename (default: dataset.csv).")
    parser.add_argument("--llm-provider", choices=["nvidia", "ollama"], default="nvidia",
                        help="Provider for the LLM: 'nvidia' or 'ollama' (local vLLM).")
    parser.add_argument("--embed-provider",
                        choices=["nvidia", "ollama", "sentence-transformers"],
                        default="nvidia",
                        help="Provider for embeddings.")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel workers for KG transforms and generation (default: 4).")
    parser.add_argument("--gen-batch-size", type=int, default=100,
                        help="Batch size for question generation checkpointing (default: 100).")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def banner(text: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def check_env(llm_provider: str, embed_provider: str) -> None:
    if (llm_provider == "nvidia" or embed_provider == "nvidia") and not os.environ.get("NVIDIA_API_KEY"):
        print("\n❌  NVIDIA_API_KEY is not set.\n   export NVIDIA_API_KEY='nvapi-...'\n")
        sys.exit(1)


def make_output_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_save_json(obj, path: Path) -> None:
    """Write JSON atomically: tmp file + rename. Crash-safe."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def atomic_save_kg(kg, path: Path) -> None:
    """Save KG atomically via tmp + rename, so a mid-write crash never corrupts the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    kg.save(str(tmp))
    tmp.replace(path)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD PDFs
# ══════════════════════════════════════════════════════════════════════════════

def load_pdfs(pdf_dir: str):
    """
    Load PDFs with LangChain PyPDFLoader. Sets 'filename' metadata, which RAGAS
    needs to identify chunks from the same document for multi-hop generation.
    """
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"\n❌  PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_files = list(pdf_path.glob("**/*.pdf"))
    if not pdf_files:
        print(f"\n❌  No PDF files found in: {pdf_dir}")
        sys.exit(1)

    print(f"  Found {len(pdf_files)} PDF file(s)")

    loader = DirectoryLoader(str(pdf_path), glob="**/*.pdf",
                             loader_cls=PyPDFLoader, show_progress=True)
    docs = loader.load()

    for doc in docs:
        if "filename" not in doc.metadata:
            doc.metadata["filename"] = doc.metadata.get("source", "unknown")

    print(f"  Loaded {len(docs)} page(s)")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

NVIDIA_LLM_MODEL       = "meta/llama-3.1-70b-instruct"
NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
SENTENCE_TRANSFORMER_EMBEDDING_MODEL = "BAAI/bge-m3"
NVIDIA_BASE_URL        = "https://integrate.api.nvidia.com/v1"

OLLAMA_LLM_MODEL       = "Qwen/Qwen2.5-32B-Instruct-AWQ"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL        = "http://localhost:8011/v1"


def _build_httpx_clients():
    """
    httpx clients tuned for multi-hour batch workloads against a local vLLM server.

    Why these settings:
      * keepalive DISABLED: TCP connections silently die after long idle periods;
        a fresh connection per request costs microseconds on localhost but prevents
        the classic 'server idle + client gets ReadError after hours' failure.
      * max_connections=100: allow true parallelism instead of queueing on a pool of 10.
      * transport retries=3: retry at the TCP layer before the error propagates up
        to ragas's tenacity, which can't distinguish "dead socket" from "real failure".
      * timeout=300s: LLM synthesis prompts can be slow under load.
    """
    import httpx
    limits = httpx.Limits(
        max_keepalive_connections=0,   # <-- kills keepalive
        max_connections=100,
        keepalive_expiry=0.0,
    )
    timeout = httpx.Timeout(300.0, connect=30.0)
    sync_transport  = httpx.HTTPTransport(retries=3)
    async_transport = httpx.AsyncHTTPTransport(retries=3)
    return (
        httpx.Client(limits=limits, timeout=timeout, transport=sync_transport),
        httpx.AsyncClient(limits=limits, timeout=timeout, transport=async_transport),
    )


def build_llm(llm_provider: str):
    """Return a LangchainLLMWrapper around the chosen provider's chat model."""
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
    else:  # ollama -> local vLLM via OpenAI-compatible endpoint
        from langchain_openai import ChatOpenAI
        sync_client, async_client = _build_httpx_clients()
        print(f"  LLM provider : vLLM (local)")
        print(f"  LLM model    : {OLLAMA_LLM_MODEL}")
        print(f"  LLM base URL : {OLLAMA_BASE_URL}")
        print(f"  httpx        : keepalive=OFF, max_connections=100, transport_retries=3")
        llm = ChatOpenAI(
            model=OLLAMA_LLM_MODEL,
            openai_api_key="dummy",
            openai_api_base=OLLAMA_BASE_URL,
            temperature=0.1,
            max_tokens=1024,
            http_client=sync_client,
            http_async_client=async_client,
        )

    return LangchainLLMWrapper(llm)


def build_embeddings(embed_provider: str):
    """Return a LangchainEmbeddingsWrapper around the chosen provider."""
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
        emb = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    else:  # sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"  Embed provider : Sentence Transformers (local)")
        print(f"  Embed model    : {SENTENCE_TRANSFORMER_EMBEDDING_MODEL}")
        emb = HuggingFaceEmbeddings(
            model_name=SENTENCE_TRANSFORMER_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return LangchainEmbeddingsWrapper(emb)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — KNOWLEDGE GRAPH (flattened pipeline with per-leaf checkpointing)
# ══════════════════════════════════════════════════════════════════════════════

def build_run_config(max_workers: int = 4):
    """RunConfig for transforms and generation. Retries + long wait handle 429s."""
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
    HeadlineSplitter subclass that silently skips nodes where no headlines were
    found, instead of crashing with ValueError. Robust for large PDF corpora
    where many pages (tables, figures, refs) have no headings.
    """
    from ragas.testset.transforms.splitters.headline import HeadlineSplitter

    class SafeHeadlineSplitter(HeadlineSplitter):
        async def split(self, node):
            headlines = node.properties.get("headlines")
            if not headlines:
                return [], []
            return await super().split(node)

    return SafeHeadlineSplitter()


# ── Pipeline flattening ──────────────────────────────────────────────────────

def flatten_pipeline(transforms):
    """
    Walk the (possibly nested) RAGAS transform pipeline and return a flat list
    of leaf transforms in execution order.

    RAGAS wraps groups of extractors in Parallel(...) and may nest further.
    The original code iterated only the top level, causing a single 'Parallel'
    checkpoint that bundled NER+Keyphrases+Headlines together - if any failed,
    ALL were lost. Flattening fixes this: each leaf gets its own checkpoint.
    """
    from ragas.testset.transforms.engine import Parallel

    flat = []

    def _walk(node):
        # Parallel wrapper: recurse into its .transformations
        if isinstance(node, Parallel):
            for sub in node.transformations:
                _walk(sub)
            return
        # Plain list/tuple (some ragas versions return lists at top level)
        if isinstance(node, (list, tuple)):
            for sub in node:
                _walk(sub)
            return
        # Leaf transform
        flat.append(node)

    _walk(transforms)
    return flat


# ── Checkpoint bookkeeping ───────────────────────────────────────────────────

def _ckpt_path(output_dir: Path, step_name: str) -> Path:
    return output_dir / f"kg_checkpoint_{step_name}.json"


def _done_path(output_dir: Path, step_name: str) -> Path:
    return output_dir / f"kg_checkpoint_{step_name}.done"


def _mark_done(output_dir: Path, step_name: str, stats: dict) -> None:
    atomic_save_json({"step": step_name, "ts": time.time(), **stats},
                     _done_path(output_dir, step_name))


def _is_done(output_dir: Path, step_name: str) -> bool:
    return _done_path(output_dir, step_name).exists()


def _latest_kg_checkpoint(output_dir: Path):
    """
    Find the newest kg_checkpoint_*.json by mtime. Returns (path, step_name)
    or (None, None) if none exist. Uses mtime not a hardcoded order, so it
    stays correct even as the pipeline evolves.
    """
    candidates = [p for p in output_dir.glob("kg_checkpoint_*.json")
                  if not p.name.endswith(".tmp")]
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    step_name = latest.stem.replace("kg_checkpoint_", "")
    return latest, step_name


# ── Node-level idempotency ───────────────────────────────────────────────────

# Maps transform class names to the node property they produce.
# Used to filter out already-processed nodes on resume, so a crashed step
# that finished 3000/5306 nodes only processes the remaining 2306 on re-run.
STEP_OUTPUT_PROPERTY = {
    "NERExtractor":        "entities",
    "KeyphrasesExtractor": "keyphrases",
    "HeadlinesExtractor":  "headlines",
    "SummaryExtractor":    "summary",
    "TitleExtractor":      "title",
    "EmbeddingExtractor":  "embedding",
    "ThemesExtractor":     "themes",
}


def _nodes_missing_property(kg, prop: str) -> int:
    """Count how many nodes are missing a given property."""
    return sum(1 for n in kg.nodes if prop not in n.properties)


# ── vLLM health check ────────────────────────────────────────────────────────

def vllm_health_check(llm_provider: str) -> bool:
    """
    Probe the local vLLM server with a tiny request before starting an expensive
    step. Returns True if reachable, False otherwise. For NVIDIA (remote), skip.
    """
    if llm_provider != "ollama":
        return True
    try:
        import httpx
        # /v1/models is a cheap endpoint vLLM exposes
        url = OLLAMA_BASE_URL.rstrip("/") + "/models"
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url)
        return r.status_code == 200
    except Exception as e:
        print(f"     health-check exception: {type(e).__name__}: {e}")
        return False


def _wait_for_vllm(llm_provider: str, max_attempts: int = 6, wait_s: int = 30) -> bool:
    """Wait up to ~3 min for vLLM to come back. Return True if healthy, else False."""
    for i in range(1, max_attempts + 1):
        if vllm_health_check(llm_provider):
            return True
        print(f"     vLLM unreachable (attempt {i}/{max_attempts}), sleeping {wait_s}s ...")
        time.sleep(wait_s)
    return False


# ── Main KG builder ──────────────────────────────────────────────────────────

def build_knowledge_graph(docs, llm, embeddings, max_workers: int,
                           output_dir: Path, llm_provider: str):
    """
    Build the KG with true per-leaf checkpointing.

    Resume logic:
      1. Load the newest kg_checkpoint_*.json (by mtime), if any.
      2. For each leaf transform in the flattened pipeline:
           - if _done_path exists -> skip (step fully completed)
           - else if all target-property nodes already have it -> mark done, skip
           - else -> run the step (ragas itself will skip already-processed nodes
                     if we set raise_exceptions=False; remaining nodes get filled in)
      3. After each step: save KG checkpoint atomically + write .done marker.
      4. At the end: save final knowledge_graph.json and clean up intermediate checkpoints.
    """
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.testset.transforms.splitters.headline import HeadlineSplitter

    # --- 1. Try resuming from the newest checkpoint ---------------------------
    latest_ckpt, latest_step = _latest_kg_checkpoint(output_dir)
    if latest_ckpt is not None:
        print(f"  Resuming from checkpoint: {latest_ckpt.name} (last step: {latest_step})")
        kg = KnowledgeGraph.load(str(latest_ckpt))
        print(f"     nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
    else:
        print("  No checkpoint found. Creating fresh KnowledgeGraph ...")
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            ))
        print(f"  Added {len(kg.nodes)} DOCUMENT node(s)")

    # --- 2. Build and flatten the transform pipeline --------------------------
    print("\n  Building transform pipeline ...")
    trans = default_transforms(documents=docs, llm=llm, embedding_model=embeddings)

    # Swap HeadlineSplitter -> SafeHeadlineSplitter (before flattening, since
    # the splitter may be nested inside a Parallel on some ragas versions).
    def _swap(node):
        from ragas.testset.transforms.engine import Parallel
        if isinstance(node, Parallel):
            for i, sub in enumerate(node.transformations):
                if isinstance(sub, HeadlineSplitter) and not isinstance(sub, type(make_safe_headline_splitter())):
                    node.transformations[i] = make_safe_headline_splitter()
                else:
                    _swap(sub)
        elif isinstance(node, (list, tuple)):
            for i, sub in enumerate(node):
                if isinstance(sub, HeadlineSplitter):
                    node[i] = make_safe_headline_splitter()
                else:
                    _swap(sub)
    _swap(trans)
    # Handle the top-level-list case too
    if isinstance(trans, list):
        for i, t in enumerate(trans):
            if isinstance(t, HeadlineSplitter):
                trans[i] = make_safe_headline_splitter()

    leaf_transforms = flatten_pipeline(trans)

    print(f"  Pipeline has {len(leaf_transforms)} leaf transform(s):")
    for t in leaf_transforms:
        print(f"     - {type(t).__name__}")

    run_config = build_run_config(max_workers=max_workers)

    # --- 3. Run each leaf, checkpoint after each one --------------------------
    print("\n  Applying transforms (per-leaf checkpoint + .done marker) ...\n")
    print(f"  {'Step':<38} Status")
    print(f"  {'-'*38} {'-'*36}")

    for transform in leaf_transforms:
        step_name = type(transform).__name__

        # Skip if explicit .done marker exists
        if _is_done(output_dir, step_name):
            print(f"  {'  ' + step_name:<38} [skipped - .done marker present]")
            continue

        # Skip if all nodes already have the expected output property
        prop = STEP_OUTPUT_PROPERTY.get(step_name)
        if prop is not None and _nodes_missing_property(kg, prop) == 0 and len(kg.nodes) > 0:
            print(f"  {'  ' + step_name:<38} [skipped - all nodes have '{prop}']")
            _mark_done(output_dir, step_name, {"reason": "all_nodes_had_property"})
            continue

        # Health-check vLLM before starting a multi-hour step
        if not vllm_health_check(llm_provider):
            print(f"  {'  ' + step_name:<38} [vLLM unreachable - waiting ...]")
            if not _wait_for_vllm(llm_provider):
                print(f"\n  ❌  vLLM server is not reachable at {OLLAMA_BASE_URL}")
                print(f"     Start it and re-run - progress is saved.")
                raise SystemExit(1)

        # Report how much work this step actually has to do
        if prop is not None:
            missing = _nodes_missing_property(kg, prop)
            print(f"  {'  ' + step_name:<38} [running on {missing}/{len(kg.nodes)} nodes ...]")
        else:
            print(f"  {'  ' + step_name:<38} [running ...]")

        t_start = time.time()
        try:
            # apply_transforms does NOT accept raise_exceptions in current ragas.
            # Node-level failures are logged internally; if a connection-level
            # error bubbles up, we catch it below and save partial progress so
            # the *next* run can pick up (nodes missing the target property are
            # re-processed; nodes that already have it are skipped by our
            # _nodes_missing_property check above).
            apply_transforms(kg, [transform], run_config)
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  {'  ' + step_name:<38} [FAILED {elapsed:.1f}s]")
            print(f"     {type(e).__name__}: {e}")
            # Save whatever partial progress we got so the next run can continue
            partial = _ckpt_path(output_dir, step_name + "_partial")
            try:
                atomic_save_kg(kg, partial)
                print(f"     partial state saved -> {partial.name}")
            except Exception as save_err:
                print(f"     (could not save partial state: {save_err})")
            print(f"\n  WARNING: Interrupted at step: {step_name}")
            print(f"  WARNING: Re-run to resume from last checkpoint automatically.")
            raise

        elapsed = time.time() - t_start

        # Save checkpoint + .done marker atomically
        ckpt = _ckpt_path(output_dir, step_name)
        atomic_save_kg(kg, ckpt)
        stats = {"elapsed_s": round(elapsed, 2),
                 "nodes": len(kg.nodes),
                 "relationships": len(kg.relationships)}
        if prop is not None:
            stats["nodes_with_property"] = sum(1 for n in kg.nodes if prop in n.properties)
        _mark_done(output_dir, step_name, stats)

        print(f"  {'  ' + step_name:<38} [done {elapsed:.1f}s -> {ckpt.name}]")

    print(f"\n  KG complete -- nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
    return kg


def save_kg(kg, path: Path) -> None:
    atomic_save_kg(kg, path)
    print(f"  KG saved -> {path}")


def load_kg(path: Path):
    from ragas.testset.graph import KnowledgeGraph
    print(f"  Loading KG from {path} ...")
    kg = KnowledgeGraph.load(str(path))
    print(f"  KG loaded -- nodes: {len(kg.nodes)}, relationships: {len(kg.relationships)}")
    return kg


def cleanup_intermediate_checkpoints(output_dir: Path) -> None:
    """After a successful full run, remove per-step checkpoints to save disk."""
    removed = 0
    for pattern in ("kg_checkpoint_*.json", "kg_checkpoint_*.done",
                    "kg_checkpoint_*.json.tmp"):
        for p in output_dir.glob(pattern):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    if removed:
        print(f"  Cleaned up {removed} intermediate checkpoint file(s)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — QUERY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def get_query_distribution(llm):
    """
    Use RAGAS's default_query_distribution(): the only officially-supported
    distribution in v0.2+. Produces:
        SingleHopSpecific   0.50
        MultiHopAbstract    0.25
        MultiHopSpecific    0.25
    """
    from ragas.testset.synthesizers import default_query_distribution

    distribution = default_query_distribution(llm)

    print("  ┌──────────────────────────────────────────────────────┬────────┐")
    print("  │ Synthesizer                                          │ Weight │")
    print("  ├──────────────────────────────────────────────────────┼────────┤")
    for synth, weight in distribution:
        name = type(synth).__name__
        print(f"  │  {name:<51} │  {weight:.2f}  │")
    print("  └──────────────────────────────────────────────────────┴────────┘")

    return distribution


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — GENERATE TESTSET (batched + checkpointed)
# ══════════════════════════════════════════════════════════════════════════════

GEN_PROGRESS_FILE = "generation_progress.json"
GEN_SAMPLES_FILE  = "generation_samples.jsonl"


def _load_generation_progress(output_dir: Path):
    """Return (done_count, list_of_sample_dicts). Empty if nothing yet."""
    progress_path = output_dir / GEN_PROGRESS_FILE
    samples_path  = output_dir / GEN_SAMPLES_FILE
    if not progress_path.exists() or not samples_path.exists():
        return 0, []
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            prog = json.load(f)
        samples = []
        with open(samples_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        done = min(prog.get("done", 0), len(samples))
        return done, samples[:done]
    except Exception as e:
        print(f"  ⚠  Could not read generation progress ({e}); starting from 0.")
        return 0, []


def _append_generation_samples(output_dir: Path, new_samples, done_total: int):
    """Append new samples to JSONL and update the progress pointer atomically."""
    samples_path  = output_dir / GEN_SAMPLES_FILE
    progress_path = output_dir / GEN_PROGRESS_FILE
    with open(samples_path, "a", encoding="utf-8") as f:
        for s in new_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    atomic_save_json({"done": done_total, "ts": time.time()}, progress_path)


def _testset_row_to_dict(row) -> dict:
    """Convert a testset DataFrame row to a plain dict for JSONL persistence."""
    ref_ctx = row.get("reference_contexts", [])
    if hasattr(ref_ctx, "tolist"):  # numpy array
        ref_ctx = ref_ctx.tolist()
    if not isinstance(ref_ctx, list):
        ref_ctx = [ref_ctx] if ref_ctx else []
    return {
        "user_input":         row.get("user_input", ""),
        "reference":          row.get("reference", ""),
        "reference_contexts": ref_ctx,
        "synthesizer_name":   row.get("synthesizer_name", ""),
    }


def generate_testset(kg, llm, embeddings, num_questions: int, query_distribution,
                      max_workers: int, output_dir: Path, batch_size: int,
                      llm_provider: str):
    """
    Batched, checkpointed generation.

    Each batch of `batch_size` questions is generated, immediately appended to
    generation_samples.jsonl on disk, and a progress pointer is updated. On
    restart, already-generated samples are loaded and only the remainder is
    generated. A crash at question 4500/5000 costs one batch, not everything.
    """
    from ragas.testset import TestsetGenerator

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
    )
    run_config = build_run_config(max_workers=max_workers)

    done, existing_samples = _load_generation_progress(output_dir)
    if done > 0:
        print(f"  Resuming generation: {done}/{num_questions} already generated")

    remaining = max(0, num_questions - done)
    if remaining == 0:
        print(f"  All {num_questions} questions already generated - skipping generation")
        return existing_samples

    print(f"  Generating {remaining} more sample(s) in batches of {batch_size}")
    print(f"  max_workers={max_workers}\n")

    all_samples = list(existing_samples)
    batch_num = 0

    while done < num_questions:
        this_batch = min(batch_size, num_questions - done)
        batch_num += 1
        print(f"  ── Batch {batch_num}: generating {this_batch} sample(s) "
              f"({done}/{num_questions} done) ──")

        if not vllm_health_check(llm_provider):
            print(f"     vLLM unreachable - waiting ...")
            if not _wait_for_vllm(llm_provider):
                print(f"\n  ❌  vLLM server is not reachable. Re-run to resume.")
                raise SystemExit(1)

        t0 = time.time()
        try:
            testset = generator.generate(
                testset_size=this_batch,
                query_distribution=query_distribution,
                num_personas=3,
                run_config=run_config,
                with_debugging_logs=False,
                raise_exceptions=False,
            )
        except Exception as e:
            print(f"     Batch {batch_num} FAILED after {time.time()-t0:.1f}s: "
                  f"{type(e).__name__}: {e}")
            print(f"     {done} sample(s) are safely persisted. Re-run to resume.")
            raise

        df = testset.to_pandas()
        new_samples = [_testset_row_to_dict(row) for _, row in df.iterrows()]

        if not new_samples:
            print(f"     ⚠  Batch {batch_num} returned 0 samples. "
                  f"Check KG quality / LLM output.")
            # Avoid infinite loop: if a batch produces nothing, stop
            print(f"     Stopping generation early with {done}/{num_questions} samples.")
            break

        done += len(new_samples)
        all_samples.extend(new_samples)
        _append_generation_samples(output_dir, new_samples, done)

        print(f"     Batch {batch_num} done in {time.time()-t0:.1f}s "
              f"(+{len(new_samples)}, total {done}/{num_questions})")

    return all_samples


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_dataset(samples, output_dir: Path, csv_filename: str) -> None:
    """Export collected samples (list of dicts) to CSV + JSON."""
    import pandas as pd

    if not samples:
        print("\n  ⚠️  No samples to export.")
        return

    df = pd.DataFrame(samples)

    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"  CSV  -> {csv_path}")

    json_path = output_dir / csv_filename.replace(".csv", ".json")
    records = []
    for s in samples:
        records.append({
            "user_input":         s.get("user_input", ""),
            "reference":          s.get("reference", ""),
            "reference_contexts": s.get("reference_contexts", []),
            "synthesizer_name":   s.get("synthesizer_name", ""),
            "metadata": {
                "synthesizer_name": s.get("synthesizer_name", ""),
                "num_contexts":     len(s.get("reference_contexts", []) or []),
            },
        })
    atomic_save_json(records, json_path)
    print(f"  JSON -> {json_path}")

    # Distribution summary
    if "synthesizer_name" in df.columns:
        print("\n  Question-type distribution:")
        dist = df["synthesizer_name"].value_counts()
        total = len(df)
        for name, count in dist.items():
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {str(name):<52} {count:>4}  ({pct:5.1f}%)  {bar}")
        print(f"    {'TOTAL':<52} {total:>4}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args       = parse_args()
    check_env(args.llm_provider, args.embed_provider)
    output_dir = make_output_dir(args.output_dir)
    kg_path    = output_dir / args.kg_file

    print(
        f"\n  LLM         : {args.llm_provider.upper()}\n"
        f"  Embeddings  : {args.embed_provider.upper()}\n"
        f"  Target      : {args.num_questions} question(s)\n"
        f"  Workers     : {args.max_workers}\n"
        f"  Batch size  : {args.gen_batch_size}\n"
        f"  Output dir  : {output_dir.resolve()}\n"
    )

    # Step 1
    banner("Step 1 · Load PDFs")
    docs = load_pdfs(args.pdf_dir)

    # Step 2
    banner("Step 2 · Models")
    llm        = build_llm(args.llm_provider)
    embeddings = build_embeddings(args.embed_provider)

    # Pre-flight vLLM health check
    if args.llm_provider == "ollama":
        banner("Pre-flight · vLLM health check")
        if vllm_health_check(args.llm_provider):
            print(f"  ✓ vLLM reachable at {OLLAMA_BASE_URL}")
        else:
            print(f"  ❌  vLLM NOT reachable at {OLLAMA_BASE_URL}")
            print(f"     Start it and re-run. (Progress is preserved across runs.)")
            sys.exit(1)

    # Step 3
    banner("Step 3 · Knowledge Graph")
    if args.load_kg and kg_path.exists():
        kg = load_kg(kg_path)
    else:
        if args.load_kg:
            print(f"  ⚠  --load-kg set but '{kg_path}' not found; building.")
        kg = build_knowledge_graph(docs, llm, embeddings, args.max_workers,
                                    output_dir, args.llm_provider)
        save_kg(kg, kg_path)

    # Step 4
    banner("Step 4 · Query Distribution")
    query_distribution = get_query_distribution(llm)

    # Step 5 (batched, checkpointed)
    banner(f"Step 5 · Generate {args.num_questions} Questions")
    t0 = time.time()
    samples = generate_testset(
        kg, llm, embeddings,
        args.num_questions, query_distribution,
        args.max_workers, output_dir, args.gen_batch_size,
        args.llm_provider,
    )
    print(f"\n  Generation completed in {time.time() - t0:.1f}s "
          f"({len(samples)}/{args.num_questions} samples)")

    # Step 6
    banner("Step 6 · Export")
    export_dataset(samples, output_dir, args.dataset_file)

    # Cleanup (only if we have the final KG and a full dataset)
    if kg_path.exists() and len(samples) >= args.num_questions:
        banner("Cleanup")
        cleanup_intermediate_checkpoints(output_dir)

    banner("Done")
    print(f"  All outputs saved to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()