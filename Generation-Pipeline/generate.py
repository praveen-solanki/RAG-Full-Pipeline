"""
generate.py — Stage B of the comparative RAG pipeline.

Reads the two retrieval JSONL files produced by Stage A, normalizes both
into the same Context object, runs a single *frozen* generator (same
model, prompt, temperature, seed) on every (query, retriever) pair, and
writes one results.jsonl file that Stage C will consume.

Nothing in this file calls a retriever. If you want different retrieval
behavior, edit Stage A and re-run it — this script only sees stored files.

Usage
-----
    # 1) start a vLLM server with your generator model, e.g.:
    #    vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000
    #
    # 2) then:
    python generate.py \
        --pageindex-file ./pageindex_retrieval.jsonl \
        --bgem3-file     ./bgem3_retrieval.jsonl    \
        --output-file    ./results.jsonl            \
        --model-id       mistralai/Mistral-7B-Instruct-v0.3 \
        --base-url       http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List

from pydantic import ValidationError

from schemas import (
    BGEM3Row,
    Context,
    ContextChunk,
    PageIndexRow,
    ResultRow,
)


# --------------------------------------------------------------------------- #
# Load + normalize retrieval files                                            #
# --------------------------------------------------------------------------- #

def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_no}: {exc}"
                ) from exc


def load_pageindex_contexts(path: Path) -> List[tuple[PageIndexRow, Context]]:
    """
    Parse pageindex_retrieval.jsonl and normalize every row to a Context.

    PageIndex returns a single plain-text block per query, so the Context
    contains exactly one ContextChunk.
    """
    normalized: List[tuple[PageIndexRow, Context]] = []
    for obj in _iter_jsonl(path):
        try:
            row = PageIndexRow(**obj)
        except ValidationError as exc:
            raise ValueError(
                f"{path}: row does not match PageIndexRow schema:\n{exc}"
            ) from exc

        context = Context(
            chunks=[
                ContextChunk(
                    text=row.context_text,
                    source_id=row.source_id,
                    score=None,
                )
            ],
            retriever="pageindex",
        )
        normalized.append((row, context))
    return normalized


def load_bgem3_contexts(path: Path) -> List[tuple[BGEM3Row, Context]]:
    """
    Parse bgem3_retrieval.jsonl and normalize every row to a Context.

    BGE-M3 returns multiple ranked chunks, which are preserved in the
    order given (the retriever already sorted them by similarity).
    """
    normalized: List[tuple[BGEM3Row, Context]] = []
    for obj in _iter_jsonl(path):
        try:
            row = BGEM3Row(**obj)
        except ValidationError as exc:
            raise ValueError(
                f"{path}: row does not match BGEM3Row schema:\n{exc}"
            ) from exc

        context = Context(
            chunks=[
                ContextChunk(
                    text=chunk.text,
                    source_id=chunk.source_id,
                    score=chunk.score,
                )
                for chunk in row.context_chunks
            ],
            retriever="bge_m3",
        )
        normalized.append((row, context))
    return normalized


# --------------------------------------------------------------------------- #
# Frozen generator                                                            #
# --------------------------------------------------------------------------- #

class FrozenRAGGenerator:
    """
    A deterministic RAG generator.

    All sampling parameters are fixed so that any quality difference between
    the two retrievers is attributable to retrieval alone, not to decoder
    randomness. The same instance is used for both PageIndex and BGE-M3 rows.

    Talks to any OpenAI-compatible endpoint (vLLM, TGI, Ollama with the
    OpenAI-compat layer). The model itself is open-source (Mistral or Llama);
    the OpenAI SDK is used only for its HTTP client.
    """

    SYSTEM_PROMPT = (
        "You are a careful question-answering assistant. "
        "Answer the user's question using ONLY the information in the "
        "provided context. If the answer is not contained in the context, "
        "reply exactly: I don't know. "
        "Do not add facts that are not in the context."
    )

    USER_PROMPT_TEMPLATE = (
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        timeout: float = 120.0,
    ) -> None:
        # Imported lazily so the module can be imported in environments
        # where the openai package isn't installed yet (e.g., CI linting).
        from openai import OpenAI

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _build_messages(self, query: str, context: Context) -> list[dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_PROMPT_TEMPLATE.format(
                    context=context.as_text(),
                    query=query,
                ),
            },
        ]

    def generate(self, query: str, context: Context) -> tuple[str, float, int]:
        """Return (answer_text, latency_ms, prompt_chars)."""
        messages = self._build_messages(query, context)
        prompt_chars = sum(len(m["content"]) for m in messages)

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        latency_ms = (time.time() - start) * 1000.0

        answer = (response.choices[0].message.content or "").strip()
        return answer, latency_ms, prompt_chars


# --------------------------------------------------------------------------- #
# Main pipeline                                                               #
# --------------------------------------------------------------------------- #

def _write_result(fh, result: ResultRow) -> None:
    fh.write(result.model_dump_json() + "\n")
    fh.flush()


def run(
    pageindex_file: Path,
    bgem3_file: Path,
    output_file: Path,
    generator: FrozenRAGGenerator,
    resume: bool = False,
) -> None:
    """
    Generate answers for every retrieved context and stream to results.jsonl.

    Results are written incrementally so that a crash mid-run doesn't lose
    completed rows. Set resume=True to skip (query, retriever) pairs that
    already exist in the output file.
    """
    pi_items = load_pageindex_contexts(pageindex_file)
    bg_items = load_bgem3_contexts(bgem3_file)

    print(
        f"Loaded {len(pi_items)} PageIndex rows and {len(bg_items)} BGE-M3 rows.",
        file=sys.stderr,
    )

    already_done: set[tuple[str, str]] = set()
    if resume and output_file.exists():
        for obj in _iter_jsonl(output_file):
            already_done.add((obj.get("query", ""), obj.get("retriever", "")))
        print(
            f"Resume mode: skipping {len(already_done)} already-completed rows.",
            file=sys.stderr,
        )

    # Open in append mode when resuming, otherwise overwrite.
    mode = "a" if resume else "w"
    with output_file.open(mode, encoding="utf-8") as fh:
        total = len(pi_items) + len(bg_items)
        done = 0

        # --- PageIndex rows --------------------------------------------- #
        for row, context in pi_items:
            done += 1
            key = (row.query, "pageindex")
            if key in already_done:
                continue

            try:
                answer, latency_ms, prompt_chars = generator.generate(
                    row.query, context
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                print(
                    f"[{done}/{total}] pageindex ERROR: {exc!r} — query: "
                    f"{row.query[:80]!r}",
                    file=sys.stderr,
                )
                continue

            result = ResultRow(
                query=row.query,
                ground_truth=row.ground_truth,
                retriever="pageindex",
                context_chunks=context.chunks,
                answer=answer,
                model_id=generator.model_id,
                latency_ms=round(latency_ms, 2),
                prompt_chars=prompt_chars,
            )
            _write_result(fh, result)
            print(
                f"[{done}/{total}] pageindex OK  "
                f"({latency_ms:7.1f} ms)  {row.query[:60]!r}",
                file=sys.stderr,
            )

        # --- BGE-M3 rows ------------------------------------------------ #
        for row, context in bg_items:
            done += 1
            key = (row.query, "bge_m3")
            if key in already_done:
                continue

            try:
                answer, latency_ms, prompt_chars = generator.generate(
                    row.query, context
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                print(
                    f"[{done}/{total}] bge_m3 ERROR: {exc!r} — query: "
                    f"{row.query[:80]!r}",
                    file=sys.stderr,
                )
                continue

            result = ResultRow(
                query=row.query,
                ground_truth=row.ground_truth,
                retriever="bge_m3",
                context_chunks=context.chunks,
                answer=answer,
                model_id=generator.model_id,
                latency_ms=round(latency_ms, 2),
                prompt_chars=prompt_chars,
            )
            _write_result(fh, result)
            print(
                f"[{done}/{total}] bge_m3    OK  "
                f"({latency_ms:7.1f} ms)  {row.query[:60]!r}",
                file=sys.stderr,
            )

    print(f"Done. Wrote {output_file}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pageindex-file", type=Path, required=True)
    p.add_argument("--bgem3-file", type=Path, required=True)
    p.add_argument("--output-file", type=Path, default=Path("results.jsonl"))
    p.add_argument(
        "--model-id",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model name as it is served on the OpenAI-compatible endpoint.",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint (vLLM/TGI/Ollama).",
    )
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append to the output file and skip (query, retriever) pairs "
             "already present.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.pageindex_file.exists():
        sys.exit(f"error: {args.pageindex_file} does not exist")
    if not args.bgem3_file.exists():
        sys.exit(f"error: {args.bgem3_file} does not exist")

    generator = FrozenRAGGenerator(
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    run(
        pageindex_file=args.pageindex_file,
        bgem3_file=args.bgem3_file,
        output_file=args.output_file,
        generator=generator,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()