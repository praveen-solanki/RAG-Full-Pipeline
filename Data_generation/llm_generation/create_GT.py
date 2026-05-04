"""
Ground Truth Dataset Builder
PageIndex (self-hosted) + RAGAS >= 0.2 + NVIDIA NIM endpoint (streaming)

Directory layout expected:
  /data/pageindex_trees/   ← one JSON per PDF (PageIndex ingestion output)
  /data/pdfs/              ← original PDF files (same base name as JSONs)

Output:
  /data/gt_output/gt_dataset.json    ← full GT with all metadata
  /data/gt_output/ragas_eval_ready/  ← HuggingFace Dataset for ragas.evaluate()
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from openai import OpenAI
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

# ── NVIDIA NIM client ─────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)

# ── Model ─────────────────────────────────────────────────────────────────────
# Best in your catalog for structured Q&A generation.
# Fallback if rate-limited: "nvidia/llama-3.1-nemotron-70b-instruct"
GENERATOR_MODEL = "meta/llama-3.1-405b-instruct"

# ── Paths ─────────────────────────────────────────────────────────────────────
TREES_DIR  = Path("/home/olj3kor/praveen/PageIndex/results")
PDFS_DIR   = Path("/home/olj3kor/praveen/Image_dataset_generation/pdfs")
OUTPUT_DIR = Path("./gt_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
QA_PAIRS_PER_NODE   = 2
MIN_NODE_WORD_COUNT = 40
GT_VERSION          = "v1.0"
RETRY_LIMIT         = 3
RETRY_DELAY_SEC     = 2


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tree walking
# ─────────────────────────────────────────────────────────────────────────────

def walk_nodes(node, parent_title="", depth=0, collector=None):
    """
    Recursively walk a PageIndex tree node.

    Actual key mapping (from inspecting the JSON output):
      content  → "summary"
      pages    → "start_index" .. "end_index"
      children → "children" (may or may not exist; structure is often flat)
    """
    if collector is None:
        collector = []

    content    = node.get("summary") or ""
    word_count = len(content.split())
    sub_nodes  = node.get("nodes", [])   # PageIndex uses "nodes", not "children"

    if word_count >= MIN_NODE_WORD_COUNT:
        collector.append({
            "node_id":              node.get("node_id", ""),
            "node_title":           node.get("title", ""),
            "start_page":           node.get("start_index"),
            "end_page":             node.get("end_index"),
            "content":              content,
            "tree_depth":           depth,
            "parent_title":         parent_title,
            "content_length_words": word_count,
            "is_leaf":              len(sub_nodes) == 0,
        })

    for child in sub_nodes:
        walk_nodes(child, parent_title=node.get("title", ""), depth=depth + 1, collector=collector)

    return collector


def load_tree(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Q&A generation  — streaming, exactly matching the NIM usage pattern
# ─────────────────────────────────────────────────────────────────────────────

QA_SYSTEM = (
    "You are a precise evaluation dataset builder. "
    "You always respond with valid JSON only — no prose, no markdown fences, "
    "no explanation before or after the JSON."
)

QA_USER = """Generate exactly {n} question-answer pairs from the document section below.

Rules:
- Every question must be answerable ONLY from the provided text.
- Vary the types: include factual, reasoning, and multi_step where the content allows.
- Answers must be concise and directly grounded in the text.
- Return a JSON array. Each element must have exactly three keys:
    "question"      : string
    "answer"        : string
    "question_type" : one of "factual" | "reasoning" | "multi_step"

Section title: {title}
Section text:
{content}

Return the JSON array now, starting with [ and ending with ]. Nothing else."""


def call_nim_streaming(messages: list[dict]) -> str:
    """
    Call the NVIDIA NIM endpoint with stream=True (exact usage pattern).
    Collects all chunks and returns the full response string.
    """
    completion = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True,
    )

    full_response = ""
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta is not None:
            full_response += delta

    return full_response


def extract_json_array(raw: str) -> list:
    """
    Robustly extract a JSON array from LLM output.
    Handles markdown fences and models that wrap the array in a dict.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Model wrapped array in a key — find the first list value
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    # Fallback: find the first [...] block in the text
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def generate_qa_pairs(node: dict, n: int = QA_PAIRS_PER_NODE) -> list[dict]:
    """
    Stream from NVIDIA NIM, collect full response, parse and validate Q&A pairs.
    Retries up to RETRY_LIMIT times on parse failure.
    """
    messages = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user",   "content": QA_USER.format(
            n=n,
            title=node["node_title"],
            content=node["content"][:3000],     # guard against huge nodes
        )},
    ]

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            raw   = call_nim_streaming(messages)
            pairs = extract_json_array(raw)
            valid = [p for p in pairs if "question" in p and "answer" in p]

            if valid:
                return valid

            print(f"      ⚠️  Attempt {attempt}: got {len(pairs)} items, none valid. Raw: {raw[:200]}")

        except Exception as e:
            print(f"      ⚠️  Attempt {attempt} error: {e}")

        if attempt < RETRY_LIMIT:
            time.sleep(RETRY_DELAY_SEC)

    return []


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build a single GT row
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_row(qa: dict, node: dict, pdf_name: str, doc_title: str) -> dict:
    """
    Merge one Q&A pair with node metadata into a complete GT record.

    reference_contexts = the node's summary text — provably correct ground truth
    because the question was generated directly from it.
    RAGAS context_recall compares this against retrieved_contexts at eval time.
    """
    return {
        # ── RAGAS >= 0.2 required fields ───────────────────────────────────
        "user_input":             qa["question"],
        "reference":              qa["answer"],
        "reference_contexts":     [node["content"]],   # gold context (summary)
        "retrieved_contexts":     [],                  # filled at eval time

        # ── Source traceability ─────────────────────────────────────────────
        "source_pdf":             pdf_name,
        "doc_title":              doc_title,

        # ── PageIndex node metadata ─────────────────────────────────────────
        "node_id":                node["node_id"],
        "node_title":             node["node_title"],
        "start_page":             node["start_page"],   # start_index from JSON
        "end_page":               node["end_page"],     # end_index from JSON
        "tree_depth":             node["tree_depth"],
        "parent_title":           node["parent_title"],
        "is_leaf_node":           node["is_leaf"],
        "node_word_count":        node["content_length_words"],

        # ── Generation metadata ─────────────────────────────────────────────
        "question_type":          qa.get("question_type", "unknown"),
        "generator_model":        GENERATOR_MODEL,
        "creation_date":          datetime.utcnow().isoformat(),
        "gt_version":             GT_VERSION,

        # ── Retrieval eval columns (empty — filled at eval time) ────────────
        "retrieved_node_ids":     [],    # node_ids PageIndex actually returned
        "retrieved_page_indexes": [],    # pages PageIndex actually fetched
        "retrieval_hit":          None,  # bool: did retrieved nodes cover reference?
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-document processing
# ─────────────────────────────────────────────────────────────────────────────

def process_document(json_path: Path) -> list[dict]:
    tree = load_tree(json_path)

    # Actual root keys from PageIndex output
    doc_title = tree.get("doc_name") or json_path.stem
    pdf_name  = tree.get("doc_name") or (json_path.stem + ".pdf")

    print(f"\n📄 {pdf_name}  |  '{doc_title}'")

    # Root list is under "structure" key — a flat or semi-flat list of nodes
    root_nodes = tree.get("structure", [])
    if not root_nodes:
        print("   ⚠️  No 'structure' key found or it is empty — check JSON format")
        return []

    all_nodes = []
    for node in root_nodes:
        walk_nodes(node, collector=all_nodes)

    print(f"   Nodes with >= {MIN_NODE_WORD_COUNT} words: {len(all_nodes)}")

    gt_rows = []
    for node in all_nodes:
        pairs = generate_qa_pairs(node)
        if not pairs:
            print(f"   ⚠️  Skipped node: '{node['node_title']}'")
            continue
        for qa in pairs:
            gt_rows.append(build_gt_row(qa, node, pdf_name, doc_title))

    print(f"   ✅ {len(gt_rows)} GT rows")
    return gt_rows


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    json_files = sorted(TREES_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {TREES_DIR}")

    all_rows = []
    for json_path in json_files:
        all_rows.extend(process_document(json_path))

    print(f"\n📊 Total GT rows: {len(all_rows)}")

    # Save 1 — full JSON with all metadata (human readable, for debugging)
    gt_json_path = OUTPUT_DIR / "gt_dataset.json"
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"💾 Full GT     → {gt_json_path}")

    # Save 2 — RAGAS EvaluationDataset (pass directly to ragas.evaluate())
    samples = [
        SingleTurnSample(
            user_input         = row["user_input"],
            reference          = row["reference"],
            reference_contexts = row["reference_contexts"],
            retrieved_contexts = row["retrieved_contexts"],
        )
        for row in all_rows
    ]
    eval_dataset = EvaluationDataset(samples=samples)

    # Save 3 — HuggingFace arrow format (fast reload, keeps all metadata)
    hf_path = OUTPUT_DIR / "ragas_eval_ready"
    Dataset.from_list(all_rows).save_to_disk(str(hf_path))
    print(f"💾 HF Dataset  → {hf_path}")

    return eval_dataset, all_rows


if __name__ == "__main__":
    eval_dataset, rows = main()

    print("\n── Example GT row ──")
    example = {k: v for k, v in rows[0].items() if k != "reference_contexts"}
    example["reference_contexts_preview"] = rows[0]["reference_contexts"][0][:120] + "..."
    print(json.dumps(example, indent=2, default=str))