"""
Split a multi-source evaluation JSON into per-document rag-chunk compatible files.

Each unique "source_document" gets its own output JSON file in rag-chunk format:
  {
    "questions": [
      { "question": "...", "relevant": [...] },
      ...
    ]
  }

Usage:
    python split_by_source.py <input.json> [output_dir]

Example:
    python split_by_source.py Q_2.json ./ragchunk_questions/

Output files are named after the source PDF, e.g.:
    Utilization_of_Crypto_Services.json
    Crypto_Driver_Specification.json
    ...
"""

import json
import os
import sys
import re
from collections import defaultdict


def safe_filename(name: str) -> str:
    """Turn a PDF filename into a safe output JSON filename."""
    # Remove .pdf extension
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    # Replace spaces and special chars with underscores
    name = re.sub(r"[^\w\-]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("_")
    return name + ".json"


def clean_snippets(snippets: list) -> list:
    """Strip whitespace and drop empty entries."""
    return [s.strip() for s in snippets if s.strip()]


def extract_keywords(snippets: list) -> list:
    """Pull meaningful words (len > 4) from snippets as fallback keywords."""
    keywords = set()
    for snippet in snippets:
        for word in snippet.split():
            clean = word.strip(".,()[]:;\"'")
            if len(clean) > 4:
                keywords.add(clean)
    return sorted(keywords)


def convert(input_path: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if not questions:
        print("❌ No 'questions' key found in input JSON.")
        sys.exit(1)

    # Group questions by source_document
    grouped = defaultdict(list)
    for q in questions:
        source = q.get("source_document", "unknown_source")
        grouped[source].append(q)

    print(f"\n📂 Found {len(grouped)} unique source document(s):\n")

    summary = []

    for source_doc, qs in grouped.items():
        converted_questions = []

        for q in qs:
            snippets = clean_snippets(q.get("evidence_snippets", []))
            keywords = extract_keywords(snippets)

            converted_questions.append({
                "id":            q.get("id", ""),
                "question":      q["question"],
                "relevant":      snippets,       # rag-chunk uses this for recall scoring
                "keywords":      keywords,        # informational fallback
                "difficulty":    q.get("difficulty", ""),
                "question_type": q.get("question_type", ""),
                "page_reference":q.get("page_reference", ""),
                "answer":        q.get("answer", "")
            })

        output = {
            "dataset_info": {
                "source_document": source_doc,
                "total_questions": len(converted_questions),
                "original_file": os.path.basename(input_path)
            },
            "questions": converted_questions
        }

        out_filename = safe_filename(source_doc)
        out_path = os.path.join(output_dir, out_filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        summary.append((source_doc, len(converted_questions), out_path))
        print(f"  ✅  [{len(converted_questions):>3} questions]  {source_doc}")
        print(f"       → {out_path}\n")

    # Print rag-chunk commands for each file
    print("\n" + "="*70)
    print("📋  READY-TO-RUN rag-chunk commands:")
    print("="*70)
    for source_doc, count, out_path in summary:
        # Derive a suggested docs folder name from source
        folder_hint = re.sub(r"\.pdf$", "", source_doc, flags=re.IGNORECASE)
        print(f"""
# {source_doc}  ({count} questions)
rag-chunk analyze "/path/to/{folder_hint}/pages/" \\
  --strategy all \\
  --chunk-size 300 \\
  --overlap 50 \\
  --test-file "{out_path}" \\
  --top-k 3 \\
  --use-embeddings \\
  --output table
""")

    print("="*70)
    print(f"\n✅  Done. {len(grouped)} file(s) written to: {os.path.abspath(output_dir)}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_by_source.py <input.json> [output_dir]")
        print("Example: python split_by_source.py Q_2.json ./ragchunk_questions/")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./ragchunk_questions"
    convert(input_path, output_dir)