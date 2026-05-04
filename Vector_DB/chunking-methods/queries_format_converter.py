"""
Convert your evaluation JSON (with evidence_snippets) to rag-chunk format (with relevant).
Usage: python convert_to_ragchunk.py Q_2.json > rag_chunk_questions.json
"""

import json
import sys

def convert(input_path: str, output_path: str = None):
    with open(input_path, "r") as f:
        data = json.load(f)

    converted = {"questions": []}

    for q in data["questions"]:
        # Clean up evidence snippets: strip whitespace, remove empty strings
        relevant = [s.strip() for s in q.get("evidence_snippets", []) if s.strip()]

        # Also extract key terms from the snippet as fallback keywords
        # (in case full snippet doesn't match verbatim in OCR text)
        keywords = set()
        for snippet in relevant:
            # Add individual meaningful words as fallback (length > 4)
            for word in snippet.split():
                clean = word.strip(".,()[]:;\"'")
                if len(clean) > 4:
                    keywords.add(clean)

        converted["questions"].append({
            "id": q.get("id", ""),
            "question": q["question"],
            "relevant": relevant,          # exact evidence phrases
            "keywords": list(keywords),    # fallback keyword list (informational)
            "difficulty": q.get("difficulty", ""),
            "page_reference": q.get("page_reference", "")
        })

    result = json.dumps(converted, indent=2)

    if output_path:
        with open(output_path, "w") as f:
            f.write(result)
        print(f"✅ Saved to {output_path}")
    else:
        print(result)

    return converted


# --- Run ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_ragchunk.py <input.json> [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(input_path, output_path)