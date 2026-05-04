
# """
# PageIndex RAG Pipeline
# ======================
# Uses the existing retrieve.py from the PageIndex repo.
# Reads questions from Q.json, runs:
#   1. Tree Search
#   2. Page Extraction
#   3. Answer Generation
#   4. Retrieval Evaluation  (page overlap vs gold page_reference — no LLM)
#   5. LLM Judge             (generated answer vs ground truth + evidence snippets)

# To change anything, edit ONLY the CONFIG section below.
# """

# import json
# import re
# import sys
# import os
# import time
# from openai import OpenAI
# from dotenv import load_dotenv

# # =============================================================================
# # CONFIG — Edit these to change paths, model, or behaviour
# # =============================================================================

# ENV_FILE         = "/home/olj3kor/praveen/Github_copilot/.env"

# PAGEINDEX_REPO   = "/home/olj3kor/praveen/PageIndex"
# QUERIES_FILE     = "/home/olj3kor/praveen/Github_copilot/Q.json"
# STRUCTURE_DIR    = "/home/olj3kor/praveen/PageIndex/results"
# PDF_DIR          = "/home/olj3kor/praveen/Image_dataset_generation/pdfs"
# OUTPUT_FILE      = "/home/olj3kor/praveen/Github_copilot/rag_results.json"

# MODEL            = "moonshotai/kimi-k2-instruct-0905"
# SLEEP_BETWEEN_Q  = 1.0    # seconds between questions (avoids rate limits)

# # =============================================================================
# # SETUP — import retrieve.py from the existing repo (no duplication)
# # =============================================================================

# load_dotenv(ENV_FILE)

# sys.path.insert(0, PAGEINDEX_REPO)
# from pageindex.retrieve import get_document_structure, get_page_content

# client = OpenAI(
#     api_key=os.getenv("NVIDIA_API_KEY"),
#     base_url=os.getenv("OPENAI_BASE_URL"),
# )


# # =============================================================================
# # HELPERS
# # =============================================================================

# def load_structure(pdf_name: str) -> dict:
#     """Load _structure.json for a given PDF filename."""
#     base           = os.path.splitext(pdf_name)[0]
#     structure_path = os.path.join(STRUCTURE_DIR, f"{base}_structure.json")
#     if not os.path.exists(structure_path):
#         raise FileNotFoundError(f"Structure file not found: {structure_path}")
#     with open(structure_path, "r") as f:
#         return json.load(f)


# def build_documents(pdf_name: str, structure: dict) -> tuple[dict, str]:
#     """
#     Build the documents dict expected by retrieve.py.
#     Returns (documents_dict, doc_id).
#     """
#     doc_id   = os.path.splitext(pdf_name)[0]
#     pdf_path = os.path.join(PDF_DIR, pdf_name)
#     documents = {
#         doc_id: {
#             "type":            "pdf",
#             "doc_name":        pdf_name,
#             "doc_description": structure.get("doc_description", ""),
#             "path":            pdf_path,
#             "structure":       structure.get("nodes", structure),
#         }
#     }
#     return documents, doc_id


# def tree_search(query: str, tree_structure_json: str) -> list[dict]:
#     """
#     Step 2 — Ask LLM to identify relevant nodes from the tree structure.
#     Returns list of dicts with node_id, start_index, end_index.
#     """
#     prompt = f"""You are given a query and the tree structure of a document.
# Find all nodes that are likely to contain the answer.

# Query: {query}

# Document tree structure:
# {tree_structure_json}

# Reply ONLY in this JSON format with no extra text:
# {{
#   "thinking": "<your reasoning about which nodes are relevant>",
#   "relevant_nodes": [
#     {{"node_id": "0001", "start_index": 5, "end_index": 8}},
#     ...
#   ]
# }}"""

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"},
#     )
#     result = json.loads(response.choices[0].message.content)
#     return result.get("relevant_nodes", [])


# def generate_answer(query: str, page_contents: list[dict]) -> str:
#     """
#     Step 4 — Generate final answer from extracted page content.
#     """
#     context = "\n\n".join(
#         f"[Page {p['page']}]\n{p['content']}"
#         for p in page_contents if p.get("content")
#     )
#     prompt = f"""Answer the following question using only the provided context.
# Be precise and cite the page number when possible.

# Question: {query}

# Context:
# {context}

# Answer:"""

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return response.choices[0].message.content.strip()


# def get_page_range_string(nodes: list[dict]) -> str:
#     """Convert list of node dicts into a pages string like '5-8,12-15' for get_page_content."""
#     parts = []
#     for node in nodes:
#         start = node.get("start_index")
#         end   = node.get("end_index")
#         if start is not None and end is not None:
#             parts.append(f"{start}-{end}" if start != end else str(start))
#     return ",".join(parts)


# def check_retrieval_overlap(retrieved_nodes: list[dict], page_reference: str) -> dict:
#     """
#     Step 5a — No LLM needed.
#     Compares retrieved page ranges against the gold page_reference from Q.json
#     e.g. "Pages 5-6" or "Page 12".
#     Returns hit flag, recall, and which pages overlapped.
#     """
#     gold_pages = set(int(m) for m in re.findall(r'\d+', page_reference))
#     retrieved_pages = set()
#     for node in retrieved_nodes:
#         start = node.get("start_index")
#         end   = node.get("end_index")
#         if start is not None and end is not None:
#             retrieved_pages.update(range(start, end + 1))
#     overlap = gold_pages & retrieved_pages
#     return {
#         "gold_pages":      sorted(gold_pages),
#         "retrieved_pages": sorted(retrieved_pages),
#         "overlap_pages":   sorted(overlap),
#         "retrieval_hit":   len(overlap) > 0,
#         "recall":          round(len(overlap) / len(gold_pages), 2) if gold_pages else 0.0,
#     }


# def llm_judge(question: str, ground_truth: str, generated_answer: str, evidence_snippets: list[str]) -> dict:
#     """
#     Step 5b — LLM as judge.
#     Evaluates the generated answer against ground truth and gold evidence snippets.
#     Scores: correctness, completeness, hallucination, verdict.
#     """
#     snippets_text = "\n".join(f"- {s}" for s in evidence_snippets) if evidence_snippets else "N/A"
#     prompt = f"""You are an expert evaluator. Compare the generated answer to the ground truth and evidence.

# Question: {question}

# Ground Truth Answer: {ground_truth}

# Evidence Snippets (gold):
# {snippets_text}

# Generated Answer: {generated_answer}

# Evaluate on these three criteria:
# 1. Factual correctness — does the generated answer convey the same facts as the ground truth?
# 2. Completeness — does it cover all key points in the ground truth?
# 3. Hallucination — does it add facts not supported by the ground truth or evidence?

# Reply ONLY in this JSON format with no extra text:
# {{
#   "verdict": "correct" | "partial" | "incorrect",
#   "correctness_score": <float 0.0 to 1.0>,
#   "completeness_score": <float 0.0 to 1.0>,
#   "hallucination": "none" | "minor" | "major",
#   "reasoning": "<brief explanation of your scores>"
# }}"""

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"},
#     )
#     return json.loads(response.choices[0].message.content)


# # =============================================================================
# # MAIN PIPELINE
# # =============================================================================

# def run_pipeline():
#     with open(QUERIES_FILE, "r") as f:
#         data = json.load(f)
#     questions = data["questions"]

#     results = []
#     total   = len(questions)

#     for i, q in enumerate(questions, 1):
#         qid               = q["id"]
#         query             = q["question"]
#         pdf_name          = q["source_document"]
#         ground_truth      = q.get("answer", "")
#         evidence_snippets = q.get("evidence_snippets", [])
#         page_reference    = q.get("page_reference", "")

#         print(f"\n[{i}/{total}] {qid} — {query[:80]}...")

#         try:
#             # Step 1 — Load structure + build documents dict
#             structure = load_structure(pdf_name)
#             documents, doc_id = build_documents(pdf_name, structure)

#             # Step 2 — Tree search: get structure → LLM picks relevant nodes
#             tree_json      = get_document_structure(documents, doc_id)
#             relevant_nodes = tree_search(query, tree_json)
#             print(f"  → {len(relevant_nodes)} node(s) retrieved")

#             # Step 3 — Extract page content for the retrieved nodes
#             page_range = get_page_range_string(relevant_nodes)
#             if not page_range:
#                 raise ValueError("Tree search returned no nodes with page ranges.")
#             raw_content   = get_page_content(documents, doc_id, page_range)
#             page_contents = json.loads(raw_content)

#             # Step 4 — Generate answer from extracted pages
#             answer = generate_answer(query, page_contents)
#             print(f"  → Answer: {answer[:120]}...")

#             # Step 5a — Retrieval evaluation (no LLM — page overlap vs gold page_reference)
#             retrieval_eval = check_retrieval_overlap(relevant_nodes, page_reference)
#             print(f"  → Retrieval hit: {retrieval_eval['retrieval_hit']} | Recall: {retrieval_eval['recall']}")

#             # Step 5b — LLM judge: generated answer vs GT + evidence snippets
#             evaluation = llm_judge(
#                 question          = query,
#                 ground_truth      = ground_truth,
#                 generated_answer  = answer,
#                 evidence_snippets = evidence_snippets,
#             )
#             print(f"  → Verdict: {evaluation.get('verdict')} | Correctness: {evaluation.get('correctness_score')}")

#             results.append({
#                 "id":              qid,
#                 "question":        query,
#                 "source_document": pdf_name,
#                 "difficulty":      q.get("difficulty", ""),
#                 "question_type":   q.get("question_type", ""),
#                 "retrieved_nodes": relevant_nodes,
#                 "pages_used":      page_range,
#                 "answer":          answer,
#                 "ground_truth":    ground_truth,
#                 "retrieval_eval":  retrieval_eval,
#                 "evaluation":      evaluation,
#                 "status":          "success",
#             })

#         except Exception as e:
#             print(f"  ✗ ERROR: {e}")
#             results.append({
#                 "id":              qid,
#                 "question":        query,
#                 "source_document": pdf_name,
#                 "difficulty":      q.get("difficulty", ""),
#                 "question_type":   q.get("question_type", ""),
#                 "answer":          "",
#                 "ground_truth":    ground_truth,
#                 "status":          "error",
#                 "error":           str(e),
#             })

#         time.sleep(SLEEP_BETWEEN_Q)

#     # Summary stats
#     success     = sum(1 for r in results if r["status"] == "success")
#     correct     = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "correct")
#     partial     = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "partial")
#     ret_hits    = sum(1 for r in results if r.get("retrieval_eval", {}).get("retrieval_hit"))
#     avg_correct = round(
#         sum(r.get("evaluation", {}).get("correctness_score", 0) for r in results if r["status"] == "success")
#         / max(success, 1), 3
#     )

#     output = {
#         "total":       total,
#         "successful":  success,
#         "summary": {
#             "retrieval_hits":        ret_hits,
#             "retrieval_hit_rate":    round(ret_hits / max(success, 1), 3),
#             "correct":               correct,
#             "partial":               partial,
#             "incorrect":             success - correct - partial,
#             "avg_correctness_score": avg_correct,
#         },
#         "results": results,
#     }

#     os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
#     with open(OUTPUT_FILE, "w") as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)

#     print(f"\n{'='*60}")
#     print(f"✅ Done — {success}/{total} successful")
#     print(f"   Retrieval hit rate : {ret_hits}/{success}")
#     print(f"   Correct / Partial  : {correct} / {partial}")
#     print(f"   Avg correctness    : {avg_correct}")
#     print(f"   Results saved to   : {OUTPUT_FILE}")
#     print(f"{'='*60}")


# if __name__ == "__main__":
#     run_pipeline()

#######################################################################################################################################################

"""
PageIndex RAG Pipeline v2
=========================
Updated to handle new Q.json format with fields:
  user_input, source_pdf, reference, reference_contexts,
  start_page, end_page, node_id, node_title, doc_title,
  question_type, tree_depth, is_leaf_node

Pipeline steps:
  1. Load structure + build documents dict
  2. Tree Search  (LLM picks relevant nodes)
  3. Page Extraction
  4. Answer Generation
  5a. Retrieval Evaluation  (page overlap vs gold start_page/end_page — no LLM)
  5b. LLM Judge             (generated answer vs reference + reference_contexts)

To change anything, edit ONLY the CONFIG section below.
"""

import json
import re
import sys
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# =============================================================================
# CONFIG — Edit these to change paths, model, or behaviour
# =============================================================================

ENV_FILE        = "/home/olj3kor/praveen/Github_copilot/.env"

PAGEINDEX_REPO  = "/home/olj3kor/praveen/PageIndex"
QUERIES_FILE    = "/home/olj3kor/praveen/gt_output/gt_dataset.json"
STRUCTURE_DIR   = "/home/olj3kor/praveen/hierarchies_v2"
PDF_DIR         = "/home/olj3kor/praveen/Image_dataset_generation/pdfs"
OUTPUT_FILE     = "/home/olj3kor/praveen/rag_v2_results.json"

MODEL           = "moonshotai/kimi-k2-instruct-0905"
SLEEP_BETWEEN_Q = 0.5   # seconds between questions (avoids rate limits)

# =============================================================================
# SETUP
# =============================================================================

load_dotenv(ENV_FILE)

sys.path.insert(0, PAGEINDEX_REPO)
from pageindex.retrieve import get_document_structure, get_page_content

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# =============================================================================
# HELPERS
# =============================================================================

def make_id(index: int) -> str:
    """Generate a sequential question ID like q001, q002, ..."""
    return f"q{index:03d}"


def load_structure(pdf_name: str) -> dict:
    """Load _structure.json for a given PDF filename."""
    base           = os.path.splitext(pdf_name)[0]
    structure_path = os.path.join(STRUCTURE_DIR, f"{base}_structure.json")
    if not os.path.exists(structure_path):
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    with open(structure_path, "r") as f:
        return json.load(f)


def build_documents(pdf_name: str, structure: dict) -> tuple[dict, str]:
    """
    Build the documents dict expected by retrieve.py.
    Returns (documents_dict, doc_id).
    """
    doc_id   = os.path.splitext(pdf_name)[0]
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    documents = {
        doc_id: {
            "type":            "pdf",
            "doc_name":        pdf_name,
            "doc_description": structure.get("doc_description", ""),
            "path":            pdf_path,
            "structure":       structure.get("nodes", structure),
        }
    }
    return documents, doc_id


def tree_search(query: str, tree_structure_json: str) -> list[dict]:
    """
    Step 2 — Ask LLM to identify relevant nodes from the tree structure.
    Returns list of dicts with node_id, start_index, end_index.
    """
    prompt = f"""You are given a query and the tree structure of a document.
Find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure:
{tree_structure_json}

Reply ONLY in this JSON format with no extra text:
{{
  "thinking": "<your reasoning about which nodes are relevant>",
  "relevant_nodes": [
    {{"node_id": "0001", "start_index": 5, "end_index": 8}},
    ...
  ]
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("relevant_nodes", [])


def generate_answer(query: str, page_contents: list[dict]) -> str:
    """
    Step 4 — Generate final answer from extracted page content.
    """
    context = "\n\n".join(
        f"[Page {p['page']}]\n{p['content']}"
        for p in page_contents if p.get("content")
    )
    prompt = f"""Answer the following question using only the provided context.
Be precise and cite the page number when possible.

Question: {query}

Context:
{context}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def get_page_range_string(nodes: list[dict]) -> str:
    """Convert list of node dicts into a pages string like '5-8,12-15' for get_page_content."""
    parts = []
    for node in nodes:
        start = node.get("start_index")
        end   = node.get("end_index")
        if start is not None and end is not None:
            parts.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(parts)


def check_retrieval_overlap(retrieved_nodes: list[dict], start_page: int, end_page: int) -> dict:
    """
    Step 5a — No LLM needed.
    Compares retrieved page ranges against the gold start_page/end_page from Q.json.
    Returns hit flag, recall, precision, F1, and which pages overlapped.
    """
    gold_pages = set(range(start_page, end_page + 1))

    retrieved_pages = set()
    for node in retrieved_nodes:
        s = node.get("start_index")
        e = node.get("end_index")
        if s is not None and e is not None:
            retrieved_pages.update(range(s, e + 1))

    overlap   = gold_pages & retrieved_pages
    recall    = round(len(overlap) / len(gold_pages),    2) if gold_pages    else 0.0
    precision = round(len(overlap) / len(retrieved_pages), 2) if retrieved_pages else 0.0
    f1        = round(
        2 * precision * recall / (precision + recall), 2
    ) if (precision + recall) > 0 else 0.0

    return {
        "gold_pages":      sorted(gold_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "overlap_pages":   sorted(overlap),
        "retrieval_hit":   len(overlap) > 0,
        "recall":          recall,
        "precision":       precision,
        "f1":              f1,
    }


def llm_judge(
    question:          str,
    ground_truth:      str,
    generated_answer:  str,
    evidence_snippets: list[str],
    node_title:        str,
    doc_title:         str,
) -> dict:
    """
    Step 5b — LLM as judge.
    Evaluates the generated answer against ground truth and gold evidence snippets.
    Also uses node_title and doc_title as grounding context.
    Scores: correctness, completeness, hallucination, verdict.
    """
    snippets_text = "\n".join(f"- {s}" for s in evidence_snippets) if evidence_snippets else "N/A"

    prompt = f"""You are an expert evaluator for a RAG system.

Document: {doc_title}
Section: {node_title}

Question: {question}

Ground Truth Answer:
{ground_truth}

Gold Evidence Snippets (from the source document):
{snippets_text}

Generated Answer:
{generated_answer}

Evaluate on these three criteria:
1. Factual correctness — does the generated answer convey the same facts as the ground truth?
2. Completeness — does it cover all key points in the ground truth?
3. Hallucination — does it add facts not supported by the ground truth or evidence?

Reply ONLY in this JSON format with no extra text:
{{
  "verdict": "correct" | "partial" | "incorrect",
  "correctness_score": <float 0.0 to 1.0>,
  "completeness_score": <float 0.0 to 1.0>,
  "hallucination": "none" | "minor" | "major",
  "reasoning": "<brief explanation of your scores>"
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    with open(QUERIES_FILE, "r") as f:
        data = json.load(f)

    # Support both a bare list and a {"questions": [...]} wrapper
    questions = data if isinstance(data, list) else data.get("questions", [])

    results = []
    total   = len(questions)

    for i, q in enumerate(questions, 1):
        # ── Field mapping: new format → pipeline variables ──────────────────
        qid               = make_id(i)
        query             = q.get("user_input", "")
        pdf_name          = q.get("source_pdf", "")
        ground_truth      = q.get("reference", "")
        evidence_snippets = q.get("reference_contexts", [])   # list of strings
        start_page        = q.get("start_page")               # int
        end_page          = q.get("end_page")                 # int

        # Metadata carried through to output for analysis
        doc_title         = q.get("doc_title", "")
        node_title        = q.get("node_title", "")
        gold_node_id      = q.get("node_id", "")
        question_type     = q.get("question_type", "")
        tree_depth        = q.get("tree_depth", "")
        is_leaf_node      = q.get("is_leaf_node", None)
        generator_model   = q.get("generator_model", "")

        print(f"\n[{i}/{total}] {qid} — {query[:80]}...")

        # Validate required fields before doing any work
        if not query or not pdf_name:
            print(f"  ✗ SKIP: missing user_input or source_pdf")
            results.append({
                "id":     qid,
                "status": "skipped",
                "error":  "missing user_input or source_pdf",
                **{k: q.get(k, "") for k in (
                    "user_input", "source_pdf", "reference",
                    "doc_title", "node_title", "node_id",
                    "question_type", "tree_depth", "is_leaf_node",
                )},
            })
            continue

        try:
            # Step 1 — Load structure + build documents dict
            structure = load_structure(pdf_name)
            documents, doc_id = build_documents(pdf_name, structure)

            # Step 2 — Tree search: get structure → LLM picks relevant nodes
            tree_json      = get_document_structure(documents, doc_id)
            relevant_nodes = tree_search(query, tree_json)
            print(f"  → {len(relevant_nodes)} node(s) retrieved")

            # Step 3 — Extract page content for the retrieved nodes
            page_range = get_page_range_string(relevant_nodes)
            if not page_range:
                raise ValueError("Tree search returned no nodes with page ranges.")
            raw_content   = get_page_content(documents, doc_id, page_range)
            page_contents = json.loads(raw_content)

            # Step 4 — Generate answer from extracted pages
            answer = generate_answer(query, page_contents)
            print(f"  → Answer: {answer[:120]}...")

            # Step 5a — Retrieval evaluation (no LLM)
            if start_page is not None and end_page is not None:
                retrieval_eval = check_retrieval_overlap(relevant_nodes, start_page, end_page)
            else:
                retrieval_eval = {"retrieval_hit": None, "note": "no gold page range in source"}
            print(f"  → Retrieval hit: {retrieval_eval.get('retrieval_hit')} | "
                  f"Recall: {retrieval_eval.get('recall', 'N/A')}")

            # Step 5b — LLM judge
            evaluation = llm_judge(
                question          = query,
                ground_truth      = ground_truth,
                generated_answer  = answer,
                evidence_snippets = evidence_snippets,
                node_title        = node_title,
                doc_title         = doc_title,
            )
            print(f"  → Verdict: {evaluation.get('verdict')} | "
                  f"Correctness: {evaluation.get('correctness_score')}")

            results.append({
                # ── IDs & question ──────────────────────────────────────────
                "id":               qid,
                "question":         query,
                "question_type":    question_type,
                # ── Source metadata ─────────────────────────────────────────
                "source_document":  pdf_name,
                "doc_title":        doc_title,
                "gold_node_id":     gold_node_id,
                "node_title":       node_title,
                "tree_depth":       tree_depth,
                "is_leaf_node":     is_leaf_node,
                "gold_start_page":  start_page,
                "gold_end_page":    end_page,
                # ── Retrieval ───────────────────────────────────────────────
                "retrieved_nodes":  relevant_nodes,
                "pages_used":       page_range,
                # ── Answer & ground truth ───────────────────────────────────
                "answer":           answer,
                "ground_truth":     ground_truth,
                "evidence_snippets": evidence_snippets,
                # ── Evaluations ─────────────────────────────────────────────
                "retrieval_eval":   retrieval_eval,
                "evaluation":       evaluation,
                # ── Bookkeeping ─────────────────────────────────────────────
                "generator_model":  generator_model,
                "status":           "success",
            })

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({
                "id":               qid,
                "question":         query,
                "question_type":    question_type,
                "source_document":  pdf_name,
                "doc_title":        doc_title,
                "gold_node_id":     gold_node_id,
                "node_title":       node_title,
                "tree_depth":       tree_depth,
                "is_leaf_node":     is_leaf_node,
                "gold_start_page":  start_page,
                "gold_end_page":    end_page,
                "answer":           "",
                "ground_truth":     ground_truth,
                "generator_model":  generator_model,
                "status":           "error",
                "error":            str(e),
            })

        time.sleep(SLEEP_BETWEEN_Q)

    # ── Summary stats ────────────────────────────────────────────────────────
    success  = sum(1 for r in results if r["status"] == "success")
    correct  = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "correct")
    partial  = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "partial")
    ret_hits = sum(1 for r in results if r.get("retrieval_eval", {}).get("retrieval_hit"))

    avg_correctness = round(
        sum(r.get("evaluation", {}).get("correctness_score", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )
    avg_completeness = round(
        sum(r.get("evaluation", {}).get("completeness_score", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )
    avg_recall = round(
        sum(r.get("retrieval_eval", {}).get("recall", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )
    avg_precision = round(
        sum(r.get("retrieval_eval", {}).get("precision", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )
    avg_f1 = round(
        sum(r.get("retrieval_eval", {}).get("f1", 0)
            for r in results if r["status"] == "success")
        / max(success, 1), 3
    )

    # Break down verdict by question_type
    qtypes = {}
    for r in results:
        if r["status"] != "success":
            continue
        qt = r.get("question_type", "unknown") or "unknown"
        verdict = r.get("evaluation", {}).get("verdict", "unknown")
        if qt not in qtypes:
            qtypes[qt] = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
        qtypes[qt]["total"] += 1
        if verdict in qtypes[qt]:
            qtypes[qt][verdict] += 1

    output = {
        "total":      total,
        "successful": success,
        "errors":     total - success,
        "summary": {
            "retrieval_hits":         ret_hits,
            "retrieval_hit_rate":     round(ret_hits / max(success, 1), 3),
            "avg_retrieval_recall":   avg_recall,
            "avg_retrieval_precision": avg_precision,
            "avg_retrieval_f1":       avg_f1,
            "correct":                correct,
            "partial":                partial,
            "incorrect":              success - correct - partial,
            "avg_correctness_score":  avg_correctness,
            "avg_completeness_score": avg_completeness,
        },
        "breakdown_by_question_type": qtypes,
        "results": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Done — {success}/{total} successful  ({total - success} errors)")
    print(f"   Retrieval hit rate    : {ret_hits}/{success}")
    print(f"   Avg recall / prec / F1: {avg_recall} / {avg_precision} / {avg_f1}")
    print(f"   Correct / Partial     : {correct} / {partial}")
    print(f"   Avg correctness       : {avg_correctness}")
    print(f"   Avg completeness      : {avg_completeness}")
    print(f"   Results saved to      : {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_pipeline()


#######################################################################################################################################################