# """
# Parallel rag-chunk benchmark runner.
# Runs all (document × chunk_size × top_k) combinations concurrently.

# Total runs: 9 docs × 3 chunk_sizes × 3 top_k = 81 runs

# Output:
#     results/
#     ├── chunk_512__topk_3/
#     │   ├── Utilization_of_Crypto_Services__512_k3.txt
#     │   └── ...
#     ├── chunk_512__topk_5/
#     ├── chunk_512__topk_10/
#     ├── chunk_768__topk_3/
#     │   └── ...
#     ├── ...
#     └── summary.txt

# Usage:
#     python run_benchmark.py
# """

# import subprocess
# import os
# import re
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from datetime import datetime

# # ── Config ────────────────────────────────────────────────────────────────────

# COMMANDS = [
#     {
#         "name": "Technical_Report_on_Operating_System_Tracing_Interface",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/Technical Report on Operating System Tracing Interface/pages",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/ragchunk_questions/Technical_Report_on_Operating_System_Tracing_Interface.json"
#     },
#     {
#         "name": "Specification_of_Raw_Data_Stream",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/Specification of Raw Data Stream/pages",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/ragchunk_questions/Specification_of_Raw_Data_Stream.json"
#     },
#     {
#         "name": "Specification_of_Firewall_for_Adaptive_Platform",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/Specification of Firewall for Adaptive Platform/pages",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/ragchunk_questions/Specification_of_Firewall_for_Adaptive_Platform.json"
#     },
#     {
#         "name": "Requirements_on_Operating_System_Interface",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/Requirements on Operating System Interface/pages",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/ragchunk_questions/Requirements_on_Operating_System_Interface.json"
#     },
#     {
#         "name": "Explanation_of_Service-Oriented_Vehicle_Diagnostics",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/Explanation of Service-Oriented Vehicle Diagnostics/pages",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/ragchunk_questions/Explanation_of_Service-Oriented_Vehicle_Diagnostics.json"
#     },
#     {
#         "name": "General_Specification_of_Transformers",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/General Specification of Transformers/pages",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/ragchunk_questions/General_Specification_of_Transformers.json"
#     },
#     {
#         "name": "Explanation_of_Sensor_Interfaces",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/Explanation of Sensor Interfaces/pages",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/ragchunk_questions/Explanation_of_Sensor_Interfaces.json"
#     },
#     {
#         "name": "Adaptive_Platform_Machine_Configuration",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/Adaptive Platform Machine Configuration/pages",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/ragchunk_questions/Adaptive_Platform_Machine_Configuration.json"
#     },
#     {
#         "name": "Utilization_of_Crypto_Services",
#         "pdf_path": "/home/olj3kor/praveen/GLM_OCR_OUTPUT/Utilization of Crypto Services/pages/",
#         "test_file": "/home/olj3kor/praveen/chunk_methods/rag-chunk/Q_2_ragchunk.json"
#     },
# ]

# CHUNK_SIZES = [512, 768, 1024]
# TOP_K_VALUES = [3, 5, 10]
# OUTPUT_DIR  = "./results"
# MAX_WORKERS = 32   # tune down to 2 if RAM is tight

# # ── Helpers ───────────────────────────────────────────────────────────────────

# def overlap_for(chunk_size: int) -> int:
#     return int(chunk_size * 0.1)


# def strip_ansi(text: str) -> str:
#     """Remove terminal color/style escape codes from rich output."""
#     return re.sub(r"\x1b\[[0-9;]*m", "", text)


# def run_single(doc: dict, chunk_size: int, overlap: int, top_k: int, out_dir: str) -> dict:
#     """Run one rag-chunk command, save output to txt, return result dict."""
#     name     = doc["name"]
#     out_path = os.path.join(out_dir, f"{name}__{chunk_size}_k{top_k}.txt")

#     cmd = [
#         "rag-chunk", "analyze",
#         doc["pdf_path"],
#         "--strategy",   "all",
#         "--chunk-size", str(chunk_size),
#         "--overlap",    str(overlap),
#         "--test-file",  doc["test_file"],
#         "--top-k",      str(top_k),
#         "--use-embeddings",
#         "--output",     "table",
#     ]

#     start = time.time()
#     header = (
#         f"{'='*70}\n"
#         f"Document   : {name}\n"
#         f"Chunk size : {chunk_size}  |  Overlap : {overlap}  |  Top-K : {top_k}\n"
#         f"Command    : {' '.join(cmd)}\n"
#         f"Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
#         f"{'='*70}\n"
#     )

#     status = "SUCCESS"
#     stdout_text = ""
#     stderr_text = ""

#     try:
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             timeout=600,
#         )
#         elapsed     = time.time() - start
#         stdout_text = strip_ansi(result.stdout)
#         stderr_text = strip_ansi(result.stderr)
#         status      = "SUCCESS" if result.returncode == 0 else f"ERROR(rc={result.returncode})"

#         content = header + stdout_text
#         if stderr_text.strip():
#             content += f"\n--- STDERR ---\n{stderr_text}"
#         content += f"\n[{status}  |  {elapsed:.1f}s]\n"

#     except subprocess.TimeoutExpired:
#         elapsed = time.time() - start
#         status  = "TIMEOUT"
#         content = header + f"\n⚠️  Timed out after {elapsed:.0f}s\n"

#     except Exception as exc:
#         elapsed = time.time() - start
#         status  = f"EXCEPTION"
#         content = header + f"\n❌  {exc}\n"

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(content)

#     icon = "✅" if status == "SUCCESS" else "❌"
#     print(f"  {icon}  chunk={chunk_size} top_k={top_k:<3} | {name}  [{elapsed:.1f}s]")

#     return {
#         "name":       name,
#         "chunk_size": chunk_size,
#         "top_k":      top_k,
#         "status":     status,
#         "elapsed":    elapsed,
#         "out_path":   out_path,
#     }


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Build all jobs
#     jobs = []
#     for chunk_size in CHUNK_SIZES:
#         overlap = overlap_for(chunk_size)
#         for top_k in TOP_K_VALUES:
#             out_dir = os.path.join(OUTPUT_DIR, f"chunk_{chunk_size}__topk_{top_k}")
#             os.makedirs(out_dir, exist_ok=True)
#             for doc in COMMANDS:
#                 jobs.append((doc, chunk_size, overlap, top_k, out_dir))

#     total = len(jobs)

#     print(f"\n{'='*60}")
#     print(f"  RAG-CHUNK PARALLEL BENCHMARK")
#     print(f"{'='*60}")
#     print(f"  Documents   : {len(COMMANDS)}")
#     print(f"  Chunk sizes : {CHUNK_SIZES}")
#     print(f"  Top-K values: {TOP_K_VALUES}")
#     print(f"  Total runs  : {len(COMMANDS)} × {len(CHUNK_SIZES)} × {len(TOP_K_VALUES)} = {total}")
#     print(f"  Workers     : {MAX_WORKERS}")
#     print(f"  Output dir  : {os.path.abspath(OUTPUT_DIR)}")
#     print(f"{'='*60}\n")

#     all_results = []
#     run_start   = time.time()

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
#         futures = {
#             pool.submit(run_single, doc, cs, ov, tk, od): (doc["name"], cs, tk)
#             for doc, cs, ov, tk, od in jobs
#         }
#         completed = 0
#         for future in as_completed(futures):
#             completed += 1
#             try:
#                 all_results.append(future.result())
#             except Exception as exc:
#                 name, cs, tk = futures[future]
#                 print(f"  ❌  Unhandled [{name} / chunk={cs} / top_k={tk}]: {exc}")
#             print(f"  📊  Progress: {completed}/{total}", end="\r")

#     total_elapsed = time.time() - run_start

#     # ── Write summary ─────────────────────────────────────────────────────────
#     summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
#     all_results_sorted = sorted(
#         all_results, key=lambda x: (x["chunk_size"], x["top_k"], x["name"])
#     )

#     with open(summary_path, "w", encoding="utf-8") as f:
#         f.write("RAG-CHUNK BENCHMARK SUMMARY\n")
#         f.write(f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Total runs : {total}  |  Wall time: {total_elapsed:.1f}s\n")
#         f.write(f"Documents  : {len(COMMANDS)}\n")
#         f.write(f"Chunk sizes: {CHUNK_SIZES}\n")
#         f.write(f"Top-K      : {TOP_K_VALUES}\n")
#         f.write("="*95 + "\n\n")

#         # Table header
#         f.write(f"{'Document':<52} {'Chunk':>6} {'TopK':>5} {'Status':<14} {'Time':>7}\n")
#         f.write("-"*95 + "\n")

#         for r in all_results_sorted:
#             f.write(
#                 f"{r['name']:<52} {r['chunk_size']:>6} {r['top_k']:>5} "
#                 f"{r['status']:<14} {r['elapsed']:>6.1f}s\n"
#             )

#         successes = sum(1 for r in all_results if r["status"] == "SUCCESS")
#         f.write("\n" + "="*95 + "\n")
#         f.write(f"✅  Succeeded : {successes}/{total}\n")
#         f.write(f"❌  Failed    : {total - successes}/{total}\n")
#         f.write(f"⏱️   Wall time : {total_elapsed:.1f}s\n")

#     print(f"\n\n{'='*60}")
#     print(f"✅  All {total} jobs done in {total_elapsed:.1f}s")
#     successes = sum(1 for r in all_results if r["status"] == "SUCCESS")
#     print(f"   Succeeded : {successes}/{total}")
#     print(f"   Failed    : {total - successes}/{total}")
#     print(f"📄  Summary   → {summary_path}")
#     print(f"📁  Results   → {os.path.abspath(OUTPUT_DIR)}/")
#     print(f"{'='*60}\n")

#     # Print folder breakdown
#     print("Output folders:")
#     for chunk_size in CHUNK_SIZES:
#         for top_k in TOP_K_VALUES:
#             folder = os.path.join(OUTPUT_DIR, f"chunk_{chunk_size}__topk_{top_k}")
#             count  = len([r for r in all_results if r["chunk_size"] == chunk_size and r["top_k"] == top_k])
#             print(f"  {folder:<50}  ({count} files)")
#     print()


# if __name__ == "__main__":
#     main()





import subprocess
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

BASE_PATH = "/home/olj3kor/praveen/DocLing/MinerU_output/final_md_structured"
TEST_BASE = "/home/olj3kor/praveen/chunk_methods/ragchunk_questions"

COMMAND_NAMES = [
    "Technical_Report_on_Operating_System_Tracing_Interface",
    "Specification_of_Raw_Data_Stream",
    "Specification_of_Firewall_for_Adaptive_Platform",
    "Requirements_on_Operating_System_Interface",
    "Explanation_of_Service-Oriented_Vehicle_Diagnostics",
    "General_Specification_of_Transformers",
    "Explanation_of_Sensor_Interfaces",
    "Adaptive_Platform_Machine_Configuration",
    "Utilization_of_Crypto_Services",
]

def build_pdf_path(name: str) -> str:
    readable_name = name.replace("_", " ")
    return os.path.join(BASE_PATH, readable_name)

def build_test_path(name: str) -> str:
    return os.path.join(TEST_BASE, f"{name}.json")

COMMANDS = [
    {
        "name": name,
        "pdf_path": build_pdf_path(name),
        "test_file": build_test_path(name)
    }
    for name in COMMAND_NAMES
]

# Special override (if needed)
COMMANDS[-1]["test_file"] = "/home/olj3kor/praveen/chunk_methods/ragas_dataset/rag_gold_pipeline/output/stage_c_finalization/gold_v1.0.json"

CHUNK_SIZES = [512, 1024]
TOP_K_VALUES = [5, 10]
OUTPUT_DIR  = "./results_2"
MAX_WORKERS = 8

# ── Helpers ───────────────────────────────────────────────────────────────────

def overlap_for(chunk_size: int) -> int:
    return int(chunk_size * 0.1)

def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)

def validate_paths(doc):
    """Ensure folder exists and contains at least one .md file"""
    if not os.path.isdir(doc["pdf_path"]):
        return False, f"Folder not found: {doc['pdf_path']}"

    md_files = [f for f in os.listdir(doc["pdf_path"]) if f.endswith(".md")]
    if not md_files:
        return False, f"No .md files in: {doc['pdf_path']}"

    if not os.path.exists(doc["test_file"]):
        return False, f"Missing test file: {doc['test_file']}"

    return True, ""

def run_single(doc: dict, chunk_size: int, overlap: int, top_k: int, out_dir: str) -> dict:
    name     = doc["name"]
    out_path = os.path.join(out_dir, f"{name}__{chunk_size}_k{top_k}.txt")

    valid, msg = validate_paths(doc)
    if not valid:
        content = f"❌ VALIDATION FAILED\n{msg}\n"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(content)

        print(f"❌ {name}: {msg}")
        return {"name": name, "chunk_size": chunk_size, "top_k": top_k, "status": "INVALID", "elapsed": 0}

    cmd = [
        "rag-chunk", "analyze",
        doc["pdf_path"],
        "--strategy", "all",
        "--chunk-size", str(chunk_size),
        "--overlap", str(overlap),
        "--test-file", doc["test_file"],
        "--top-k", str(top_k),
        "--use-embeddings",
        "--output", "table",
    ]

    start = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5000)
        elapsed = time.time() - start

        stdout_text = strip_ansi(result.stdout)
        stderr_text = strip_ansi(result.stderr)

        status = "SUCCESS" if result.returncode == 0 else f"ERROR({result.returncode})"

        content = stdout_text
        if stderr_text.strip():
            content += "\n--- STDERR ---\n" + stderr_text

        content += f"\n[{status} | {elapsed:.1f}s]"

    except Exception as e:
        elapsed = time.time() - start
        status = "EXCEPTION"
        content = f"❌ {e}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)

    icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{icon} chunk={chunk_size} top_k={top_k:<3} | {name} [{elapsed:.1f}s]")

    return {
        "name": name,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "status": status,
        "elapsed": elapsed,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jobs = []
    for chunk_size in CHUNK_SIZES:
        overlap = overlap_for(chunk_size)

        for top_k in TOP_K_VALUES:
            out_dir = os.path.join(OUTPUT_DIR, f"chunk_{chunk_size}__topk_{top_k}")
            os.makedirs(out_dir, exist_ok=True)

            for doc in COMMANDS:
                jobs.append((doc, chunk_size, overlap, top_k, out_dir))

    total = len(jobs)

    print(f"\n{'='*60}")
    print(f"RAG-CHUNK PARALLEL BENCHMARK")
    print(f"{'='*60}")
    print(f"Total runs: {total}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_single, *job) for job in jobs]

        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            print(f"📊 Progress: {i}/{total}", end="\r")

    total_time = time.time() - start_time

    print("\n\n✅ DONE")
    print(f"Total time: {total_time:.1f}s")

if __name__ == "__main__":
    main()