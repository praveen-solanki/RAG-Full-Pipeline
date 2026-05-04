"""
filter_gt.py

Removes records from a Ground Truth (GT) JSON file in two modes:
  1. --n            : Remove N records whose questions failed in the retrieval results file.
  2. --random_remove: Remove N records chosen randomly from the GT (ignores results file).

Both flags can be used together — failures are removed first, then random ones from what's left.

Usage examples:
    # Remove 50 failed queries (based on results file)
    python filter_gt.py --gt gt.json --results eval.json --output out.json --n 50

    # Remove 30 random records (no results file needed)
    python filter_gt.py --gt gt.json --output out.json --random_remove 30

    # Both together
    python filter_gt.py --gt gt.json --results eval.json --output out.json --n 50 --random_remove 30
"""

import argparse
import json
import random
import sys


def load_json(path: str) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_failed_questions(results: dict) -> list[dict]:
    """Extract all failed questions from detailed_results where found=False."""
    failed = []
    for entry in results.get("detailed_results", []):
        if not entry.get("metrics", {}).get("found", True):
            failed.append({
                "question_index": entry.get("question_index"),
                "question": entry.get("question", "").strip(),
            })
    return failed


def normalize(text: str) -> str:
    """Normalize whitespace for comparison."""
    return " ".join(text.lower().split())


def match_failures_to_gt(gt_records: list, failed_questions: list, n: int) -> set:
    """
    Match the first n failed questions to GT records by comparing
    'question' (results) with 'user_input' (GT).
    Returns a set of GT record indices to remove.
    """
    gt_lookup = {normalize(r.get("user_input", "")): idx for idx, r in enumerate(gt_records)}

    to_remove = set()
    matched_count = 0

    for failure in failed_questions:
        if matched_count >= n:
            break
        key = normalize(failure["question"])
        if key in gt_lookup:
            gt_idx = gt_lookup[key]
            if gt_idx not in to_remove:
                to_remove.add(gt_idx)
                matched_count += 1
                print(f"  [MATCH] question_index={failure['question_index']} -> GT index={gt_idx}")
        else:
            print(f"  [NO MATCH] question_index={failure['question_index']}: "
                  f"{failure['question'][:80]}...")

    return to_remove


def main():
    parser = argparse.ArgumentParser(
        description="Remove records from a GT JSON file by failure matching or randomly."
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to the Ground Truth JSON file (array of QA records)."
    )
    parser.add_argument(
        "--results",
        required=False,
        default=None,
        help="Path to the retrieval evaluation results JSON file (required when using --n)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the filtered GT JSON file."
    )
    parser.add_argument(
        "--n",
        type=int,
        required=False,
        default=None,
        help="Number of failed queries (from --results file) to remove from the GT."
    )
    parser.add_argument(
        "--random_remove",
        type=int,
        required=False,
        default=None,
        help="Number of records to randomly remove from GT (independent of results file)."
    )
    args = parser.parse_args()

    # Validate argument combinations
    if args.n is None and args.random_remove is None:
        print("ERROR: Provide at least one of --n or --random_remove.")
        sys.exit(1)

    if args.n is not None and args.results is None:
        print("ERROR: --results is required when using --n.")
        sys.exit(1)

    # --- Load GT ---
    print(f"\nLoading GT file: {args.gt}")
    gt_records = load_json(args.gt)
    if not isinstance(gt_records, list):
        print("ERROR: GT file must be a JSON array of records.")
        sys.exit(1)
    print(f"  Total GT records: {len(gt_records)}")

    to_remove = set()

    # --- Mode 1: Remove based on failed queries from results file ---
    if args.n is not None:
        print(f"\nLoading retrieval results: {args.results}")
        results = load_json(args.results)
        if not isinstance(results, dict):
            print("ERROR: Results file must be a JSON object.")
            sys.exit(1)

        failed_questions = get_failed_questions(results)
        total_failures = len(failed_questions)
        print(f"Total failed queries found in results: {total_failures}")

        if args.n > total_failures:
            print(f"WARNING: Requested --n={args.n} but only {total_failures} failures exist. "
                  f"Will remove up to {total_failures}.")

        print(f"\nMatching first {args.n} failures to GT records...")
        failure_removals = match_failures_to_gt(gt_records, failed_questions, args.n)
        to_remove.update(failure_removals)
        print(f"Matched {len(failure_removals)} GT records from failure matching.")

    # --- Mode 2: Randomly remove records (from whatever remains) ---
    if args.random_remove is not None:
        available = [i for i in range(len(gt_records)) if i not in to_remove]
        count = min(args.random_remove, len(available))
        if count < args.random_remove:
            print(f"\nWARNING: Only {len(available)} records available for random removal "
                  f"(requested {args.random_remove}). Removing all available.")
        random_indices = random.sample(available, count)
        to_remove.update(random_indices)
        print(f"\nRandomly selected {count} records to remove.")

    # --- Filter and save ---
    print(f"\nTotal records to remove: {len(to_remove)}")
    filtered_gt = [r for idx, r in enumerate(gt_records) if idx not in to_remove]
    print(f"GT records after filtering: {len(filtered_gt)} "
          f"(removed {len(gt_records) - len(filtered_gt)})")

    save_json(filtered_gt, args.output)
    print(f"\nFiltered GT saved to: {args.output}\n")


if __name__ == "__main__":
    main()