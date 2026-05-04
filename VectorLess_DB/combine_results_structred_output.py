"""
combine_metrics.py

1. Combines all metrics_summary.json files → combined_metrics.json
2. Generates a flat comparison table  → metrics_comparison.csv

Usage:
    python combine_metrics.py --root /path/to/Structured_files_outputs
    python combine_metrics.py          # uses current directory by default
"""

import json
import os
import csv
import argparse
from datetime import datetime


# ── Flat columns we want in the comparison table ──────────────────────────────
SUMMARY_FIELDS = [
    "total",
    "successful",
    "errors",
    "avg_page_recall",
    "avg_evidence_recall",
    "accuracy",
    "avg_correctness_score",
    "avg_completeness_score",
    "correct",
    "partial",
    "incorrect",
    "hallucination_none",
    "hallucination_minor",
    "hallucination_major",
]


def extract_row(folder_name: str, data: dict) -> dict:
    """Pull the comparison fields out of one metrics_summary.json payload."""
    summary = data.get("summary", {})
    hallucination = summary.get("hallucination_counts", {})

    row = {"source": folder_name}

    # Top-level fields
    for field in ("total", "successful", "errors"):
        row[field] = data.get(field, "")

    # Summary fields
    for field in (
        "avg_page_recall",
        "avg_evidence_recall",
        "accuracy",
        "avg_correctness_score",
        "avg_completeness_score",
        "correct",
        "partial",
        "incorrect",
    ):
        row[field] = summary.get(field, "")

    # Hallucination sub-fields (flattened)
    row["hallucination_none"]  = hallucination.get("none",  "")
    row["hallucination_minor"] = hallucination.get("minor", "")
    row["hallucination_major"] = hallucination.get("major", "")

    return row


def combine_metrics(root_dir: str, output_filename: str = "combined_metrics.json",
                    csv_filename: str = "metrics_comparison.csv"):
    root_dir = os.path.abspath(root_dir)
    combined = {}
    source_files = []
    missing = []
    rows = []

    subdirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    print(f"Found {len(subdirs)} subdirectories.\n")

    for folder_name in subdirs:
        metrics_path = os.path.join(root_dir, folder_name, "metrics_summary.json")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)

            combined[folder_name] = data
            source_files.append(metrics_path)
            rows.append(extract_row(folder_name, data))
            print(f"  [OK]   {folder_name}")
        else:
            missing.append(folder_name)
            print(f"  [SKIP] {folder_name} — metrics_summary.json not found")

    # ── 1. Write combined_metrics.json ────────────────────────────────────────
    output_json = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "root_directory": root_dir,
            "total_sources": len(combined),
            "source_files": source_files,
            "skipped_folders": missing,
        },
        "combined_metrics": combined,
    }

    json_path = os.path.join(root_dir, output_filename)
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\n[1/2] Combined JSON  → {json_path}")

    # ── 2. Write metrics_comparison.csv ───────────────────────────────────────
    csv_path = os.path.join(root_dir, csv_filename)
    csv_columns = ["source"] + SUMMARY_FIELDS

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[2/2] Comparison CSV → {csv_path}")

    if missing:
        print(f"\nSkipped {len(missing)} folder(s) (no metrics_summary.json): {missing}")

    print(f"\nDone! Processed {len(combined)} source(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine metrics_summary.json files and produce a comparison table."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory containing the subdirectories (default: current dir)",
    )
    parser.add_argument(
        "--output-json",
        default="combined_metrics.json",
        help="Name of the merged JSON output file",
    )
    parser.add_argument(
        "--output-csv",
        default="metrics_comparison.csv",
        help="Name of the comparison CSV output file",
    )
    args = parser.parse_args()

    print(f"\nScanning: {os.path.abspath(args.root)}\n")
    combine_metrics(args.root, args.output_json, args.output_csv)