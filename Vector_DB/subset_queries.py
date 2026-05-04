import json

# File paths
big_file = "/home/olj3kor/praveen/chunk_methods/ragas_dataset/rag_gold_pipeline/output/stage_c_finalization/gold_v5.0.json"
subset_file = "/home/olj3kor/praveen/full_pipeline/clean_GT/gold_gt_1000.json"
output_file = "/home/olj3kor/praveen/full_pipeline/clean_GT/gold_gt_300.json"

# Load data
with open(big_file, "r") as f:
    big_data = json.load(f)

with open(subset_file, "r") as f:
    subset_data = json.load(f)

# Extract IDs from subset
subset_ids = {item["id"] for item in subset_data}

# Filter entries present only in big_data
difference = [item for item in big_data if item["id"] not in subset_ids]

# Save result
with open(output_file, "w") as f:
    json.dump(difference, f, indent=2)

print(f"Extracted {len(difference)} entries not in subset.")