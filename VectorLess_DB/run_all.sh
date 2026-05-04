#!/bin/bash

STRUCTURED_DIR="/home/olj3kor/praveen/Structured_files"
OUTPUT_BASE="/home/olj3kor/praveen/Structured_files_outputs"
QUERY="/home/olj3kor/praveen/Github_copilot/Q_copy.json"
PDF_DIR="/home/olj3kor/praveen/Image_dataset_generation/pdfs"
PROVIDER="ollama"
MODEL="qwen2.5:32b"
JUDGE_PROVIDER="ollama"
JUDGE_MODEL="llama3:8b"
PAGEINDEX_REPO="/home/olj3kor/praveen/PageIndex"
DOMAIN="autosar"
MODE="eval"

for folder in "$STRUCTURED_DIR"/*/; do
    folder_name=$(basename "$folder")
    output_dir="$OUTPUT_BASE/$folder_name"

    echo "=========================================="
    echo "Running for: $folder_name"
    echo "Output dir : $output_dir"
    echo "=========================================="

    mkdir -p "$output_dir"

    python Pageindex_retrival.py \
        --query "$QUERY" \
        --tree_dir "$folder" \
        --pdf_dir "$PDF_DIR" \
        --output_dir "$output_dir" \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --judge_provider "$JUDGE_PROVIDER" \
        --judge_model "$JUDGE_MODEL" \
        --pageindex_repo "$PAGEINDEX_REPO" \
        --domain "$DOMAIN" \
        --mode "$MODE"

    echo "Done: $folder_name"
    echo ""
done

echo "All folders processed!"



# #!/bin/bash

# STRUCTURED_DIR="/home/olj3kor/praveen/Structured_files"
# OUTPUT_BASE="/home/olj3kor/praveen/Structured_files_outputs"
# QUERY="/home/olj3kor/praveen/Github_copilot/Q_2.json"
# PDF_DIR="/home/olj3kor/praveen/Image_dataset_generation/pdfs"
# PROVIDER="ollama"
# MODEL="qwen2.5:7b"
# JUDGE_PROVIDER="ollama"
# JUDGE_MODEL="llama3:8b"
# PAGEINDEX_REPO="/home/olj3kor/praveen/PageIndex"
# DOMAIN="autosar"
# MODE="eval"

# TARGET_FILE="Utilization of Crypto Services_structure.json"

# for folder in "$STRUCTURED_DIR"/*/; do
#     folder_name=$(basename "$folder")
#     tree_file="$folder/$TARGET_FILE"

#     # Skip if the target file doesn't exist in this folder
#     if [ ! -f "$tree_file" ]; then
#         echo "Skipping $folder_name — target file not found."
#         continue
#     fi

#     output_dir="$OUTPUT_BASE/$folder_name"

#     echo "=========================================="
#     echo "Running for: $folder_name"
#     echo "Tree file  : $tree_file"
#     echo "Output dir : $output_dir"
#     echo "=========================================="

#     mkdir -p "$output_dir"

#     python Pageindex_retrival.py \
#         --query "$QUERY" \
#         --tree_dir "$folder" \
#         --pdf_dir "$PDF_DIR" \
#         --output_dir "$output_dir" \
#         --provider "$PROVIDER" \
#         --model "$MODEL" \
#         --judge_provider "$JUDGE_PROVIDER" \
#         --judge_model "$JUDGE_MODEL" \
#         --pageindex_repo "$PAGEINDEX_REPO" \
#         --domain "$DOMAIN" \
#         --mode "$MODE"

#     echo "Done: $folder_name"
#     echo ""
# done

# echo "All folders processed!"