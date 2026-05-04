#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MinerU v3 — One-shot setup for AUTOSAR PDF-to-Markdown pipeline
# Tested on: Ubuntu 22.04 / 24.04, Python 3.10–3.12, NVIDIA A-series GPU
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "======================================================"
echo "  AUTOSAR MinerU Setup Script"
echo "======================================================"

# ── 0. Check Python version ───────────────────────────────────────────────────
PYTHON=$(which python3)
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "[INFO] Python: $PY_VER at $PYTHON"

# ── 1. Install uv (fast pip replacement recommended by MinerU) ────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] Installing uv ..."
    pip install --upgrade pip uv
fi

# ── 2. Install MinerU with all extras ────────────────────────────────────────
# mineru[all] includes: pipeline backend, VLM backend, gradio UI, and router
echo "[INFO] Installing MinerU[all] ..."
uv pip install -U "mineru[all]"

# ── 3. Download models ────────────────────────────────────────────────────────
echo "[INFO] Downloading MinerU models (pipeline + VLM) ..."
echo "       This is a one-time download (~5-10 GB). Be patient ..."

# Use ModelScope if HuggingFace is slow in your region:
# export MINERU_MODEL_SOURCE=modelscope

mineru-models-download

echo "[INFO] Models downloaded successfully."

# ── 4. Verify installation ────────────────────────────────────────────────────
echo "[INFO] Verifying mineru CLI ..."
mineru --help | head -5

echo ""
echo "======================================================"
echo "  Setup complete!  Run the converter with:"
echo ""
echo "  # Single PDF:"
echo "  python convert.py --input spec.pdf --output ./output"
echo ""
echo "  # Full directory:"
echo "  python convert.py --input ./pdfs/ --output ./output"
echo ""
echo "  # Best quality (hybrid VLM mode):"
echo "  python convert.py --input ./pdfs/ --output ./output --backend hybrid"
echo ""
echo "  # Multi-GPU (both 48GB cards):"
echo "  python convert.py --input ./pdfs/ --output ./output --multi-gpu --gpu-ids 0,1"
echo ""
echo "  # CPU only (no GPU needed):"
echo "  python convert.py --input ./pdfs/ --output ./output --cpu"
echo "======================================================"