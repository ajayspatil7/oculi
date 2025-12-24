#!/bin/bash
# run_mistral_experiment.sh
# =========================
# Automates the Spectra Phase1-M experiment on Mistral-7B-v0.1

set -e  # Exit on error

echo "========================================================"
echo "    Spectra Phase1-M: Mistral-7B-v0.1 Experiment"
echo "========================================================"

# 1. Install dependencies (ensure environment is ready)
echo "[1/3] Checking dependencies..."
# Fix GLIBCXX error (point to conda's newer libstdc++)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib

# Fix broken torchvision (RuntimeError: operator torchvision::nms does not exist)
echo "[1/3] Fixing environment (forcing reinstall of torchvision)..."
pip install --force-reinstall torchvision

pip install -r requirements.txt > /dev/null

# 2. Prepare Mistral Data (using LLaMA sources for exact replication)
# This script reads samples.json from LLaMA run and re-tokenizes the same text
echo "[2/3] Generating Mistral-7B data (replicating LLaMA sample text)..."
python scripts/prepare_mistral_data.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --lengths "128,512,1024,2048" \
    --llama-data-dir "data" \
    --output-dir "data/mistral"

# 3. Run Pipeline 
# Note: config/experiment.yaml is already configured for Mistral and data/mistral
echo "[3/3] Running Experiment Pipeline..."
python run_pipeline.py

echo "========================================================"
echo "✅ Experiment Complete!"
echo "Results saved in: results/Mistral-7B-v0.1/"
echo "========================================================"
