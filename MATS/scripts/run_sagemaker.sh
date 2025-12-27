#!/bin/bash
# MATS 10.0 SageMaker Execution Script
# =====================================

set -e

echo "======================================"
echo "MATS 10.0: Sycophancy Entropy Control"
echo "======================================"

# Navigate to MATS directory
cd /home/ec2-user/SageMaker/Spectra/MATS

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Verify installation
echo "Verifying installation..."
python -c "from transformer_lens import HookedTransformer; print('✓ TransformerLens OK')"
python -c "from datasets import load_dataset; print('✓ Datasets OK')"

# Run pipeline
echo ""
echo "Starting pipeline..."
echo ""

if [ "$1" == "--phase" ]; then
    python run_pipeline.py --phase $2
elif [ "$1" == "--dry-run" ]; then
    python run_pipeline.py --dry-run
else
    python run_pipeline.py
fi

echo ""
echo "Pipeline complete!"
echo "Results saved to: results/"
