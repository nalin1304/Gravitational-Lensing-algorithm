#!/bin/bash
# ============================================================================
# reproduce.sh — One-command reproducibility verification
# ============================================================================
#
# Builds the Docker container, runs the full test suite, executes
# benchmark scripts, and outputs a verification hash.
#
# Usage:
#   chmod +x scripts/reproduce.sh
#   ./scripts/reproduce.sh
#
# Requirements:
#   - Docker (or uv/pip for local execution)
#   - ~5 minutes for full run
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"

echo "============================================================================"
echo "  REPRODUCIBILITY VERIFICATION"
echo "  Project: Gravitational Lensing Toolkit"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================================"

mkdir -p "$RESULTS_DIR"

# Python runner: use uv if available, else fall back to python3
if command -v uv &>/dev/null; then
    PY="uv run python"
else
    PY="python3"
fi

# ---- Step 1: Run test suite ----
echo ""
echo "▶ Step 1/8: Running test suite..."
cd "$PROJECT_DIR"

$PY -m pytest tests/ -q 2>&1 | tee "$RESULTS_DIR/test_output.txt"

# ---- Step 2: Run ablation study ----
echo ""
echo "▶ Step 2/8: Running ablation study..."
$PY scripts/ablation_study.py --grid 64 --n-trials 3 --n-calibration 6 --systems-per-trial 8 --outdir "$RESULTS_DIR" 2>&1 \
    | tee -a "$RESULTS_DIR/ablation_output.txt"

# ---- Step 3: Run real data validation ----
echo ""
echo "▶ Step 3/8: Running real data validation..."
$PY scripts/validate_real_data.py --grid 64 --use-real --strict-observational --outdir "$RESULTS_DIR/real_data" 2>&1 \
    | tee -a "$RESULTS_DIR/real_data_output.txt"

# ---- Step 4: Run SOTA comparison ----
echo ""
echo "▶ Step 4/8: Running SOTA comparison..."
$PY scripts/sota_comparison.py --grid 64 --n-lenses 10 --n-calibration 6 --outdir "$RESULTS_DIR" 2>&1 \
    | tee -a "$RESULTS_DIR/sota_output.txt"

# ---- Step 5: Run scalability benchmark ----
echo ""
echo "▶ Step 5/8: Running scalability benchmark..."
$PY scripts/scalability_benchmark.py --outdir "$RESULTS_DIR" 2>&1 \
    | tee -a "$RESULTS_DIR/scalability_output.txt"

# ---- Step 6: Run uncertainty calibration ----
echo ""
echo "▶ Step 6/8: Running uncertainty calibration..."
echo "  Training/loading checkpoint-backed Bayesian surrogate for held-out synthetic NFW analog calibration."
$PY scripts/uncertainty_calibration.py --grid 64 --n-samples 30 --seed 21 --dropout-rate 0.04 --outdir "$RESULTS_DIR" --model "$PROJECT_DIR/models/bayesian_uq_synthetic.pt" 2>&1 \
    | tee -a "$RESULTS_DIR/calibration_output.txt"

# ---- Step 7: Run Pareto benchmark ----
echo ""
echo "▶ Step 7/8: Running Pareto benchmark..."
$PY scripts/pareto_benchmark.py --outdir "$RESULTS_DIR" 2>&1 \
    | tee -a "$RESULTS_DIR/pareto_output.txt"

# ---- Step 8: Run multi-messenger demo ----
echo ""
echo "▶ Step 8/8: Running multi-messenger demo..."
$PY scripts/multi_messenger_demo.py --outdir "$RESULTS_DIR" 2>&1 \
    | tee -a "$RESULTS_DIR/multi_messenger_output.txt"

# ---- Verification hash ----
echo ""
echo "============================================================================"
echo "  VERIFICATION HASH"
echo "============================================================================"

HASH=$(find "$RESULTS_DIR" -name "*.json" | sort | xargs cat 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
echo "  SHA-256: $HASH"
echo "  Results: $RESULTS_DIR/"
echo ""

# List generated outputs
echo "  Generated files:"
find "$RESULTS_DIR" -type f -name "*.tex" -o -name "*.json" -o -name "*.png" | sort | \
    while read f; do
        size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo "?")
        echo "    $(basename "$f") (${size} bytes)"
    done

echo ""
echo "✓ Reproducibility verification complete."
echo "============================================================================"
