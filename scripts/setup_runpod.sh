#!/usr/bin/env bash
# setup_runpod.sh — One-shot environment setup for piml-scaling on a RunPod instance.
#
# Usage (from the repo root):
#   bash scripts/setup_runpod.sh
#
# What this script does:
#   1. Installs all Python runtime and dev dependencies.
#   2. Installs the scaling_piml package in editable mode.
#   3. Generates datasets for all three data seeds (seeds 11, 22, 33).
#   4. Runs unit tests to confirm the installation is healthy.
#   5. Runs a cheap end-to-end smoke test (configs/smoke.yaml, one run).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/5] Installing dependencies ==="
pip install --quiet -e ".[dev]" -r requirements-dev.txt

echo "=== [2/5] Package installed — checking import ==="
python -c "import scaling_piml; print('  scaling_piml OK')"

echo "=== [3/5] Generating datasets (data seeds 11, 22, 33) ==="
# generate_datasets.py iterates over all data_seeds listed in the config.
# Skip if all three seed directories already exist.
ALL_SEEDS_EXIST=true
for SEED in 11 22 33; do
    if [ ! -d "data/data_seed=${SEED}" ]; then
        ALL_SEEDS_EXIST=false
        break
    fi
done

if [ "$ALL_SEEDS_EXIST" = true ]; then
    echo "  All data seed directories already exist, skipping."
else
    echo "  Generating datasets for all data seeds ..."
    python scripts/generate_datasets.py \
        --config configs/default.yaml \
        --out data
fi

echo "=== [4/5] Running unit tests ==="
python -m pytest -q

echo "=== [5/5] Smoke test (single run, 5 epochs) ==="
SMOKE_OUT="data-smoke"
python scripts/generate_datasets.py \
    --config configs/smoke.yaml \
    --out "$SMOKE_OUT"

python scripts/run_experiment.py \
    --config configs/smoke.yaml \
    --data-root "${SMOKE_OUT}/data_seed=11" \
    --D 64 \
    --train-seed 101 \
    --model plain

echo ""
echo "=== Setup complete ==="
echo "Environment is healthy. To run a full sweep:"
echo "  python scripts/run_sweep.py --config configs/default.yaml \\"
echo "      --data-dir data --models plain,piml --out runs-progress"
