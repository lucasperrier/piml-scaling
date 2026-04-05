#!/usr/bin/env bash
# Parallel worker 3: Duffing data gen + sweep (Section 26)
set -euo pipefail
cd /workspace/projects/piml-scaling
export PYTHONUNBUFFERED=1
PY=".venv/bin/python"
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

echo "=== WORKER 3: Duffing system ==="
echo "Started at $(date)"

# Section 26b: Duffing data generation
$PY scripts/generate_datasets.py \
  --config configs/duffing.yaml \
  --out data-duffing \
  --system duffing

# Section 26c: Duffing sweeps
for model in plain piml piml-simpson piml-conservation; do
  echo "--- Duffing sweep: $model ---"
  $PY scripts/run_sweep.py --config configs/duffing.yaml \
    --data-dir data-duffing --out runs-duffing --models $model \
    --system duffing \
    --capacities tiny,small,medium,large,xlarge \
    --dataset-sizes 64,128,256,512,1024,2048,4096,8192
done

echo "=== WORKER 3 DONE at $(date) ==="
