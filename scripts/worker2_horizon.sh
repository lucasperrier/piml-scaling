#!/usr/bin/env bash
# Parallel worker 2: Horizon data gen + sweep (Section 22)
# Safe to run alongside the dense sweep since output dirs are separate.
set -euo pipefail
cd /workspace/projects/piml-scaling
export PYTHONUNBUFFERED=1
PY=".venv/bin/python"
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

echo "=== WORKER 2: Horizon sweep + diagnostics ==="
echo "Started at $(date)"

# Section 22: Horizon sweep
$PY scripts/run_horizon_sweep.py \
  --horizons 0.5 1.0 2.0 \
  --models plain piml piml-simpson \
  --data-base-dir data-horizon \
  --out-base-dir runs-horizon \
  --generate-data \
  --capacities tiny,small,medium,large,xlarge \
  --dataset-sizes 64,128,256,512,1024,2048,4096,8192

echo "=== WORKER 2 DONE at $(date) ==="
