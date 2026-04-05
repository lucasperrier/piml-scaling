#!/usr/bin/env bash
# Worker 1: Dense sweep (Sections 21b + 25) - plain, piml, piml-simpson, piml-simpson-true
set -euo pipefail
cd /workspace/projects/piml-scaling
export PYTHONUNBUFFERED=1
PY=".venv/bin/python"
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

echo "=== WORKER 1: Dense sweep ==="
echo "Started at $(date)"

$PY scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models plain

$PY scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models piml

$PY scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models piml-simpson

# Section 25: Simpson's-rule prior
echo "--- Section 25a: Diagnose physics (Simpson GT residual) ---"
$PY scripts/diagnose_physics.py --data-root data/data_seed=11

$PY scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models piml-simpson-true

echo "=== WORKER 1 DONE at $(date) ==="
