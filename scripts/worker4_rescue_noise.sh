#!/usr/bin/env bash
# Parallel worker 4: Rescue + gradient dynamics + noise (Sections 24, 28, 27)
set -euo pipefail
cd /workspace/projects/piml-scaling
export PYTHONUNBUFFERED=1
PY=".venv/bin/python"
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

echo "=== WORKER 4: Rescue + Gradient + Noise ==="
echo "Started at $(date)"

# Section 24: Rescue study
$PY scripts/run_rescue_sweep.py \
  --config configs/default.yaml \
  --data-dir data \
  --out runs-rescue \
  --capacities large,xlarge \
  --dataset-sizes 256,512,1024 \
  --lambda-phys 0.01,0.1,1.0

# Section 28: Gradient dynamics
for model in plain piml piml-simpson piml-conservation; do
  for cap in large medium; do
    for D in 1024 512; do
      for tseed in 101 202 303; do
        echo "--- grad: $model $cap D=$D tseed=$tseed ---"
        $PY scripts/run_experiment.py \
          --config configs/default.yaml \
          --data-root data/data_seed=11 \
          --D $D --train-seed $tseed \
          --model $model --capacity $cap \
          --out runs-grad-dynamics \
          --log-grad-decomposition 2>&1 || true
      done
    done
  done
done

# Section 27: Noise robustness
for sigma in 0.0 0.01 0.05 0.1; do
  echo "--- noise sigma=$sigma ---"
  for model in plain piml piml-simpson; do
    $PY scripts/run_sweep.py --config configs/default.yaml \
      --data-dir data --out "runs-noise/obs/sigma=$sigma" \
      --models $model --obs-noise $sigma \
      --capacities tiny,small,medium,large,xlarge \
      --dataset-sizes 64,128,256,512,1024,2048,4096,8192
  done
done

# Section 27d: Prior mismatch
for delta_label in 0.05 0.10 0.20; do
  case $delta_label in
    0.05) pp="1.575,1.05,1.05,3.15" ;;
    0.10) pp="1.65,1.10,1.10,3.30" ;;
    0.20) pp="1.80,1.20,1.20,3.60" ;;
  esac
  echo "--- mismatch delta=$delta_label ---"
  for model in piml piml-simpson; do
    $PY scripts/run_sweep.py --config configs/default.yaml \
      --data-dir data --out "runs-noise/mismatch/delta=$delta_label" \
      --models $model --prior-params "$pp" \
      --capacities tiny,small,medium,large,xlarge \
      --dataset-sizes 64,128,256,512,1024,2048,4096,8192
  done
done

echo "=== WORKER 4 DONE at $(date) ==="
