#!/usr/bin/env bash
# Master script for all GPU-dependent Phase 2 work.
# Run with nohup on the Runpod pod.
# Skips already-completed runs automatically (metrics.json exists).
set -euo pipefail

cd /workspace/projects/piml-scaling
export PYTHONUNBUFFERED=1
PY=".venv/bin/python"

# Raise file descriptor limit to avoid "Too many open files"
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

log() { echo ""; echo "========================================"; echo "$(date '+%Y-%m-%d %H:%M:%S') $1"; echo "========================================"; }

# ============================================================
# SECTION 21b: Dense Lotka-Volterra sweep (plain, piml, piml-simpson)
# ~2,079 runs total. Resumes from where it left off.
# ============================================================
log "SECTION 21b: Dense sweep - plain (430 remaining)"
$PY scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models plain

log "SECTION 21b: Dense sweep - piml/midpoint (601 remaining)"
$PY scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models piml

log "SECTION 21b: Dense sweep - piml-simpson/composite (616 remaining)"
$PY scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models piml-simpson

# ============================================================
# SECTION 25: Simpson's-rule prior (4th-order)
# First verify ground-truth residual, then run dense sweep.
# ============================================================
log "SECTION 25a: Verify Simpson ground-truth residual"
$PY scripts/diagnose_physics.py --data-root data/data_seed=11 2>&1 | tee logs/diagnose_physics_simpson.log

log "SECTION 25b: Dense sweep - piml-simpson-true (693 runs)"
$PY scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models piml-simpson-true

# ============================================================
# SECTION 22: Horizon sweep (T=0.5, 1.0, 2.0)
# Generates data at T=0.5 and T=2.0, then runs 3 models x 3 horizons.
# Uses original grid: 5 capacities x 8 dataset sizes x 3 data seeds x 3 train seeds = 360/model/horizon
# ============================================================
log "SECTION 22: Horizon sweep - data generation + runs"
$PY scripts/run_horizon_sweep.py \
  --horizons 0.5 1.0 2.0 \
  --models plain piml piml-simpson \
  --data-base-dir data-horizon \
  --out-base-dir runs-horizon \
  --generate-data \
  --capacities tiny,small,medium,large,xlarge \
  --dataset-sizes 64,128,256,512,1024,2048,4096,8192

log "SECTION 22: Horizon - diagnose physics at each T"
for T in 0.5 1.0 2.0; do
  echo "--- T=$T ---"
  $PY scripts/diagnose_physics.py --data-root "data-horizon/T=$T/data_seed=11" 2>&1 || true
done | tee logs/diagnose_physics_horizons.log

# ============================================================
# SECTION 24: Rescue study for conservation prior
# 5 variants x 3 lambda x 2 capacities x 3 dataset sizes x 3 data seeds x 3 train seeds = 810 runs
# ============================================================
log "SECTION 24: Rescue study for conservation prior"
$PY scripts/run_rescue_sweep.py \
  --config configs/default.yaml \
  --data-dir data \
  --out runs-rescue \
  --capacities large,xlarge \
  --dataset-sizes 256,512,1024 \
  --lambda-phys 0.01,0.1,1.0

# ============================================================
# SECTION 28: Gradient dynamics visualization
# 4 models x 2 configs x 3 train seeds = 24 runs with --log-grad-decomposition
# ============================================================
log "SECTION 28: Gradient dynamics diagnostic runs"
for model in plain piml piml-simpson piml-conservation; do
  for cap in large medium; do
    for D in 1024 512; do
      for dseed in 11; do
        for tseed in 101 202 303; do
          echo "--- $model $cap D=$D dseed=$dseed tseed=$tseed ---"
          $PY scripts/run_experiment.py \
            --config configs/default.yaml \
            --data-root "data/data_seed=$dseed" \
            --D $D --train-seed $tseed \
            --model $model --capacity $cap \
            --out runs-grad-dynamics \
            --log-grad-decomposition 2>&1 || true
        done
      done
    done
  done
done

# ============================================================
# SECTION 26: Duffing oscillator replication
# Generate data, then 4 models × 5 capacities × 8 dataset sizes × 3 data seeds × 3 train seeds = 1440 runs
# ============================================================
log "SECTION 26b: Duffing data generation"
$PY scripts/generate_datasets.py \
  --config configs/duffing.yaml \
  --out data-duffing \
  --system duffing

log "SECTION 26b: Duffing - diagnose physics"
$PY scripts/diagnose_physics.py --data-root data-duffing/data_seed=11 --config configs/duffing.yaml 2>&1 | tee logs/diagnose_physics_duffing.log || true

log "SECTION 26c: Duffing sweep - plain"
$PY scripts/run_sweep.py --config configs/duffing.yaml \
  --data-dir data-duffing --out runs-duffing --models plain \
  --system duffing \
  --capacities tiny,small,medium,large,xlarge \
  --dataset-sizes 64,128,256,512,1024,2048,4096,8192

log "SECTION 26c: Duffing sweep - piml/midpoint"
$PY scripts/run_sweep.py --config configs/duffing.yaml \
  --data-dir data-duffing --out runs-duffing --models piml \
  --system duffing \
  --capacities tiny,small,medium,large,xlarge \
  --dataset-sizes 64,128,256,512,1024,2048,4096,8192

log "SECTION 26c: Duffing sweep - piml-simpson/composite"
$PY scripts/run_sweep.py --config configs/duffing.yaml \
  --data-dir data-duffing --out runs-duffing --models piml-simpson \
  --system duffing \
  --capacities tiny,small,medium,large,xlarge \
  --dataset-sizes 64,128,256,512,1024,2048,4096,8192

log "SECTION 26c: Duffing sweep - piml-conservation"
$PY scripts/run_sweep.py --config configs/duffing.yaml \
  --data-dir data-duffing --out runs-duffing --models piml-conservation \
  --system duffing \
  --capacities tiny,small,medium,large,xlarge \
  --dataset-sizes 64,128,256,512,1024,2048,4096,8192

# ============================================================
# SECTION 27: Noise robustness
# Observation noise: 3 models x 4 noise levels x 360 = 4320 runs
# Prior mismatch: 2 models x 3 levels x 360 = 2160 runs
# ============================================================
log "SECTION 27c: Noise robustness - observation noise"
for sigma in 0.0 0.01 0.05 0.1; do
  log "  Obs noise sigma=$sigma"
  for model in plain piml piml-simpson; do
    $PY scripts/run_sweep.py --config configs/default.yaml \
      --data-dir data --out "runs-noise/obs/sigma=$sigma" \
      --models $model \
      --obs-noise $sigma \
      --capacities tiny,small,medium,large,xlarge \
      --dataset-sizes 64,128,256,512,1024,2048,4096,8192
  done
done

log "SECTION 27d: Noise robustness - prior mismatch"
# Perturb LV params: true=(1.5, 1.0, 1.0, 3.0)
# +5%: (1.575, 1.05, 1.05, 3.15)
# +10%: (1.65, 1.10, 1.10, 3.30)
# +20%: (1.80, 1.20, 1.20, 3.60)
for delta_label in 0.05 0.10 0.20; do
  case $delta_label in
    0.05) pp="1.575,1.05,1.05,3.15" ;;
    0.10) pp="1.65,1.10,1.10,3.30" ;;
    0.20) pp="1.80,1.20,1.20,3.60" ;;
  esac
  log "  Prior mismatch delta=$delta_label params=$pp"
  for model in piml piml-simpson; do
    $PY scripts/run_sweep.py --config configs/default.yaml \
      --data-dir data --out "runs-noise/mismatch/delta=$delta_label" \
      --models $model \
      --prior-params "$pp" \
      --capacities tiny,small,medium,large,xlarge \
      --dataset-sizes 64,128,256,512,1024,2048,4096,8192
  done
done

log "ALL GPU WORK COMPLETE"
echo "Finished at $(date)"
