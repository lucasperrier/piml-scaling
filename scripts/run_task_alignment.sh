#!/usr/bin/env bash
# =============================================================================
# run_task_alignment.sh — Master script for task-alignment experiments on RunPod
#
# Run Phases A→C on an RTX 3090 pod, ~25h GPU time, ~$5.50 total.
#
# Usage:
#   bash scripts/run_task_alignment.sh [phase]
#   phase: a | b | c | all (default: all)
# =============================================================================
set -euo pipefail

PHASE="${1:-all}"
SYSTEM="lotka-volterra"
CONFIG="configs/default.yaml"
DATA_DIR="data-trajectories/lotka-volterra"
OUT_DIR="runs-aligned"

echo "=========================================="
echo " Task-Alignment Experiments"
echo " System: $SYSTEM"
echo " Phase: $PHASE"
echo "=========================================="

# -------------------------------------------------------------------
# Step 0: Generate trajectory data (CPU, ~5 min)
# -------------------------------------------------------------------
if [[ ! -d "$DATA_DIR/data_seed=11" ]]; then
    echo ""
    echo ">>> Generating trajectory data..."
    python scripts/generate_trajectory_datasets.py \
        --config "$CONFIG" \
        --system "$SYSTEM" \
        --seeds 11,22 \
        --T 5.0 --dt-fine 0.001 \
        --out "$DATA_DIR"
    echo ">>> Trajectory data generated."
else
    echo ">>> Trajectory data already exists, skipping generation."
fi

# -------------------------------------------------------------------
# Phase A: Sanity Check (4 runs, ~8 min)
# -------------------------------------------------------------------
run_phase_a() {
    echo ""
    echo ">>> Phase A: Sanity Check (4 runs)"
    echo "    S1: onestep dt=0.05 plain medium D=2048"
    echo "    S2: onestep dt=0.05 midpoint medium D=2048"
    echo "    S3: flowmap T=1.0 plain medium D=2048"
    echo "    S4: flowmap T=1.0 midpoint medium D=2048"

    # S1 + S2: one-step at dt=0.05
    python scripts/run_sweep.py \
        --config "$CONFIG" --system "$SYSTEM" \
        --task onestep --dt 0.05 \
        --data-dir "$DATA_DIR" --out "$OUT_DIR" \
        --models plain,piml \
        --capacities medium \
        --dataset-sizes 2048 \
        --data-seeds 11 --train-seeds 101

    # S3 + S4: flow-map at T=1.0
    python scripts/run_sweep.py \
        --config "$CONFIG" --system "$SYSTEM" \
        --task flowmap \
        --data-dir data --out "$OUT_DIR" \
        --models plain,piml \
        --capacities medium \
        --dataset-sizes 2048 \
        --data-seeds 11 --train-seeds 101

    # Print results for decision
    echo ""
    echo ">>> Phase A Results:"
    echo "--- One-step (dt=0.05) ---"
    for m in plain piml; do
        f="$OUT_DIR/task=onestep/dt=0.05/model=$m/capacity=medium/D=2048/data_seed=11/train_seed=101/metrics.json"
        if [[ -f "$f" ]]; then
            echo "  $m: $(python -c "import json; d=json.load(open('$f')); print(f'test_rel_l2={d[\"test_rel_l2\"]:.6f}  status={d[\"status\"]}')")"
        fi
    done
    echo "--- Flow-map (T=1.0) ---"
    for m in plain piml; do
        f="$OUT_DIR/model=$m/capacity=medium/D=2048/data_seed=11/train_seed=101/metrics.json"
        if [[ -f "$f" ]]; then
            echo "  $m: $(python -c "import json; d=json.load(open('$f')); print(f'test_rel_l2={d[\"test_rel_l2\"]:.6f}  status={d[\"status\"]}')")"
        fi
    done

    echo ""
    echo ">>> Decision: If piml < plain on one-step AND piml > plain on flow-map → proceed to Phase B."
}

# -------------------------------------------------------------------
# Phase B: Lean Pilot (96 runs, ~3h)
# -------------------------------------------------------------------
run_phase_b() {
    echo ""
    echo ">>> Phase B: Lean Pilot (96 one-step + 48 flow-map runs)"

    for DT in 0.05 0.1; do
        echo "  Running one-step at dt=$DT..."
        python scripts/run_sweep.py \
            --config "$CONFIG" --system "$SYSTEM" \
            --task onestep --dt "$DT" \
            --data-dir "$DATA_DIR" --out "$OUT_DIR" \
            --models plain,piml \
            --capacities small,medium,large \
            --dataset-sizes 128,512,2048,8192 \
            --data-seeds 11 --train-seeds 101,202
    done

    echo "  Running flow-map baseline..."
    python scripts/run_sweep.py \
        --config "$CONFIG" --system "$SYSTEM" \
        --task flowmap \
        --data-dir data --out "$OUT_DIR" \
        --models plain,piml \
        --capacities small,medium,large \
        --dataset-sizes 128,512,2048,8192 \
        --data-seeds 11 --train-seeds 101,202

    echo ">>> Phase B complete. Aggregating..."
    python scripts/aggregate_runs.py --runs-root "$OUT_DIR"
}

# -------------------------------------------------------------------
# Phase C: Modest Confirmation (648 runs, ~22h)
# -------------------------------------------------------------------
run_phase_c() {
    echo ""
    echo ">>> Phase C: Modest Confirmation (648 runs)"

    for DT in 0.05 0.1 0.2; do
        echo "  Running one-step at dt=$DT..."
        python scripts/run_sweep.py \
            --config "$CONFIG" --system "$SYSTEM" \
            --task onestep --dt "$DT" \
            --data-dir "$DATA_DIR" --out "$OUT_DIR" \
            --models plain,piml,piml-trapezoidal \
            --capacities small,medium,large \
            --dataset-sizes 64,128,512,2048,4096,8192 \
            --data-seeds 11,22 --train-seeds 101,202
    done

    echo "  Running flow-map baseline for Phase C..."
    python scripts/run_sweep.py \
        --config "$CONFIG" --system "$SYSTEM" \
        --task flowmap \
        --data-dir data --out "$OUT_DIR" \
        --models plain,piml,piml-trapezoidal \
        --capacities small,medium,large \
        --dataset-sizes 64,128,512,2048,4096,8192 \
        --data-seeds 11,22 --train-seeds 101,202

    echo ">>> Phase C complete. Aggregating..."
    python scripts/aggregate_runs.py --runs-root "$OUT_DIR"
}

# -------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------
case "$PHASE" in
    a)   run_phase_a ;;
    b)   run_phase_a; run_phase_b ;;
    c)   run_phase_a; run_phase_b; run_phase_c ;;
    all) run_phase_a; run_phase_b; run_phase_c ;;
    *)   echo "Unknown phase: $PHASE (use a, b, c, or all)"; exit 1 ;;
esac

echo ""
echo "=========================================="
echo " All requested phases complete!"
echo "=========================================="
