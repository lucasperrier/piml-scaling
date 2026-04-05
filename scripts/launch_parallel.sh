#!/usr/bin/env bash
# Launch all 4 workers in parallel
set -u
cd /workspace/projects/piml-scaling
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)

nohup bash scripts/worker1_dense.sh        > "logs/worker1_${TS}.log" 2>&1 &
echo "W1 PID=$!"
nohup bash scripts/worker2_horizon.sh      > "logs/worker2_${TS}.log" 2>&1 &
echo "W2 PID=$!"
nohup bash scripts/worker3_duffing.sh      > "logs/worker3_${TS}.log" 2>&1 &
echo "W3 PID=$!"
nohup bash scripts/worker4_rescue_noise.sh > "logs/worker4_${TS}.log" 2>&1 &
echo "W4 PID=$!"

echo "All workers launched at $(date). Logs: logs/worker{1,2,3,4}_${TS}.log"
