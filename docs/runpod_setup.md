# Runpod Training Setup

## 1. SSH config

Add to `~/.ssh/config` (update HostName and Port each time you spin up a new pod):

```
Host runpod
  HostName <POD_IP>
  User root
  Port <POD_PORT>
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 120

Host runpod-piml
  HostName <POD_IP>
  User root
  Port <POD_PORT>
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  RequestTTY yes
  RemoteCommand cd /workspace/projects/piml-scaling && exec bash -l
  ServerAliveInterval 30
  ServerAliveCountMax 120
```

Get `<POD_IP>` and `<POD_PORT>` from the Runpod dashboard under **SSH over exposed TCP**.

Test with:

```bash
ssh runpod 'echo OK && hostname'
```

## 2. First-time pod setup

```bash
ssh runpod

# Clone repo
cd /workspace/projects
git clone https://github.com/lucasperrier/piml-scaling.git
cd piml-scaling

# Create venv and install
python -m venv .venv
.venv/bin/python -m pip install -e .

# Install GPU-compatible torch (match pod driver version)
# Check driver: nvidia-smi | head -3
# For CUDA 12.4 driver:
.venv/bin/python -m pip install --force-reinstall "torch==2.4.1+cu124" \
  --index-url https://download.pytorch.org/whl/cu124

# Verify GPU
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Sync data to pod

The `data/` directory is gitignored. Sync it from local:

```bash
# From local machine
tar czf - -C /path/to/scaling-piml data | \
  ssh runpod 'cd /workspace/projects/piml-scaling && tar xzf - --no-same-owner'
```

Verify:

```bash
ssh runpod 'ls /workspace/projects/piml-scaling/data/'
# Should show: data_seed=11  data_seed=22  data_seed=33
```

If `src/scaling_piml/data/` is missing from the remote checkout (not tracked in git), sync it too:

```bash
tar czf - -C /path/to/scaling-piml src/scaling_piml/data | \
  ssh runpod 'cd /workspace/projects/piml-scaling && tar xzf - --no-same-owner'
```

## 4. Pulling latest code

```bash
ssh runpod 'cd /workspace/projects/piml-scaling && git pull origin main'
```

If you changed dependencies locally, also re-run:

```bash
ssh runpod 'cd /workspace/projects/piml-scaling && .venv/bin/python -m pip install -e .'
```

## 5. Launching training runs

### Interactive (stays alive only while connected)

```bash
ssh runpod-piml
.venv/bin/python scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --out runs-dense --models plain
```

### Detached (survives disconnect)

```bash
ssh runpod 'cd /workspace/projects/piml-scaling && mkdir -p logs && \
  nohup bash -lc '"'"'\
    cd /workspace/projects/piml-scaling && \
    export PYTHONUNBUFFERED=1 && \
    .venv/bin/python scripts/run_sweep.py --config configs/default.yaml \
      --data-dir data --out runs-dense --models plain && \
    .venv/bin/python scripts/run_sweep.py --config configs/default.yaml \
      --data-dir data --out runs-dense --models piml && \
    .venv/bin/python scripts/run_sweep.py --config configs/default.yaml \
      --data-dir data --out runs-dense --models piml-simpson \
  '"'"' > logs/sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 & \
  echo "pid=$! started"'
```

The sweep script skips existing `metrics.json` files, so it is safe to restart after a crash.

## 6. Monitoring

### Follow live log

```bash
ssh runpod 'tail -f /workspace/projects/piml-scaling/logs/<LOG_FILE>'
```

### Check progress counts

```bash
ssh runpod 'cd /workspace/projects/piml-scaling && \
  for m in plain piml piml-simpson; do \
    c=$(find runs-dense/model=$m -name metrics.json 2>/dev/null | wc -l); \
    echo "$m: $c completed"; \
  done'
```

### Check GPU usage

```bash
ssh runpod 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader'
```

### Check if sweep process is alive

```bash
ssh runpod "ps -ef | grep 'run_sweep.py' | grep -v grep"
```

## 7. Pulling results back to local

```bash
# Sync finished runs
ssh runpod 'cd /workspace/projects/piml-scaling && tar czf - runs-dense' | \
  tar xzf - -C /path/to/scaling-piml

# Or just the aggregated CSV
scp runpod:/workspace/projects/piml-scaling/runs-dense/grouped_metrics.csv \
  /path/to/scaling-piml/runs-dense/
```

## 8. VS Code Remote-SSH

For full IDE experience on the pod:

1. Install the **Remote - SSH** extension in VS Code.
2. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → select `runpod`.
3. Open folder `/workspace/projects/piml-scaling`.
4. Select Python interpreter: `.venv/bin/python`.

Copilot and all extensions work as if local.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: scaling_piml` | Run `.venv/bin/python -m pip install -e .` on the pod |
| `torch.cuda.is_available() == False` | Install torch matching driver CUDA version (see step 2) |
| `NVIDIA driver too old` | Downgrade torch: `pip install torch==2.4.1+cu124 --index-url ...` |
| SSH hangs / drops | Check pod is running in dashboard; update IP/port in `~/.ssh/config` |
| `rsync: command not found` | Use `tar \| ssh` method instead (rsync not installed on default pods) |
| Sweep died mid-run | Just relaunch — it skips completed runs automatically |

## 10. Pod lifecycle notes

- **IP and port change** every time a pod is restarted. Update `~/.ssh/config` accordingly.
- Data on `/workspace` persists across pod restarts (network volume). Data outside `/workspace` is ephemeral.
- Always run long jobs with `nohup` or in a `screen`/`tmux` session (install with `apt-get install -y tmux` if needed).
- Stop the pod in the dashboard when not training to save credits.
