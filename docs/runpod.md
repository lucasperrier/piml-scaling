# Running piml-scaling on RunPod

This guide explains how to set up a [RunPod](https://www.runpod.io/) instance, connect via SSH, and run experiments from this repository.

## 1. Create a RunPod instance

1. Log in to [runpod.io](https://www.runpod.io/) and click **Deploy**.
2. Choose a GPU pod. The scaling sweeps are CPU-light but benefit from a GPU for PyTorch training. A single A100/RTX 4090 is more than sufficient.
3. Select a PyTorch template (e.g. `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`) so that CUDA and a compatible Python are pre-installed.
4. Under **Advanced**, add your **SSH public key** in the *Public Key* field. This lets you connect with key-based authentication.
5. Click **Deploy On-Demand** (or Spot for cheaper runs).

## 2. Connect via SSH

Once the pod is running, click **Connect** in the RunPod console to get the SSH command. It will look like:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

Replace `~/.ssh/id_rsa` with the path to the private key that matches the public key you uploaded.

You can also add a convenient alias to `~/.ssh/config` on your local machine:

```
Host runpod-piml
    HostName <pod-ip>
    Port     <port>
    User     root
    IdentityFile ~/.ssh/id_rsa
```

Then connect with:

```bash
ssh runpod-piml
```

## 3. Set up the environment on the pod

Clone the repository and run the setup script in one step:

```bash
git clone https://github.com/<your-github-username>/piml-scaling.git
cd piml-scaling
bash scripts/setup_runpod.sh
```

The script installs all Python dependencies into the current environment, generates the datasets, and runs a quick smoke test to confirm everything works end-to-end.

## 4. Run a single experiment

```bash
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --data-root data/data_seed=11 \
    --D 256 \
    --train-seed 101 \
    --model plain \
    --capacity small
```

## 5. Run the full scaling sweep

```bash
python scripts/run_sweep.py \
    --config configs/default.yaml \
    --data-dir data \
    --models plain,piml \
    --out runs-progress
```

Large sweeps (720+ runs) can be left running in a persistent terminal session using `tmux` or `screen`:

```bash
tmux new -s sweep
python scripts/run_sweep.py --config configs/default.yaml \
    --data-dir data --models plain,piml --out runs-progress
# Detach with Ctrl-B D; reattach with: tmux attach -t sweep
```

## 6. Retrieve results

Copy results back to your local machine with `scp` or `rsync`:

```bash
rsync -avz runpod-piml:/root/piml-scaling/runs-progress/ ./runs-progress/
```

## 7. Persistent storage

RunPod pod storage is ephemeral by default. To keep data across pod restarts:

- Use a **Network Volume** (attach it at pod creation, typically mounted at `/workspace`).
- Clone the repo and store datasets/results under `/workspace/piml-scaling/`.
- Update paths accordingly, e.g. `--data-root /workspace/piml-scaling/data/data_seed=11`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ssh: connect to host … port … Connection refused` | Pod is still starting; wait 30–60 s then retry. |
| `Permission denied (publickey)` | Confirm the *public* key (`.pub` file) was pasted into the RunPod console. |
| `ModuleNotFoundError: No module named 'scaling_piml'` | Re-run `pip install -e .` from the repo root. |
| CUDA out of memory | Reduce `batch_size_cap` in the config or switch to a larger GPU. |
| Sweep killed mid-run | Run inside `tmux` so the process survives SSH disconnects. |
