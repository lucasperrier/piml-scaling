# Empirical Scaling Laws for Physics-Informed Priors in SciML

This repository implements a minimal, reproducible experiment for a first foundational paper in scientific machine learning.

## Core question

> Does adding a simple physics prior change empirical scaling behavior with respect to model size and dataset size in a controlled SciML setting?

The project is intentionally narrow. It is designed to produce one clean paper-quality result, not a broad SciML benchmark suite.

## Scope

### Included in v1
- System: Lotka–Volterra ODE
- Task: fixed-horizon flow-map prediction
- Models: plain MLP and physics-regularized MLP
- Scaling axes: model capacity `N` and dataset size `D`
- Primary metric: relative L2 error at prediction horizon `T`

### Excluded from v1
- PDEs
- DeepONet, FNO, Neural ODE training
- autoregressive rollouts
- chaotic systems
- real-world data
- distributed training
- hyperparameter search

## Task

Lotka–Volterra dynamics:

\[
\dot{x} = \alpha x - \beta x y,\qquad
\dot{y} = \delta x y - \gamma y
\]

State:
\[
u(t) = [x(t), y(t)] \in \mathbb{R}^2
\]

Learning problem:
\[
f_\theta(u_0) \mapsto u(T)
\]

This is a fixed-horizon flow-map prediction task.

## Main scaling relation

The central analysis object is:

\[
E(N, D) \approx E_\infty + aN^{-\alpha} + bD^{-\beta}
\]

where:
- `E(N, D)` = test error
- `N` = parameter count
- `D` = training set size
- `E_inf` = irreducible error floor
- `alpha` = capacity scaling exponent
- `beta` = data scaling exponent

## Default experiment settings

### Lotka–Volterra parameters
- `alpha = 1.5`
- `beta = 1.0`
- `delta = 1.0`
- `gamma = 3.0`

### Prediction horizon
- `T = 1.0`

### Initial conditions
- `x0 ~ Uniform(0.5, 2.5)`
- `y0 ~ Uniform(0.5, 2.5)`

### ODE solver
- `solve_ivp`
- method: `DOP853` preferred
- `rtol = 1e-9`
- `atol = 1e-11`

## Dataset protocol
### Per data seed
- train pool: `20000`
- validation: `2000`
- test: `2000`
### Data seeds
- `11`
- `22`
- `33`
### Dataset sizes for scaling

```text
D ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192}
```

Constraints:
- Training subsets must be **nested slices** of the same shuffled master train pool.

Normalization:
- Compute input and target mean/std from the **full train pool only**.
- Reuse the same normalization for all subset sizes within a data seed.

## Models

### Plain MLP
- Input: normalized $u_0$
- Output: normalized $u(T)$
- Activation: ReLU or GELU
- Use one consistent architecture family across all runs

### Physics-regularized MLP

Same architecture as the plain MLP. Only the loss changes.
Loss terms (with a single physics weight $\lambda$):

- Data loss:
  \[
  \mathcal{L}_{\text{data}} = \lVert \hat{u}(T) - u(T) \rVert_2^2
  \]

- Physics loss (implicit midpoint residual using the vector field $F$):
  \[
  \mathcal{L}_{\text{phys}} = \left\lVert \hat{u}(T) - u_0 - T\,F\left(\tfrac{\hat{u}(T) + u_0}{2}\right) \right\rVert_2^2
  \]

- Total loss:
  \[
  \mathcal{L} = \mathcal{L}_{\text{data}} + \lambda\,\mathcal{L}_{\text{phys}}
  \]

Default:
- `lambda_phys = 0.1`

## Capacity grid

| name   | hidden widths |
|--------|--------------|
| tiny   | `[32, 32]` |
| small  | `[64, 64]` |
| medium | `[128, 128]` |
| large  | `[256, 256]` |
| xlarge | `[256, 256, 256]` |

## Training protocol

- Optimizer: Adam
- Learning rate: `1e-3`
- Weight decay: `1e-6`
- Batch size: `min(128, D)`
- Max epochs: `400`
- Early stopping patience: `40`
- Monitor: validation relative L2

### Training seeds

- `101`
- `202`
- `303`

## Experiment matrix

Full matrix: `2` models × `5` capacities × `8` dataset sizes × `3` data seeds × `3` train seeds

$$2 \times 5 \times 8 \times 3 \times 3 = 720\text{ runs}$$

Reduced acceptable matrix: `2` models × `4` capacities × `6` dataset sizes × `2` data seeds × `3` train seeds

$$2 \times 4 \times 6 \times 2 \times 3 = 288\text{ runs}$$

## Metrics

```text
model_name
is_physics_informed
parameter_count
dataset_size
data_seed
train_seed
best_epoch
train_rel_l2
val_rel_l2
test_rel_l2
train_mse
val_mse
test_mse
runtime_seconds
diverged
nan_detected
```

\[
E = \frac{\lVert \hat{u}(T) - u(T) \rVert_2}{\lVert u(T) \rVert_2 + \varepsilon}
\]

## Analysis requirements

Aggregate over data seeds and train seeds for each (model, N, D).
- Capacity scaling: error vs $N$ at fixed $D$
- Data scaling: error vs $D$ at fixed $N$
- Full surface fit: $E(N, D)$
- nonlinear least squares
- bootstrap confidence intervals for exponents

## Required outputs

### Figures
- Task setup schematic
- Capacity scaling plot
- Data scaling plot
- 2D error heatmap / contour
- Exponent comparison with confidence intervals
- Stability summary

### Per-run artifacts
- Frozen config
- Metrics JSON
- Training history CSV
- Optional saved predictions and checkpoint

## Experiment contract

The project should be run under a strict artifact contract. Every stage must produce the same minimal evidence so runs are reproducible, comparable, and easy to aggregate.

### Run-level contract

Each training run is identified by:
- `model` in `{plain, piml}`
- `capacity_name` such as `tiny`, `small`, `medium`, `large`, `xlarge`
- exact `hidden_widths`
- `dataset_size`
- `data_seed`
- `train_seed`

Each run directory must contain:
- frozen `config.yaml`
- `history.csv`
- `metrics.json`
- `best.pt` checkpoint if training produced at least one valid best epoch

Each run must be assigned a status:
- `success`: completed, finite metrics, eligible for aggregation
- `diverged`: optimization or evaluation diverged
- `nan`: NaN or Inf detected in loss or metrics
- `failed`: run crashed before producing valid final metrics

Each run must also record enough metadata to reproduce and diagnose the result:
- model identifiers and parameter count
- dataset identifiers and seeds
- wall-clock runtime
- best epoch
- divergence / NaN flags
- artifact paths or deterministic artifact locations

### Sweep-level contract

One canonical aggregate table should be produced with one row per attempted run. This table is the source of truth for downstream filtering and grouped summaries.

The aggregate table must include:
- all run identifiers
- final metrics
- run status
- divergence / NaN indicators
- parameter count
- runtime
- artifact paths
- an `eligible_for_fit` flag

Grouped summaries for each `(model, parameter_count, dataset_size)` should then compute:
- number of completed runs
- mean / std / stderr for each metric of interest
- divergence rate
- NaN rate

### Promotion gates

Do not move to the next phase until the current phase passes its gate.

1. Smoke gate: scripts run end-to-end and write all required artifacts.
2. Sanity gate: a single run converges, metrics are finite, and checkpoint reload works.
3. Pilot gate: trends versus `D` and `N` are directionally sensible and divergence is acceptable.
4. Full-sweep gate: the pilot sweep is stable enough that the full matrix is worth running.
5. Analysis gate: aggregate tables, fits, confidence intervals, and figures regenerate from scripts only.

### Required manual inspection

Before a full sweep, inspect at least the following:
- training and validation convergence curves from `history.csv`
- relative scale of physics loss and data loss
- train / val / test metrics from `metrics.json`
- best-epoch distribution and early stopping behavior
- seed variance across repeated runs
- divergence and NaN rates by model family

## Artifact schemas

The schemas below define the expected contract for the current implementation and the next missing analysis layer.

### `metrics.json`

One JSON object per run with the following fields:

```text
model_name              str
is_physics_informed     bool
capacity_name           str
hidden_widths           list[int]
parameter_count         int
dataset_size            int
data_seed               int
train_seed              int
status                  str     # success | diverged | nan | failed
failure_reason          str     # empty on success
best_epoch              int
train_rel_l2            float
val_rel_l2              float
test_rel_l2             float
train_mse               float
val_mse                 float
test_mse                float
runtime_seconds         float
diverged                bool
nan_detected            bool
eligible_for_fit        bool
data_root               str
run_dir                 str
config_path             str
history_path            str
checkpoint_path         str
metrics_path            str
device                  str
```

Not yet implemented:

```text
git_commit              str
```

### `history.csv`

One row per epoch with the following columns:

```text
epoch
train_loss
train_data_loss
train_phys_loss
val_rel_l2
val_mse
```

Recommended additions for better diagnostics:

```text
lr
epoch_seconds
train_rel_l2
grad_norm
checkpoint_saved
```

### Aggregate results CSV

The canonical aggregate table should contain one row per attempted run. Suggested filename:

```text
runs_aggregate.csv
```

Required columns:

```text
model_name
is_physics_informed
capacity_name
hidden_widths
parameter_count
dataset_size
data_seed
train_seed
status
failure_reason
best_epoch
train_rel_l2
val_rel_l2
test_rel_l2
train_mse
val_mse
test_mse
runtime_seconds
diverged
nan_detected
eligible_for_fit
data_root
run_dir
config_path
history_path
checkpoint_path
metrics_path
```

This table should be created before any grouped scaling analysis. It is the contract between execution and analysis.

### Grouped summary CSV

After the per-run aggregate table exists, grouped summaries should be generated over `(model_name, parameter_count, dataset_size)` with a suggested filename:

```text
grouped_metrics.csv
```

Suggested columns:

```text
model_name
is_physics_informed
capacity_name
hidden_widths
parameter_count
dataset_size
n_attempted
n_runs
divergence_rate
nan_rate
test_rel_l2_mean
test_rel_l2_std
test_rel_l2_stderr
val_rel_l2_mean
val_rel_l2_std
val_rel_l2_stderr
test_mse_mean
test_mse_std
test_mse_stderr
runtime_seconds_mean
runtime_seconds_std
runtime_seconds_stderr
```

This grouped table is the contract used by the scaling-law fitting and plotting scripts.

## Build order

1. Implement Lotka–Volterra system and data generation
2. Freeze datasets for each data seed
3. Implement MLP
4. Implement physics loss
5. Implement training + evaluation
6. Run sanity checks
7. Run pilot sweep
8. Run full sweep
9. Aggregate results
10. Fit scaling laws
11. Generate figures

## Verify the repository

Create or activate a Python environment, then install the package in editable mode so the scripts in `scripts/` can import `scaling_piml` correctly:

```bash
python -m pip install -e . -r requirements-dev.txt
```

Run the unit tests:

```bash
python -m pytest -q
```

Run lint:

```bash
python -m ruff check .
```

Run a cheap end-to-end smoke test:

```bash
python scripts/generate_datasets.py --config configs/smoke.yaml --out data-smoke
python scripts/run_experiment.py --config configs/smoke.yaml --data-root data-smoke/data_seed=11 --D 64 --train-seed 101 --model plain
```

Launch the live dashboard (auto-refreshing):

```bash
streamlit run dashboard/app.py
```

The smoke config in `configs/smoke.yaml` uses a single small dataset and five training epochs so you can confirm data generation, loading, training, checkpointing, and metrics writing without starting a full sweep.

## Success criterion
- clean scaling curves,
- fitted exponents with uncertainty,
- a clear statement about whether the physics prior changes data efficiency, capacity efficiency, error floor, or training stability.

This repository is a paper engine for one sharply defined experiment. Prefer simple, reproducible, analyzable choices.