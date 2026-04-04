# Empirical Scaling Laws for Physics-Informed Priors in SciML

This repository implements a controlled empirical study of how physics-informed priors change scaling behavior in scientific machine learning.

## Core question

> Do different physics priors produce distinguishable scaling signatures, and can these signatures diagnose the quality of the prior?

## Current status (2026-04-04)

### Phase 1 — Lotka–Volterra pilot (complete)

| Experiment | Runs | Status |
|-----------|------|--------|
| Full scaling sweep (plain + midpoint) | 720 | Done |
| Composite (simpson) sweep | ~360 | Done |
| Midpoint λ sweep | 324 | Done |
| Conservation λ sweep | 126 | Done |
| Data validation | — | Done |

#### Key findings (Phase 1)

| Model | E∞ | α (capacity) | β (data) | Qualitative signature |
|-------|-----|------|------|--------|
| Plain MLP | ≈ 0 | 0.91 | 0.85 | Clean power-law scaling |
| PIML (midpoint) | ≈ 0.011 | 0.16 | 0.53 | Floor raised by 15% irreducible prior bias |
| PIML (composite) | ≈ 0.001 | ~2.0 | ~1.0 | Reduced bias → steeper exponents |
| PIML (conservation) | — | — | — | Bimodal optimization failure |

Three-way taxonomy:
- **Biased prior** (midpoint): raises error floor, compresses exponents
- **Reduced-bias prior** (composite): recovers and steepens scaling
- **Exact-but-weak prior** (conservation): corrupts scaling surface stochastically

### Phase 2 — Extended experiments (planned)

| # | Experiment | Purpose | Est. runs |
|---|-----------|---------|-----------|
| 1 | Dense LV sweep | Statistical power for main comparisons | ~2,079 |
| 2 | Horizon sweep | Mechanistic test: discretization error ↔ E∞ | ~3,240 |
| 3 | Alternative ansatzes + held-out validation | Fit robustness and predictive adequacy | 0 (analysis) |
| 4 | Optimization rescue (conservation) | Resolve bimodality ambiguity | ~150 |
| 5 | Van der Pol replication | Cross-system: smooth nonlinear oscillator | ~1,080–1,440 |
| 6 | Duffing replication | Cross-system: harder/chaotic regime | ~1,080–1,440 |
| 7 | Noise robustness | Observation noise + prior parameter mismatch | ~6,480 |

See `IMPLEMENTATION_CHECKLIST.md` for detailed task breakdowns.

### Data validation summary
- Solver accuracy: max relative error ~1.9e-9 (DOP853, rtol=1e-9, atol=1e-11)
- Conservation law on data: max |ΔH|/|H| ~2.0e-9
- Float32 normalization roundtrip: max error ~1e-7
- All data strictly positive; flow map injective (T/period ≈ 0.30)
- Oscillation period ~3.1–3.5; T=1.0 gives 0.92 mean relative displacement
- Flow map Lipschitz constant: max ~3.67, mean ~1.54

## Scope

### Systems

| System | Role | Status |
|--------|------|--------|
| **Lotka–Volterra** | Primary system. Non-chaotic, has exact first integral. | Implemented |
| **Van der Pol** | Smooth nonlinear oscillator, no polynomial first integral. Tests whether taxonomy generalizes beyond conservative systems. | Planned (Phase 2) |
| **Duffing** | Stronger nonlinearity, potential chaos (forced case). Stress test for scaling signatures. | Planned (Phase 2) |

### Task

Fixed-horizon flow-map prediction: learn $f_\theta(u_0) \mapsto u(T)$.

Applies identically to all three systems. No autoregressive rollout, no PDE, no operator learning.

### Physics priors

| Prior | Type | Bias | Systems |
|-------|------|------|---------|
| None (plain) | Baseline | — | All |
| Midpoint ODE residual | Approximate, biased | ~15% at T=1.0 (LV) | All |
| Composite 2-step midpoint | Approximate, reduced bias | ~0.6% at T=1.0 (LV) | All |
| Conservation law | Exact, scalar constraint | 0 (exact) | LV (H invariant), Duffing (energy, undamped) |
| Dissipation rate / energy bound | Exact, scalar constraint | 0 (exact) | Van der Pol (TBD) |

### Scaling axes
- Model capacity $N$ (parameter count): 5–7 levels from tiny to xlarge
- Dataset size $D$: 8–11 levels from 48 to 8192

### Scaling ansatz

Primary:
$$E(N, D) \approx E_\infty + aN^{-\alpha} + bD^{-\beta}$$

Alternative ansatzes compared in Phase 2:
- No-floor power law: $E = aN^{-\alpha} + bD^{-\beta}$
- Multiplicative separable: $E = cN^{-\alpha}D^{-\beta}$
- Additive with interaction: $E = E_\infty + aN^{-\alpha} + bD^{-\beta} + dN^{-\alpha}D^{-\beta}$
- Nonparametric (GP or thin-plate spline on log-log grid)

### Experiment inclusion criterion

Every experiment answers at least one of:
1. Does the signature replicate?
2. Does the fit remain statistically stable?
3. Does a causal intervention move the expected parameter?
4. Does the ansatz fail in an interpretable way?

## Default experiment settings

### Lotka–Volterra parameters
- $\alpha = 1.5$, $\beta = 1.0$, $\delta = 1.0$, $\gamma = 3.0$

### Prediction horizons
- Primary: $T = 1.0$
- Horizon sweep: $T \in \{0.5, 1.0, 2.0\}$

### Initial conditions
- $x_0 \sim \text{Uniform}(0.5, 2.5)$, $y_0 \sim \text{Uniform}(0.5, 2.5)$

### ODE solver
- `solve_ivp`, method `DOP853`, rtol=$10^{-9}$, atol=$10^{-11}$

## Dataset protocol

### Per data seed
- Train pool: 20,000
- Validation: 2,000
- Test: 2,000

### Data seeds
- 11, 22, 33

### Dataset sizes (Phase 1)

```
D ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192}
```

### Dataset sizes (Phase 2 dense sweep)

```
D ∈ {48, 64, 96, 128, 256, 384, 512, 1024, 2048, 4096, 8192}
```

Constraints:
- Training subsets are **nested slices** of the same shuffled master train pool.
- Normalization computed from the **full train pool only**, reused for all subset sizes within a data seed.

## Models

### Plain MLP
- Input: normalized $u_0$
- Output: normalized $u(T)$
- Activation: ReLU (default) or GELU
- One consistent architecture family across all runs

### Physics-regularized MLP (midpoint residual)

Same architecture. Loss:
$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda\,\mathcal{L}_{\text{phys}}, \quad \mathcal{L}_{\text{phys}} = \left\lVert \hat{u}(T) - u_0 - T\,F\!\left(\tfrac{\hat{u}(T) + u_0}{2}\right) \right\rVert_2^2$$

**Status**: Irreducible ground-truth residual ~15% at T=1.0. Degrades performance at all λ ≥ 0.01.

### Physics-regularized MLP (composite 2-step midpoint)

Same architecture. Applies midpoint rule on $[0, T/2]$ and $[T/2, T]$ separately, using a learned midpoint $u(T/2)$ (4D output).

**Status**: Ground-truth residual ~24× lower than single-step. Recovers and steepens scaling.

### Physics-regularized MLP (conservation law)

Same architecture. Loss uses the Lotka–Volterra first integral:
$$H(u,v) = \delta u - \gamma \ln u + \beta v - \alpha \ln v$$
$$\mathcal{L}_{\text{cons}} = \left( H(\hat{u}(T)) - H(u_0) \right)^2$$

**Status**: Exact but scalar constraint on vector output. Bimodal optimization failure. Large models trap at ~0.09.

## Capacity grid

| Name | Hidden widths | Status |
|------|--------------|--------|
| tiny | `[32, 32]` | Implemented |
| small | `[64, 64]` | Implemented |
| small-med | `[96, 96]` | Planned (Phase 2) |
| medium | `[128, 128]` | Implemented |
| med-large | `[192, 192]` | Planned (Phase 2) |
| large | `[256, 256]` | Implemented |
| xlarge | `[256, 256, 256]` | Implemented |

## Directory structure

```
configs/           YAML configs (default, pilot, smoke, per-system)
data/              Lotka–Volterra datasets (3 data seeds)
data-vdp/          Van der Pol datasets (planned)
data-duffing/      Duffing datasets (planned)
paper/             LaTeX draft
runs-progress/     Phase 1 main results (plain + midpoint)
runs-simpson/      Composite (simpson) results
runs-dense/        Phase 2 dense sweep (planned)
runs-horizon/      Phase 2 horizon sweep (planned)
runs-rescue/       Phase 2 optimization rescue (planned)
runs-vdp/          Van der Pol results (planned)
runs-duffing/      Duffing results (planned)
runs-noise/        Noise robustness results (planned)
scripts/           All experiment, analysis, and figure scripts
src/scaling_piml/  Source package
tests/             Pytest tests
```

## Quickstart

```bash
# Install
python -m pip install -e .

# Generate datasets
python scripts/generate_datasets.py --config configs/default.yaml --out data

# Run a single experiment
python scripts/run_experiment.py --config configs/default.yaml \
  --data-root data --D 256 --train-seed 101 --model plain --capacity small

# Run a full sweep
python scripts/run_sweep.py --config configs/default.yaml \
  --data-dir data --models plain,piml --out runs-progress

# Aggregate results
python scripts/aggregate_runs.py --runs-root runs-progress --out-dir runs-progress

# Fit scaling laws
python scripts/fit_scaling.py --grouped-metrics runs-progress/grouped_metrics.csv \
  --out-dir runs-progress
```

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

### Full scaling sweep (completed)
`2` models (plain + midpoint PIML) × `5` capacities × `8` dataset sizes × `3` data seeds × `3` train seeds

$$2 \times 5 \times 8 \times 3 \times 3 = 720\text{ runs}$$

### Midpoint lambda sweep (completed)
`6` λ values × `2` capacities × `3` dataset sizes × `3` data seeds × `3` train seeds = **324 runs**

### Conservation lambda sweep (completed)
`7` λ values × `2` capacities × `3` dataset sizes × `1` data seed × `3` train seeds = **126 runs**

### Remaining
- [ ] Conservation full scaling sweep (pending: no beneficial λ found)
- [ ] Figure generation

## Metrics

```text
model_name
is_physics_informed
physics_prior          # "none", "midpoint", or "conservation"
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