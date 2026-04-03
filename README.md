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
Per data seed:
- train pool: `20000`
- validation: `2000`
- test: `2000`
### Data seeds
Data seeds:
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
Physics-Regularized MLP

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
lambda_phys = 0.1
Capacity grid

| name   | hidden widths |
|--------|--------------|
| tiny   | `[32, 32]` |
| small  | `[64, 64]` |
| medium | `[128, 128]` |
| large  | `[256, 256]` |
| xlarge | `[256, 256, 256]` |
large   : [256, 256]
xlarge  : [256, 256, 256]

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
303
Experiment matrix

`2` models × `5` capacities × `8` dataset sizes × `3` data seeds × `3` train seeds
3 data seeds
3 train seeds

$$2 \times 5 \times 8 \times 3 \times 3 = 720\text{ runs}$$

2 × 5 × 8 × 3 × 3 = 720 runs

$$2 \times 4 \times 6 \times 2 \times 3 = 288\text{ runs}$$

## Metrics
2 × 4 × 6 × 2 × 3 = 288 runs
Metrics

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
diverged
nan_detected

\[
E = \frac{\lVert \hat{u}(T) - u(T) \rVert_2}{\lVert u(T) \rVert_2 + \varepsilon}
\]

## Analysis requirements
	​

Analysis requirements

Aggregate over data seeds and train seeds for each (model, N, D).
- Capacity scaling: error vs $N$ at fixed $D$
- Data scaling: error vs $D$ at fixed $N$
- Full surface fit: $E(N, D)$
Capacity scaling: error vs N at fixed D
Data scaling: error vs D at fixed N
Full surface fit: E(N,D)
- nonlinear least squares
- bootstrap confidence intervals for exponents

nonlinear least squares
bootstrap confidence intervals for exponents
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

## Success criterion
fit scaling laws
generate figures
Success criterion
- clean scaling curves,
- fitted exponents with uncertainty,
- a clear statement about whether the physics prior changes data efficiency, capacity efficiency, error floor, or training stability.
clean scaling curves,
fitted exponents with uncertainty,
a clear statement about whether the physics prior changes data efficiency, capacity efficiency, error floor, or training stability.

This repository is a paper engine for one sharply defined experiment. Prefer simple, reproducible, analyzable choices.