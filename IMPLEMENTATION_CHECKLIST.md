
# Implementation Checklist

Status is synced to the current repository state as of 2026-04-04. Checked items are implemented or directly validated in the repo; unchecked items are still missing, not automated yet, or not fully verified.

---

## Phase 1 — Completed Foundation (sections 1–20)

These sections document the initial pilot study. All checked items are implemented and verified.

### 1. Project setup
- [x] Create repository structure
- [x] Add `pyproject.toml` or `requirements.txt`
- [x] Add config system for data, model, train, and experiments
- [x] Add reproducible seeding utilities
- [x] Add logging / artifact saving utilities

### 2. Core system
- [x] Implement Lotka–Volterra RHS function
- [x] Implement ODE solve helper using `solve_ivp`
- [x] Expose system parameters through config
- [x] Add simple validation test for solver output
- [x] Validate solver accuracy: max relative error ~1.9e-9 vs tighter tolerances
- [x] Validate conservation law: max |ΔH|/|H| ~2.0e-9 on ground-truth data

### 3. Dataset generation
- [x] Implement initial condition sampler
- [x] Generate train / val / test splits for one data seed
- [x] Save frozen splits to disk
- [x] Implement nested train subsets for all `D`
- [x] Compute normalization stats from full train pool only
- [x] Save normalization stats to disk
- [x] Repeat for data seeds `11`, `22`, `33` (all 3 generated under `data/`)

### 4. Data loading
- [x] Implement dataset class for frozen splits
- [x] Implement subset loading by dataset size `D`
- [x] Apply saved normalization consistently
- [x] Verify no leakage from val/test into normalization
- [x] Verify normalization roundtrip precision: max error ~1e-7 in float32
- [x] Verify positivity of all inputs and outputs (required for conservation log terms)
- [x] Verify flow map injectivity: 0 suspicious pairs (T < period guarantees diffeomorphism)

### 5. Models
- [x] Implement plain MLP
- [x] Implement configurable hidden layer sizes
- [x] Implement parameter count utility
- [x] Define default capacity grid
- [x] Verify forward output shape is correct

### 6. Losses
- [x] Implement data loss (MSE in normalized space)
- [x] Implement Lotka–Volterra midpoint-rule physics residual
- [x] Implement conservation-law physics loss (H invariant)
- [x] Implement total loss with `lambda_phys` for midpoint
- [x] Implement total loss with `lambda_phys` for conservation
- [x] Verify physics-informed model uses same architecture as plain MLP
- [x] Verify conservation loss on ground truth: ~3.4e-14 (exact to machine precision)
- [x] Verify midpoint residual on ground truth: ~0.151 (15% irreducible bias at T=1.0)

### 7. Training
- [x] Implement training loop
- [x] Implement validation loop
- [x] Implement early stopping
- [x] Log train / val metrics each epoch
- [x] Log grad_norm and phys_data_ratio per epoch
- [x] Save best model checkpoint
- [x] Save training history CSV
- [x] Save final metrics JSON
- [x] Support `physics_prior` parameter: `"none"`, `"midpoint"`, `"conservation"`

### 8. Evaluation
- [x] Implement relative L2 metric (per-sample, physical space)
- [x] Implement MSE metric
- [x] Evaluate on test set
- [ ] Save test predictions optionally
- [x] Flag diverged / NaN runs

### 9. CLI / scripts
- [x] Script to generate datasets (`scripts/generate_datasets.py`)
- [x] Script to run one experiment (`scripts/run_experiment.py`)
- [x] Script to run a sweep (`scripts/run_sweep.py`)
- [x] Script to aggregate run results (`scripts/aggregate_runs.py`)
- [x] Script to fit scaling curves (`scripts/fit_scaling.py`)
- [x] Script to run lambda sweep (`scripts/run_lambda_sweep.py`, supports `--model piml|piml-conservation`)
- [x] Script to analyze lambda sweep (`scripts/analyze_lambda_sweep.py`)
- [x] Script to diagnose physics loss (`scripts/diagnose_physics.py`)
- [ ] Script to generate figures (`scripts/generate_figures.py` — exists but needs updates)

### 10. Sanity checks
- [x] Single-run convergence test
- [ ] Tiny-set overfit test with `D=32`
- [x] Pilot monotonicity check for error vs `D`
- [x] Pilot monotonicity check for error vs `N`
- [x] Check scale of physics loss vs data loss (midpoint: ratio ~1.2× at init)
- [x] Check exact reproducibility with same seed (λ=0 PIML ≡ plain, bit-for-bit)

### 11. Pilot sweep
- [x] Run 2 models × 2 capacities × 3 dataset sizes × 1 data seed × 2 train seeds (24 runs)
- [x] Inspect logs and saved metrics
- [x] Inspect capacity scaling trend
- [x] Inspect data scaling trend
- [x] Fix only genuine bugs before full sweep

### 12. Full experiment
- [x] Run full target matrix: 2 models × 5 capacities × 8 dataset sizes × 3 data seeds × 3 train seeds = **720 runs** (plain + midpoint PIML)
- [x] Confirm all runs save artifacts correctly
- [x] Track divergence rate: **0% divergence** across all 720 runs
- [x] Verify parameter counts and dataset sizes are logged correctly

### 13. Aggregation
- [x] Aggregate by `(model, physics_prior, N, D)`
- [x] Compute mean, std, stderr
- [x] Compute divergence rate
- [x] Export aggregated results table (`runs-progress/runs_aggregate.csv`, `grouped_metrics.csv`)

### 14. Scaling analysis
- [x] Fit capacity scaling curves at fixed `D`
- [x] Fit data scaling curves at fixed `N`
- [x] Fit full `E(N,D)` surface
- [x] Bootstrap confidence intervals (1000 samples)
- [x] Exclude unstable points from fitting when divergence > 30%
- [x] Save fit summaries to disk (`runs-progress/scaling_fits.json`)
- Plain: E∞ ≈ 0, α ≈ 0.91, β ≈ 0.85
- Midpoint PIML: E∞ ≈ 0.011, α ≈ 0.16, β ≈ 0.53

### 15. Lambda sweeps
- [x] Midpoint lambda sweep: 6 λ values × 2 capacities × 3 dataset sizes × 3 data seeds × 3 train seeds = **324 runs**
- [x] Conservation lambda sweep: 7 λ values × 2 capacities × 3 dataset sizes × 1 data seed × 3 train seeds = **126 runs**
- [x] Midpoint result: no λ improves over plain; monotonic degradation above λ=1e-2
- [x] Conservation result: bimodal failure (good convergence or trap at ~0.09); large model highly susceptible; no λ consistently improves

### 16. Data validation
- [x] Solver accuracy verified (re-integration at tighter tolerances: max rel. error ~1.9e-9)
- [x] Conservation law verified on data: max |ΔH|/|H| ~2.0e-9
- [x] Normalization stats match recomputed values exactly
- [x] Normalized data: zero mean, unit std (to machine precision)
- [x] Float32 roundtrip: max error ~1e-7 (negligible)
- [x] All data strictly positive (required for log in conservation loss)
- [x] Flow map injective (T/period ≈ 0.30, well under 1 full orbit)
- [x] No near-duplicate outputs from distant inputs (0 suspicious pairs)
- [x] Cross-seed consistency verified (all 3 data seeds)
- [x] Lipschitz constant of flow map: max ~3.67, mean ~1.54 (well-conditioned)
- [x] Horizon analysis: T=1.0 gives ~0.92 mean relative displacement (good nonlinear regime)
- [x] Oscillation period: ~3.1–3.5 (T/period ≈ 0.29–0.34)

### 17. Figures
- [ ] Task setup schematic
- [ ] Capacity scaling figure
- [ ] Data scaling figure
- [ ] 2D error heatmap / contour
- [ ] Exponent comparison with CI
- [ ] Stability summary figure

### 18. Paper draft
- [x] Created `paper/draft.tex` (compiles cleanly)
- [x] Plain and midpoint results filled in
- [ ] Conservation results still pending (marked with `\pending{}`)
- [ ] Figures still placeholder (marked with `\status{}`)

### 19. Final validation
- [ ] Confirm figures regenerate from scripts only
- [x] Confirm all configs are saved per run
- [x] Confirm no notebook-only results are required
- [ ] Confirm README matches actual implementation
- [ ] Confirm minimal success criterion is met

### 20. Minimal success criterion (Phase 1)
- [x] Two clean scaling plots exist (plain + midpoint, data in `runs-progress/`)
- [x] Exponent estimates with uncertainty exist (`runs-progress/scaling_fits.json`)
- [x] Stability comparison exists (0% divergence both models)
- [x] Clear conclusion can be stated: midpoint physics prior **degrades** scaling (raises E∞ 20×, flattens α from 0.91→0.16, flattens β from 0.85→0.53) due to 15% irreducible approximation error at T=1.0. Conservation prior exhibits bimodal optimization failure (trapping at ~0.09 for large models). Neither prior improves over the plain baseline at any tested configuration.

---

## Phase 2 — Extended Experiments (sections 21–27)

These experiments turn the initial Lotka–Volterra pilot into a robust study. Every experiment answers one of four questions: (1) Does the signature replicate? (2) Does the fit remain statistically stable? (3) Does a causal intervention move the expected parameter? (4) Does the ansatz fail in an interpretable way?

Execution order matches experimental dependency, not section numbering.

### 21. Dense Lotka–Volterra sweep (plain, midpoint, composite)

**Goal**: Increase statistical power on the comparisons that carry the main argument. Densify the grid for plain vs composite (the positive claim about exponent steepening) and plain vs midpoint (the negative claim about floor raising). The composite (simpson) model already exists in `losses.py` and `run_sweep.py`.

**Questions answered**: (2) statistical stability of exponent/floor estimates; (3) does reducing prior bias move α, β, E∞ as predicted?

#### 21a. Infrastructure
- [ ] Add intermediate capacities to `CAPACITY_GRID` in `models/mlp.py`: `small-med [96, 96]` between small and medium, `med-large [192, 192]` between medium and large
- [ ] Add intermediate dataset sizes to `configs/default.yaml`: extend `dataset_sizes` to include `[48, 64, 96, 128, 256, 384, 512, 1024, 2048, 4096, 8192]` (add 48, 96, 384 for low-to-mid densification)
- [ ] Verify nested subset generation works for new D values (subsets must remain nested slices of the same train pool)
- [ ] Regenerate `data/` with extended `dataset_sizes` if `train_subsets.json` doesn't already cover the new sizes

#### 21b. Runs
- [ ] Run dense sweep for `plain`: 7 capacities × 11 dataset sizes × 3 data seeds × 3 train seeds = **693 runs** → `runs-dense/model=plain/`
- [ ] Run dense sweep for `piml` (midpoint): same grid = **693 runs** → `runs-dense/model=piml/`
- [ ] Run dense sweep for `piml-simpson` (composite): same grid = **693 runs** → `runs-dense/model=piml-simpson/`
- [ ] Total: ~2,079 runs

#### 21c. Analysis
- [ ] Aggregate all three models into `runs-dense/grouped_metrics.csv`
- [ ] Fit scaling surfaces for all three models with bootstrap CIs
- [ ] Verify composite exponents remain steeper than plain (α ≈ 2.0 vs 0.91, β ≈ 1.0 vs 0.85)
- [ ] Verify midpoint floor remains elevated (E∞ ≈ 0.011)
- [ ] Compare CI widths to Phase 1 fits — confirm narrower intervals with denser grid

### 22. Horizon sweep on Lotka–Volterra

**Goal**: Test the mechanistic prediction that midpoint prior degradation worsens with horizon (time-discretization error grows) while the composite prior degrades more slowly.

**Questions answered**: (3) causal intervention — increasing T should move E∞ and exponents predictably.

#### 22a. Infrastructure
- [ ] Add `--horizon` CLI argument to `run_experiment.py` and `run_sweep.py` to override `data.T` from config
- [ ] Create `scripts/run_horizon_sweep.py` that iterates over horizons and calls sweep logic for each
- [ ] Ensure `generate_datasets.py` accepts `--horizon` to generate data at different T values
- [ ] Generate datasets at T ∈ {0.5, 1.0, 2.0} for all 3 data seeds → `data-horizon-T=0.5/`, `data-horizon-T=2.0/` (T=1.0 already exists in `data/`)
- [ ] Run `diagnose_physics.py` at each horizon to measure ground-truth midpoint residual and composite residual

#### 22b. Runs
- [ ] Run at each horizon T ∈ {0.5, 1.0, 2.0} for plain, midpoint, composite
- [ ] Use reduced grid: 5 capacities (original) × 8 dataset sizes (original) × 3 data seeds × 3 train seeds = 360 runs per model per horizon
- [ ] 3 models × 3 horizons × 360 = **3,240 runs** → `runs-horizon/T=0.5/`, `runs-horizon/T=1.0/`, `runs-horizon/T=2.0/`

#### 22c. Analysis
- [ ] Aggregate per horizon and fit scaling surfaces
- [ ] Verify midpoint E∞ increases with T (floor tracks irreducible discretization error)
- [ ] Verify composite E∞ increases more slowly than midpoint
- [ ] Plot exponents (α, β) vs horizon for all three models
- [ ] Verify T=0.5 midpoint residual is lower than T=1.0 (should be ~4× smaller for midpoint, ~16× for composite)

### 23. Alternative ansatz comparison and held-out fit prediction

**Goal**: Test whether the current scaling ansatz E(N,D) = E∞ + aN^{-α} + bD^{-β} is the best effective description, and whether it can predict unseen grid cells.

**Questions answered**: (2) fit stability across ansatzes; (4) does the ansatz fail interpretably?

#### 23a. Infrastructure — alternative ansatzes
- [ ] Implement in `src/scaling_piml/analysis/scaling.py` (or new file `analysis/ansatz_comparison.py`):
  - **Ansatz A (current)**: E = E∞ + a·N^{-α} + b·D^{-β} (additive, separable, with floor)
  - **Ansatz B (no-floor power law)**: E = a·N^{-α} + b·D^{-β} (E∞ fixed to 0)
  - **Ansatz C (multiplicative separable)**: E = c · N^{-α} · D^{-β}
  - **Ansatz D (additive with interaction)**: E = E∞ + a·N^{-α} + b·D^{-β} + d·N^{-α}·D^{-β}
  - **Ansatz E (nonparametric baseline)**: 2D thin-plate spline or Gaussian-process fit on log(N) × log(D) → log(E)
- [ ] For each ansatz, implement: fit function, bootstrap CI, residual diagnostics
- [ ] Implement model comparison metrics: AIC/BIC for parametric ansatzes, leave-one-out cross-validation error for all

#### 23b. Infrastructure — held-out fit prediction
- [ ] Create `scripts/validate_scaling_fits.py` that:
  - Takes a `grouped_metrics.csv` and an ansatz name
  - Performs leave-one-column-out (hold out one D) and leave-one-row-out (hold out one N)
  - Fits on the remaining grid cells
  - Reports prediction error on held-out cells
  - Repeats for all ansatzes
- [ ] Also implement leave-one-corner-out: hold out the largest (N, D) cell and predict from the rest (extrapolation test)

#### 23c. Analysis
- [ ] Run ansatz comparison on Phase 1 data (`runs-progress/grouped_metrics.csv`) for all models
- [ ] Run ansatz comparison on dense sweep data (section 21) once available
- [ ] Report AIC/BIC ranking per model
- [ ] Report held-out prediction error per ansatz per model
- [ ] Identify whether Ansatz A (current) wins or is dominated, and under what conditions
- [ ] Check whether interacted or nonparametric ansatzes reveal structure missed by the additive form

### 24. Optimization rescue study for conservation prior

**Goal**: Resolve the ambiguity around the conservation prior's bimodal failure. Determine whether it is an intrinsic surface corruption or a recoverable optimizer artifact.

**Questions answered**: (3) causal intervention — if targeted optimizer changes eliminate bimodality, the interpretation changes fundamentally.

#### 24a. Infrastructure
- [ ] Add `--warm-start` option to `run_experiment.py`: load a trained plain-model checkpoint as initialization before training with conservation loss
- [ ] Add `--grad-clip` option to `run_experiment.py`: apply gradient clipping (e.g., max_norm ∈ {1.0, 5.0})
- [ ] Add `--lambda-schedule` option to `run_experiment.py`: support linear ramp-up of λ_phys over the first K epochs (e.g., K = 50)
- [ ] Add `--two-stage` option: train phase 1 with data-only loss for M epochs, then add conservation loss for remaining epochs
- [ ] Create `scripts/run_rescue_sweep.py` that runs all rescue variants on the configurations where bimodality is strongest (large capacity, moderate-to-high λ)

#### 24b. Identify target configurations
- [ ] From conservation λ sweep results (`runs-conservation-lambda-sweep/`), identify the (capacity, D, λ) triples with highest bimodality rate (e.g., large model, λ ≥ 0.01)
- [ ] Select top 6–10 configurations as the rescue target set

#### 24c. Runs
- [ ] For each target configuration, run 5 rescue variants × 3 train seeds:
  - Baseline (existing conservation training)
  - Warm start from plain checkpoint
  - Gradient clipping (max_norm=1.0)
  - λ ramp-up over 50 epochs
  - Two-stage (200 epochs data-only, then 200 epochs with conservation)
- [ ] ~10 configs × 5 variants × 3 seeds = **150 runs** → `runs-rescue/`

#### 24d. Analysis
- [ ] For each variant, measure: fraction of runs that avoid the ~0.09 trap, final test error, training stability (grad norm, loss curves)
- [ ] Compare bimodality rate across variants
- [ ] If warm start or λ scheduling eliminates bimodality → interpretation: optimizer-sensitive, not intrinsically harmful
- [ ] If bimodality persists across all variants → interpretation: surface corruption from weak exact prior
- [ ] Report rescue success rate table

### 25. Cross-system replication: Van der Pol oscillator

**Goal**: Test whether the scaling signatures (floor from biased prior, steepening from reduced-bias prior, bimodality from exact-but-weak prior) replicate on a smooth non-chaotic nonlinear oscillator with different geometry.

**Questions answered**: (1) does the signature replicate on a qualitatively different system?

#### 25a. System implementation
- [ ] Implement Van der Pol RHS in `src/scaling_piml/systems/van_der_pol.py`:
  - ẋ = y, ẏ = μ(1 − x²)y − x
  - Default parameter: μ = 1.0 (weakly nonlinear regime)
- [ ] Implement Van der Pol midpoint physics loss in `src/scaling_piml/losses.py` (or `losses_vdp.py`): same midpoint-rule structure, using Van der Pol RHS
- [ ] Implement Van der Pol composite (2-step midpoint) physics loss
- [ ] Identify an appropriate physics prior for the "exact-but-weak" role:
  - Van der Pol has no polynomial first integral in general, so use energy-related bound or known asymptotic invariant for the limit cycle
  - Alternatively, use the dissipation-rate constraint: d/dt[½(x² + y²)] = μ(1 − x²)y² (known exactly)
  - Choose whichever gives a nontrivial scalar constraint analogous to the conservation law
- [ ] Add `--system vdp` support to `run_experiment.py`, `run_sweep.py`, and `generate_datasets.py`
- [ ] Create `configs/vdp.yaml` with appropriate system parameters, initial condition ranges, horizon, and solver settings

#### 25b. Data generation and validation
- [ ] Choose initial condition domain (e.g., x₀ ∈ [−2, 2], y₀ ∈ [−2, 2]) and horizon T
- [ ] Choose T such that the flow map is still diffeomorphic (T < period or well below transient timescale)
- [ ] Generate datasets for 3 data seeds → `data-vdp/data_seed={11,22,33}/`
- [ ] Validate solver accuracy at chosen T
- [ ] Measure ground-truth midpoint residual at chosen T
- [ ] Measure ground-truth composite residual at chosen T
- [ ] Verify dataset properties: injectivity, Lipschitz bound, no degeneracies

#### 25c. Runs
- [ ] Run full sweep: plain + midpoint + composite + exact-prior (if applicable) × 5 capacities × 8 dataset sizes × 3 data seeds × 3 train seeds
- [ ] ~3–4 models × 360 = **1,080–1,440 runs** → `runs-vdp/`

#### 25d. Analysis
- [ ] Fit scaling surfaces for each model variant
- [ ] Compare qualitative pattern: does midpoint raise E∞? Does composite steepen exponents? Does the exact prior produce bimodality?
- [ ] Tabulate (E∞, α, β) side-by-side with Lotka–Volterra results
- [ ] If signatures replicate → strong evidence that taxonomy is tied to prior properties, not system geometry
- [ ] If signatures differ → document which aspect differs and why (different nonlinearity, different discretization error structure)

### 26. Cross-system replication: Duffing oscillator (harder regime)

**Goal**: Stress-test the scaling taxonomy on a system with richer dynamics (potential chaos for driven Duffing, or strong nonlinearity for unforced).

**Questions answered**: (1) does the signature replicate in a harder dynamical regime?

#### 26a. System implementation
- [ ] Implement Duffing RHS in `src/scaling_piml/systems/duffing.py`:
  - Unforced: ẍ + δẋ + αx + βx³ = 0 (rewrite as 2D first-order system)
  - Or forced (autonomous embedding): add phase variable θ̇ = ω for periodic forcing
  - Choose parameter regime: unforced with strong cubic nonlinearity, or lightly damped + forced for mild chaos
- [ ] Implement Duffing midpoint and composite physics losses
- [ ] Identify exact-but-weak prior candidate:
  - Unforced undamped Duffing has energy E = ½ẋ² + ½αx² + ¼βx⁴ (exact invariant)
  - This is a good analog to the Lotka–Volterra conservation law
- [ ] Add `--system duffing` support to CLI scripts
- [ ] Create `configs/duffing.yaml`

#### 26b. Data generation and validation
- [ ] Choose parameter regime and initial condition domain
- [ ] Choose T (stay below full period or chaos onset)
- [ ] Generate datasets for 3 data seeds → `data-duffing/data_seed={11,22,33}/`
- [ ] Validate solver accuracy, measure ground-truth residuals
- [ ] Verify dataset properties

#### 26c. Runs
- [ ] Full sweep: same protocol as section 25c
- [ ] ~1,080–1,440 runs → `runs-duffing/`

#### 26d. Analysis
- [ ] Fit scaling surfaces, compare taxonomy
- [ ] Tabulate alongside Lotka–Volterra and Van der Pol
- [ ] If Duffing is chaotic/near-chaotic: check whether all priors degrade, and whether the relative ordering is preserved
- [ ] Cross-system summary table: (system, prior, E∞, α, β) for all 3 systems

### 27. Noise robustness

**Goal**: Test scaling signatures under realistic corruption. Two sub-experiments: observation noise on supervised targets, and prior mismatch (wrong physics parameters in the loss).

**Questions answered**: (3) does noise move exponents/floor as expected? (1) do signatures persist under corruption?

#### 27a. Infrastructure — observation noise
- [ ] Add `--obs-noise` argument to `run_experiment.py` and `run_sweep.py`: adds Gaussian noise to u(T) targets at training time
- [ ] Noise levels: σ ∈ {0, 0.01, 0.05, 0.1} (fraction of per-component std of the targets)
- [ ] Ensure noise is added after normalization and is reproducible (seeded per sample)
- [ ] Ensure test/validation data remains clean (noise only on training targets)

#### 27b. Infrastructure — prior mismatch
- [ ] Add `--prior-params` argument to `run_experiment.py` to override system parameters used in the physics loss (while keeping the data-generating parameters fixed)
- [ ] Mismatch levels: perturb each Lotka–Volterra parameter by ±5%, ±10%, ±20%
- [ ] The data remains generated with true parameters; only the RHS in the physics loss uses perturbed parameters

#### 27c. Runs — observation noise
- [ ] Run on Lotka–Volterra, plain + midpoint + composite, at 4 noise levels
- [ ] Use standard grid: 5 capacities × 8 dataset sizes × 3 data seeds × 3 train seeds = 360 runs per model per noise level
- [ ] 3 models × 4 noise levels × 360 = **4,320 runs** → `runs-noise/obs/sigma=0.00/`, etc.

#### 27d. Runs — prior mismatch
- [ ] Run on Lotka–Volterra, midpoint + composite only (plain is unaffected), at 3 mismatch levels
- [ ] 2 models × 3 mismatch levels × 360 = **2,160 runs** → `runs-noise/mismatch/delta=0.05/`, etc.

#### 27e. Analysis
- [ ] For observation noise: plot E∞, α, β vs noise level for each model. Does the prior change the noise sensitivity of the scaling surface?
- [ ] For prior mismatch: plot E∞, α, β vs mismatch level. Does mismatch create a new floor even for the composite prior?
- [ ] Compare: does noise affect data-axis (β) more than capacity-axis (α)? Does prior mismatch primarily affect the floor?
- [ ] Report whether the qualitative taxonomy (biased/reduced-bias/exact-but-weak signatures) survives under corruption

---

## Phase 2 summary — execution order

| Step | Section | Description | Est. runs | Depends on |
|------|---------|-------------|-----------|------------|
| 1 | 21 | Dense LV sweep (plain/midpoint/composite) | ~2,079 | — |
| 2 | 22 | Horizon sweep on LV | ~3,240 | New data at T=0.5, T=2.0 |
| 3 | 23 | Alternative ansatz + held-out validation | 0 (analysis only) | Steps 1–2 data |
| 4 | 24 | Optimization rescue for conservation prior | ~150 | — |
| 5 | 25 | Van der Pol replication | ~1,080–1,440 | System implementation |
| 6 | 26 | Duffing replication | ~1,080–1,440 | System implementation |
| 7 | 27 | Noise robustness | ~6,480 | — |

Total Phase 2: ~14,000–15,000 runs.

### Phase 2 completion criteria
- [ ] At least 2 additional ODE systems with full scaling fits
- [ ] Horizon variation confirms mechanistic link between discretization error and E∞
- [ ] Alternative ansatzes compared; current ansatz wins or a better one is identified
- [ ] Conservation prior ambiguity resolved (optimizer artifact vs surface corruption)
- [ ] Held-out prediction validates the scaling ansatz as more than descriptive
- [ ] Noise experiments confirm taxonomy survives under realistic corruption
- [ ] All results reproducible from scripts and configs alone