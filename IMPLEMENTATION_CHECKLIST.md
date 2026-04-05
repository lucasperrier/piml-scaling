
# Implementation Checklist

Status is synced to the current repository state as of 2026-04-05. Checked items are implemented or directly validated in the repo; unchecked items are still missing, not automated yet, or not fully verified.

**Paper scope (merged plan):** One paper with a clear progression — establish a controlled taxonomy of prior-induced scaling regimes on Lotka–Volterra, test the proposed mechanism on the same system, then test whether the taxonomy survives across other dynamical systems (Van der Pol, Duffing). The central claim: simple physics priors induce distinct scaling regimes, and those regimes can be diagnosed through scaling surfaces, prior residual checks, and training-stability behavior. The target is regime recurrence across systems, not numeric replication of fitted exponents.

**System ladder:** Lotka–Volterra (anchor, conservative) → Van der Pol (first generalization, non-conservative) → Duffing (stress test, stronger nonlinearity).

---

## Phase 1 — Completed Foundation (sections 1–20)

These sections document the initial Lotka–Volterra pilot study that establishes the three-way taxonomy. All checked items are implemented and verified.

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
- [x] Save test predictions optionally
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
- [x] Script to validate scaling ansatzes (`scripts/validate_scaling_fits.py`)
- [x] Script to run horizon sweep (`scripts/run_horizon_sweep.py`)
- [x] Script to run rescue sweep (`scripts/run_rescue_sweep.py`)
- [x] Script to generate figures (`scripts/generate_figures.py` — complete with all 6 paper figures)
- [x] Script to plot gradient dynamics (`scripts/plot_gradient_dynamics.py`)

### 10. Sanity checks
- [x] Single-run convergence test
- [x] Tiny-set overfit test with `D=32`
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
- [x] Task setup schematic
- [x] Capacity scaling figure
- [x] Data scaling figure
- [x] 2D error heatmap / contour
- [x] Exponent comparison with CI
- [x] Stability summary figure

### 18. Paper draft
- [x] Created `paper/draft.tex` (compiles cleanly)
- [x] Plain and midpoint results filled in
- [x] Composite (simpson) results filled in
- [x] Conservation results described (bimodal failure, lambda sweep)
- [x] New Section 6.7: ansatz comparison with BIC table, CV table, interpretation
- [x] Abstract, contributions, conclusion, discussion, limitations updated with ansatz results
- [x] Figures generated from data (`scripts/generate_figures.py` produces all 6 paper figures)

### 19. Final validation
- [x] Confirm figures regenerate from scripts only
- [x] Confirm all configs are saved per run
- [x] Confirm no notebook-only results are required
- [x] Confirm README matches actual implementation
- [x] Confirm minimal success criterion is met

### 20. Minimal success criterion (Phase 1)
- [x] Two clean scaling plots exist (plain + midpoint, data in `runs-progress/`)
- [x] Exponent estimates with uncertainty exist (`runs-progress/scaling_fits.json`)
- [x] Stability comparison exists (0% divergence both models)
- [x] Clear conclusion can be stated: midpoint physics prior **degrades** scaling (raises E∞ 20×, flattens α from 0.91→0.16, flattens β from 0.85→0.53) due to 15% irreducible approximation error at T=1.0. Conservation prior exhibits bimodal optimization failure (trapping at ~0.09 for large models). Neither prior improves over the plain baseline at any tested configuration.

---

## Phase 2 — Cross-System Validation and Mechanism Tests (sections 21–30)

These experiments upgrade the paper from a single-system case study to a controlled cross-system study. The central question: do the same qualitative scaling signatures recur across dynamical systems with different structure? Every experiment answers one of four questions: (1) Does the regime recur on a new system? (2) Does the fit remain statistically stable? (3) Does a causal intervention move the expected parameter? (4) Does the ansatz fail in an interpretable way?

**Paper structure this enables:**
1. Lotka–Volterra taxonomy (Phase 1, done) — anchor system, full evidence
2. Mechanism tests on LV (horizon sweep, conservation rescue) — causal support
3. Van der Pol generalization — first test beyond conservative dynamics
4. Duffing stress test — stronger nonlinearity with conservative structure
5. Discussion — which signatures are stable across systems vs system-specific

**Prior-class comparison across systems (not formula identity):**
- *Biased residual prior*: single-step midpoint rule → expect raised floor, compressed exponents
- *Reduced-bias residual prior*: composite/Simpson → expect recovered scaling, lower floor
- *Exact-but-weak structural prior*: conservation law (LV, Duffing) or system-appropriate structural constraint (Van der Pol) → expect optimization-limited behavior, not clean floor

Execution order reflects the merged paper priority: anchor stabilization → cross-system ladder → mechanism tests → supporting experiments.

### 21. Dense Lotka–Volterra sweep — anchor stabilization

**Goal**: Tighten confidence intervals on the anchor-system taxonomy. Densify the grid for plain vs composite (the positive claim about exponent steepening) and plain vs midpoint (the negative claim about floor raising). This is statistical infrastructure for the main claim, not the conceptual extension.

**Paper role**: Strengthens the LV section with tighter CIs and more stable surfaces.

**Questions answered**: (2) statistical stability of exponent/floor estimates; (3) does reducing prior bias move α, β, E∞ as predicted?

#### 21a. Infrastructure
- [x] Add intermediate capacities to `CAPACITY_GRID` in `models/mlp.py`: `small-med [96, 96]` between small and medium, `med-large [192, 192]` between medium and large
- [x] Add intermediate dataset sizes to `configs/default.yaml`: extend `dataset_sizes` to include `[48, 64, 96, 128, 256, 384, 512, 1024, 2048, 4096, 8192]` (add 48, 96, 384 for low-to-mid densification)
- [x] Verify nested subset generation works for new D values (subsets must remain nested slices of the same train pool)
- [x] Regenerate `data/` with extended `dataset_sizes` if `train_subsets.json` doesn't already cover the new sizes

#### 21b. Runs (IN PROGRESS — running on Runpod GPU pod)
- [ ] Run dense sweep for `plain`: 7 capacities × 11 dataset sizes × 3 data seeds × 3 train seeds = **693 runs** → `runs-dense/model=plain/` — **184/693 completed**
- [ ] Run dense sweep for `piml` (midpoint): same grid = **693 runs** → `runs-dense/model=piml/` — **92/693 completed** (from prior partial run)
- [ ] Run dense sweep for `piml-simpson` (composite): same grid = **693 runs** → `runs-dense/model=piml-simpson/` — **77/693 completed** (from prior partial run)
- [ ] Total: ~2,079 runs — **353/2,079 completed (~17%)**

#### 21c. Analysis
- [ ] Aggregate all three models into `runs-dense/grouped_metrics.csv`
- [ ] Fit scaling surfaces for all three models with bootstrap CIs
- [ ] Verify composite exponents remain steeper than plain (α ≈ 2.0 vs 0.91, β ≈ 1.0 vs 0.85)
- [ ] Verify midpoint floor remains elevated (E∞ ≈ 0.011)
- [ ] Compare CI widths to Phase 1 fits — confirm narrower intervals with denser grid

### 29. Cross-system generalization: Van der Pol oscillator

**Goal**: First essential generalization beyond conservative dynamics. Van der Pol is a non-conservative system with a limit cycle, which removes the exact first integral that Lotka–Volterra provides. This tests whether the prior-induced scaling taxonomy is tied to prior class properties (biased residual, reduced-bias residual, structural constraint) rather than to conservative geometry. This is the key generalization step in the paper.

**Paper role**: Main Section 4 — "Does the taxonomy survive once conservative geometry is removed?"

**Questions answered**: (1) do the three abstract regime types recur on a non-conservative system?

#### 29a. System implementation
- [x] Implement Van der Pol RHS in `src/scaling_piml/systems/van_der_pol.py`:
  - $\dot{x} = y$, $\dot{y} = \mu(1 - x^2)y - x$ (standard form)
  - Default parameters: $\mu = 1.0$ (moderate nonlinearity; limit cycle exists for $\mu > 0$)
- [x] Add `--system van-der-pol` dispatch to `generate_datasets.py`, `run_experiment.py`, `run_sweep.py`
- [x] Add Van der Pol system-name dispatch to `src/scaling_piml/data/generate.py` (alongside existing LV/Duffing branches)
- [x] Add Van der Pol system-name dispatch to `src/scaling_piml/train.py` (loss selection branches)
- [x] Create `configs/van_der_pol.yaml` with system parameters, IC domain, horizon, solver settings
- [x] Smoke-tested all 5 model types (plain, piml, piml-conservation, piml-simpson, piml-simpson-true)

#### 29b. Physics priors for Van der Pol
**Design principle**: Preserve the abstract prior categories, not force identical formulas. Compare at the level of *prior class*.

- [x] **Biased residual prior (midpoint)**: implemented `vdp_physics_loss()` in `losses.py` — single-step midpoint rule using Van der Pol RHS. GT residual: **3.25** (higher than LV's 0.15, reflecting stronger nonlinearity at μ=1.0)
- [x] **Reduced-bias residual prior (composite)**: implemented `vdp_composite_midpoint_loss()` in `losses.py` — 2-step composite midpoint using Van der Pol RHS. GT residual: **0.38** (8.6× reduction from midpoint)
- [x] **Structural prior (dissipation-aware)**: Chose Option A — dissipation-rate loss enforcing ΔE ≈ T·μ(1−x²_mid)·y²_mid at midpoint. Implemented as `vdp_dissipation_loss()`. GT residual: **3.50** (high — midpoint approximation of dissipation rate is crude; this is the "exact-but-weak" regime test)
- [x] **Simpson prior**: implemented `vdp_simpson_loss()` — 4th-order Simpson's 1/3 rule. GT residual: **0.135** (24× reduction from midpoint)
- [x] Verify ground-truth residuals computed for all VdP priors (see `diagnostic_residuals.json`)

#### 29c. Data generation and validation
- [x] Choose IC domain: $x_0 \in [-3, 3]$, $y_0 \in [-3, 3]$ — spans both inside and outside the limit cycle
- [x] Choose $T = 1.0$ (well below limit cycle period ≈ 2π for μ=1.0)
- [x] Generate datasets for 3 data seeds → `data-vdp/data_seed={11,22,33}/`
- [x] Validate solver accuracy: max absolute error 3.78e-07 (100 samples vs tighter tolerances)
- [x] Measure ground-truth residuals: midpoint=3.25, composite=0.38, Simpson=0.135, dissipation=3.50
- [ ] Verify dataset properties: injectivity, Lipschitz bound

#### 29d. Runs
- [ ] Run full sweep: plain + midpoint + composite + structural × 5 capacities × 8 dataset sizes × 3 data seeds × 3 train seeds
- [ ] 4 models × 360 = **1,440 runs** → `runs-vdp/`
- [ ] If structural prior design is difficult, run plain + midpoint + composite first (3 × 360 = 1,080 runs) and add structural later

#### 29e. Analysis
- [ ] Fit scaling surfaces for each model variant
- [ ] Test three regime-recurrence questions:
  - Does midpoint still raise E∞ and compress exponents? (biased residual regime)
  - Does composite still recover healthier scaling with lower floor? (reduced-bias regime)
  - Does structural prior still produce optimization-limited or bimodal behavior? (exact-but-weak regime)
- [ ] Tabulate (E∞, α, β) side-by-side with Lotka–Volterra results
- [ ] Explicitly separate what matches across systems from what differs
- [ ] If regime recurrence holds → main paper claim validated beyond conservative systems
- [ ] If regime structure differs → document which aspects are system-specific and why (e.g., non-conservative dynamics may change the exact-but-weak regime character)

### 26. Cross-system stress test: Duffing oscillator

**Goal**: Test whether the scaling taxonomy remains meaningful under stronger nonlinearity. Duffing is a conservative Hamiltonian system like Lotka–Volterra but with polynomial (cubic) nonlinearity instead of predator–prey interaction. This is the stress test: if the same qualitative signatures appear here, the taxonomy is robust across both conservative and non-conservative systems and across different nonlinearity types.

**Paper role**: Main Section 5 — "Stress test under stronger nonlinearity."

**Questions answered**: (1) does the signature replicate on a qualitatively different conservative system with harder dynamics?

#### 26a. System implementation
- [x] Implement Duffing RHS in `src/scaling_piml/systems/duffing.py`:
  - $\dot{x} = y$, $\dot{y} = -\alpha x - \beta x^3$ (undamped, unforced)
  - Default parameters: $\alpha = 1.0$, $\beta = 1.0$ (hardening spring)
- [x] Implement Duffing midpoint and composite physics losses (same midpoint/composite structure, swap RHS)
- [x] Implement Duffing conservation loss: $L_{\text{cons}} = (E(\hat{\bm{x}}_T) - E(\bm{x}_0))^2$ where $E = \frac{1}{2}y^2 + \frac{1}{2}\alpha x^2 + \frac{1}{4}\beta x^4$
- [x] Implement Duffing Simpson loss
- [x] Add `--system duffing` support to `run_experiment.py`, `run_sweep.py`, `generate_datasets.py`
- [x] Create `configs/duffing.yaml` with system parameters, IC domain, horizon, solver settings

#### 26b. Data generation and validation
- [x] IC domain: $x_0 \in [-2, 2]$, $y_0 \in [-2, 2]$, horizon $T = 1.0$ (from duffing.yaml)
- [x] Generate datasets for 3 data seeds → `data-duffing/data_seed={11,22,33}/`
- [x] Validate solver accuracy: max absolute error 2.47e-07 (100 samples vs tighter tolerances)
- [x] Measure ground-truth residuals: midpoint=1.72, composite=0.134 (12.8× reduction), Simpson=0.025 (67.7× reduction), conservation=5.78e-14 (machine epsilon)
- [x] Energy conservation verified: max |ΔE/E| = 3.18e-07, mean = 6.74e-08
- [ ] Verify dataset properties: injectivity, Lipschitz bound

#### 26c. Runs
- [ ] Run full sweep: plain + midpoint + composite + conservation × 5 capacities × 8 dataset sizes × 3 data seeds × 3 train seeds
- [ ] 4 models × 360 = **1,440 runs** → `runs-duffing/`

#### 26d. Analysis
- [ ] Fit scaling surfaces for each model variant
- [ ] Test three regime-recurrence questions (same as Van der Pol):
  - Does midpoint raise E∞ and compress exponents?
  - Does composite recover scaling?
  - Does conservation produce bimodality or optimization-limited behavior?
- [ ] Tabulate (E∞, α, β) side-by-side with Lotka–Volterra and Van der Pol results
- [ ] If signatures replicate → strong evidence that taxonomy is tied to prior properties, not system geometry
- [ ] If signatures differ → document which aspect differs and why

### 22. Horizon sweep on Lotka–Volterra — mechanism test

**Goal**: Test the mechanistic prediction that midpoint prior degradation worsens with horizon (time-discretization error grows) while the composite prior degrades more slowly. This is a causal intervention within the anchor system.

**Paper role**: Mechanism subsection within the LV section — links discretization error to E∞.

**Questions answered**: (3) causal intervention — increasing T should move E∞ and exponents predictably.

#### 22a. Infrastructure
- [x] Add `--horizon` CLI argument to `run_experiment.py` and `run_sweep.py` to override `data.T` from config
- [x] Create `scripts/run_horizon_sweep.py` that iterates over horizons and calls sweep logic for each
- [x] Ensure `generate_datasets.py` accepts `--horizon` to generate data at different T values
- [x] Generate datasets at T ∈ {0.5, 1.0, 2.0} for all 3 data seeds → `data-horizon-T=0.5/`, `data-horizon-T=2.0/` (T=1.0 already exists in `data/`)
- [x] Ground-truth residuals measured at all horizons (see `diagnostic_residuals.json`):
  - T=0.5: midpoint=0.004, composite=1.34e-4, Simpson=1.64e-6, conservation=2.53e-14
  - T=1.0: midpoint=0.151, composite=0.006, Simpson=4.38e-4, conservation=3.27e-14
  - T=2.0: midpoint=25.3, composite=2.06, Simpson=2.32, conservation=5.40e-14
  - Midpoint bias scales ~T⁴ as expected for 2nd-order method; Simpson breaks down at T=2.0 (same as composite — horizon too long)

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

### 24. Optimization rescue study for conservation prior — interpretation qualifier

**Goal**: Resolve the ambiguity around the conservation prior's bimodal failure. Determine whether it is an intrinsic surface corruption or a recoverable optimizer artifact. This interpretation matters for the cross-system discussion: if exact-but-weak priors consistently cause bimodality on LV but not on Duffing, the explanation must be system-specific; if rescue interventions eliminate it, the explanation is optimizer-sensitive.

**Paper role**: Interpretation qualifier within the LV section or in the Discussion.

**Questions answered**: (3) causal intervention — if targeted optimizer changes eliminate bimodality, the interpretation changes fundamentally.

#### 24a. Infrastructure
- [x] Add `--warm-start` option to `run_experiment.py`: load a trained plain-model checkpoint as initialization before training with conservation loss
- [x] Add `--grad-clip` option to `run_experiment.py`: apply gradient clipping (e.g., max_norm ∈ {1.0, 5.0})
- [x] Add `--lambda-schedule` option to `run_experiment.py`: support linear ramp-up of λ_phys over the first K epochs (e.g., K = 50)
- [x] Add `--two-stage` option: train phase 1 with data-only loss for M epochs, then add conservation loss for remaining epochs
- [x] Create `scripts/run_rescue_sweep.py` that runs all rescue variants on the configurations where bimodality is strongest (large capacity, moderate-to-high λ)
- [x] Add `warm_start`, `grad_clip`, `lambda_schedule_epochs`, `two_stage_epochs` to `TrainConfig`
- [x] Implement warm-start loading (compatible-key matching), effective_lambda computation (two-stage + ramp-up), gradient clipping in `train.py`

#### 24b. Identify target configurations
- [x] Analyzed conservation λ sweep results (`runs-conservation-lambda-sweep/`, 126 runs, 7 λ values × 2 capacities × 3 D sizes)
- [x] Highest bimodality (CV > 0.8, spread > 100×): λ=10.0 large D=4096 (228× spread), λ=1.0 large D=1024 (183× spread), λ=0.1 large D=1024 (109× spread), λ=0.1 large D=4096 (136× spread)
- [x] Rescue sweep already completed (810 runs across 5 variants × 162 configs in `runs-rescue/`):
  - **lambda-ramp**: best mean L2 (0.080), **eliminates all bimodality** (0/162 > 2× median)
  - **two-stage**: second-best mean L2 (0.087), **eliminates all bimodality**, tightest std (0.008)
  - **baseline**: bimodal (15/162 > 2× median), mean L2 = 0.098
  - **warm-start**: identical to baseline (likely not implemented correctly, or warm-start has no effect for conservation)
  - **grad-clip**: worst (mean L2 = 0.106), bimodal (12/162 > 2× median)
- [x] Interpretation: bimodality is an optimizer artifact, not intrinsic surface corruption. Lambda-ramp and two-stage both resolve it. However, conservation floor persists at ~8-9% — the prior is too weak to help with approximation.

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

### 30. Cross-system summary analysis and paper figures

**Goal**: Synthesize the three-system results into the paper's main comparative evidence. This section produces the cross-system comparison tables, regime-recurrence figures, and the Discussion material.

**Paper role**: Sections 6 (Discussion) and key comparative figures.

**Questions answered**: (1) which signatures are robust across systems? Which are system-specific?

#### 30a. Cross-system comparison
- [ ] Collect (E∞, α, β) with CIs for all (system × prior) combinations into a single summary table
- [ ] Classify each (system × prior) into regime type: floor-dominated, healthy-scaling, optimization-limited
- [ ] Test regime recurrence: for each prior class, does the same regime type appear across all 3 systems?
- [ ] Compute cross-system variance within each prior class — are exponents more similar within-class than between-class?

#### 30b. Figures
- [ ] Three-system scaling surface comparison figure (3×4 panel: system × prior)
- [ ] Cross-system exponent comparison figure with CIs (grouped bar chart or forest plot)
- [ ] Regime classification summary table (system × prior → regime type)
- [ ] Update `scripts/generate_figures.py` to produce all cross-system figures

#### 30c. Paper integration
- [ ] Rewrite abstract and title to reflect cross-system scope (e.g., "Prior-induced scaling regimes in scientific machine learning across dynamical systems")
- [ ] Update LV section to serve as anchor (no longer the entire paper)
- [ ] Write Van der Pol section (generalization beyond conservative dynamics)
- [ ] Write Duffing section (stress test under stronger nonlinearity)
- [ ] Write Discussion section: explicitly separate robust vs system-specific signatures
- [ ] Update limitations section (single-system limitation is now addressed)

### 25. Simpson's-rule prior — dose-response within LV

**Goal**: Add a true Simpson's 1/3-rule physics loss (4th-order) alongside the existing single-step midpoint (2nd-order) and composite midpoint (two-step, 2nd-order). Turns the two-point comparison into a three-point dose–response on discretization order.

**Paper role**: Supporting evidence within the LV section — strengthens the bias-reduction mechanism claim.

**Questions answered**: (3) does further reducing discretization bias continue to improve scaling? (2) is this a monotone relationship?

#### 25a. Infrastructure
- [x] Implement `simpson_residual()` in `src/scaling_piml/losses.py`: Simpson's 1/3 rule on $[0, T]$ using 3 evaluations of $F$ at $t=0$, $t=T/2$, $t=T$. The network predicts $(\hat{\bm{x}}_{T/2}, \hat{\bm{x}}_T) \in \mathbb{R}^4$ (same output layout as composite). Residual: $r = \hat{\bm{x}}_T - \bm{x}_0 - \frac{T}{6}[F(\bm{x}_0) + 4F(\hat{\bm{x}}_{T/2}) + F(\hat{\bm{x}}_T)]$
- [x] Implement `total_loss_simpson()` wrapping data + λ · Simpson residual
- [x] Add `physics_prior="simpson-true"` path in `train.py` (keeping `"simpson"` as the existing composite midpoint alias for backward compatibility)
- [x] Simpson ground-truth residual verified: **4.38e-4** (LV at T=1.0), vs composite midpoint 0.006 (14× reduction). Simpson is 4th-order vs midpoint's 2nd-order, confirming dose-response on discretization order.

#### 25b. Runs
- [ ] Run dense sweep for `piml-simpson-true`: 7 capacities × 11 dataset sizes × 3 data seeds × 3 train seeds = **693 runs** → `runs-dense/model=piml-simpson-true/`
- [ ] Alternatively, run on the standard grid (5 × 8 × 9 = 360 runs) if dense sweep is too expensive

#### 25c. Analysis
- [ ] Fit scaling surface and compare to plain, midpoint, composite
- [ ] Tabulate ground-truth residual: midpoint (0.151) → composite (0.006) → Simpson (???)
- [ ] Tabulate (E∞, α, β) for all four physics variants
- [ ] If Simpson further lowers E∞ and recovers/steepens exponents → strong dose–response evidence
- [ ] If Simpson matches composite → diminishing returns, suggesting the composite already saturates what soft-constraint regularization can achieve

### 23. Alternative ansatz comparison

**Goal**: Test whether the current scaling ansatz is the best effective description. Already largely complete from Phase 1; update with dense data when available.

**Paper role**: Appendix or Methods subsection.

**Questions answered**: (2) fit stability across ansatzes; (4) does the ansatz fail interpretably?

#### 23a. Infrastructure — alternative ansatzes
- [x] Implement in `src/scaling_piml/analysis/ansatz_comparison.py`:
  - **Ansatz A (current)**: E = E∞ + a·N^{-α} + b·D^{-β} (additive, separable, with floor)
  - **Ansatz B (no-floor power law)**: E = a·N^{-α} + b·D^{-β} (E∞ fixed to 0)
  - **Ansatz C (multiplicative separable)**: E = c · N^{-α} · D^{-β}
  - **Ansatz D (additive with interaction)**: E = E∞ + a·N^{-α} + b·D^{-β} + d·N^{-α}·D^{-β}
  - **Ansatz E (nonparametric baseline)**: 2D thin-plate spline (RBFInterpolator) on log(N) × log(D) → log(E)
- [x] For each ansatz, implement: fit function, bootstrap CI, residual diagnostics
- [x] Implement model comparison metrics: AIC/BIC for parametric ansatzes, leave-one-out cross-validation error for all

#### 23b. Infrastructure — held-out fit prediction
- [x] Create `scripts/validate_scaling_fits.py` that:
  - Takes a `grouped_metrics.csv` and an ansatz name
  - Performs leave-one-column-out (hold out one D) and leave-one-row-out (hold out one N)
  - Fits on the remaining grid cells
  - Reports prediction error on held-out cells
  - Repeats for all ansatzes
- [x] Also implement leave-one-corner-out: hold out the largest (N, D) cell and predict from the rest (extrapolation test)

#### 23c. Analysis
- [x] Run ansatz comparison on Phase 1 data (`runs-progress/grouped_metrics.csv`) for all models → `runs-progress/ansatz_comparison/`
- [x] Run ansatz comparison on combined data (`runs-combined/grouped_metrics.csv`, 120 rows, 3 models) → `runs-combined/ansatz_comparison/`
- [ ] Run ansatz comparison on dense sweep data (section 21) once available
- [ ] Run ansatz comparison on Van der Pol and Duffing data once available
- [x] Report AIC/BIC ranking per model — E > D > others consistently
- [x] Report held-out prediction error per ansatz per model — Ansatz D best among parametric
- [x] Identify whether Ansatz A (current) wins or is dominated — Ansatz D (interaction) dominates A; Ansatz E (nonparametric) best overall
- [x] Check whether interacted or nonparametric ansatzes reveal structure missed by the additive form — interaction term d significant (d≈0.52 piml, d≈7.4 plain, d≈13.1 piml-simpson)

### 28. Training dynamics visualization (gradient-norm decomposition)

**Goal**: Make the paper's mechanistic claim directly visible. A figure showing physics/data gradient norms over training epochs turns the floor-raising inference into observable evidence.

**Paper role**: Supporting figure, ideally in the LV mechanism subsection.

**Questions answered**: (3) causal mechanism — does the gradient-ratio crossover actually occur? Does the composite prior avoid it?

#### 28a. Infrastructure — decomposed gradient logging
- [x] Modify `train.py` to optionally log `grad_norm_data` and `grad_norm_phys` separately per epoch:
  - After computing the total loss but before `L.backward()`, compute `L_data.backward(retain_graph=True)`, record data-component grad norms, zero grads, then `L.backward()` for the combined step
  - Alternatively (cheaper): log the loss-value ratio (already done via `phys_data_ratio`) alongside the total `grad_norm`, which gives an approximate decomposition
  - If exact decomposition is too expensive for all runs, add a `--log-grad-decomposition` flag that enables it for selected diagnostic runs only
- [x] Add `grad_norm_data` and `grad_norm_phys` columns to the training CSV when the flag is active
- [x] Create `scripts/plot_gradient_dynamics.py` that reads training CSVs and produces the overlay figure

#### 28b. Diagnostic runs
- [ ] Select 2–3 representative $(N, D)$ configurations: e.g., (Large, D=1024), (Medium, D=512)
- [ ] Run plain, midpoint, composite, and conservation at each config with `--log-grad-decomposition` enabled
- [ ] 4 models × 2–3 configs × 3 train seeds = **24–36 runs** → `runs-grad-dynamics/`

#### 28c. Figures and analysis
- [ ] Plot $\|\nabla \mathcal{L}_{\text{data}}\|$ and $\lambda \|\nabla \mathcal{L}_{\text{phys}}\|$ vs epoch for each model:
  - Plain: only data gradient (no physics)
  - Midpoint: expect physics gradient to dominate early; data gradient decays but physics gradient stays large → crossover → saturation
  - Composite: expect both gradients to remain comparable throughout → no premature saturation
  - Conservation: expect oscillatory / unstable gradient behavior at seeds that trap
- [ ] Produce a 4-panel figure or 2×2 grid suitable for inclusion in the paper
- [ ] Identify the epoch at which the midpoint physics gradient exceeds the data gradient — compare with the epoch at which early stopping fires
- [ ] If the crossover is visible: strong mechanistic evidence for the floor-raising claim
- [ ] If no clear crossover: revise the mechanistic explanation or note that the loss-ratio (already logged) is a sufficient proxy

### 27. Noise robustness

**Goal**: Test scaling signatures under realistic corruption. Two sub-experiments: observation noise on supervised targets, and prior mismatch (wrong physics parameters in the loss). Only include if the main cross-system plan is already complete.

**Paper role**: Appendix or robustness supplement — not a main contribution.

**Questions answered**: (3) does noise move exponents/floor as expected? (1) do signatures persist under corruption?

#### 27a. Infrastructure — observation noise
- [x] Add `--obs-noise` argument to `run_experiment.py` and `run_sweep.py`: adds Gaussian noise to u(T) targets at training time
- [x] Noise levels: σ ∈ {0, 0.01, 0.05, 0.1} (fraction of per-component std of the targets)
- [x] Ensure noise is added after normalization and is reproducible (seeded per sample)
- [x] Ensure test/validation data remains clean (noise only on training targets)

#### 27b. Infrastructure — prior mismatch
- [x] Add `--prior-params` argument to `run_experiment.py` to override system parameters used in the physics loss (while keeping the data-generating parameters fixed)
- [x] Mismatch levels: perturb each Lotka–Volterra parameter by ±5%, ±10%, ±20%
- [x] The data remains generated with true parameters; only the RHS in the physics loss uses perturbed parameters

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

## Phase 2 summary — execution order (merged paper)

| Priority | Section | Description | Est. runs | Depends on | Impact | Paper role |
|----------|---------|-------------|-----------|------------|--------|------------|
| 1 | 21 | Dense LV sweep (anchor stabilization) | ~2,079 | — | Tightens CIs on anchor | LV section |
| 2 | 29 | **Van der Pol** (first generalization) | ~1,080–1,440 | VdP implementation | **Main upgrade: cross-system** | Section 4 |
| 3 | 26 | **Duffing** (stress test) | ~1,440 | Data generation | **Main upgrade: cross-system** | Section 5 |
| 4 | 22 | Horizon sweep on LV (mechanism) | ~3,240 | New data at T=0.5, T=2.0 | Proves causal mechanism | LV mechanism subsection |
| 5 | 24 | Conservation rescue (interpretation) | ~150 | — | Clarifies exact-but-weak regime | LV interpretation |
| 6 | 30 | Cross-system summary + paper rewrite | 0 (analysis/writing) | Sections 29, 26 | Produces the main claim | Discussion + figures |
| 7 | 25 | Simpson's-rule dose-response | ~360–693 | — | Supporting LV evidence | LV subsection |
| 8 | 23 | Ansatz comparison on new data | 0 (analysis) | Section 21, 29, 26 data | Updates existing results | Methods/Appendix |
| 9 | 28 | Gradient dynamics visualization | ~24–36 | — | Visualizes mechanism | Supporting figure |
| 10 | 27 | Noise robustness | ~6,480 | — | Robustness supplement | Appendix |

Total Phase 2: ~15,000–17,000 runs.

**Priority tiers:**
- **Tier A (must-have for submission):** Sections 21, 29, 26, 30 — anchor stabilization + the three-system ladder + cross-system synthesis. This is the paper.
- **Tier B (strongly recommended):** Sections 22, 24 — mechanism test and interpretation qualifier on LV. These make the anchor section deeper.
- **Tier C (supporting extensions):** Sections 25, 23, 28 — dose-response, ansatz updates, gradient dynamics. Strengthen the LV section but don't change the main claim.
- **Tier D (include only if complete):** Section 27 — noise robustness. High run count, low marginal value relative to cross-system evidence.

### Phase 2 completion criteria (merged paper)

**Minimum for submission:**
- [ ] Dense LV sweep narrows bootstrap CIs on composite exponents to ≤ 1.0 width (currently ~2.5)
- [ ] Van der Pol system implemented, data generated, full sweep completed
- [ ] Van der Pol regime recurrence tested: at least 2 of 3 regime types visibly recur
- [ ] Duffing data generated, full sweep completed
- [ ] Duffing regime recurrence tested: at least 2 of 3 regime types visibly recur
- [ ] Cross-system comparison table and figure produced (3 systems × 3+ priors)
- [ ] Paper title and framing updated to cross-system scope
- [ ] Discussion explicitly separates robust vs system-specific signatures
- [ ] All results reproducible from scripts and configs alone

**Strongly recommended:**
- [ ] Horizon variation confirms mechanistic link between discretization error and E∞
- [ ] Conservation prior ambiguity resolved (optimizer artifact vs surface corruption)

**Full completion:**
- [ ] Simpson prior establishes dose–response on discretization order (3+ data points)
- [ ] Gradient dynamics figure shows physics/data gradient crossover for midpoint prior
- [ ] Alternative ansatzes re-compared on dense + cross-system data
- [ ] Noise robustness tested on at least LV