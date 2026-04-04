
# Implementation Checklist

Status is synced to the current repository state as of 2026-04-04. Checked items are implemented or directly validated in the repo; unchecked items are still missing, not automated yet, or not fully verified.

## 1. Project setup
- [x] Create repository structure
- [x] Add `pyproject.toml` or `requirements.txt`
- [x] Add config system for data, model, train, and experiments
- [x] Add reproducible seeding utilities
- [x] Add logging / artifact saving utilities

## 2. Core system
- [x] Implement Lotka–Volterra RHS function
- [x] Implement ODE solve helper using `solve_ivp`
- [x] Expose system parameters through config
- [x] Add simple validation test for solver output
- [x] Validate solver accuracy: max relative error ~1.9e-9 vs tighter tolerances
- [x] Validate conservation law: max |ΔH|/|H| ~2.0e-9 on ground-truth data

## 3. Dataset generation
- [x] Implement initial condition sampler
- [x] Generate train / val / test splits for one data seed
- [x] Save frozen splits to disk
- [x] Implement nested train subsets for all `D`
- [x] Compute normalization stats from full train pool only
- [x] Save normalization stats to disk
- [x] Repeat for data seeds `11`, `22`, `33` (all 3 generated under `data/`)

## 4. Data loading
- [x] Implement dataset class for frozen splits
- [x] Implement subset loading by dataset size `D`
- [x] Apply saved normalization consistently
- [x] Verify no leakage from val/test into normalization
- [x] Verify normalization roundtrip precision: max error ~1e-7 in float32
- [x] Verify positivity of all inputs and outputs (required for conservation log terms)
- [x] Verify flow map injectivity: 0 suspicious pairs (T < period guarantees diffeomorphism)

## 5. Models
- [x] Implement plain MLP
- [x] Implement configurable hidden layer sizes
- [x] Implement parameter count utility
- [x] Define default capacity grid
- [x] Verify forward output shape is correct

## 6. Losses
- [x] Implement data loss (MSE in normalized space)
- [x] Implement Lotka–Volterra midpoint-rule physics residual
- [x] Implement conservation-law physics loss (H invariant)
- [x] Implement total loss with `lambda_phys` for midpoint
- [x] Implement total loss with `lambda_phys` for conservation
- [x] Verify physics-informed model uses same architecture as plain MLP
- [x] Verify conservation loss on ground truth: ~3.4e-14 (exact to machine precision)
- [x] Verify midpoint residual on ground truth: ~0.151 (15% irreducible bias at T=1.0)

## 7. Training
- [x] Implement training loop
- [x] Implement validation loop
- [x] Implement early stopping
- [x] Log train / val metrics each epoch
- [x] Log grad_norm and phys_data_ratio per epoch
- [x] Save best model checkpoint
- [x] Save training history CSV
- [x] Save final metrics JSON
- [x] Support `physics_prior` parameter: `"none"`, `"midpoint"`, `"conservation"`

## 8. Evaluation
- [x] Implement relative L2 metric (per-sample, physical space)
- [x] Implement MSE metric
- [x] Evaluate on test set
- [ ] Save test predictions optionally
- [x] Flag diverged / NaN runs

## 9. CLI / scripts
- [x] Script to generate datasets (`scripts/generate_datasets.py`)
- [x] Script to run one experiment (`scripts/run_experiment.py`)
- [x] Script to run a sweep (`scripts/run_sweep.py`)
- [x] Script to aggregate run results (`scripts/aggregate_runs.py`)
- [x] Script to fit scaling curves (`scripts/fit_scaling.py`)
- [x] Script to run lambda sweep (`scripts/run_lambda_sweep.py`, supports `--model piml|piml-conservation`)
- [x] Script to analyze lambda sweep (`scripts/analyze_lambda_sweep.py`)
- [x] Script to diagnose physics loss (`scripts/diagnose_physics.py`)
- [ ] Script to generate figures (`scripts/generate_figures.py` — exists but needs updates)

## 10. Sanity checks
- [x] Single-run convergence test
- [ ] Tiny-set overfit test with `D=32`
- [x] Pilot monotonicity check for error vs `D`
- [x] Pilot monotonicity check for error vs `N`
- [x] Check scale of physics loss vs data loss (midpoint: ratio ~1.2× at init)
- [x] Check exact reproducibility with same seed (λ=0 PIML ≡ plain, bit-for-bit)

## 11. Pilot sweep
- [x] Run 2 models × 2 capacities × 3 dataset sizes × 1 data seed × 2 train seeds (24 runs)
- [x] Inspect logs and saved metrics
- [x] Inspect capacity scaling trend
- [x] Inspect data scaling trend
- [x] Fix only genuine bugs before full sweep

## 12. Full experiment
- [x] Run full target matrix: 2 models × 5 capacities × 8 dataset sizes × 3 data seeds × 3 train seeds = **720 runs** (plain + midpoint PIML)
- [x] Confirm all runs save artifacts correctly
- [x] Track divergence rate: **0% divergence** across all 720 runs
- [x] Verify parameter counts and dataset sizes are logged correctly

## 13. Aggregation
- [x] Aggregate by `(model, physics_prior, N, D)`
- [x] Compute mean, std, stderr
- [x] Compute divergence rate
- [x] Export aggregated results table (`runs-progress/runs_aggregate.csv`, `grouped_metrics.csv`)

## 14. Scaling analysis
- [x] Fit capacity scaling curves at fixed `D`
- [x] Fit data scaling curves at fixed `N`
- [x] Fit full `E(N,D)` surface
- [x] Bootstrap confidence intervals (1000 samples)
- [x] Exclude unstable points from fitting when divergence > 30%
- [x] Save fit summaries to disk (`runs-progress/scaling_fits.json`)
- Plain: E∞ ≈ 0, α ≈ 0.91, β ≈ 0.85
- Midpoint PIML: E∞ ≈ 0.011, α ≈ 0.16, β ≈ 0.53

## 15. Lambda sweeps
- [x] Midpoint lambda sweep: 6 λ values × 2 capacities × 3 dataset sizes × 3 data seeds × 3 train seeds = **324 runs**
- [x] Conservation lambda sweep: 7 λ values × 2 capacities × 3 dataset sizes × 1 data seed × 3 train seeds = **126 runs**
- [x] Midpoint result: no λ improves over plain; monotonic degradation above λ=1e-2
- [x] Conservation result: bimodal failure (good convergence or trap at ~0.09); large model highly susceptible; no λ consistently improves

## 16. Data validation
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

## 17. Figures
- [ ] Task setup schematic
- [ ] Capacity scaling figure
- [ ] Data scaling figure
- [ ] 2D error heatmap / contour
- [ ] Exponent comparison with CI
- [ ] Stability summary figure

## 18. Paper draft
- [x] Created `paper/draft.tex` (compiles cleanly)
- [x] Plain and midpoint results filled in
- [ ] Conservation results still pending (marked with `\pending{}`)
- [ ] Figures still placeholder (marked with `\status{}`)

## 19. Final validation
- [ ] Confirm figures regenerate from scripts only
- [x] Confirm all configs are saved per run
- [x] Confirm no notebook-only results are required
- [ ] Confirm README matches actual implementation
- [ ] Confirm minimal success criterion is met

## 20. Minimal success criterion
- [x] Two clean scaling plots exist (plain + midpoint, data in `runs-progress/`)
- [x] Exponent estimates with uncertainty exist (`runs-progress/scaling_fits.json`)
- [x] Stability comparison exists (0% divergence both models)
- [x] Clear conclusion can be stated: midpoint physics prior **degrades** scaling (raises E∞ 20×, flattens α from 0.91→0.16, flattens β from 0.85→0.53) due to 15% irreducible approximation error at T=1.0. Conservation prior exhibits bimodal optimization failure (trapping at ~0.09 for large models). Neither prior improves over the plain baseline at any tested configuration.