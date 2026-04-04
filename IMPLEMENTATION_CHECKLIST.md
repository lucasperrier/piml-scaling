
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

## 3. Dataset generation
- [x] Implement initial condition sampler
- [x] Generate train / val / test splits for one data seed
- [x] Save frozen splits to disk
- [x] Implement nested train subsets for all `D`
- [x] Compute normalization stats from full train pool only
- [x] Save normalization stats to disk
- [ ] Repeat for data seeds `11`, `22`, `33`

## 4. Data loading
- [x] Implement dataset class for frozen splits
- [x] Implement subset loading by dataset size `D`
- [x] Apply saved normalization consistently
- [x] Verify no leakage from val/test into normalization

## 5. Models
- [x] Implement plain MLP
- [x] Implement configurable hidden layer sizes
- [x] Implement parameter count utility
- [x] Define default capacity grid
- [x] Verify forward output shape is correct

## 6. Losses
- [x] Implement data loss
- [x] Implement Lotka–Volterra physics residual
- [x] Implement total loss with `lambda_phys`
- [x] Verify physics-informed model uses same architecture as plain MLP

## 7. Training
- [x] Implement training loop
- [x] Implement validation loop
- [x] Implement early stopping
- [x] Log train / val metrics each epoch
- [x] Save best model checkpoint
- [x] Save training history CSV
- [x] Save final metrics JSON

## 8. Evaluation
- [x] Implement relative L2 metric
- [x] Implement MSE metric
- [x] Evaluate on test set
- [ ] Save test predictions optionally
- [x] Flag diverged / NaN runs

## 9. CLI / scripts
- [x] Script to generate datasets
- [x] Script to run one experiment
- [x] Script to run a sweep
- [x] Script to aggregate run results
- [ ] Script to fit scaling curves
- [ ] Script to generate figures

## 10. Sanity checks
- [x] Single-run convergence test
- [ ] Tiny-set overfit test with `D=32`
- [x] Pilot monotonicity check for error vs `D`
- [x] Pilot monotonicity check for error vs `N`
- [ ] Check scale of physics loss vs data loss
- [ ] Check exact reproducibility with same seed

## 11. Pilot sweep
- [x] Run 2 models × 2 capacities × 3 dataset sizes × 1 data seed × 2 train seeds
- [x] Inspect logs and saved metrics
- [x] Inspect capacity scaling trend
- [x] Inspect data scaling trend
- [ ] Fix only genuine bugs before full sweep

## 12. Full experiment
- [ ] Run reduced acceptable matrix or full target matrix
- [ ] Confirm all runs save artifacts correctly
- [ ] Track divergence rate
- [ ] Verify parameter counts and dataset sizes are logged correctly

## 13. Aggregation
- [ ] Aggregate by `(model, N, D)`
- [ ] Compute mean, std, stderr
- [ ] Compute divergence rate
- [ ] Export aggregated results table

## 14. Scaling analysis
- [ ] Fit capacity scaling curves at fixed `D`
- [ ] Fit data scaling curves at fixed `N`
- [ ] Fit full `E(N,D)` surface
- [ ] Bootstrap confidence intervals
- [ ] Exclude unstable points from fitting when divergence > 30%
- [ ] Save fit summaries to disk

## 15. Figures
- [ ] Task setup schematic
- [ ] Capacity scaling figure
- [ ] Data scaling figure
- [ ] 2D error heatmap / contour
- [ ] Exponent comparison with CI
- [ ] Stability summary figure

## 16. Final validation
- [ ] Confirm figures regenerate from scripts only
- [x] Confirm all configs are saved per run
- [x] Confirm no notebook-only results are required
- [ ] Confirm README matches actual implementation
- [ ] Confirm minimal success criterion is met

## 17. Minimal success criterion
- [ ] Two clean scaling plots exist
- [ ] Exponent estimates with uncertainty exist
- [ ] Stability comparison exists
- [ ] Clear conclusion can be stated: physics prior affects data efficiency, capacity efficiency, error floor, and/or stability