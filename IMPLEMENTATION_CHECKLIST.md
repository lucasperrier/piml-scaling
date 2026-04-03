
`IMPLEMENTATION_CHECKLIST.md`
```markdown
# Implementation Checklist

## 1. Project setup
- [ ] Create repository structure
- [ ] Add `pyproject.toml` or `requirements.txt`
- [ ] Add config system for data, model, train, and experiments
- [ ] Add reproducible seeding utilities
- [ ] Add logging / artifact saving utilities

## 2. Core system
- [ ] Implement Lotka–Volterra RHS function
- [ ] Implement ODE solve helper using `solve_ivp`
- [ ] Expose system parameters through config
- [ ] Add simple validation test for solver output

## 3. Dataset generation
- [ ] Implement initial condition sampler
- [ ] Generate train / val / test splits for one data seed
- [ ] Save frozen splits to disk
- [ ] Implement nested train subsets for all `D`
- [ ] Compute normalization stats from full train pool only
- [ ] Save normalization stats to disk
- [ ] Repeat for data seeds `11`, `22`, `33`

## 4. Data loading
- [ ] Implement dataset class for frozen splits
- [ ] Implement subset loading by dataset size `D`
- [ ] Apply saved normalization consistently
- [ ] Verify no leakage from val/test into normalization

## 5. Models
- [ ] Implement plain MLP
- [ ] Implement configurable hidden layer sizes
- [ ] Implement parameter count utility
- [ ] Define default capacity grid
- [ ] Verify forward output shape is correct

## 6. Losses
- [ ] Implement data loss
- [ ] Implement Lotka–Volterra physics residual
- [ ] Implement total loss with `lambda_phys`
- [ ] Verify physics-informed model uses same architecture as plain MLP

## 7. Training
- [ ] Implement training loop
- [ ] Implement validation loop
- [ ] Implement early stopping
- [ ] Log train / val metrics each epoch
- [ ] Save best model checkpoint
- [ ] Save training history CSV
- [ ] Save final metrics JSON

## 8. Evaluation
- [ ] Implement relative L2 metric
- [ ] Implement MSE metric
- [ ] Evaluate on test set
- [ ] Save test predictions optionally
- [ ] Flag diverged / NaN runs

## 9. CLI / scripts
- [ ] Script to generate datasets
- [ ] Script to run one experiment
- [ ] Script to run a sweep
- [ ] Script to aggregate run results
- [ ] Script to fit scaling curves
- [ ] Script to generate figures

## 10. Sanity checks
- [ ] Single-run convergence test
- [ ] Tiny-set overfit test with `D=32`
- [ ] Pilot monotonicity check for error vs `D`
- [ ] Pilot monotonicity check for error vs `N`
- [ ] Check scale of physics loss vs data loss
- [ ] Check exact reproducibility with same seed

## 11. Pilot sweep
- [ ] Run 2 models × 2 capacities × 3 dataset sizes × 1 data seed × 2 train seeds
- [ ] Inspect logs and saved metrics
- [ ] Inspect capacity scaling trend
- [ ] Inspect data scaling trend
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
- [ ] Confirm all configs are saved per run
- [ ] Confirm no notebook-only results are required
- [ ] Confirm README matches actual implementation
- [ ] Confirm minimal success criterion is met

## 17. Minimal success criterion
- [ ] Two clean scaling plots exist
- [ ] Exponent estimates with uncertainty exist
- [ ] Stability comparison exists
- [ ] Clear conclusion can be stated:
      physics prior affects data efficiency, capacity efficiency, error floor, and/or stability