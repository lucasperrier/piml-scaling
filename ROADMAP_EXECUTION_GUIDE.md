# Execution Guide: Task-Alignment Roadmap

## Overview

This guide covers the practical steps to complete the task-alignment experiments from `ROADMAP_TASK_ALIGNMENT.md`, from branch setup through final results. The core question: does the midpoint prior help when the supervised task is aligned (one-step at small Δt) even though it hurts on the coarse flow-map (T=1.0)?

---

## 1. Branch & Environment Setup

```bash
# Create feature branch from current main
cd /home/lucas-perrier/Documents/projects/scaling-piml
git checkout -b task-alignment

# Verify environment
source .venv/bin/activate
python -c "import scaling_piml; print('OK')"
```

Keep `main` untouched for Task 21 (dense LV sweep). All roadmap work goes on `task-alignment`.

---

## 2. GPU Strategy

### Why cheap GPUs are fine

Your models are tiny (largest = 132K params, uses <1MB VRAM). An A100 is wildly overkill — the bottleneck is Python/PyTorch overhead, not GPU compute. Any 24GB card will train at effectively the same speed.

### Recommended GPU (RunPod Community Cloud, April 2026 prices)

| GPU | VRAM | $/hr | Notes |
|-----|------|------|-------|
| **RTX A5000** | 24 GB | **$0.16** | Cheapest. First choice if available. |
| **RTX 3090** | 24 GB | **$0.22** | Good availability. Reliable fallback. |
| RTX 4090 | 24 GB | $0.34 | Faster but no practical difference for tiny MLPs |
| A40 | 48 GB | $0.35 | Overkill VRAM |
| A100 PCIe | 80 GB | $1.19 | **7× the cost, no speed benefit** — avoid |

**Pick RTX A5000 or RTX 3090.** Do not rent anything more expensive.

### Parallel execution (optional, cuts wall-clock time 4–8×)

Each model uses <1MB VRAM. You can run multiple training jobs concurrently on one GPU:

```bash
# Example: run 4 jobs in parallel using GNU parallel
cat run_commands.txt | parallel -j 4
```

Or modify `run_sweep.py` to use Python `multiprocessing`. This is optional but reduces a 3-hour pilot to ~45 minutes.

---

## 3. Phased Experiment Plan

### Do NOT run the full 14,400-run matrix upfront.

The roadmap's Tier 1 matrix (14,400 runs) is designed for bulletproof statistics. You don't need that to confirm the hypothesis. Follow this escalation ladder:

### Phase A: Sanity Check (4 runs, ~8 min, ~$0.03)

Confirm the alignment effect exists at a single operating point before investing further.

| Run | Task | Δt / T | Prior | Capacity | D |
|-----|------|--------|-------|----------|---|
| S1 | onestep | 0.05 | plain | medium | 2048 |
| S2 | onestep | 0.05 | midpoint | medium | 2048 |
| S3 | flowmap | T=1.0 | plain | medium | 2048 |
| S4 | flowmap | T=1.0 | midpoint | medium | 2048 |

**Decision rule:** If S2 < S1 (midpoint helps on one-step) AND S4 > S3 (midpoint hurts on flow-map), the hypothesis is confirmed at one point. Proceed to Phase B.

If S2 ≥ S1, stop and diagnose before spending more. Check normalization, loss computation, Δt value.

### Phase B: Lean Pilot (96 runs, ~3h, ~$0.51 on A5000)

Establish scaling curves at two Δt values.

| Axis | Values |
|------|--------|
| System | LV only |
| Task | onestep |
| Δt | 0.05, 0.1 |
| Prior | plain, midpoint |
| Capacity | small, medium, large |
| Dataset size | 128, 512, 2048, 8192 |
| Data seed | 11 |
| Train seeds | 101, 202 |

**Total: 2 × 2 × 3 × 4 × 1 × 2 = 96 runs**

Also run the matching flow-map grid (already available from existing runs, or add 48 runs for plain+midpoint × 3 caps × 4 D × 2 seeds).

**Decision rule:** Fit log-log data-scaling slopes for each (task, Δt, prior). If midpoint slope on one-step is steeper than midpoint slope on flow-map (ideally close to plain slope), proceed to Phase C. Plot the hero figure (Figure 1 from roadmap).

### Phase C: Modest Confirmation (648 runs, ~22h, ~$4.75 on 3090)

Add Δt variation, trapezoidal prior, and more seeds.

| Axis | Values |
|------|--------|
| System | LV only |
| Task | onestep |
| Δt | 0.05, 0.1, 0.2 |
| Prior | plain, midpoint, trapezoidal |
| Capacity | small, medium, large |
| Dataset size | 64, 128, 512, 2048, 4096, 8192 |
| Data seeds | 11, 22 |
| Train seeds | 101, 202 |

**Total: 3 × 3 × 3 × 6 × 2 × 2 = 648 runs**

**Decision rule:** This is enough for the Δt transition figure (Figure 2) and a preliminary scaling exponent comparison (Figure 3). If the alignment effect is clear and monotonic across Δt, you have a workshop paper. Decide whether to extend to Duffing.

### Phase D: Cross-System + Full Statistics (→ conference paper)

Scale up to Duffing, add conservation prior, add all seeds and capacities.

| Block | Runs | GPU-hours | Cost (3090) |
|-------|------|-----------|-------------|
| LV full one-step grid | ~3,500 | ~117h | ~$26 |
| Duffing one-step grid | ~3,500 | ~117h | ~$26 |
| Rollout evaluation (inference only) | ~3,500 | ~10h | ~$2 |
| **Phase D total** | **~10,500** | **~244h** | **~$54** |

### Total cost to conference paper

| Phase | Runs | Cost (RTX 3090) |
|-------|------|-----------------|
| A: Sanity check | 4 | $0.03 |
| B: Lean pilot | 96 | $0.70 |
| C: Modest confirmation | 648 | $4.75 |
| D: Cross-system + full | ~10,500 | ~$54 |
| **Total** | **~11,248** | **~$60** |

Compare to the roadmap's original estimate of $960 on A100 — **16× cheaper** by using the right GPU and phased execution.

---

## 4. Implementation Steps

Complete these on the `task-alignment` branch. All are CPU-side work (no GPU needed).

### Step 1: Trajectory Data Generation (~2h dev)

Add `solve_trajectory()` to `src/scaling_piml/systems/ode.py`:

```python
def solve_trajectory(rhs, u0, *, T, dt, method="DOP853", rtol=1e-9, atol=1e-11):
    t_eval = np.linspace(0, T, int(round(T / dt)) + 1)
    sol = solve_ivp(rhs, (0, T), u0, t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    return sol.t, sol.y.T
```

Create `scripts/generate_trajectory_datasets.py`:
- For each IC, solve full trajectory at `dt_fine=0.001`, `T=5.0`
- Save `trajectories.npy` (N, K, d) and `times.npy` (K,)
- Reuse existing IC sampling and seed logic from `data/generate.py`

Generate for LV seeds 11, 22, 33 → `data-trajectories/lotka-volterra/data_seed={11,22,33}/`

### Step 2: OneStepDataset (~2h dev)

Create `src/scaling_piml/data/onestep_dataset.py`:
- Load trajectory data, extract `(x_t, x_{t+Δt})` pairs via strided indexing
- `D` controls number of *trajectories* used (not time-step pairs) — keeps D interpretation consistent with flow-map
- Normalize using training-fold statistics

### Step 3: Wire into Training (~1h dev)

Minimal changes to `scripts/run_sweep.py` and `src/scaling_piml/train.py`:
- Add `--task` flag: `"flowmap"` (default) or `"onestep"`
- Add `--dt` flag for one-step task
- If `task == "onestep"`, use `OneStepDataset` instead of `FlowMapDataset`
- The midpoint loss formula is identical — just with `T = Δt` instead of `T = 1.0`
- Run directory gains task/dt path components: `runs-aligned/task=onestep/dt=0.1/model=piml/...`

### Step 4: Trapezoidal Prior (~30min dev)

Add to `src/scaling_piml/losses.py`:

```python
def trapezoidal_residual(u0, uT_hat, F, T):
    return uT_hat - u0 - (T / 2) * (F(u0) + F(uT_hat))
```

Wire into loss dispatch with `physics_prior = "trapezoidal"`.

### Step 5: Validation (~30min)

- Run existing flow-map experiment through new code path with `--task flowmap` — verify identical `metrics.json`
- Generate one-step data for LV seed 11, extract pairs at Δt=1.0 — verify they match existing flow-map targets to solver tolerance

---

## 5. Running the Experiments

### RunPod Pod Setup

```bash
# 1. Rent RTX 3090 community pod ($0.22/hr)
# 2. Select a PyTorch template (e.g., runpod/pytorch:2.1.0-py3.10-cuda12.1.0)
# 3. Clone repo
git clone <your-repo-url>
cd scaling-piml
git checkout task-alignment

# 4. Install
pip install -e .
pip install -r requirements.txt

# 5. Generate trajectory data (CPU, ~5 min)
python scripts/generate_trajectory_datasets.py \
  --system lotka-volterra \
  --seeds 11,22,33 \
  --T 5.0 --dt-fine 0.001 \
  --out data-trajectories/lotka-volterra

# 6. Run sanity check
python scripts/run_sweep.py \
  --task onestep --dt 0.05 \
  --config configs/default.yaml \
  --system lotka-volterra \
  --data-dir data-trajectories/lotka-volterra \
  --out runs-aligned \
  --models plain,piml \
  --capacities medium \
  --dataset-sizes 2048 \
  --data-seeds 11 \
  --train-seeds 101

# 7. Compare S1 vs S2 (check metrics.json in each run dir)
```

### Sweep Commands (Phase B)

```bash
# One-step experiments
for DT in 0.05 0.1; do
  python scripts/run_sweep.py \
    --task onestep --dt $DT \
    --config configs/default.yaml \
    --system lotka-volterra \
    --data-dir data-trajectories/lotka-volterra \
    --out runs-aligned \
    --models plain,piml \
    --capacities small,medium,large \
    --dataset-sizes 128,512,2048,8192 \
    --data-seeds 11 \
    --train-seeds 101,202
done
```

### Aggregation

```bash
python scripts/aggregate_runs.py --root runs-aligned
python scripts/fit_scaling.py --root runs-aligned
```

The aggregate script needs a minor extension to extract `task_name` and `dt` from the run directory path and include them as columns.

---

## 6. Analysis & Figures

After each phase, generate the key plots:

### After Phase A (sanity check)
- Just compare 4 numbers manually. Print `test_rel_l2` from each `metrics.json`.

### After Phase B (lean pilot)
- **Hero figure**: 2×1 panel. Left = flow-map scaling, right = one-step scaling. Plain vs midpoint curves on each.
- **Scaling slopes table**: Log-log slope of `test_rel_l2` vs D at large capacity for each (task, prior).

### After Phase C (modest confirmation)
- **Δt transition figure**: Fixed capacity, test error vs D, one curve per Δt + flow-map T=1.0.
- **Exponent comparison**: Bar chart of fitted β per (task, Δt, prior) with bootstrap CIs (if enough seeds).

### After Phase D (full)
- All 7 figures from roadmap Section 4
- Cross-system taxonomy table

---

## 7. Paper Integration

The roadmap results extend rather than replace the current paper.

### Where they go in draft.tex

Insert a new section between the current cross-system comparison (Section 8) and Discussion (Section 9):

```
\section{Results: Task Alignment}
\subsection{One-step task formulation}
\subsection{Prior alignment at small Δt}
\subsection{Δt as an alignment knob}
\subsection{Cross-system replication (if Phase D complete)}
```

### Title update

Current: "Prior-Induced Scaling Regimes in Scientific Machine Learning: A Cross-System Study"

Updated: "Prior-Induced Scaling Regimes in Scientific Machine Learning: Task Alignment Across Dynamical Systems"

### Abstract update

Add one sentence: "We further show that the same midpoint prior that raises error floors on coarse flow-map prediction recovers healthy scaling when the supervised task is aligned — predicting one-step evolution at small Δt — establishing task alignment as the missing variable in prior evaluation."

### Extended taxonomy

The current 3-type taxonomy (floor-dominated, partial recovery, system-dependent structural) gains two new types:
- **Type 2: Aligned strong** — midpoint on one-step at small Δt
- **Type 5: Locally aligned, globally fragile** — if rollout degrades despite good one-step error

---

## 8. Timeline

| Week | Activity | Compute needed |
|------|----------|---------------|
| 1 | Implement Steps 1–5 (all CPU-side dev) | None |
| 1 | Phase A sanity check | 8 min GPU |
| 1–2 | Phase B lean pilot + first figures | 3h GPU |
| 2–3 | Phase C modest confirmation | 22h GPU |
| 3–4 | Write task-alignment section for paper | None |
| 4–6 | Phase D cross-system (if pursuing conference) | 244h GPU |
| 6–8 | Final figures, paper revision | None |

Total GPU rental for workshop-ready result (Phases A–C): **~25h × $0.22 = $5.50**

---

## 9. Risk Checkpoints

After each phase, evaluate before spending more:

| Checkpoint | Question | If no |
|------------|----------|-------|
| After Phase A | Does midpoint help on one-step (S2 < S1)? | Stop. Diagnose normalization, loss, Δt. Try Δt=0.01. |
| After Phase B | Are scaling slopes steeper for midpoint on one-step vs flow-map? | Run λ-sweep on one-step midpoint. If λ doesn't help, the prior may be genuinely uninformative — still publishable as stronger negative result. |
| After Phase C | Is the Δt transition monotonic? | Check for optimization issues at intermediate Δt. May indicate non-smooth alignment boundary. |
| After Phase D | Does the effect replicate on Duffing? | Paper is still interesting (alignment is system-dependent). Frame accordingly. |

---

## 10. Checklist

```
[ ] Create task-alignment branch
[ ] Implement solve_trajectory() in systems/ode.py
[ ] Create generate_trajectory_datasets.py
[ ] Generate LV trajectory data (seeds 11, 22, 33)
[ ] Implement OneStepDataset
[ ] Add --task/--dt flags to run_sweep.py and train.py
[ ] Implement trapezoidal prior in losses.py
[ ] Validate: flowmap path produces identical results
[ ] Validate: one-step at Δt=1.0 matches flow-map targets
[ ] Rent RTX 3090 pod on RunPod
[ ] Run Phase A sanity check (4 runs)
[ ] Evaluate: does midpoint help on one-step?
[ ] Run Phase B lean pilot (96 runs)
[ ] Generate hero figure + scaling slopes
[ ] Run Phase C modest confirmation (648 runs)
[ ] Generate Δt transition figure
[ ] Draft task-alignment section for paper
[ ] Decide: proceed to Phase D?
[ ] Merge task-alignment branch into main
```
