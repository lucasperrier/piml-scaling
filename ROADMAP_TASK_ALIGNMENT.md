# Research Roadmap: Task-Aligned Empirical Scaling for Physics Priors

**One-line thesis:** A physics prior only improves empirical scaling when the supervised object matches the semantic level of the prior.

---

## 0. Current Codebase: End-to-End Pipeline Map

```
generate_datasets.py
  └─ data/generate.py::generate_dataset_for_seed()
       └─ systems/ode.py::solve_flow_map()   ← solve_ivp, returns u(T) only
       └─ saves: u0_all.npy, uT_all.npy, train_idx, val_idx, test_idx,
                 train_subsets.json, normalization.json

run_sweep.py / run_experiment.py
  └─ data/dataset.py::FlowMapDataset         ← returns (u0_norm, uT_norm) pairs
  └─ models/mlp.py::MLP                      ← in_dim=2, out_dim=2|4
  └─ train.py::train_one_run()
       └─ losses.py (mse, midpoint, composite, simpson, conservation, per-system)
       └─ metrics.py (mse, relative_l2)
       └─ saves: config.yaml, history.csv, best.pt, metrics.json

aggregate_runs.py
  └─ Walks runs/**/ metrics.json → runs_aggregate.csv, grouped_metrics.csv

fit_scaling.py
  └─ analysis/scaling.py (power-law fits, bootstrap CIs)
  └─ saves: scaling_fits.json

generate_figures.py / dashboard/app.py
  └─ reads grouped_metrics.csv, scaling_fits.json
```

### Where task definition is currently hard-coded

| Component | Hard-coded assumption | File:Line |
|---|---|---|
| **Data generation** | `solve_flow_map` returns only `u(T)` — no trajectory, no derivative | `data/generate.py:77-84` |
| **Dataset class** | `FlowMapDataset` returns `(u0, uT)` pairs only | `data/dataset.py:55-99` |
| **Normalization** | Stats computed on `(u0, uT)` — no derivative or intermediate-state stats | `data/generate.py:94-99` |
| **Model output dim** | `out_dim=2` for plain/midpoint/conservation, `out_dim=4` for simpson/composite | `train.py:114` |
| **Loss dispatch** | Giant if/elif block branches on `physics_prior` string, all assume coarse flow-map structure | `train.py:185-330` |
| **Evaluation** | `evaluate()` computes MSE/rel-L2 on `(pred, target)` — no rollout | `train.py:49-73` |
| **Metrics logging** | Only `test_rel_l2`, `test_mse` — no rollout error, no invariant drift | `train.py:380-395` |
| **Aggregate / scaling** | Groups by `model_name`, `capacity_name`, `dataset_size` — no task axis | `aggregate_runs.py`, `analysis/scaling.py` |

---

## 1. Refactor Plan

### 1.1 Task Abstraction Layer

**Goal:** Replace the hard-coded flow-map assumption with a pluggable `Task` interface so the same sweep runner, training loop, and scaling analysis work across task formulations.

```python
# src/scaling_piml/tasks/base.py
@dataclass
class TaskSpec:
    """Fully describes a supervised task for the scaling study."""
    name: str                         # e.g. "flowmap", "onestep", "vecfield"
    input_dim: int                    # model input dimension
    output_dim: int                   # model output dimension (before prior-specific expansion)
    has_rollout: bool = False         # whether rollout evaluation is defined

class Task(ABC):
    spec: TaskSpec

    @abstractmethod
    def build_dataset(self, data_root, split, *, D=None, normalize=True, **kw) -> Dataset:
        ...

    @abstractmethod
    def compute_loss(self, model, batch, *, cfg, physics_prior, lambda_phys) -> tuple[Tensor, dict]:
        ...

    @abstractmethod
    def evaluate(self, model, loader, device, **kw) -> dict:
        ...

    def rollout(self, model, x0, n_steps, *, device) -> Tensor:
        """Optional: autoregressive rollout for one-step tasks."""
        raise NotImplementedError
```

**Concrete implementations to create:**

| Task class | Input | Target | File |
|---|---|---|---|
| `FlowMapTask` | `x₀` | `x(T)` | `tasks/flowmap.py` — wraps existing `FlowMapDataset` and loss logic |
| `OneStepTask` | `x_t` | `x_{t+Δt}` | `tasks/onestep.py` — new dataset + loss |
| `VecFieldTask` | `x_t` | `ẋ_t` | `tasks/vecfield.py` — new dataset + loss |

The current `FlowMapDataset`, loss dispatch, and `evaluate()` should be migrated into `FlowMapTask` as a first step, keeping behavior identical.

### 1.2 Data Generation Extension

**Current:** `generate_dataset_for_seed()` calls `solve_flow_map()` which runs `solve_ivp` and returns only `u(T)`.

**New:** Add `solve_trajectory()` to `systems/ode.py`:

```python
def solve_trajectory(
    rhs, u0, *, T, dt, method="DOP853", rtol=1e-9, atol=1e-11
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, states) arrays along the full trajectory.

    times: shape (K,) where K = int(T/dt) + 1
    states: shape (K, state_dim)
    """
    t_eval = np.linspace(0, T, int(round(T / dt)) + 1)
    sol = solve_ivp(rhs, (0, T), u0, t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    return sol.t, sol.y.T  # (K,), (K, d)
```

**New generation script** (`generate_trajectory_datasets.py`):
- For each IC, solve the full trajectory at fine resolution (e.g. `dt_fine = 0.001`).
- Save `trajectories.npy` (shape `(N_total, K, d)`) and `times.npy` (shape `(K,)`).
- Derivative targets `ẋ_t` can be computed from the RHS evaluated at grid points (exact, not finite-differenced).
- One-step pairs `(x_t, x_{t+Δt})` are extracted at multiple Δt values at dataset-load time via strided indexing.
- The same trajectory data supports all three task formulations.

**Key design decision:** Store fine-resolution trajectories once. Extract task-specific pairs on the fly in the `Dataset.__getitem__`, parameterized by `dt` or `task_type`. This avoids combinatorial blowup of stored files.

### 1.3 Dataset Classes

```python
# tasks/onestep.py
class OneStepDataset(Dataset):
    """Yields (x_t, x_{t+Δt}) pairs from stored trajectories."""
    def __init__(self, root, split, *, D=None, dt=0.1, normalize=True):
        # Load trajectories, extract pairs at stride = round(dt / dt_fine)
        # Normalization: compute from training fold state values
        ...

# tasks/vecfield.py
class VecFieldDataset(Dataset):
    """Yields (x_t, f(x_t)) pairs."""
    def __init__(self, root, split, *, D=None, normalize=True):
        # Load trajectory states, evaluate RHS to get exact derivatives
        ...
```

**Nested subsets:** For scaling, `D` controls the number of *trajectory-level* samples used (not individual time-step pairs). This keeps the interpretation of `D` consistent with the flow-map case and avoids inflating effective dataset size by extracting many pairs per trajectory.

**Normalization:** Compute input/target statistics from the training fold. For one-step tasks, input stats come from all `x_t` values in training trajectories; target stats from `x_{t+Δt}` values. For vector-field tasks, target stats come from `ẋ_t` values.

### 1.4 Prior Dispatch Refactor

**Current pain point:** `train.py` lines 185-330 are a 150-line if/elif block that duplicates logic per system × prior combination.

**Proposed structure:**

```python
# src/scaling_piml/priors/base.py
class PhysicsPrior(ABC):
    @abstractmethod
    def residual(self, *, u0, uT_hat, cfg) -> Tensor:
        """Return residual vector (B, d). Loss = mean(||r||²)."""
        ...

# src/scaling_piml/priors/midpoint.py
class MidpointResidual(PhysicsPrior):
    def __init__(self, vector_field_fn, T):
        self.f = vector_field_fn
        self.T = T

    def residual(self, *, u0, uT_hat, cfg) -> Tensor:
        mid = 0.5 * (u0 + uT_hat)
        return uT_hat - u0 - self.T * self.f(mid)
```

**Critical insight for task alignment:** The *same* `MidpointResidual` class works for both flow-map and one-step tasks — only `T` changes. For one-step tasks, `T = Δt`. This is exactly the alignment axis we want to test.

**Prior registry per system:**

```python
PRIOR_REGISTRY = {
    "lotka-volterra": {
        "midpoint": lambda cfg: MidpointResidual(lv_field, cfg.data.T),
        "composite": lambda cfg: CompositeMidpointResidual(lv_field, cfg.data.T),
        "trapezoidal": lambda cfg: TrapezoidalResidual(lv_field, cfg.data.T),
        "conservation": lambda cfg: ConservationPrior(lv_invariant),
        "simpson": lambda cfg: SimpsonResidual(lv_field, cfg.data.T),
    },
    "duffing": { ... },
    "van-der-pol": { ... },
}
```

### 1.5 New Priors to Implement

| Prior | Formula | Task alignment | Priority |
|---|---|---|---|
| **Midpoint** (exists) | `r = û - u₀ - T·f((u₀+û)/2)` | Local: aligned to one-step at small Δt, misaligned to coarse flow map | Already implemented |
| **Trapezoidal** (new) | `r = û - u₀ - (T/2)·[f(u₀) + f(û)]` | Same order as midpoint, different linearization; tests whether alignment is structural or coincidental | **High** |
| **RK2 Heun** (new) | `r = û - u₀ - (T/2)·[f(u₀) + f(u₀ + T·f(u₀))]` | Explicit-method residual; no implicit midpoint evaluation | Medium |
| **Composite** (exists) | 2-step midpoint on `[0,T/2]` and `[T/2,T]` | Reduces discretization bias by 4×; still coarse for large T | Already implemented |
| **Simpson** (exists) | `r = û - u₀ - (T/6)[f(u₀) + 4f(u_{T/2}) + f(û)]` | 4th-order; lower bias ceiling | Already implemented |
| **Conservation** (exists) | `|H(û) - H(u₀)|²` | Structural prior, task-agnostic; applies equally to all tasks | Already implemented |

For the one-step task, midpoint and trapezoidal are the highest-priority priors because:
- At small Δt, the midpoint residual should have very low bias (O(Δt³)).
- This directly tests whether the current negative result (midpoint hurts on coarse T=1.0 flow map) reverses when Δt is small.

### 1.6 Evaluation Extension

```python
# src/scaling_piml/eval/rollout.py
@torch.no_grad()
def rollout_evaluate(
    model, x0_batch, *, n_steps, dt, true_trajectories, device
) -> dict:
    """Autoregressive rollout from one-step model.

    Returns:
        one_step_rel_l2: error on individual steps
        rollout_rel_l2_vs_t: array of rel-L2 at each future time step
        rollout_final_rel_l2: rel-L2 at terminal time
        invariant_drift: |H(x_k) - H(x_0)| averaged over rollout
        rollout_stable_steps: number of steps before error exceeds threshold
    """
```

**Invariant drift computation:**

```python
def invariant_drift(trajectory, invariant_fn) -> np.ndarray:
    """Return |H(x_k) - H(x_0)| for k = 0, ..., K."""
    H = invariant_fn(trajectory)  # (K,)
    return np.abs(H - H[0])
```

### 1.7 Config Extension

```yaml
# configs/onestep_lv.yaml
task:
  name: onestep           # "flowmap" | "onestep" | "vecfield"
  dt: 0.1                 # Δt for one-step task (ignored for flowmap)
  dt_fine: 0.001          # integration resolution for trajectory generation
  rollout_horizon: 1.0    # total rollout time for evaluation
  rollout_threshold: 1.0  # rel-L2 threshold to count "stable steps"

# ... rest of config unchanged
```

Add `task.dt` to the sweep grid via a new CLI flag `--dt` in `run_sweep.py`.

### 1.8 Sweep Runner Changes

**Minimal change to `run_sweep.py`:**
- Add `--task` flag: `"flowmap"` (default, backward-compatible), `"onestep"`, `"vecfield"`.
- Add `--dt` flag for one-step/vecfield tasks.
- Task object is instantiated once per sweep and passed to `train_one_run`.
- Run directory gains a `task=` path component: `runs/task=onestep/dt=0.1/model=piml/capacity=small/D=256/...`

**Minimal change to `train_one_run`:**
- Accept a `task: Task` argument.
- Replace the loss if/elif block with `task.compute_loss(model, batch, ...)`.
- Replace `evaluate()` with `task.evaluate(model, loader, device)`.

### 1.9 Aggregation and Scaling Analysis Changes

**`aggregate_runs.py`:**
- Add `task_name` and `dt` columns extracted from run directory path (`task=onestep`, `dt=0.1`).
- Group by `(task_name, dt, model_name, capacity_name, dataset_size)`.

**`analysis/scaling.py`:**
- `run_scaling_analysis()` already groups by `model_name`. Extend to group by `(task_name, dt, model_name)` so that scaling fits are produced per task configuration.

---

## 2. Experiment Plan

### 2.1 Experiment Matrix

#### Tier 1: Minimal Viable Paper (MVP)

| Axis | Values | Count |
|---|---|---|
| System | Lotka–Volterra, Duffing | 2 |
| Task | flowmap (T=1.0), onestep | 2 |
| Δt (onestep only) | 0.01, 0.05, 0.1, 0.2 | 4 |
| Prior | plain, midpoint, trapezoidal, conservation | 4 |
| Capacity | tiny, small, medium, large, xlarge | 5 |
| Dataset size | 64, 128, 256, 512, 1024, 2048, 4096, 8192 | 8 |
| Data seeds | 11, 22, 33 | 3 |
| Train seeds | 101, 202, 303 | 3 |

**Run count:**
- Flowmap: 2 systems × 4 priors × 5 caps × 8 D × 3 × 3 = 2,880
- Onestep: 2 systems × 4 Δt × 4 priors × 5 caps × 8 D × 3 × 3 = 11,520
- **Total: 14,400 runs**

At ~2 min/run average (mix of small and large), this is ~480 GPU-hours. On a single A100 with parallelism, ~2 weeks.

**Minimal publishable subset (Phase 1 pilot):**
Reduce to 2 capacities (small, large), 4 dataset sizes (128, 512, 2048, 8192), 2 Δt values (0.05, 0.1), 1 system (LV), 2 seeds each → 2 × (2×4×2×4×2×2 + 4×2×4×2×2×2) = **enough to establish the alignment effect with ~500 runs**, achievable in 1-2 days.

#### Tier 2: Full Paper

Add to Tier 1:
- **Rollout evaluation** for all one-step-trained models (no additional training, only inference).
- **Vector-field task** with plain and midpoint priors on LV and Duffing.
- **Simpson/composite priors** on the one-step task to test whether higher-order priors improve further.
- **Van der Pol** system to test generality.

Additional runs: ~5,000–8,000.

#### Tier 3: Thesis Extension

- PDE systems (Burgers, KdV, or 2D Navier–Stokes via neural operator).
- Hamiltonian / symplectic priors for pendulum / Kepler.
- Learned priors (hypernetwork-based prior that learns task alignment).
- Formal analysis connecting prior bias to scaling exponent shift.

### 2.2 Δt Selection Rationale

For Lotka–Volterra with the current IC range `[0.5, 2.5]²` and parameters `α=1.5, β=1.0, δ=1.0, γ=3.0`:
- Characteristic timescale: T_char ≈ 2π/√(αγ) ≈ 3.0 (oscillation period).
- System is stiff near the saddle point — largest eigenvalue of Jacobian ≈ max(α, γ) = 3.

| Δt | Δt / T_char | Expected quality |
|---|---|---|
| 0.01 | ~0.003 | Well-resolved; midpoint residual bias ≈ O(Δt³) ≈ 10⁻⁶ |
| 0.05 | ~0.017 | Good resolution; bias ≈ 10⁻⁴ |
| 0.1 | ~0.033 | Moderate; bias ≈ 10⁻³ |
| 0.2 | ~0.067 | Coarsening — approaching current flow-map regime |

This grid should clearly show the transition from "prior is aligned" to "prior is misaligned" as Δt grows.

For Duffing (`α=1, β=1`, IC range `[-2,2]²`): characteristic frequency √α = 1, period ≈ 6.3. Same Δt grid is reasonable; 0.2 is still well within resolution. Consider adding Δt=0.5 for Duffing to see the misalignment boundary.

### 2.3 Priority Ordering

```
╔══════════════════════════════════════════════════════════════════╗
║ PRIORITY 1 (Weeks 1-2): Core alignment experiment              ║
╠══════════════════════════════════════════════════════════════════╣
║ 1a. Implement Task abstraction + OneStepTask                   ║
║ 1b. Implement trajectory data generation (LV only)             ║
║ 1c. Implement trapezoidal residual prior                       ║
║ 1d. Run pilot: LV, onestep, Δt={0.05, 0.1}, plain+midpoint,  ║
║     2 capacities × 4 D-sizes × 2 seeds (~200 runs)            ║
║ 1e. Fit scaling laws per (task, Δt, prior) → first evidence    ║
╠══════════════════════════════════════════════════════════════════╣
║ PRIORITY 2 (Weeks 3-4): Full LV study                         ║
╠══════════════════════════════════════════════════════════════════╣
║ 2a. Full capacity × D grid for LV onestep at all 4 Δt values  ║
║ 2b. Add trapezoidal + conservation priors                      ║
║ 2c. Rollout evaluation for onestep-trained models              ║
║ 2d. Scaling comparison: flowmap vs onestep across priors       ║
╠══════════════════════════════════════════════════════════════════╣
║ PRIORITY 3 (Weeks 5-6): Cross-system replication               ║
╠══════════════════════════════════════════════════════════════════╣
║ 3a. Generate trajectory data for Duffing                       ║
║ 3b. Run same experiment matrix on Duffing                      ║
║ 3c. Cross-system comparison of alignment effects               ║
╠══════════════════════════════════════════════════════════════════╣
║ PRIORITY 4 (Weeks 7-8): Paper writeup                          ║
╠══════════════════════════════════════════════════════════════════╣
║ 4a. Vector-field task (if time permits)                        ║
║ 4b. Final figures and analysis                                 ║
║ 4c. Draft paper                                                ║
╚══════════════════════════════════════════════════════════════════╝
```

### 2.4 Decision Rules

- **After Priority 1d:** If midpoint scaling exponents on one-step (small Δt) are significantly better than midpoint on flow-map (same system), the core hypothesis is supported → proceed to full study.
- **If no difference:** Check whether the issue is optimization (loss landscape), not alignment. Run λ-sweep on one-step midpoint. If λ-sweep doesn't help, consider that the prior may be genuinely uninformative for this system class.
- **After Priority 2c:** If one-step scaling is good but rollout degrades quickly, this defines Type 5 (locally aligned, globally fragile). Record as a result.
- **After Priority 3:** If alignment effects replicate on Duffing, the paper is strong. If they don't, the paper is still interesting (alignment is system-dependent).

---

## 3. Failure-Risk Analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Midpoint still hurts on one-step tasks** (prior genuinely uninformative for any finite Δt on these systems) | Low-Medium | Still publishable as a stronger negative result. The taxonomy argument holds: midpoint is a misaligned-biased prior regardless of task, not just on flow maps. Trapezoidal or RK2 might differ. |
| **One-step training is trivially easy** (models achieve near-zero error at all capacities, no scaling signal) | Medium | Reduce dataset sizes further (32, 16) or increase Δt to create a harder prediction problem. Also test with smaller models. |
| **Normalization issues** with trajectory-based datasets (different magnitudes for small vs large Δt targets) | Medium | Normalize per-Δt. Keep input normalization consistent. Validate on a known case before sweeping. |
| **Data leakage between time steps of same trajectory** | Low | Split at the trajectory level, not the time-step level. Already planned. |
| **Rollout instability masks one-step gains** | Medium | This is scientifically interesting (Type 5 prior). Report one-step and rollout metrics separately. |
| **Compute budget insufficient for full matrix** | Medium | Priority ordering is designed for early stopping. The pilot (Priority 1d) is ~200 runs (~7h on A100). Even just this is publishable as a workshop paper. |
| **Trapezoidal prior behaves identically to midpoint** | High | Expected for well-resolved Δt (both are O(Δt²) methods). The interesting axis is Δt variation, not midpoint vs trapezoidal. Keep trapezoidal for completeness but don't depend on it. |

---

## 4. Figure Plan

### Figure 1: The Alignment Effect (Hero Figure)
**2×2 panel.** Rows = task (flow-map T=1.0, one-step Δt=0.05). Columns = scaling dimension (capacity N, data D).
Each panel: `test_rel_l2` vs N (or D), with curves for plain (blue) and midpoint (red).
**Expected story:** Top row shows midpoint above plain (current negative result). Bottom row shows midpoint below plain (alignment effect).

### Figure 2: Δt Transition
**Single panel** or 1×4 strip. For LV, midpoint prior, fixed capacity.
X-axis: dataset size D. Y-axis: test_rel_l2.
Separate curves for Δt = 0.01, 0.05, 0.1, 0.2, and T=1.0 flow map.
**Expected story:** Monotonic transition from prior helping (small Δt) to prior hurting (large T).

### Figure 3: Scaling Exponent Comparison
**Bar chart or table.** For each (task, prior) combination, show fitted β (data-scaling exponent) with bootstrap CIs.
**Expected story:** β is largest for aligned (prior, task) pairs.

### Figure 4: Scaling Surface
**3D surface or heatmap.** E(N, D) for plain vs midpoint on one-step task.
Shows how the floor E∞ and exponents differ.

### Figure 5: Rollout Stability (if Type 5 result appears)
**Time-series panel.** Rollout error vs time steps for one-step-trained models.
Prior colors. Shows whether prior helps or hurts long-term stability.

### Figure 6: Invariant Drift
**Error vs rollout time.** |H(x_k) - H(x_0)| for plain vs midpoint vs conservation prior.
LV and Duffing side by side.

### Figure 7: Cross-System Comparison
**2-panel.** Same metric (e.g. best-in-class test_rel_l2 at D=8192) shown for LV and Duffing across all (task, prior) combos.
Demonstrates generality of alignment effect.

### Table 1: Full Taxonomy Classification
**Matrix** of (prior × task) with assigned taxonomy type (1–5) and supporting evidence.

---

## 5. Prior–Task Alignment Taxonomy (Target)

| Type | Name | Signature | Example |
|---|---|---|---|
| **1** | Misaligned biased | Floor ↑, exponents ↓, hurts everywhere | Midpoint on coarse flow-map (current result) |
| **2** | Aligned strong | Floor ↓ or exponents ↑, improves absolute performance | Midpoint on one-step at small Δt (hypothesis) |
| **3** | Aligned but weak | Little change in α, β; minor offset improvement | Conservation prior on one-step (hypothesis — low info content) |
| **4** | Exact but underdetermined | No bias, but optimization instability or multimodality | Simpson residual if it creates optimization difficulties |
| **5** | Locally aligned, globally fragile | Good one-step, poor rollout | Midpoint on one-step if rollout degrades (to be tested) |

---

## 6. Candidate Paper Outline

**Title:** "When Does a Physics Prior Improve Scaling? Task Alignment in Scientific Dynamics Learning"

1. **Introduction**
   - Empirical scaling laws as diagnostic tools for SciML.
   - Prior work: the negative result on fixed-horizon flow-map prediction.
   - Research question: does the result reverse when the task is aligned?

2. **Background**
   - Flow-map learning (Churchill & Xiu; MNO).
   - One-step operator learning and neural ODEs.
   - TI-DeepONet and temporal granularity.
   - Physics-informed losses: midpoint residual, conservation, composite.

3. **Methodology**
   - Scaling protocol: capacity grid, data-size grid, multi-seed, power-law fits.
   - Task formulations: flow-map (A), one-step (B), vector-field (C).
   - Prior library and alignment theory.
   - Evaluation protocol: one-step, rollout, invariant drift.

4. **Results**
   - 4.1 Prior underperforms on misaligned task (current result, preserved).
   - 4.2 Prior improves scaling under aligned one-step task.
   - 4.3 Δt as an alignment knob: continuous transition.
   - 4.4 Rollout vs one-step: locally aligned, globally fragile?
   - 4.5 Cross-system replication (Duffing).
   - 4.6 Taxonomy classification.

5. **Discussion**
   - Why alignment matters: the prior's semantic level must match the learning objective.
   - Implications for SciML practice: choose your task before choosing your prior.
   - Limitations: ODE systems only, specific prior families.

6. **Conclusion**

---

## 7. Implementation: Phase-by-Phase Breakdown

### Phase 1: Task Abstraction (est. 2-3 days of dev)

**Files to create:**
```
src/scaling_piml/tasks/__init__.py
src/scaling_piml/tasks/base.py          # Task ABC
src/scaling_piml/tasks/flowmap.py       # Wraps existing FlowMapDataset + loss logic
src/scaling_piml/tasks/onestep.py       # New OneStepDataset + loss
src/scaling_piml/tasks/vecfield.py      # VecFieldDataset + loss (Phase 4+)
```

**Files to modify:**
```
src/scaling_piml/train.py               # Accept Task, delegate loss/eval to it
scripts/run_experiment.py               # Add --task, --dt flags
scripts/run_sweep.py                    # Add --task, --dt flags
```

**Validation:** Run existing flow-map experiments through the new `FlowMapTask` and verify identical metrics.json output to within floating-point tolerance.

### Phase 2: Trajectory Data Generation (est. 1-2 days)

**Files to create:**
```
src/scaling_piml/data/generate_trajectories.py
scripts/generate_trajectory_datasets.py
```

**Files to modify:**
```
src/scaling_piml/systems/ode.py          # Add solve_trajectory()
src/scaling_piml/config.py               # Add TaskConfig dataclass
configs/onestep_lv.yaml                  # New config for LV one-step
configs/onestep_duffing.yaml             # New config for Duffing one-step
```

**Validation:** Generate trajectory data for LV with seed=11, extract one-step pairs at Δt=1.0, verify they match existing flow-map data to solver tolerance.

### Phase 3: Prior Refactor (est. 1-2 days)

**Files to create:**
```
src/scaling_piml/priors/__init__.py
src/scaling_piml/priors/base.py
src/scaling_piml/priors/midpoint.py
src/scaling_piml/priors/trapezoidal.py
src/scaling_piml/priors/composite.py
src/scaling_piml/priors/conservation.py
src/scaling_piml/priors/simpson.py
src/scaling_piml/priors/registry.py
```

**Key design:** Each prior takes a `vector_field_fn` callable and a time horizon. The same prior class works for flow-map and one-step tasks — the time horizon is the differentiator.

**Validation:** Reproduce existing loss values from `losses.py` using the new prior classes.

### Phase 4: Rollout Evaluation (est. 1 day)

**Files to create:**
```
src/scaling_piml/eval/__init__.py
src/scaling_piml/eval/rollout.py
src/scaling_piml/eval/invariants.py
scripts/evaluate_rollout.py
```

### Phase 5: Sweep and Analysis Extension (est. 1 day)

**Files to modify:**
```
scripts/aggregate_runs.py               # Add task_name, dt columns
scripts/fit_scaling.py                  # Group by task
src/scaling_piml/analysis/scaling.py    # Accept task grouping
scripts/generate_figures.py             # New figure types
```

---

## 8. Backward Compatibility

All changes must be backward-compatible:
- `run_sweep.py` without `--task` defaults to `flowmap`, preserving existing behavior.
- Existing `runs/` directories are not affected; new experiments go into `runs-aligned/` or similar.
- `FlowMapDataset` remains usable directly; it becomes one implementation of `Task.build_dataset()`.
- Existing `losses.py` functions are not deleted — they are wrapped by `FlowMapTask.compute_loss()`.

---

## 9. Compute Budget Estimate

| Experiment block | Runs | Est. GPU-hours (A100) |
|---|---|---|
| Priority 1 pilot (LV, 2 priors, 2 caps, 4 D, 2 Δt, 2 seeds) | ~200 | ~7h |
| Priority 2 full LV one-step | ~3,500 | ~120h |
| Priority 2 LV rollout eval | ~3,500 (inference only) | ~10h |
| Priority 3 Duffing | ~3,500 | ~120h |
| Existing flowmap runs (already done) | ~720+ | 0 (reuse) |
| **Total for full paper** | **~11,000** | **~260h** |
| Vector-field extension | ~2,000 | ~70h |

A single A100 at ~$2/hr ≈ $520 for the full paper, or ~$14 for the pilot.

---

## 10. What to Run First

**Immediate next step (< 1 day of compute):**

1. Implement `solve_trajectory()` in `systems/ode.py`.
2. Generate LV trajectory data for seeds 11, 22, 33 with `dt_fine=0.001`, `T=5.0`.
3. Implement `OneStepDataset` that extracts `(x_t, x_{t+Δt})` pairs.
4. Wire it into `train_one_run` via a minimal task flag.
5. Run a 4-point sanity check:

| Run | Task | Δt | Prior | Capacity | D |
|---|---|---|---|---|---|
| S1 | onestep | 0.05 | plain | medium | 2048 |
| S2 | onestep | 0.05 | midpoint | medium | 2048 |
| S3 | flowmap | T=1.0 | plain | medium | 2048 |
| S4 | flowmap | T=1.0 | midpoint | medium | 2048 |

If S2 < S1 (midpoint helps on one-step) while S4 > S3 (midpoint hurts on flow-map), the alignment hypothesis is confirmed at one point. Proceed to full scaling study.

---

## 11. Summary of Deliverables Hierarchy

```
Minimal Viable Paper (Workshop / Short Paper)
├── LV: flowmap vs onestep, plain vs midpoint, 2 Δt values
├── Scaling exponents per condition (bootstrap CIs)
├── Hero figure (Figure 1)
├── Δt transition figure (Figure 2)
└── Taxonomy table (partial)

Full Paper (Conference)
├── Everything in MVP, plus:
├── Duffing replication
├── 4 Δt values, 4 priors
├── Rollout evaluation
├── Invariant drift analysis
├── Complete taxonomy table
├── Vector-field task (if results are interesting)
└── 7 figures + 2 tables

Thesis Extension
├── PDE systems
├── Formal scaling-exponent analysis
├── Learned priors
└── Hamiltonian/symplectic structure
```
