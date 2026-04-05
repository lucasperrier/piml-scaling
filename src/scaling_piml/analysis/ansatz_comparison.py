"""Alternative scaling-law ansatzes with model comparison.

Implements five ansatzes for E(N, D):
  A: E = E∞ + a·N^{-α} + b·D^{-β}         (additive with floor, current default)
  B: E = a·N^{-α} + b·D^{-β}               (no floor, E∞ ≡ 0)
  C: E = c·N^{-α}·D^{-β}                   (multiplicative separable)
  D: E = E∞ + a·N^{-α} + b·D^{-β} + d·N^{-α}·D^{-β}  (additive + interaction)
  E: GP on log(N)×log(D) → log(E)           (nonparametric baseline)

Each ansatz provides: fit, predict, bootstrap CI, residual diagnostics, AIC/BIC.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Stores a single fit result."""
    ansatz_name: str
    params: dict[str, float]
    residuals: np.ndarray
    rss: float
    n_obs: int
    n_params: int
    aic: float
    bic: float
    converged: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ansatz": self.ansatz_name,
            "params": self.params,
            "rss": self.rss,
            "n_obs": self.n_obs,
            "n_params": self.n_params,
            "aic": self.aic,
            "bic": self.bic,
            "converged": self.converged,
            "residual_mean": float(np.mean(self.residuals)),
            "residual_std": float(np.std(self.residuals)),
            "residual_max_abs": float(np.max(np.abs(self.residuals))),
        }


@dataclass
class BootstrapResult:
    """Stores bootstrap summary for one ansatz."""
    ansatz_name: str
    n_boot_success: int
    param_summary: dict[str, dict[str, float]]  # param -> {mean, std, ci_lo, ci_hi}
    aic_summary: dict[str, float] = field(default_factory=dict)
    bic_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ansatz": self.ansatz_name,
            "n_boot_success": self.n_boot_success,
        }
        for pname, pstats in self.param_summary.items():
            for stat, val in pstats.items():
                d[f"{pname}_{stat}"] = val
        if self.aic_summary:
            d["aic_mean"] = self.aic_summary.get("mean", float("nan"))
            d["aic_std"] = self.aic_summary.get("std", float("nan"))
        if self.bic_summary:
            d["bic_mean"] = self.bic_summary.get("mean", float("nan"))
            d["bic_std"] = self.bic_summary.get("std", float("nan"))
        return d


# ---------------------------------------------------------------------------
# AIC / BIC helpers
# ---------------------------------------------------------------------------

def _aic(rss: float, n: int, k: int) -> float:
    """AIC assuming Gaussian errors. k = number of model parameters + 1 (for variance)."""
    if n <= 0 or rss <= 0:
        return float("inf")
    k_full = k + 1  # +1 for error variance
    return n * np.log(rss / n) + 2 * k_full


def _bic(rss: float, n: int, k: int) -> float:
    """BIC assuming Gaussian errors."""
    if n <= 0 or rss <= 0:
        return float("inf")
    k_full = k + 1
    return n * np.log(rss / n) + k_full * np.log(n)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Ansatz(ABC):
    """Abstract base for a scaling-law ansatz."""

    name: str
    param_names: list[str]

    @abstractmethod
    def _func(self, ND: np.ndarray, *params) -> np.ndarray:
        """Evaluate the ansatz. ND shape (2, n): ND[0]=N, ND[1]=D."""

    @abstractmethod
    def _p0(self, E: np.ndarray) -> list[float]:
        """Initial guess."""

    @abstractmethod
    def _bounds(self) -> tuple[list[float], list[float]]:
        """Parameter bounds (lower, upper)."""

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def fit(
        self, N: np.ndarray, D: np.ndarray, E: np.ndarray, *, maxfev: int = 20000
    ) -> FitResult | None:
        """Fit ansatz to data. Returns FitResult or None on failure."""
        if len(E) < self.n_params + 1:
            return None
        ND = np.vstack([N.astype(float), D.astype(float)])
        E = E.astype(float)
        p0 = self._p0(E)
        lo, hi = self._bounds()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, _ = curve_fit(
                    self._func, ND, E, p0=p0, bounds=(lo, hi), maxfev=maxfev
                )
            except (RuntimeError, ValueError):
                return None
        pred = self._func(ND, *popt)
        resid = E - pred
        rss = float(np.sum(resid ** 2))
        n = len(E)
        k = self.n_params
        return FitResult(
            ansatz_name=self.name,
            params=dict(zip(self.param_names, [float(v) for v in popt])),
            residuals=resid,
            rss=rss,
            n_obs=n,
            n_params=k,
            aic=_aic(rss, n, k),
            bic=_bic(rss, n, k),
        )

    def predict(self, N: np.ndarray, D: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """Predict E from fitted parameters."""
        ND = np.vstack([N.astype(float), D.astype(float)])
        p = [params[name] for name in self.param_names]
        return self._func(ND, *p)

    def bootstrap(
        self,
        N: np.ndarray,
        D: np.ndarray,
        E: np.ndarray,
        *,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> BootstrapResult:
        """Bootstrap confidence intervals for all parameters + AIC/BIC."""
        rng = np.random.default_rng(seed)
        n = len(E)
        param_samples: list[dict[str, float]] = []
        aic_samples: list[float] = []
        bic_samples: list[float] = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            result = self.fit(N[idx], D[idx], E[idx])
            if result is not None and result.converged:
                param_samples.append(result.params)
                aic_samples.append(result.aic)
                bic_samples.append(result.bic)

        n_success = len(param_samples)
        if n_success == 0:
            return BootstrapResult(
                ansatz_name=self.name,
                n_boot_success=0,
                param_summary={},
            )

        param_summary: dict[str, dict[str, float]] = {}
        for pname in self.param_names:
            vals = np.array([s[pname] for s in param_samples])
            param_summary[pname] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_lo": float(np.percentile(vals, 2.5)),
                "ci_hi": float(np.percentile(vals, 97.5)),
            }

        aic_arr = np.array(aic_samples)
        bic_arr = np.array(bic_samples)
        return BootstrapResult(
            ansatz_name=self.name,
            n_boot_success=n_success,
            param_summary=param_summary,
            aic_summary={"mean": float(np.mean(aic_arr)), "std": float(np.std(aic_arr))},
            bic_summary={"mean": float(np.mean(bic_arr)), "std": float(np.std(bic_arr))},
        )


# ---------------------------------------------------------------------------
# Ansatz A: Additive with floor (current default)
# ---------------------------------------------------------------------------

class AnsatzA(Ansatz):
    """E = E∞ + a·N^{-α} + b·D^{-β}"""

    name = "A_additive_floor"
    param_names = ["E_inf", "a", "alpha", "b", "beta"]

    def _func(self, ND, E_inf, a, alpha, b, beta):
        N, D = ND[0].astype(float), ND[1].astype(float)
        return E_inf + a * np.power(N, -alpha) + b * np.power(D, -beta)

    def _p0(self, E):
        return [float(E.min()), 1.0, 0.5, 1.0, 0.5]

    def _bounds(self):
        return ([0, 0, 0.01, 0, 0.01], [np.inf, np.inf, 5.0, np.inf, 5.0])


# ---------------------------------------------------------------------------
# Ansatz B: No-floor power law
# ---------------------------------------------------------------------------

class AnsatzB(Ansatz):
    """E = a·N^{-α} + b·D^{-β}   (E∞ ≡ 0)"""

    name = "B_no_floor"
    param_names = ["a", "alpha", "b", "beta"]

    def _func(self, ND, a, alpha, b, beta):
        N, D = ND[0].astype(float), ND[1].astype(float)
        return a * np.power(N, -alpha) + b * np.power(D, -beta)

    def _p0(self, E):
        return [1.0, 0.5, 1.0, 0.5]

    def _bounds(self):
        return ([0, 0.01, 0, 0.01], [np.inf, 5.0, np.inf, 5.0])


# ---------------------------------------------------------------------------
# Ansatz C: Multiplicative separable
# ---------------------------------------------------------------------------

class AnsatzC(Ansatz):
    """E = c·N^{-α}·D^{-β}"""

    name = "C_multiplicative"
    param_names = ["c", "alpha", "beta"]

    def _func(self, ND, c, alpha, beta):
        N, D = ND[0].astype(float), ND[1].astype(float)
        return c * np.power(N, -alpha) * np.power(D, -beta)

    def _p0(self, E):
        return [float(np.median(E)), 0.5, 0.5]

    def _bounds(self):
        return ([0, 0.01, 0.01], [np.inf, 5.0, 5.0])


# ---------------------------------------------------------------------------
# Ansatz D: Additive with interaction
# ---------------------------------------------------------------------------

class AnsatzD(Ansatz):
    """E = E∞ + a·N^{-α} + b·D^{-β} + d·N^{-α}·D^{-β}"""

    name = "D_interaction"
    param_names = ["E_inf", "a", "alpha", "b", "beta", "d"]

    def _func(self, ND, E_inf, a, alpha, b, beta, d):
        N, D = ND[0].astype(float), ND[1].astype(float)
        N_term = np.power(N, -alpha)
        D_term = np.power(D, -beta)
        return E_inf + a * N_term + b * D_term + d * N_term * D_term

    def _p0(self, E):
        return [float(E.min()), 1.0, 0.5, 1.0, 0.5, 0.0]

    def _bounds(self):
        return (
            [0, 0, 0.01, 0, 0.01, -np.inf],
            [np.inf, np.inf, 5.0, np.inf, 5.0, np.inf],
        )


# ---------------------------------------------------------------------------
# Ansatz E: Nonparametric GP baseline
# ---------------------------------------------------------------------------

class AnsatzE(Ansatz):
    """GP on log(N)×log(D) → log(E).

    Uses scipy RBF interpolation as a lightweight GP surrogate.
    Prediction is exp(f(log N, log D)).
    """

    name = "E_gp_nonparametric"
    param_names: list[str] = []  # no explicit parameters

    def _func(self, ND, *params):
        raise NotImplementedError("GP ansatz does not use _func for fitting")

    def _p0(self, E):
        return []

    def _bounds(self):
        return ([], [])

    def fit(
        self, N: np.ndarray, D: np.ndarray, E: np.ndarray, *, maxfev: int = 20000
    ) -> FitResult | None:
        from scipy.interpolate import RBFInterpolator

        N = N.astype(float)
        D = D.astype(float)
        E = E.astype(float)

        if len(E) < 4:
            return None

        log_N = np.log(N)
        log_D = np.log(D)
        log_E = np.log(np.clip(E, 1e-12, None))

        X = np.column_stack([log_N, log_D])
        try:
            self._rbf = RBFInterpolator(X, log_E, kernel="thin_plate_spline", smoothing=0.1)
        except (ValueError, np.linalg.LinAlgError):
            return None

        pred_log = self._rbf(X)
        pred = np.exp(pred_log)
        resid = E - pred
        rss = float(np.sum(resid ** 2))
        n = len(E)
        # Effective degrees of freedom ≈ n for a flexible GP; use n/2 as a rough estimate
        k_eff = max(n // 2, 3)
        return FitResult(
            ansatz_name=self.name,
            params={},
            residuals=resid,
            rss=rss,
            n_obs=n,
            n_params=k_eff,
            aic=_aic(rss, n, k_eff),
            bic=_bic(rss, n, k_eff),
        )

    def predict(self, N: np.ndarray, D: np.ndarray, params: dict[str, float]) -> np.ndarray:
        log_N = np.log(N.astype(float))
        log_D = np.log(D.astype(float))
        X = np.column_stack([log_N, log_D])
        return np.exp(self._rbf(X))

    def bootstrap(
        self,
        N: np.ndarray,
        D: np.ndarray,
        E: np.ndarray,
        *,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> BootstrapResult:
        """Bootstrap for GP: track AIC/BIC but no explicit parameters."""
        rng = np.random.default_rng(seed)
        n = len(E)
        aic_samples: list[float] = []
        bic_samples: list[float] = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            result = self.fit(N[idx], D[idx], E[idx])
            if result is not None:
                aic_samples.append(result.aic)
                bic_samples.append(result.bic)

        n_success = len(aic_samples)
        aic_summary = {}
        bic_summary = {}
        if n_success > 0:
            aic_arr = np.array(aic_samples)
            bic_arr = np.array(bic_samples)
            aic_summary = {"mean": float(np.mean(aic_arr)), "std": float(np.std(aic_arr))}
            bic_summary = {"mean": float(np.mean(bic_arr)), "std": float(np.std(bic_arr))}
        return BootstrapResult(
            ansatz_name=self.name,
            n_boot_success=n_success,
            param_summary={},
            aic_summary=aic_summary,
            bic_summary=bic_summary,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_ANSATZES: dict[str, Ansatz] = {
    "A": AnsatzA(),
    "B": AnsatzB(),
    "C": AnsatzC(),
    "D": AnsatzD(),
    "E": AnsatzE(),
}


def get_ansatz(name: str) -> Ansatz:
    """Look up ansatz by short name (A-E)."""
    if name not in ALL_ANSATZES:
        raise ValueError(f"Unknown ansatz '{name}'. Choose from {list(ALL_ANSATZES)}")
    return ALL_ANSATZES[name]


# ---------------------------------------------------------------------------
# High-level comparison driver
# ---------------------------------------------------------------------------

def compare_ansatzes(
    N: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    *,
    ansatz_names: list[str] | None = None,
    n_boot: int = 1000,
    boot_seed: int = 42,
) -> dict[str, Any]:
    """Fit all (or selected) ansatzes and return comparison summary.

    Returns dict with keys: fits (list of FitResult dicts),
    bootstraps (list of BootstrapResult dicts), ranking (sorted by BIC).
    """
    names = ansatz_names or list(ALL_ANSATZES.keys())
    fits: list[dict[str, Any]] = []
    bootstraps: list[dict[str, Any]] = []

    for name in names:
        ansatz = get_ansatz(name)
        result = ansatz.fit(N, D, E)
        if result is not None:
            fits.append(result.to_dict())
            boot = ansatz.bootstrap(N, D, E, n_boot=n_boot, seed=boot_seed)
            bootstraps.append(boot.to_dict())
        else:
            fits.append({"ansatz": ansatz.name, "converged": False})
            bootstraps.append({"ansatz": ansatz.name, "n_boot_success": 0})

    # Rank by BIC (lower is better)
    converged = [f for f in fits if f.get("converged", False)]
    ranking = sorted(converged, key=lambda f: f.get("bic", float("inf")))
    ranking_summary = [
        {"ansatz": f["ansatz"], "bic": f["bic"], "aic": f["aic"], "n_params": f["n_params"]}
        for f in ranking
    ]

    return {
        "fits": fits,
        "bootstraps": bootstraps,
        "ranking": ranking_summary,
    }


def run_ansatz_comparison(
    grouped_df: "pd.DataFrame",
    *,
    metric_col: str = "test_rel_l2_mean",
    max_divergence_rate: float = 0.30,
    ansatz_names: list[str] | None = None,
    n_boot: int = 1000,
    boot_seed: int = 42,
) -> dict[str, Any]:
    """Run ansatz comparison for all models in the grouped DataFrame.

    Returns dict keyed by model_name, each containing compare_ansatzes output.
    """
    import pandas as pd

    df = grouped_df.copy()
    df = df[df["divergence_rate"] <= max_divergence_rate]
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    results: dict[str, Any] = {}
    for model_name, model_df in df.groupby("model_name"):
        N = model_df["parameter_count"].values.astype(float)
        D = model_df["dataset_size"].values.astype(float)
        E = model_df[metric_col].values.astype(float)
        results[str(model_name)] = compare_ansatzes(
            N, D, E,
            ansatz_names=ansatz_names,
            n_boot=n_boot,
            boot_seed=boot_seed,
        )

    return results


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def leave_one_column_out(
    N: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    ansatz: Ansatz,
) -> dict[str, Any]:
    """Hold out each unique D value, fit on rest, predict held-out."""
    unique_D = np.unique(D)
    errors: list[dict[str, Any]] = []
    for d_held in unique_D:
        mask_train = D != d_held
        mask_test = D == d_held
        if mask_train.sum() < ansatz.n_params + 1 or mask_test.sum() == 0:
            continue
        result = ansatz.fit(N[mask_train], D[mask_train], E[mask_train])
        if result is None:
            continue
        pred = ansatz.predict(N[mask_test], D[mask_test], result.params)
        err = E[mask_test] - pred
        errors.append({
            "held_out_D": float(d_held),
            "n_test": int(mask_test.sum()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "mean_rel_error": float(np.mean(np.abs(err) / np.clip(E[mask_test], 1e-12, None))),
        })
    return {"strategy": "leave_one_column_out", "folds": errors}


def leave_one_row_out(
    N: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    ansatz: Ansatz,
) -> dict[str, Any]:
    """Hold out each unique N value, fit on rest, predict held-out."""
    unique_N = np.unique(N)
    errors: list[dict[str, Any]] = []
    for n_held in unique_N:
        mask_train = N != n_held
        mask_test = N == n_held
        if mask_train.sum() < ansatz.n_params + 1 or mask_test.sum() == 0:
            continue
        result = ansatz.fit(N[mask_train], D[mask_train], E[mask_train])
        if result is None:
            continue
        pred = ansatz.predict(N[mask_test], D[mask_test], result.params)
        err = E[mask_test] - pred
        errors.append({
            "held_out_N": float(n_held),
            "n_test": int(mask_test.sum()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "mean_rel_error": float(np.mean(np.abs(err) / np.clip(E[mask_test], 1e-12, None))),
        })
    return {"strategy": "leave_one_row_out", "folds": errors}


def leave_one_corner_out(
    N: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    ansatz: Ansatz,
) -> dict[str, Any]:
    """Hold out the largest (N, D) cell — extrapolation test."""
    max_N = N.max()
    max_D = D.max()
    mask_test = (N == max_N) & (D == max_D)
    mask_train = ~mask_test
    if mask_train.sum() < ansatz.n_params + 1 or mask_test.sum() == 0:
        return {"strategy": "leave_one_corner_out", "folds": []}
    result = ansatz.fit(N[mask_train], D[mask_train], E[mask_train])
    if result is None:
        return {"strategy": "leave_one_corner_out", "folds": []}
    pred = ansatz.predict(N[mask_test], D[mask_test], result.params)
    err = E[mask_test] - pred
    return {
        "strategy": "leave_one_corner_out",
        "folds": [{
            "held_out_N": float(max_N),
            "held_out_D": float(max_D),
            "n_test": int(mask_test.sum()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "mean_rel_error": float(np.mean(np.abs(err) / np.clip(E[mask_test], 1e-12, None))),
            "true_E": float(np.mean(E[mask_test])),
            "pred_E": float(np.mean(pred)),
        }],
    }
