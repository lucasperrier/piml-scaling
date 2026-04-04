"""Scaling-law fitting utilities.

Fits power-law models of the form E(N,D) = E_inf + a*N^{-alpha} + b*D^{-beta}
to aggregated experiment results, with bootstrap confidence intervals.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Functional forms
# ---------------------------------------------------------------------------

def _capacity_law(N: np.ndarray, E_inf: float, a: float, alpha: float) -> np.ndarray:
    """E(N) = E_inf + a * N^{-alpha}"""
    return E_inf + a * np.power(N.astype(float), -alpha)


def _data_law(D: np.ndarray, E_inf: float, b: float, beta: float) -> np.ndarray:
    """E(D) = E_inf + b * D^{-beta}"""
    return E_inf + b * np.power(D.astype(float), -beta)


def _full_law(ND: np.ndarray, E_inf: float, a: float, alpha: float, b: float, beta: float) -> np.ndarray:
    """E(N,D) = E_inf + a*N^{-alpha} + b*D^{-beta}

    *ND* has shape (2, n) with ND[0]=N, ND[1]=D.
    """
    N = ND[0].astype(float)
    D = ND[1].astype(float)
    return E_inf + a * np.power(N, -alpha) + b * np.power(D, -beta)


# ---------------------------------------------------------------------------
# Single curve fitting
# ---------------------------------------------------------------------------

def _safe_curve_fit(func, xdata, ydata, p0, bounds, maxfev: int = 20000):
    """Wrapper around curve_fit that returns None on failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, pcov = curve_fit(func, xdata, ydata, p0=p0, bounds=bounds, maxfev=maxfev)
            return popt, pcov
        except (RuntimeError, ValueError):
            return None


def fit_capacity_scaling(
    N: np.ndarray,
    E: np.ndarray,
) -> dict[str, Any] | None:
    """Fit E(N) = E_inf + a * N^{-alpha} to data."""
    if len(N) < 3:
        return None
    p0 = [E.min(), 1.0, 0.5]
    bounds = ([0, 0, 0.01], [np.inf, np.inf, 5.0])
    result = _safe_curve_fit(_capacity_law, N, E, p0, bounds)
    if result is None:
        return None
    popt, pcov = result
    return {"E_inf": popt[0], "a": popt[1], "alpha": popt[2], "pcov": pcov}


def fit_data_scaling(
    D: np.ndarray,
    E: np.ndarray,
) -> dict[str, Any] | None:
    """Fit E(D) = E_inf + b * D^{-beta} to data."""
    if len(D) < 3:
        return None
    p0 = [E.min(), 1.0, 0.5]
    bounds = ([0, 0, 0.01], [np.inf, np.inf, 5.0])
    result = _safe_curve_fit(_data_law, D, E, p0, bounds)
    if result is None:
        return None
    popt, pcov = result
    return {"E_inf": popt[0], "b": popt[1], "beta": popt[2], "pcov": pcov}


def fit_full_surface(
    N: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
) -> dict[str, Any] | None:
    """Fit E(N,D) = E_inf + a*N^{-alpha} + b*D^{-beta}."""
    if len(E) < 5:
        return None
    p0 = [E.min(), 1.0, 0.5, 1.0, 0.5]
    bounds = ([0, 0, 0.01, 0, 0.01], [np.inf, np.inf, 5.0, np.inf, 5.0])
    ND = np.vstack([N.astype(float), D.astype(float)])
    result = _safe_curve_fit(_full_law, ND, E, p0, bounds)
    if result is None:
        return None
    popt, pcov = result
    return {
        "E_inf": popt[0], "a": popt[1], "alpha": popt[2],
        "b": popt[3], "beta": popt[4], "pcov": pcov,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_fit(fit_func, *arrays, n_boot: int = 1000, seed: int = 42):
    """Bootstrap a fit function over row-indices and return parameter samples."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    all_params: list[dict[str, Any]] = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sampled = [arr[idx] for arr in arrays]
        result = fit_func(*sampled)
        if result is not None:
            all_params.append({k: v for k, v in result.items() if k != "pcov"})
    return all_params


def bootstrap_capacity(N, E, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    samples = _bootstrap_fit(fit_capacity_scaling, N, E, n_boot=n_boot, seed=seed)
    return _summarize_bootstrap(samples, params=["E_inf", "a", "alpha"])


def bootstrap_data(D, E, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    samples = _bootstrap_fit(fit_data_scaling, D, E, n_boot=n_boot, seed=seed)
    return _summarize_bootstrap(samples, params=["E_inf", "b", "beta"])


def bootstrap_full(N, D, E, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    samples = _bootstrap_fit(fit_full_surface, N, D, E, n_boot=n_boot, seed=seed)
    return _summarize_bootstrap(samples, params=["E_inf", "a", "alpha", "b", "beta"])


def _summarize_bootstrap(samples: list[dict], *, params: list[str]) -> dict[str, Any]:
    if not samples:
        return {"n_boot_success": 0}
    summary: dict[str, Any] = {"n_boot_success": len(samples)}
    for p in params:
        vals = np.array([s[p] for s in samples])
        summary[f"{p}_mean"] = float(np.mean(vals))
        summary[f"{p}_std"] = float(np.std(vals))
        summary[f"{p}_ci_lo"] = float(np.percentile(vals, 2.5))
        summary[f"{p}_ci_hi"] = float(np.percentile(vals, 97.5))
    return summary


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def run_scaling_analysis(
    grouped_df: pd.DataFrame,
    *,
    metric_col: str = "test_rel_l2_mean",
    max_divergence_rate: float = 0.30,
    n_boot: int = 1000,
    boot_seed: int = 42,
) -> dict[str, Any]:
    """Run all scaling fits and bootstraps for all models.

    Returns a dict with keys: capacity_fits, data_fits, full_fits, each
    being a list of per-model/per-slice fit records.
    """
    df = grouped_df.copy()
    df = df[df["divergence_rate"] <= max_divergence_rate]
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    capacity_fits: list[dict[str, Any]] = []
    data_fits: list[dict[str, Any]] = []
    full_fits: list[dict[str, Any]] = []

    for model_name, model_df in df.groupby("model_name"):
        # Capacity scaling: for each dataset_size, fit E vs N
        for ds_size, sub in model_df.groupby("dataset_size"):
            N = sub["parameter_count"].values.astype(float)
            E = sub[metric_col].values.astype(float)
            point_fit = fit_capacity_scaling(N, E)
            boot = bootstrap_capacity(N, E, n_boot=n_boot, seed=boot_seed)
            rec = {
                "model_name": str(model_name),
                "dataset_size": int(ds_size),
                "n_points": len(N),
                "N_values": N.tolist(),
                "E_values": E.tolist(),
            }
            if point_fit:
                rec.update({k: v for k, v in point_fit.items() if k != "pcov"})
            rec["bootstrap"] = boot
            capacity_fits.append(rec)

        # Data scaling: for each capacity, fit E vs D
        for cap_name, sub in model_df.groupby("capacity_name"):
            D = sub["dataset_size"].values.astype(float)
            E = sub[metric_col].values.astype(float)
            point_fit = fit_data_scaling(D, E)
            boot = bootstrap_data(D, E, n_boot=n_boot, seed=boot_seed)
            rec = {
                "model_name": str(model_name),
                "capacity_name": str(cap_name),
                "n_points": len(D),
                "D_values": D.tolist(),
                "E_values": E.tolist(),
            }
            if point_fit:
                rec.update({k: v for k, v in point_fit.items() if k != "pcov"})
            rec["bootstrap"] = boot
            data_fits.append(rec)

        # Full surface fit: all (N, D) for this model
        N_all = model_df["parameter_count"].values.astype(float)
        D_all = model_df["dataset_size"].values.astype(float)
        E_all = model_df[metric_col].values.astype(float)
        point_fit = fit_full_surface(N_all, D_all, E_all)
        boot = bootstrap_full(N_all, D_all, E_all, n_boot=n_boot, seed=boot_seed)
        rec = {
            "model_name": str(model_name),
            "n_points": len(E_all),
        }
        if point_fit:
            rec.update({k: v for k, v in point_fit.items() if k != "pcov"})
        rec["bootstrap"] = boot
        full_fits.append(rec)

    return {
        "capacity_fits": capacity_fits,
        "data_fits": data_fits,
        "full_fits": full_fits,
    }
