from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _check_sequence(frame: pd.DataFrame, *, x_col: str, metric_col: str, relative_tolerance: float) -> dict[str, Any]:
    ordered = frame.sort_values(x_col)
    xs = [int(value) for value in ordered[x_col].tolist()]
    ys = [float(value) for value in ordered[metric_col].tolist()]

    if len(xs) < 2:
        return {
            "status": "insufficient",
            "x_values": xs,
            "metric_values": ys,
            "n_points": len(xs),
            "reversals": 0,
            "max_relative_increase": None,
        }

    reversals = 0
    max_relative_increase = 0.0
    for prev, cur in zip(ys, ys[1:]):
        allowed = prev * (1.0 + relative_tolerance)
        if cur > allowed:
            reversals += 1
            relative_increase = (cur - prev) / max(abs(prev), 1e-12)
            max_relative_increase = max(max_relative_increase, relative_increase)

    return {
        "status": "pass" if reversals == 0 else "fail",
        "x_values": xs,
        "metric_values": ys,
        "n_points": len(xs),
        "reversals": reversals,
        "max_relative_increase": max_relative_increase,
    }


def _sequence_record(base: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        **base,
        **result,
    }


def build_pilot_summary(
    grouped_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    *,
    metric_col: str = "test_rel_l2_mean",
    relative_tolerance: float = 0.05,
    max_divergence_rate: float = 0.30,
) -> dict[str, Any]:
    if metric_col not in grouped_df.columns:
        raise ValueError(f"Metric column not found: {metric_col}")

    grouped = grouped_df.copy()
    aggregate = aggregate_df.copy()

    grouped[metric_col] = pd.to_numeric(grouped[metric_col], errors="coerce")
    grouped = grouped.dropna(subset=[metric_col])

    status_counts = Counter(str(status) for status in aggregate["status"].fillna("unknown"))
    total_runs = int(len(aggregate))
    divergence_rate = float(pd.to_numeric(aggregate["diverged"], errors="coerce").fillna(0).astype(float).mean())
    nan_rate = float(pd.to_numeric(aggregate["nan_detected"], errors="coerce").fillna(0).astype(float).mean())

    data_checks: list[dict[str, Any]] = []
    for keys, frame in grouped.groupby(["model_name", "capacity_name"], dropna=False):
        model_name, capacity_name = keys
        result = _check_sequence(frame, x_col="dataset_size", metric_col=metric_col, relative_tolerance=relative_tolerance)
        data_checks.append(
            _sequence_record(
                {
                    "model_name": str(model_name),
                    "capacity_name": str(capacity_name),
                    "axis": "dataset_size",
                },
                result,
            )
        )

    capacity_checks: list[dict[str, Any]] = []
    for keys, frame in grouped.groupby(["model_name", "dataset_size"], dropna=False):
        model_name, dataset_size = keys
        result = _check_sequence(frame, x_col="parameter_count", metric_col=metric_col, relative_tolerance=relative_tolerance)
        capacity_checks.append(
            _sequence_record(
                {
                    "model_name": str(model_name),
                    "dataset_size": int(dataset_size),
                    "axis": "parameter_count",
                },
                result,
            )
        )

    sufficient_data_checks = [item for item in data_checks if item["status"] != "insufficient"]
    sufficient_capacity_checks = [item for item in capacity_checks if item["status"] != "insufficient"]

    all_data_checks_pass = all(item["status"] == "pass" for item in sufficient_data_checks)
    all_capacity_checks_pass = all(item["status"] == "pass" for item in sufficient_capacity_checks)
    enough_data_for_gate = bool(sufficient_data_checks) and bool(sufficient_capacity_checks)
    stability_ok = divergence_rate <= max_divergence_rate and nan_rate <= max_divergence_rate

    return {
        "metric": metric_col,
        "relative_tolerance": relative_tolerance,
        "max_divergence_rate": max_divergence_rate,
        "run_summary": {
            "total_runs": total_runs,
            "status_counts": dict(status_counts),
            "divergence_rate": divergence_rate,
            "nan_rate": nan_rate,
        },
        "checks": {
            "error_vs_D": data_checks,
            "error_vs_N": capacity_checks,
        },
        "gate": {
            "enough_data": enough_data_for_gate,
            "all_error_vs_D_checks_pass": all_data_checks_pass,
            "all_error_vs_N_checks_pass": all_capacity_checks_pass,
            "stability_ok": stability_ok,
            "ready_for_full_sweep": bool(
                enough_data_for_gate and all_data_checks_pass and all_capacity_checks_pass and stability_ok
            ),
        },
        "best_epoch_summary": {
            "mean": _float_or_none(pd.to_numeric(aggregate["best_epoch"], errors="coerce").replace(-1, pd.NA).mean()),
            "std": _float_or_none(pd.to_numeric(aggregate["best_epoch"], errors="coerce").replace(-1, pd.NA).std(ddof=1)),
        },
    }