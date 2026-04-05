from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "model_name",
    "is_physics_informed",
    "physics_prior",
    "capacity_name",
    "hidden_widths",
    "parameter_count",
    "dataset_size",
    "data_seed",
    "train_seed",
    "status",
    "failure_reason",
    "best_epoch",
    "train_rel_l2",
    "val_rel_l2",
    "test_rel_l2",
    "train_mse",
    "val_mse",
    "test_mse",
    "runtime_seconds",
    "diverged",
    "nan_detected",
    "eligible_for_fit",
    "data_root",
    "run_dir",
    "config_path",
    "history_path",
    "checkpoint_path",
    "metrics_path",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _normalize_hidden_widths(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def _infer_from_path(metrics_path: Path) -> dict[str, Any]:
    run_dir = metrics_path.parent
    inferred: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "config_path": str((run_dir / "config.yaml").resolve()),
        "history_path": str((run_dir / "history.csv").resolve()) if (run_dir / "history.csv").exists() else "",
        "checkpoint_path": str((run_dir / "best.pt").resolve()) if (run_dir / "best.pt").exists() else "",
    }
    for part in run_dir.parts:
        if part.startswith("capacity="):
            inferred.setdefault("capacity_name", part.split("=", 1)[1])
        elif part.startswith("D="):
            inferred.setdefault("dataset_size", int(part.split("=", 1)[1]))
        elif part.startswith("data_seed="):
            inferred.setdefault("data_seed", int(part.split("=", 1)[1]))
        elif part.startswith("train_seed="):
            inferred.setdefault("train_seed", int(part.split("=", 1)[1]))
        elif part.startswith("model="):
            inferred.setdefault("model_name", part.split("=", 1)[1])
        elif part.startswith("task="):
            inferred.setdefault("task_name", part.split("=", 1)[1])
        elif part.startswith("dt="):
            inferred.setdefault("dt", float(part.split("=", 1)[1]))
    # Default task/dt for legacy flow-map runs
    inferred.setdefault("task_name", "flowmap")
    inferred.setdefault("dt", None)
    return inferred


def _collect_records(runs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.rglob("metrics.json")):
        payload = _load_json(metrics_path)
        inferred = _infer_from_path(metrics_path)
        row = {**inferred, **payload}
        row["hidden_widths"] = _normalize_hidden_widths(row.get("hidden_widths"))
        row["failure_reason"] = str(row.get("failure_reason", ""))
        row["capacity_name"] = str(row.get("capacity_name", "custom"))
        row["status"] = str(row.get("status", "success"))
        row["eligible_for_fit"] = bool(row.get("eligible_for_fit", row["status"] == "success"))
        records.append(row)
    return records


def _group_records(df: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        "model_name",
        "is_physics_informed",
        "physics_prior",
        "capacity_name",
        "hidden_widths",
        "parameter_count",
        "dataset_size",
    ]
    # Include task_name and dt in grouping when present
    if "task_name" in df.columns:
        group_keys = ["task_name", "dt"] + group_keys

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_keys, dropna=False):
        key_map = dict(zip(group_keys, keys))
        eligible = group[group["eligible_for_fit"]]

        row: dict[str, Any] = {
            **key_map,
            "n_attempted": int(len(group)),
            "n_runs": int(len(eligible)),
            "divergence_rate": float(group["diverged"].astype(float).mean()) if len(group) else 0.0,
            "nan_rate": float(group["nan_detected"].astype(float).mean()) if len(group) else 0.0,
        }

        for metric in ["test_rel_l2", "val_rel_l2", "test_mse", "runtime_seconds"]:
            series = pd.to_numeric(eligible[metric], errors="coerce")
            row[f"{metric}_mean"] = float(series.mean()) if not series.empty else float("nan")
            row[f"{metric}_std"] = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
            row[f"{metric}_stderr"] = (
                float(series.std(ddof=1) / (len(series) ** 0.5)) if len(series) > 1 else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    out_dir = Path(args.out_dir) if args.out_dir else runs_root
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _collect_records(runs_root)
    if not records:
        raise FileNotFoundError(f"No metrics.json files found under {runs_root}")

    aggregate_df = pd.DataFrame(records)
    for column in REQUIRED_COLUMNS:
        if column not in aggregate_df.columns:
            aggregate_df[column] = ""
    aggregate_df = aggregate_df[REQUIRED_COLUMNS].sort_values(
        ["model_name", "capacity_name", "dataset_size", "data_seed", "train_seed"]
    )

    grouped_df = _group_records(aggregate_df)

    aggregate_path = out_dir / "runs_aggregate.csv"
    grouped_path = out_dir / "grouped_metrics.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)

    print(f"Wrote {aggregate_path}")
    print(f"Wrote {grouped_path}")


if __name__ == "__main__":
    main()