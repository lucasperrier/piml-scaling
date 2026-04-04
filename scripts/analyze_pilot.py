from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scaling_piml.analysis import build_pilot_summary


def _render_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Pilot Analysis Summary")
    lines.append("")
    gate = summary["gate"]
    run_summary = summary["run_summary"]
    lines.append(f"ready_for_full_sweep: {gate['ready_for_full_sweep']}")
    lines.append(f"enough_data: {gate['enough_data']}")
    lines.append(f"stability_ok: {gate['stability_ok']}")
    lines.append(f"divergence_rate: {run_summary['divergence_rate']:.4f}")
    lines.append(f"nan_rate: {run_summary['nan_rate']:.4f}")
    lines.append("")
    lines.append("## Error vs D")
    for item in summary["checks"]["error_vs_D"]:
        lines.append(
            f"- model={item['model_name']} capacity={item['capacity_name']} status={item['status']} points={item['n_points']} reversals={item['reversals']}"
        )
    lines.append("")
    lines.append("## Error vs N")
    for item in summary["checks"]["error_vs_N"]:
        lines.append(
            f"- model={item['model_name']} D={item['dataset_size']} status={item['status']} points={item['n_points']} reversals={item['reversals']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-aggregate", type=str, required=True)
    parser.add_argument("--grouped-metrics", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--metric", type=str, default="test_rel_l2_mean")
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--max-divergence-rate", type=float, default=0.30)
    args = parser.parse_args()

    aggregate_path = Path(args.runs_aggregate)
    grouped_path = Path(args.grouped_metrics)
    out_dir = Path(args.out_dir) if args.out_dir else grouped_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_df = pd.read_csv(aggregate_path)
    grouped_df = pd.read_csv(grouped_path)

    summary = build_pilot_summary(
        grouped_df,
        aggregate_df,
        metric_col=args.metric,
        relative_tolerance=args.relative_tolerance,
        max_divergence_rate=args.max_divergence_rate,
    )

    json_path = out_dir / "pilot_summary.json"
    markdown_path = out_dir / "pilot_summary.md"

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    markdown_path.write_text(_render_markdown(summary))

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"ready_for_full_sweep={summary['gate']['ready_for_full_sweep']}")


if __name__ == "__main__":
    main()