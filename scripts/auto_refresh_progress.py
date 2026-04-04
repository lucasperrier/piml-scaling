from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from watchfiles import watch


def count_metrics(runs_root: Path) -> int:
    return sum(1 for _ in runs_root.rglob("metrics.json"))


def run_refresh(repo_root: Path, out_dir: Path, n_boot: int) -> None:
    python = str((repo_root / ".venv" / "bin" / "python3").resolve())

    cmd1 = [python, "scripts/aggregate_runs.py", "--runs-root", "runs", "--out-dir", str(out_dir)]
    cmd2 = [
        python,
        "scripts/fit_scaling.py",
        "--grouped-metrics",
        str(out_dir / "grouped_metrics.csv"),
        "--out-dir",
        str(out_dir),
        "--n-boot",
        str(n_boot),
    ]
    cmd3 = [
        python,
        "scripts/generate_figures.py",
        "--grouped-metrics",
        str(out_dir / "grouped_metrics.csv"),
        "--scaling-fits",
        str(out_dir / "scaling_fits.json"),
        "--out-dir",
        "figures-progress",
    ]

    print("[refresh] aggregate_runs.py")
    subprocess.run(cmd1, cwd=repo_root, check=True)
    print("[refresh] fit_scaling.py")
    subprocess.run(cmd2, cwd=repo_root, check=True)
    print("[refresh] generate_figures.py")
    subprocess.run(cmd3, cwd=repo_root, check=True)
    print("[refresh] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-refresh progress analysis every N completed runs")
    parser.add_argument("--runs-root", default="runs", type=str)
    parser.add_argument("--out-dir", default="runs-progress", type=str)
    parser.add_argument("--step", default=50, type=int)
    parser.add_argument("--target", default=720, type=int)
    parser.add_argument("--n-boot", default=300, type=int)
    args = parser.parse_args()

    repo_root = Path.cwd()
    runs_root = (repo_root / args.runs_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        print(f"[error] runs root not found: {runs_root}")
        sys.exit(1)

    current = count_metrics(runs_root)
    next_threshold = ((current // args.step) + 1) * args.step
    print(f"[monitor] current={current}, next_threshold={next_threshold}, target={args.target}")

    for changes in watch(str(runs_root), recursive=True):
        if not any(path.endswith("metrics.json") for _, path in changes):
            continue

        current = count_metrics(runs_root)
        print(f"[monitor] detected update, current={current}")

        while current >= next_threshold:
            print(f"[monitor] threshold reached: {next_threshold}")
            try:
                run_refresh(repo_root, out_dir, args.n_boot)
            except subprocess.CalledProcessError as exc:
                print(f"[error] refresh failed at threshold {next_threshold}: {exc}")
                break
            next_threshold += args.step
            print(f"[monitor] next_threshold={next_threshold}")

        if current >= args.target:
            print(f"[monitor] target reached ({current} >= {args.target}), final refresh")
            run_refresh(repo_root, out_dir, args.n_boot)
            print("[monitor] complete")
            break


if __name__ == "__main__":
    main()
