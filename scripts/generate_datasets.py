from __future__ import annotations

import argparse
from pathlib import Path

from scaling_piml.config_loader import load_experiment_config
from scaling_piml.data.generate import generate_dataset_for_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out", type=str, default="data")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    out = Path(args.out)

    for s in cfg.data.data_seeds:
        root = generate_dataset_for_seed(
            data_seed=s,
            out_dir=out,
            system=cfg.system,
            solver=cfg.solver,
            data=cfg.data,
        )
        print(f"Saved {s} -> {root}")


if __name__ == "__main__":
    main()
