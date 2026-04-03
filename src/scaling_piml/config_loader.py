from __future__ import annotations

from dataclasses import fields
from typing import Any, TypeVar

from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ODESolverConfig,
    SystemConfig,
    TrainConfig,
)
from .utils.io import load_yaml

T = TypeVar("T")


def _from_dict(dc_type: type[T], d: dict[str, Any] | None) -> T:
    d = {} if d is None else dict(d)
    allowed = {f.name for f in fields(dc_type)}
    filtered = {k: v for k, v in d.items() if k in allowed}
    return dc_type(**filtered)  # type: ignore[arg-type]


def load_experiment_config(path: str) -> ExperimentConfig:
    raw = load_yaml(path)
    return ExperimentConfig(
        system=_from_dict(SystemConfig, raw.get("system")),
        solver=_from_dict(ODESolverConfig, raw.get("solver")),
        data=_from_dict(DataConfig, raw.get("data")),
        model=_from_dict(ModelConfig, raw.get("model")),
        train=_from_dict(TrainConfig, raw.get("train")),
        out_dir=str(raw.get("out_dir", "runs")),
    )
