from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    # System name: "lotka-volterra" or "duffing"
    name: str = "lotka-volterra"
    # Shared parameters (alpha, beta used by both systems)
    alpha: float = 1.5
    beta: float = 1.0
    # Lotka–Volterra only
    delta: float = 1.0
    gamma: float = 3.0


@dataclass
class ODESolverConfig:
    method: str = "DOP853"
    rtol: float = 1e-9
    atol: float = 1e-11


@dataclass
class DataConfig:
    # Task setup
    T: float = 1.0

    # Initial condition distribution (Uniform)
    x0_low: float = 0.5
    x0_high: float = 2.5
    y0_low: float = 0.5
    y0_high: float = 2.5

    # Protocol sizes
    train_pool: int = 20000
    val_size: int = 2000
    test_size: int = 2000

    # Scaling dataset sizes (nested subsets of train pool)
    dataset_sizes: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    )

    data_seeds: list[int] = field(default_factory=lambda: [11, 22, 33])


@dataclass
class ModelConfig:
    activation: str = "relu"  # relu | gelu
    hidden_widths: list[int] = field(default_factory=lambda: [64, 64])


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size_cap: int = 128
    max_epochs: int = 400
    early_stopping_patience: int = 40

    lambda_phys: float = 0.1

    train_seeds: list[int] = field(default_factory=lambda: [101, 202, 303])

    # Rescue study options (Section 24)
    warm_start: str = ""  # Path to a plain-model checkpoint to initialize from
    grad_clip: float = 0.0  # Max gradient norm (0 = disabled)
    lambda_schedule_epochs: int = 0  # Linear ramp-up of lambda_phys over this many epochs (0 = disabled)
    two_stage_epochs: int = 0  # Data-only phase before adding physics loss (0 = disabled)

    # Gradient decomposition logging (Section 28)
    log_grad_decomposition: bool = False  # Log grad_norm_data and grad_norm_phys separately


@dataclass
class ExperimentConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    solver: ODESolverConfig = field(default_factory=ODESolverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Artifacts
    out_dir: str = "runs"
