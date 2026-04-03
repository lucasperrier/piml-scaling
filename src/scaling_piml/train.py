from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .losses import mse_loss, total_loss
from .metrics import mse, relative_l2
from .models.mlp import MLP, parameter_count
from .utils.io import ensure_dir, save_json, save_yaml
from .utils.seed import seed_everything


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rels = []
    mses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        rels.append(relative_l2(yhat, y).detach().cpu())
        mses.append(mse(yhat, y).detach().cpu())
    return {"rel_l2": float(torch.stack(rels).mean()), "mse": float(torch.stack(mses).mean())}


def train_one_run(
    *,
    cfg: ExperimentConfig,
    run_dir: str | Path,
    model_name: str,
    is_physics_informed: bool,
    train_seed: int,
    data_root: str | Path,
    dataset_size: int,
    train_dataset,
    val_dataset,
    test_dataset,
) -> dict[str, object]:
    seed_everything(train_seed)

    run_dir = ensure_dir(run_dir)
    save_yaml(run_dir / "config.yaml", asdict(cfg))

    device = _device()

    model = MLP(2, 2, hidden_widths=cfg.model.hidden_widths, activation=cfg.model.activation).to(device)
    n_params = parameter_count(model)

    batch_size = min(cfg.train.batch_size_cap, dataset_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_data_loss",
                "train_phys_loss",
                "val_rel_l2",
                "val_mse",
            ],
        )
        writer.writeheader()

        start = time.time()
        diverged = False
        nan_detected = False

        for epoch in range(cfg.train.max_epochs):
            model.train()
            losses = []
            data_losses = []
            phys_losses = []

            for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(x)

                if is_physics_informed:
                    L, parts = total_loss(
                        pred=pred,
                        target=y,
                        u0=x,
                        T=cfg.data.T,
                        alpha=cfg.system.alpha,
                        beta=cfg.system.beta,
                        delta=cfg.system.delta,
                        gamma=cfg.system.gamma,
                        lambda_phys=cfg.train.lambda_phys,
                    )
                    losses.append(parts["loss"])
                    data_losses.append(parts["data"])
                    phys_losses.append(parts["phys"])
                else:
                    L = mse_loss(pred, y)
                    loss_val = float(L.detach().cpu())
                    losses.append(loss_val)
                    data_losses.append(loss_val)
                    phys_losses.append(0.0)

                if torch.isnan(L) or torch.isinf(L):
                    nan_detected = True
                    diverged = True
                    break

                L.backward()
                opt.step()

            if diverged:
                break

            val_metrics = evaluate(model, val_loader, device)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": float(sum(losses) / max(1, len(losses))),
                    "train_data_loss": float(sum(data_losses) / max(1, len(data_losses))),
                    "train_phys_loss": float(sum(phys_losses) / max(1, len(phys_losses))),
                    "val_rel_l2": val_metrics["rel_l2"],
                    "val_mse": val_metrics["mse"],
                }
            )
            f.flush()

            if val_metrics["rel_l2"] < best_val:
                best_val = val_metrics["rel_l2"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save({"state_dict": model.state_dict()}, run_dir / "best.pt")
            else:
                bad_epochs += 1

            if bad_epochs >= cfg.train.early_stopping_patience:
                break

        runtime = time.time() - start

    # Load best checkpoint for final metrics
    if (run_dir / "best.pt").exists():
        ckpt = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    metrics_out = {
        "model_name": model_name,
        "is_physics_informed": bool(is_physics_informed),
        "parameter_count": int(n_params),
        "dataset_size": int(dataset_size),
        "data_seed": int(str(Path(data_root).name).split("=")[-1]),
        "train_seed": int(train_seed),
        "best_epoch": int(best_epoch),
        "train_rel_l2": float(train_metrics["rel_l2"]),
        "val_rel_l2": float(val_metrics["rel_l2"]),
        "test_rel_l2": float(test_metrics["rel_l2"]),
        "train_mse": float(train_metrics["mse"]),
        "val_mse": float(val_metrics["mse"]),
        "test_mse": float(test_metrics["mse"]),
        "runtime_seconds": float(runtime),
        "diverged": bool(diverged),
        "nan_detected": bool(nan_detected),
    }

    save_json(run_dir / "metrics.json", metrics_out)
    return metrics_out
