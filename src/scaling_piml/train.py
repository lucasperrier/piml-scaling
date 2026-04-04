from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .losses import mse_loss, total_loss, total_loss_conservation, total_loss_composite
from .metrics import mse, relative_l2
from .models.mlp import MLP, parameter_count
from .utils.io import ensure_dir, save_json, save_yaml
from .utils.seed import seed_everything


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _physical_batch(loader: DataLoader, x: torch.Tensor, y: torch.Tensor, yhat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = loader.dataset
    if hasattr(dataset, "denormalize_inputs") and hasattr(dataset, "denormalize_targets"):
        x_phys = dataset.denormalize_inputs(x)
        y_phys = dataset.denormalize_targets(y)
        yhat_phys = dataset.denormalize_targets(yhat)
        return x_phys, y_phys, yhat_phys
    return x, y, yhat


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, slice_last2: bool = False) -> dict[str, float]:
    model.eval()
    rels = []
    mses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        if slice_last2:
            yhat = yhat[:, 2:]
        _, y_phys, yhat_phys = _physical_batch(loader, x, y, yhat)
        rels.append(relative_l2(yhat_phys, y_phys).detach().cpu())
        mses.append(mse(yhat_phys, y_phys).detach().cpu())
    return {"rel_l2": float(torch.stack(rels).mean()), "mse": float(torch.stack(mses).mean())}


def train_one_run(
    *,
    cfg: ExperimentConfig,
    run_dir: str | Path,
    model_name: str,
    capacity_name: str | None,
    physics_prior: str,
    train_seed: int,
    data_root: str | Path,
    dataset_size: int,
    train_dataset,
    val_dataset,
    test_dataset,
) -> dict[str, object]:
    seed_everything(train_seed)

    run_dir = ensure_dir(run_dir)
    config_path = run_dir / "config.yaml"
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.json"

    save_yaml(config_path, asdict(cfg))

    device = _device()

    out_dim = 4 if physics_prior == "simpson" else 2
    model = MLP(
        2,
        out_dim,
        hidden_widths=cfg.model.hidden_widths,
        activation=cfg.model.activation,
    ).to(device)
    n_params = parameter_count(model)

    batch_size = min(cfg.train.batch_size_cap, dataset_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_data_loss",
                "train_phys_loss",
                "phys_data_ratio",
                "grad_norm",
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
            epoch_grad_norms = []

            for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(x)

                if physics_prior == "midpoint":
                    x_phys, _, pred_phys = _physical_batch(train_loader, x, y, pred)
                    L, parts = total_loss(
                        pred=pred,
                        target=y,
                        u0=x_phys,
                        uT_hat_phys=pred_phys,
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
                elif physics_prior == "conservation":
                    x_phys, _, pred_phys = _physical_batch(train_loader, x, y, pred)
                    L, parts = total_loss_conservation(
                        pred=pred,
                        target=y,
                        u0=x_phys,
                        uT_hat_phys=pred_phys,
                        alpha=cfg.system.alpha,
                        beta=cfg.system.beta,
                        delta=cfg.system.delta,
                        gamma=cfg.system.gamma,
                        lambda_phys=cfg.train.lambda_phys,
                    )
                    losses.append(parts["loss"])
                    data_losses.append(parts["data"])
                    phys_losses.append(parts["phys"])
                elif physics_prior == "simpson":
                    # pred is (B, 4): first 2 = u_{T/2}, last 2 = u_T
                    ds = train_loader.dataset
                    x_phys = ds.denormalize_inputs(x) if hasattr(ds, "denormalize_inputs") else x
                    pred_T2_phys = ds.denormalize_targets(pred[:, :2])
                    pred_T_phys = ds.denormalize_targets(pred[:, 2:])
                    L, parts = total_loss_composite(
                        pred_full=pred,
                        pred_target=pred[:, 2:],
                        target=y,
                        u0=x_phys,
                        uT2_hat_phys=pred_T2_phys,
                        uT_hat_phys=pred_T_phys,
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
                grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item() ** 2)
                epoch_grad_norms.append(sum(grad_norms) ** 0.5)
                opt.step()

            if diverged:
                break

            val_metrics = evaluate(model, val_loader, device, slice_last2=(physics_prior == "simpson"))

            avg_data = float(sum(data_losses) / max(1, len(data_losses)))
            avg_phys = float(sum(phys_losses) / max(1, len(phys_losses)))
            ratio = avg_phys / max(avg_data, 1e-15) if avg_data > 0 else 0.0

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": float(sum(losses) / max(1, len(losses))),
                    "train_data_loss": avg_data,
                    "train_phys_loss": avg_phys,
                    "phys_data_ratio": ratio,
                    "grad_norm": float(sum(epoch_grad_norms) / max(1, len(epoch_grad_norms))),
                    "val_rel_l2": val_metrics["rel_l2"],
                    "val_mse": val_metrics["mse"],
                }
            )
            f.flush()

            if val_metrics["rel_l2"] < best_val:
                best_val = val_metrics["rel_l2"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save({"state_dict": model.state_dict()}, checkpoint_path)
            else:
                bad_epochs += 1

            if bad_epochs >= cfg.train.early_stopping_patience:
                break

        runtime = time.time() - start

    # Load best checkpoint for final metrics
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    _is_composite = (physics_prior == "simpson")
    train_metrics = evaluate(model, train_loader, device, slice_last2=_is_composite)
    val_metrics = evaluate(model, val_loader, device, slice_last2=_is_composite)
    test_metrics = evaluate(model, test_loader, device, slice_last2=_is_composite)

    scalar_metrics = {
        "train_rel_l2": float(train_metrics["rel_l2"]),
        "val_rel_l2": float(val_metrics["rel_l2"]),
        "test_rel_l2": float(test_metrics["rel_l2"]),
        "train_mse": float(train_metrics["mse"]),
        "val_mse": float(val_metrics["mse"]),
        "test_mse": float(test_metrics["mse"]),
        "runtime_seconds": float(runtime),
    }

    if nan_detected:
        status = "nan"
        failure_reason = "NaN or Inf detected during training"
    elif diverged:
        status = "diverged"
        failure_reason = "Training diverged before completing an epoch"
    elif best_epoch < 0:
        status = "failed"
        failure_reason = "No valid checkpoint was produced"
    elif not all(math.isfinite(value) for value in scalar_metrics.values()):
        status = "nan"
        failure_reason = "Non-finite evaluation metrics"
    else:
        status = "success"
        failure_reason = ""

    metrics_out = {
        "model_name": model_name,
        "is_physics_informed": physics_prior != "none",
        "physics_prior": physics_prior,
        "capacity_name": str(capacity_name or "custom"),
        "hidden_widths": list(cfg.model.hidden_widths),
        "parameter_count": int(n_params),
        "dataset_size": int(dataset_size),
        "data_seed": int(str(Path(data_root).name).split("=")[-1]),
        "train_seed": int(train_seed),
        "status": status,
        "failure_reason": failure_reason,
        "best_epoch": int(best_epoch),
        **scalar_metrics,
        "diverged": bool(diverged),
        "nan_detected": bool(nan_detected),
        "eligible_for_fit": bool(status == "success"),
        "data_root": str(Path(data_root).resolve()),
        "run_dir": str(run_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "history_path": str(history_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path.exists() else "",
        "metrics_path": str(metrics_path.resolve()),
        "device": str(device),
    }

    save_json(metrics_path, metrics_out)
    return metrics_out
