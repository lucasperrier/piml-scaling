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
from .losses import (
    mse_loss, total_loss, total_loss_conservation, total_loss_composite, total_loss_simpson,
    duffing_physics_loss, duffing_composite_midpoint_loss, duffing_conservation_loss,
    duffing_simpson_loss,
    vdp_physics_loss, vdp_composite_midpoint_loss, vdp_dissipation_loss,
    vdp_simpson_loss,
)
from .metrics import mse, relative_l2
from .models.mlp import MLP, parameter_count
from .utils.io import ensure_dir, save_json, save_yaml
from .utils.seed import seed_everything


def _device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def _physical_batch(loader: DataLoader, x: torch.Tensor, y: torch.Tensor, yhat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = loader.dataset
    if hasattr(dataset, "denormalize_inputs") and hasattr(dataset, "denormalize_targets"):
        x_phys = dataset.denormalize_inputs(x)
        y_phys = dataset.denormalize_targets(y)
        yhat_phys = dataset.denormalize_targets(yhat)
        return x_phys, y_phys, yhat_phys
    return x, y, yhat


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    slice_last2: bool = False,
    collect_predictions: bool = False,
) -> dict[str, object]:
    model.eval()
    rels = []
    mses = []
    all_preds = [] if collect_predictions else None
    all_targets = [] if collect_predictions else None
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yhat = model(x)
        if slice_last2:
            yhat = yhat[:, 2:]
        _, y_phys, yhat_phys = _physical_batch(loader, x, y, yhat)
        rels.append(relative_l2(yhat_phys, y_phys).detach().cpu())
        mses.append(mse(yhat_phys, y_phys).detach().cpu())
        if collect_predictions:
            all_preds.append(yhat_phys.detach().cpu())
            all_targets.append(y_phys.detach().cpu())
    result: dict[str, object] = {
        "rel_l2": float(torch.stack(rels).mean()),
        "mse": float(torch.stack(mses).mean()),
    }
    if collect_predictions:
        result["predictions"] = torch.cat(all_preds, dim=0).numpy()
        result["targets"] = torch.cat(all_targets, dim=0).numpy()
    return result


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
    save_predictions: bool = False,
) -> dict[str, object]:
    seed_everything(train_seed)

    run_dir = ensure_dir(run_dir)
    config_path = run_dir / "config.yaml"
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.json"

    save_yaml(config_path, asdict(cfg))

    device = _device()
    _system_name = getattr(cfg.system, "name", "lotka-volterra")
    _is_duffing = _system_name == "duffing"
    _is_vdp = _system_name == "van-der-pol"

    out_dim = 4 if physics_prior in ("simpson", "simpson-true") else 2
    model = MLP(
        2,
        out_dim,
        hidden_widths=cfg.model.hidden_widths,
        activation=cfg.model.activation,
    ).to(device)

    # Warm start: load weights from a pre-trained checkpoint
    if cfg.train.warm_start:
        ws_path = Path(cfg.train.warm_start)
        if ws_path.exists():
            ws_ckpt = torch.load(ws_path, map_location=device)
            # Load compatible keys only (warm start may be from a plain model with out_dim=2)
            src_state = ws_ckpt.get("state_dict", ws_ckpt)
            tgt_state = model.state_dict()
            compatible = {k: v for k, v in src_state.items() if k in tgt_state and v.shape == tgt_state[k].shape}
            tgt_state.update(compatible)
            model.load_state_dict(tgt_state)

    n_params = parameter_count(model)

    gpu_batch_cap = 1024 if device.type == "cuda" else cfg.train.batch_size_cap
    batch_size = min(gpu_batch_cap, dataset_size)
    _pin = device.type == "cuda"
    _workers = 2 if _pin else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=_workers, pin_memory=_pin, persistent_workers=_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=_workers, pin_memory=_pin, persistent_workers=_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=_workers, pin_memory=_pin, persistent_workers=_workers > 0)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    _log_grad_decomp = cfg.train.log_grad_decomposition

    with history_path.open("w", newline="") as f:
        _fieldnames = [
            "epoch",
            "train_loss",
            "train_data_loss",
            "train_phys_loss",
            "phys_data_ratio",
            "grad_norm",
            "val_rel_l2",
            "val_mse",
        ]
        if _log_grad_decomp:
            _fieldnames.extend(["grad_norm_data", "grad_norm_phys"])
        writer = csv.DictWriter(f, fieldnames=_fieldnames)
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
            epoch_grad_norms_data = []
            epoch_grad_norms_phys = []

            # Compute effective lambda_phys for this epoch (rescue options)
            base_lambda = cfg.train.lambda_phys
            if cfg.train.two_stage_epochs > 0 and epoch < cfg.train.two_stage_epochs:
                effective_lambda = 0.0
            elif cfg.train.lambda_schedule_epochs > 0:
                ramp_start = cfg.train.two_stage_epochs  # ramp starts after two-stage phase
                ramp_progress = min(1.0, max(0.0, (epoch - ramp_start) / cfg.train.lambda_schedule_epochs))
                effective_lambda = base_lambda * ramp_progress
            else:
                effective_lambda = base_lambda

            for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(x)

                if physics_prior == "midpoint":
                    x_phys, _, pred_phys = _physical_batch(train_loader, x, y, pred)
                    if _is_duffing:
                        ld = mse_loss(pred, y)
                        lp = duffing_physics_loss(
                            u0=x_phys, uT_hat=pred_phys, T=cfg.data.T,
                            alpha=cfg.system.alpha, beta=cfg.system.beta,
                        )
                        L = ld + effective_lambda * lp
                        parts = {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
                    elif _is_vdp:
                        ld = mse_loss(pred, y)
                        lp = vdp_physics_loss(
                            u0=x_phys, uT_hat=pred_phys, T=cfg.data.T,
                            mu=cfg.system.mu,
                        )
                        L = ld + effective_lambda * lp
                        parts = {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
                    else:
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
                            lambda_phys=effective_lambda,
                        )
                    losses.append(parts["loss"])
                    data_losses.append(parts["data"])
                    phys_losses.append(parts["phys"])
                elif physics_prior == "conservation":
                    x_phys, _, pred_phys = _physical_batch(train_loader, x, y, pred)
                    if _is_duffing:
                        ld = mse_loss(pred, y)
                        lp = duffing_conservation_loss(
                            u0=x_phys, uT_hat=pred_phys,
                            alpha=cfg.system.alpha, beta=cfg.system.beta,
                        )
                        L = ld + effective_lambda * lp
                        parts = {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
                    elif _is_vdp:
                        ld = mse_loss(pred, y)
                        lp = vdp_dissipation_loss(
                            u0=x_phys, uT_hat=pred_phys, T=cfg.data.T,
                            mu=cfg.system.mu,
                        )
                        L = ld + effective_lambda * lp
                        parts = {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
                    else:
                        L, parts = total_loss_conservation(
                            pred=pred,
                            target=y,
                            u0=x_phys,
                            uT_hat_phys=pred_phys,
                            alpha=cfg.system.alpha,
                            beta=cfg.system.beta,
                            delta=cfg.system.delta,
                            gamma=cfg.system.gamma,
                            lambda_phys=effective_lambda,
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
                    if _is_duffing:
                        ld = mse_loss(pred[:, 2:], y)
                        lp = duffing_composite_midpoint_loss(
                            u0=x_phys, uT2_hat=pred_T2_phys, uT_hat=pred_T_phys,
                            T=cfg.data.T, alpha=cfg.system.alpha, beta=cfg.system.beta,
                        )
                        L = ld + effective_lambda * lp
                        parts = {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
                    elif _is_vdp:
                        ld = mse_loss(pred[:, 2:], y)
                        lp = vdp_composite_midpoint_loss(
                            u0=x_phys, uT2_hat=pred_T2_phys, uT_hat=pred_T_phys,
                            T=cfg.data.T, mu=cfg.system.mu,
                        )
                        L = ld + effective_lambda * lp
                        parts = {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
                    else:
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
                            lambda_phys=effective_lambda,
                        )
                    losses.append(parts["loss"])
                    data_losses.append(parts["data"])
                    phys_losses.append(parts["phys"])
                elif physics_prior == "simpson-true":
                    # pred is (B, 4): first 2 = u_{T/2}, last 2 = u_T
                    ds = train_loader.dataset
                    x_phys = ds.denormalize_inputs(x) if hasattr(ds, "denormalize_inputs") else x
                    pred_T2_phys = ds.denormalize_targets(pred[:, :2])
                    pred_T_phys = ds.denormalize_targets(pred[:, 2:])
                    if _is_duffing:
                        ld = mse_loss(pred[:, 2:], y)
                        lp = duffing_simpson_loss(
                            u0=x_phys, uT2_hat=pred_T2_phys, uT_hat=pred_T_phys,
                            T=cfg.data.T, alpha=cfg.system.alpha, beta=cfg.system.beta,
                        )
                        L = ld + effective_lambda * lp
                        parts = {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
                    elif _is_vdp:
                        ld = mse_loss(pred[:, 2:], y)
                        lp = vdp_simpson_loss(
                            u0=x_phys, uT2_hat=pred_T2_phys, uT_hat=pred_T_phys,
                            T=cfg.data.T, mu=cfg.system.mu,
                        )
                        L = ld + effective_lambda * lp
                        parts = {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
                    else:
                        L, parts = total_loss_simpson(
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
                            lambda_phys=effective_lambda,
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

                # Gradient decomposition: log data and physics grad norms separately
                if _log_grad_decomp and physics_prior != "none":
                    # Compute data-only gradient norm
                    opt.zero_grad(set_to_none=True)
                    pred_target = pred if physics_prior not in ("simpson", "simpson-true") else pred[:, 2:]
                    ld_only = mse_loss(pred_target, y)
                    ld_only.backward(retain_graph=True)
                    gn_data_sq = sum(
                        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
                    )
                    epoch_grad_norms_data.append(gn_data_sq ** 0.5)

                    # Compute physics gradient norm: L_total - L_data
                    opt.zero_grad(set_to_none=True)
                    lp_component = L - ld_only
                    lp_component.backward(retain_graph=True)
                    gn_phys_sq = sum(
                        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
                    )
                    epoch_grad_norms_phys.append(gn_phys_sq ** 0.5)

                    # Full backward for the actual optimizer step
                    opt.zero_grad(set_to_none=True)
                    L.backward()
                else:
                    L.backward()

                grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item() ** 2)
                epoch_grad_norms.append(sum(grad_norms) ** 0.5)
                if cfg.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step()

            if diverged:
                break

            val_metrics = evaluate(model, val_loader, device, slice_last2=(physics_prior in ("simpson", "simpson-true")))

            avg_data = float(sum(data_losses) / max(1, len(data_losses)))
            avg_phys = float(sum(phys_losses) / max(1, len(phys_losses)))
            ratio = avg_phys / max(avg_data, 1e-15) if avg_data > 0 else 0.0

            row = {
                "epoch": epoch,
                "train_loss": float(sum(losses) / max(1, len(losses))),
                "train_data_loss": avg_data,
                "train_phys_loss": avg_phys,
                "phys_data_ratio": ratio,
                "grad_norm": float(sum(epoch_grad_norms) / max(1, len(epoch_grad_norms))),
                "val_rel_l2": val_metrics["rel_l2"],
                "val_mse": val_metrics["mse"],
            }
            if _log_grad_decomp and epoch_grad_norms_data:
                row["grad_norm_data"] = float(sum(epoch_grad_norms_data) / len(epoch_grad_norms_data))
                row["grad_norm_phys"] = float(sum(epoch_grad_norms_phys) / len(epoch_grad_norms_phys))
            writer.writerow(row)
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

    _is_composite = (physics_prior == "simpson" or physics_prior == "simpson-true")
    train_metrics = evaluate(model, train_loader, device, slice_last2=_is_composite)
    val_metrics = evaluate(model, val_loader, device, slice_last2=_is_composite)
    test_metrics = evaluate(model, test_loader, device, slice_last2=_is_composite, collect_predictions=save_predictions)

    if save_predictions and "predictions" in test_metrics:
        import numpy as np
        np.save(run_dir / "test_predictions.npy", test_metrics["predictions"])
        np.save(run_dir / "test_targets.npy", test_metrics["targets"])

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
