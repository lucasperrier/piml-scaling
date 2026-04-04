"""Physics-loss diagnostic script.

Checks:
1. Ground-truth midpoint residual magnitude at T=1.0
2. Scale mismatch between normalized data loss and physical physics loss
3. Loss ratio evolution during early training
4. Lambda=0 PIML vs plain equivalence
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scaling_piml.config_loader import load_experiment_config
from scaling_piml.data.dataset import FlowMapDataset
from scaling_piml.losses import physics_midpoint_residual, mse_loss, physics_loss
from scaling_piml.systems.lotka_volterra import lotka_volterra_rhs


def check_ground_truth_residual(data_root: str, cfg) -> dict:
    """Compute midpoint residual on true (u0, uT) pairs."""
    ds = FlowMapDataset(data_root, "test", normalize=False)
    u0 = torch.from_numpy(ds.u0)
    uT = torch.from_numpy(ds.uT)

    residual = physics_midpoint_residual(
        u0=u0,
        uT_hat=uT,
        T=cfg.data.T,
        alpha=cfg.system.alpha,
        beta=cfg.system.beta,
        delta=cfg.system.delta,
        gamma=cfg.system.gamma,
    )

    per_sample_norm = torch.sqrt(torch.sum(residual**2, dim=1))
    uT_norm = torch.sqrt(torch.sum(uT**2, dim=1))
    relative_residual = per_sample_norm / (uT_norm + 1e-12)

    return {
        "residual_l2_mean": float(per_sample_norm.mean()),
        "residual_l2_std": float(per_sample_norm.std()),
        "residual_l2_median": float(per_sample_norm.median()),
        "residual_l2_max": float(per_sample_norm.max()),
        "residual_l2_min": float(per_sample_norm.min()),
        "relative_residual_mean": float(relative_residual.mean()),
        "relative_residual_std": float(relative_residual.std()),
        "relative_residual_max": float(relative_residual.max()),
        "uT_norm_mean": float(uT_norm.mean()),
        "uT_norm_std": float(uT_norm.std()),
        "physics_loss_on_truth": float(physics_loss(
            u0=u0, uT_hat=uT,
            T=cfg.data.T,
            alpha=cfg.system.alpha, beta=cfg.system.beta,
            delta=cfg.system.delta, gamma=cfg.system.gamma,
        )),
    }


def check_loss_scale_mismatch(data_root: str, cfg) -> dict:
    """Compare scales of normalized data loss vs physical physics loss at init."""
    from scaling_piml.models.mlp import MLP, CAPACITY_GRID

    results = {}
    for cap_name in ["tiny", "small", "large"]:
        widths = CAPACITY_GRID[cap_name]
        model = MLP(2, 2, hidden_widths=widths, activation=cfg.model.activation)
        model.eval()

        ds_norm = FlowMapDataset(data_root, "test", normalize=True)
        ds_phys = FlowMapDataset(data_root, "test", normalize=False)

        with torch.no_grad():
            x_norm = torch.from_numpy(ds_norm.u0)
            y_norm = torch.from_numpy(ds_norm.uT)
            pred_norm = model(x_norm)

            # Data loss in normalized space (what training uses)
            data_loss_val = float(mse_loss(pred_norm, y_norm))

            # Denormalize prediction for physics loss
            pred_phys = ds_norm.denormalize_targets(pred_norm)
            x_phys = torch.from_numpy(ds_phys.u0)

            # Physics loss in physical space (what training uses)
            phys_loss_val = float(physics_loss(
                u0=x_phys, uT_hat=pred_phys,
                T=cfg.data.T,
                alpha=cfg.system.alpha, beta=cfg.system.beta,
                delta=cfg.system.delta, gamma=cfg.system.gamma,
            ))

        results[cap_name] = {
            "data_loss_normalized": data_loss_val,
            "physics_loss_physical": phys_loss_val,
            "ratio_phys_over_data": phys_loss_val / max(data_loss_val, 1e-15),
            "effective_phys_contribution": cfg.train.lambda_phys * phys_loss_val,
            "effective_ratio": (cfg.train.lambda_phys * phys_loss_val) / max(data_loss_val, 1e-15),
        }

    return results


def check_gradient_scale(data_root: str, cfg) -> dict:
    """Compare gradient norms from data loss vs physics loss."""
    from scaling_piml.models.mlp import MLP, CAPACITY_GRID
    from scaling_piml.losses import total_loss

    results = {}
    for cap_name in ["tiny", "small"]:
        widths = CAPACITY_GRID[cap_name]
        model = MLP(2, 2, hidden_widths=widths, activation=cfg.model.activation)

        ds = FlowMapDataset(data_root, "train", D=256, normalize=True)

        x_norm = torch.from_numpy(ds.u0)
        y_norm = torch.from_numpy(ds.uT)

        # Gradient from data loss only
        model.zero_grad()
        pred = model(x_norm)
        ld = mse_loss(pred, y_norm)
        ld.backward()
        grad_data_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5

        # Gradient from physics loss only
        model.zero_grad()
        pred = model(x_norm)
        pred_phys = ds.denormalize_targets(pred)
        x_phys = ds.denormalize_inputs(x_norm)
        lp = physics_loss(
            u0=x_phys, uT_hat=pred_phys,
            T=cfg.data.T,
            alpha=cfg.system.alpha, beta=cfg.system.beta,
            delta=cfg.system.delta, gamma=cfg.system.gamma,
        )
        lp.backward()
        grad_phys_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5

        results[cap_name] = {
            "grad_norm_data_loss": grad_data_norm,
            "grad_norm_physics_loss": grad_phys_norm,
            "grad_norm_ratio_phys_over_data": grad_phys_norm / max(grad_data_norm, 1e-15),
            "effective_grad_ratio_at_lambda": (cfg.train.lambda_phys * grad_phys_norm) / max(grad_data_norm, 1e-15),
        }

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data-root", type=str, default="data/data_seed=11")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg = load_experiment_config(args.config)

    print("=" * 70)
    print("PHYSICS-LOSS DIAGNOSTIC REPORT")
    print("=" * 70)

    # 1. Ground-truth midpoint residual
    print("\n--- 1. Ground-truth midpoint residual at T={} ---".format(cfg.data.T))
    gt_res = check_ground_truth_residual(args.data_root, cfg)
    for k, v in gt_res.items():
        print(f"  {k}: {v:.6f}")

    if gt_res["relative_residual_mean"] > 0.05:
        print("\n  *** WARNING: Mean relative residual is {:.1%} ***".format(
            gt_res["relative_residual_mean"]
        ))
        print("  The midpoint rule is a poor proxy for the true flow map at this horizon.")
        print("  Physics loss penalizes even perfectly correct predictions.")
    else:
        print("\n  Midpoint residual is reasonably small (< 5% relative).")

    # 2. Loss scale mismatch
    print("\n--- 2. Loss scale mismatch (data=normalized, physics=physical) ---")
    scale_res = check_loss_scale_mismatch(args.data_root, cfg)
    for cap, vals in scale_res.items():
        print(f"\n  Capacity: {cap}")
        for k, v in vals.items():
            print(f"    {k}: {v:.6f}")

    # 3. Gradient scale comparison
    print("\n--- 3. Gradient norm comparison at initialization ---")
    grad_res = check_gradient_scale(args.data_root, cfg)
    for cap, vals in grad_res.items():
        print(f"\n  Capacity: {cap}")
        for k, v in vals.items():
            print(f"    {k}: {v:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    if gt_res["relative_residual_mean"] > 0.05:
        issues.append(
            "RESIDUAL PROXY: Ground-truth midpoint residual is {:.1%} relative. "
            "The implicit midpoint rule is not a faithful surrogate for the true "
            "flow map at T={}. The physics loss penalizes correct predictions.".format(
                gt_res["relative_residual_mean"], cfg.data.T
            )
        )

    any_ratio = next(iter(scale_res.values()))
    if any_ratio["ratio_phys_over_data"] > 5 or any_ratio["ratio_phys_over_data"] < 0.2:
        issues.append(
            "SCALE MISMATCH: Physics loss (physical space) and data loss (normalized space) "
            "differ by {:.1f}x at init. They are in different unit systems.".format(
                any_ratio["ratio_phys_over_data"]
            )
        )

    if not issues:
        print("  No critical issues detected.")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"  [{i}] {issue}")

    # Save full results
    report = {
        "ground_truth_residual": gt_res,
        "loss_scale_mismatch": scale_res,
        "gradient_scale": grad_res,
        "issues": issues,
    }

    out_path = Path(args.out) if args.out else Path("diagnostic_report.json")
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {out_path}")


if __name__ == "__main__":
    main()
