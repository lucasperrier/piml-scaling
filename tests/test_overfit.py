"""Tiny-set overfit test: verify that a large model can memorize D=32 samples."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from scaling_piml.models.mlp import MLP
from scaling_piml.losses import mse_loss


def test_overfit_d32():
    """A medium-capacity model should drive training MSE below 1e-3 on 32 random samples."""
    torch.manual_seed(0)

    D = 32
    x = torch.randn(D, 2)
    y = torch.randn(D, 2)

    model = MLP(in_dim=2, out_dim=2, hidden_widths=[256, 256], activation="relu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
    loader = DataLoader(TensorDataset(x, y), batch_size=D, shuffle=True)

    for _ in range(3000):
        for xb, yb in loader:
            opt.zero_grad()
            loss = mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
        scheduler.step()
        for xb, yb in loader:
            opt.zero_grad()
            loss = mse_loss(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        final_loss = mse_loss(model(x), y).item()

    assert final_loss < 1e-3, f"Model failed to overfit D=32: final MSE = {final_loss:.6f}"
