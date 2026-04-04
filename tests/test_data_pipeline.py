import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from scaling_piml.config import DataConfig, ExperimentConfig, ODESolverConfig, SystemConfig, TrainConfig
from scaling_piml.data.dataset import FlowMapDataset
from scaling_piml.data.generate import generate_dataset_for_seed
from scaling_piml.train import evaluate, train_one_run


def _write_frozen_dataset(root, *, u0_all, uT_all, train_idx, val_idx, test_idx, normalization, subsets):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "u0_all.npy", np.asarray(u0_all, dtype=np.float32))
    np.save(root / "uT_all.npy", np.asarray(uT_all, dtype=np.float32))
    np.save(root / "train_idx.npy", np.asarray(train_idx, dtype=np.int64))
    np.save(root / "val_idx.npy", np.asarray(val_idx, dtype=np.int64))
    np.save(root / "test_idx.npy", np.asarray(test_idx, dtype=np.int64))
    with (root / "train_subsets.json").open("w") as f:
        json.dump(subsets, f)
    with (root / "normalization.json").open("w") as f:
        json.dump(normalization, f)


def test_generate_dataset_normalization_uses_full_train_pool_only(tmp_path):
    root = generate_dataset_for_seed(
        data_seed=11,
        out_dir=tmp_path,
        system=SystemConfig(),
        solver=ODESolverConfig(method="RK45", rtol=1e-6, atol=1e-8),
        data=DataConfig(
            T=0.1,
            train_pool=4,
            val_size=2,
            test_size=1,
            dataset_sizes=[2, 4],
            data_seeds=[11],
        ),
    )

    u0_all = np.load(root / "u0_all.npy")
    uT_all = np.load(root / "uT_all.npy")
    train_idx = np.load(root / "train_idx.npy")

    with (root / "normalization.json").open("r") as f:
        stats = json.load(f)

    u0_train = u0_all[train_idx]
    uT_train = uT_all[train_idx]

    assert np.allclose(stats["x_mean"], u0_train.mean(axis=0))
    assert np.allclose(stats["x_std"], u0_train.std(axis=0) + 1e-12)
    assert np.allclose(stats["y_mean"], uT_train.mean(axis=0))
    assert np.allclose(stats["y_std"], uT_train.std(axis=0) + 1e-12)


class _ConstantModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.register_buffer("output", torch.tensor(output, dtype=torch.float32))

    def forward(self, x):
        return self.output.unsqueeze(0).expand(x.shape[0], -1)


def test_evaluate_uses_physical_coordinates(tmp_path):
    root = tmp_path / "data_seed=11"
    normalization = {
        "x_mean": [1.0, 2.0],
        "x_std": [2.0, 4.0],
        "y_mean": [10.0, 20.0],
        "y_std": [10.0, 5.0],
    }
    _write_frozen_dataset(
        root,
        u0_all=[[1.0, 2.0]],
        uT_all=[[15.0, 15.0]],
        train_idx=[0],
        val_idx=[0],
        test_idx=[0],
        normalization=normalization,
        subsets={"1": [0]},
    )

    dataset = FlowMapDataset(root, "val", normalize=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = _ConstantModel([0.6, -0.8])

    metrics = evaluate(model, loader, torch.device("cpu"))

    pred_phys = np.array([16.0, 16.0], dtype=np.float32)
    target_phys = np.array([15.0, 15.0], dtype=np.float32)
    expected_mse = np.mean((pred_phys - target_phys) ** 2)
    expected_rel_l2 = np.linalg.norm(pred_phys - target_phys) / np.linalg.norm(target_phys)

    assert np.isclose(metrics["mse"], expected_mse)
    assert np.isclose(metrics["rel_l2"], expected_rel_l2)


def test_train_one_run_passes_physical_states_to_physics_loss(tmp_path, monkeypatch):
    root = tmp_path / "data_seed=11"
    normalization = {
        "x_mean": [10.0, 20.0],
        "x_std": [2.0, 5.0],
        "y_mean": [30.0, 40.0],
        "y_std": [7.0, 11.0],
    }
    _write_frozen_dataset(
        root,
        u0_all=[[12.0, 15.0], [11.0, 18.0], [13.0, 14.0]],
        uT_all=[[31.0, 52.0], [32.0, 47.0], [29.0, 44.0]],
        train_idx=[0],
        val_idx=[1],
        test_idx=[2],
        normalization=normalization,
        subsets={"1": [0]},
    )

    train_ds = FlowMapDataset(root, "train", D=1, normalize=True)
    val_ds = FlowMapDataset(root, "val", normalize=True)
    test_ds = FlowMapDataset(root, "test", normalize=True)

    captured = {}

    def fake_total_loss(**kwargs):
        captured["u0"] = kwargs["u0"].detach().cpu().numpy()
        captured["uT_hat_phys"] = kwargs["uT_hat_phys"].detach().cpu().numpy()
        loss = torch.mean((kwargs["pred"] - kwargs["target"]) ** 2)
        return loss, {"loss": float(loss.detach().cpu()), "data": float(loss.detach().cpu()), "phys": 0.0}

    monkeypatch.setattr("scaling_piml.train.total_loss", fake_total_loss)

    cfg = ExperimentConfig(
        train=TrainConfig(
            max_epochs=1,
            early_stopping_patience=1,
            batch_size_cap=1,
            train_seeds=[101],
        )
    )

    run_dir = tmp_path / "run"
    metrics = train_one_run(
        cfg=cfg,
        run_dir=run_dir,
        model_name="piml_test",
        capacity_name="tiny",
        physics_prior="midpoint",
        train_seed=101,
        data_root=root,
        dataset_size=1,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )

    expected_u0 = np.array([[12.0, 15.0]], dtype=np.float32)
    assert np.allclose(captured["u0"], expected_u0)
    assert metrics["status"] == "success"
    assert metrics["eligible_for_fit"] is True
    assert metrics["capacity_name"] == "tiny"

    normalized_pred = (captured["uT_hat_phys"] - np.array(normalization["y_mean"], dtype=np.float32)) / np.array(
        normalization["y_std"], dtype=np.float32
    )
    assert np.isfinite(normalized_pred).all()
