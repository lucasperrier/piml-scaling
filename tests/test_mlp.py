import torch

from scaling_piml.models.mlp import MLP


def test_mlp_forward_output_shape():
    model = MLP(in_dim=2, out_dim=2, hidden_widths=[8, 8], activation="relu")
    x = torch.randn(5, 2)
    y = model(x)
    assert y.shape == (5, 2)