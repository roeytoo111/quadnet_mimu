import torch
from quadnet_mimu.src.models import build_model


def test_model_forward_rda():
    batch = 8
    n_imus = 4
    L = 120
    x = torch.randn(batch, 6 * n_imus, L)
    model = build_model(mode="rda", n_imus=n_imus, dropout=0.0, out_dim=1, ara_shared_weights=True, in_channels=6*n_imus)
    y = model(x)
    assert y.shape == (batch, 1)


def test_model_forward_ara_shared():
    batch = 8
    n_imus = 3
    L = 120
    x = torch.randn(batch, 6 * n_imus, L)
    model = build_model(mode="ara", n_imus=n_imus, dropout=0.0, out_dim=1, ara_shared_weights=True, in_channels=6*n_imus)
    y = model(x)
    assert y.shape == (batch, 1)

