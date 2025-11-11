import torch
from quadnet_mimu.src.datasets import QuadMIMUDataset


def test_mock_dataset_shapes():
    ds = QuadMIMUDataset(dataset_root=".", mock=True, window_length=120, stride=120)
    assert len(ds) > 0
    x, y, meta = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    c, l = x.shape
    assert l == 120
    assert c % 6 == 0
    assert y.shape[-1] in (1, 2)
    assert "trajectory_id" in meta and "imu_ids" in meta

