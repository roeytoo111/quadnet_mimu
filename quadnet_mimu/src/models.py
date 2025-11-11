from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, (nn.BatchNorm1d,)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class QuadNet(nn.Module):
    """
    1D-CNN + FC regression model.
    Input: (B, C, L)
    Output: (B, out_dim) where out_dim = 1 or 2
    """
    def __init__(
        self,
        in_channels: int,
        conv_channels: List[int] = (64, 128, 128, 256, 256, 512),
        kernels: List[int] = (5, 5, 3, 3, 3, 3),
        pools: List[int] = (1, 2, 1, 2, 1, 1),
        dropout: float = 0.2,
        out_dim: int = 1,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernels) == len(pools)
        layers: List[nn.Module] = []
        c_in = in_channels
        for c_out, k, p in zip(conv_channels, kernels, pools):
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2))
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU(inplace=True))
            if p and p > 1:
                layers.append(nn.MaxPool1d(kernel_size=p))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gap(x)
        x = self.head(x)
        return x

    def summary(self, input_shape: Tuple[int, int, int]) -> str:
        c, l = input_shape[1], input_shape[2]
        params = sum(p.numel() for p in self.parameters())
        return f"QuadNet(in={c}, L={l}) params={params/1e6:.2f}M"


class RDAModel(nn.Module):
    """
    Raw Data Average: average per-IMU signals to 6 channels then run a single QuadNet.
    """
    def __init__(self, n_imus: int, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        # After average, channels reduce to 6
        self.backbone = QuadNet(in_channels=6, dropout=dropout, out_dim=out_dim)
        self.n_imus = n_imus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6*nIMU, L)
        b, c, l = x.shape
        assert c % 6 == 0, "Channels must be multiple of 6"
        n_imus = c // 6
        x = x.view(b, n_imus, 6, l).mean(dim=1)  # (B, 6, L)
        return self.backbone(x)


class ARAModel(nn.Module):
    """
    After Regression Average: run per-IMU network and average scalar outputs.
    Supports shared-weights mode for efficiency.
    """
    def __init__(self, n_imus: int, shared_weights: bool = True, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        self.n_imus = n_imus
        self.shared = shared_weights
        self.out_dim = out_dim
        if shared_weights:
            self.shared_backbone = QuadNet(in_channels=6, dropout=dropout, out_dim=out_dim)
        else:
            self.backbones = nn.ModuleList([QuadNet(in_channels=6, dropout=dropout, out_dim=out_dim) for _ in range(n_imus)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6*nIMU, L)
        b, c, l = x.shape
        assert c % 6 == 0, "Channels must be multiple of 6"
        n_imus = c // 6
        x = x.view(b, n_imus, 6, l)
        preds = []
        if self.shared:
            for i in range(n_imus):
                preds.append(self.shared_backbone(x[:, i, :, :]))
        else:
            for i in range(n_imus):
                preds.append(self.backbones[i](x[:, i, :, :]))
        y = torch.stack(preds, dim=0).mean(dim=0)  # (B, out_dim)
        return y


def build_model(mode: str, n_imus: int, dropout: float = 0.2, out_dim: int = 1, ara_shared_weights: bool = True, in_channels: Optional[int] = None) -> nn.Module:
    mode = mode.lower()
    if mode == "rda":
        return RDAModel(n_imus=n_imus, dropout=dropout, out_dim=out_dim)
    if mode == "ara":
        return ARAModel(n_imus=n_imus, shared_weights=ara_shared_weights, dropout=dropout, out_dim=out_dim)
    # Fallback: plain QuadNet for single-IMU or already-stacked multi-IMU input
    if in_channels is None:
        raise ValueError("in_channels must be provided when using plain model mode.")
    return QuadNet(in_channels=in_channels, dropout=dropout, out_dim=out_dim)

