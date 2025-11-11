import json
import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@dataclass
class NormalizationStats:
    mean: np.ndarray  # shape (C,)
    std: np.ndarray   # shape (C,)
    per_imu: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "per_imu": self.per_imu,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NormalizationStats":
        return NormalizationStats(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
            per_imu=bool(d.get("per_imu", False)),
        )


class Normalizer:
    def __init__(self, stats: NormalizationStats):
        self.stats = stats
        self.eps = 1e-8

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) or (C, L)
        mean = torch.as_tensor(self.stats.mean, dtype=x.dtype, device=x.device).view(1, -1, 1)
        std = torch.as_tensor(self.stats.std, dtype=x.dtype, device=x.device).view(1, -1, 1)
        if x.dim() == 2:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        return (x - mean) / (std + self.eps)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.stats.mean, dtype=x.dtype, device=x.device).view(1, -1, 1)
        std = torch.as_tensor(self.stats.std, dtype=x.dtype, device=x.device).view(1, -1, 1)
        if x.dim() == 2:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        return x * (std + self.eps) + mean

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.stats.to_dict(), f)

    @staticmethod
    def load(path: str) -> "Normalizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return Normalizer(NormalizationStats.from_dict(d))


def compute_channel_stats(iterator, num_channels: int) -> NormalizationStats:
    # iterator yields torch.Tensor (C, L) or (B, C, L)
    total = 0
    sum_c = np.zeros((num_channels,), dtype=np.float64)
    sumsq_c = np.zeros((num_channels,), dtype=np.float64)
    for xb in iterator:
        if xb.dim() == 3:
            b, c, l = xb.shape
            x_np = xb.detach().cpu().numpy().transpose(1, 0, 2).reshape(c, -1)
        else:
            c, l = xb.shape
            x_np = xb.detach().cpu().numpy().reshape(c, -1)
        sum_c += x_np.mean(axis=1)
        sumsq_c += (x_np ** 2).mean(axis=1)
        total += 1
    mean = sum_c / max(total, 1)
    var = sumsq_c / max(total, 1) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32), per_imu=False)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

