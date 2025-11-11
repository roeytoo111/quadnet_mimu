import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


SUPPORTED_EXTS = {".csv", ".npy", ".npz", ".mat"}


@dataclass
class SampleMeta:
    trajectory_id: str
    timestamp_start: float
    imu_ids: List[str]


def _list_files(root: str) -> List[str]:
    files = []
    for ext in ["**/*.csv", "**/*.npy", "**/*.npz", "**/*.mat"]:
        files.extend(glob.glob(os.path.join(root, ext), recursive=True))
    return sorted(files)


def _read_file(path: str) -> Dict[str, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        return {col: df[col].to_numpy(dtype=np.float32) for col in df.columns}
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        return {"data": arr.astype(np.float32)}
    if ext == ".npz":
        npz = np.load(path, allow_pickle=False)
        return {k: v.astype(np.float32) for k, v in npz.items()}
    if ext == ".mat":
        mat = loadmat(path)
        out = {}
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray):
                out[k] = v.astype(np.float32).squeeze()
        return out
    raise ValueError(f"Unsupported file extension: {ext}")


def _stack_imus(imu_blocks: List[np.ndarray]) -> np.ndarray:
    # imu_blocks: list of arrays with shape (6, T)
    return np.concatenate(imu_blocks, axis=0)  # (6*nIMU, T)


def _make_windows(arr: np.ndarray, labels: np.ndarray, window: int, stride: int) -> Tuple[List[np.ndarray], List[float]]:
    xs, ys = [], []
    t = arr.shape[-1]
    for start in range(0, max(t - window + 1, 0), stride):
        end = start + window
        if end > t:
            break
        xs.append(arr[:, start:end])
        ys.append(labels[end - 1] - labels[start])  # delta over window
    return xs, ys


class QuadMIMUDataset(Dataset):
    """
    Flexible dataset for the Quadrotor multi-IMU repository.
    Expects per-IMU signals with columns: acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
    Ground truth should provide positions (x,y,z) or distance/altitude signals.
    """
    def __init__(
        self,
        dataset_root: str,
        split: str = "D1",
        target: str = "distance",
        window_length: int = 120,
        stride: int = 120,
        n_imus: Optional[int] = None,
        imu_ids: Optional[List[str]] = None,
        per_imu_normalization: bool = False,
        augment: Optional[Dict] = None,
        mock: bool = False,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.target = target
        self.window_length = window_length
        self.stride = stride
        self.per_imu_normalization = per_imu_normalization
        self.augment_cfg = augment or {"enabled": False}
        self.mock = mock

        if mock:
            self._build_mock()
        else:
            self._discover_and_build(imu_ids=imu_ids, n_imus=n_imus)

    def _build_mock(self) -> None:
        T = 2000
        num_imus = 4
        # Synthetic 6 channels per IMU
        xs = []
        ys = None
        for imu_idx in range(num_imus):
            ch = 6
            sig = np.random.randn(ch, T).astype(np.float32) * 0.2
            xs.append(sig)
            if ys is None:
                # simulate cumulative distance signal (random walk)
                walk = np.cumsum(np.abs(np.random.randn(T)).astype(np.float32) * 0.02)
                ys = walk
        data = _stack_imus(xs)  # (6*4, T)
        x_windows, y_windows = _make_windows(data, ys, self.window_length, self.stride)
        self.X = [torch.from_numpy(xw) for xw in x_windows]
        self.y = torch.tensor(y_windows, dtype=torch.float32).view(-1, 1)
        self.meta = [{"trajectory_id": "mock_traj", "timestamp_start": float(i), "imu_ids": ["imu0", "imu1", "imu2", "imu3"]} for i in range(len(self.X))]

    def _discover_and_build(self, imu_ids: Optional[List[str]], n_imus: Optional[int]) -> None:
        files = _list_files(self.dataset_root)
        if not files:
            raise FileNotFoundError(
                f"No data files found under {self.dataset_root}. "
                f"Expected one of extensions: {', '.join(sorted(SUPPORTED_EXTS))}."
            )
        # Heuristic grouping by directory as trajectory, and within it IMU files
        trajectories: Dict[str, List[str]] = {}
        for f in files:
            traj = os.path.dirname(f)
            trajectories.setdefault(traj, []).append(f)

        # Select IMU IDs: use filenames containing 'imu' digits or fallback to order
        all_x, all_y, all_meta = [], [], []
        for traj_dir, traj_files in sorted(trajectories.items()):
            # Load per-IMU blocks and a distance/altitude reference from any supported key
            imu_blocks: List[np.ndarray] = []
            dist_signal: Optional[np.ndarray] = None
            alt_signal: Optional[np.ndarray] = None
            picked_imu_ids: List[str] = []
            for f in sorted(traj_files):
                data = _read_file(f)
                cols = [k.lower() for k in data.keys()]
                # try to assemble 6 channels
                chs: List[np.ndarray] = []
                for name in ["acc_x", "accy", "acc_y", "accx", "acc_z", "gyro_x", "gyro_y", "gyro_z", "gyr_x", "gyr_y", "gyr_z"]:
                    # collect once per specific axis
                    pass
                # Robust channel picking
                def pick(keys: List[str]) -> Optional[np.ndarray]:
                    for k in keys:
                        if k in data:
                            v = data[k]
                            return v.squeeze().astype(np.float32)
                    return None
                acc_x = pick(["acc_x", "accx", "ax"])
                acc_y = pick(["acc_y", "accy", "ay"])
                acc_z = pick(["acc_z", "accz", "az"])
                gyr_x = pick(["gyro_x", "gyr_x", "gx"])
                gyr_y = pick(["gyro_y", "gyr_y", "gy"])
                gyr_z = pick(["gyro_z", "gyr_z", "gz"])
                distance = pick(["distance", "dist", "s", "pos_s"])
                altitude = pick(["altitude", "alt", "z", "pos_z", "height"])

                if acc_x is not None and acc_y is not None and acc_z is not None and gyr_x is not None and gyr_y is not None and gyr_z is not None:
                    T = min(len(acc_x), len(acc_y), len(acc_z), len(gyr_x), len(gyr_y), len(gyr_z))
                    block = np.stack([acc_x[:T], acc_y[:T], acc_z[:T], gyr_x[:T], gyr_y[:T], gyr_z[:T]], axis=0)  # (6, T)
                    imu_blocks.append(block)
                    picked_imu_ids.append(os.path.splitext(os.path.basename(f))[0])
                if distance is not None:
                    dist_signal = distance
                if altitude is not None:
                    alt_signal = altitude

            if not imu_blocks:
                # skip this trajectory if no IMU block could be parsed
                continue
            if n_imus is not None and len(imu_blocks) >= n_imus:
                imu_blocks = imu_blocks[:n_imus]
                picked_imu_ids = picked_imu_ids[:n_imus]
            data = _stack_imus(imu_blocks)  # (6*nIMU, T)
            if self.target == "altitude":
                labels = alt_signal if alt_signal is not None else np.zeros((data.shape[-1],), dtype=np.float32)
            elif self.target == "both":
                # For now use distance as primary delta; altitude secondary if present
                if dist_signal is None:
                    dist_signal = np.zeros((data.shape[-1],), dtype=np.float32)
                if alt_signal is None:
                    alt_signal = np.zeros((data.shape[-1],), dtype=np.float32)
                labels = np.stack([dist_signal, alt_signal], axis=1).astype(np.float32)  # (T, 2)
            else:
                labels = dist_signal if dist_signal is not None else np.zeros((data.shape[-1],), dtype=np.float32)

            if labels.ndim == 1:
                x_windows, y_windows = _make_windows(data, labels, self.window_length, self.stride)
                y_t = torch.tensor(y_windows, dtype=torch.float32).view(-1, 1)
            else:
                # multi-target: compute deltas per column
                x_windows, y_windows = [], []
                T = data.shape[-1]
                for start in range(0, max(T - self.window_length + 1, 0), self.stride):
                    end = start + self.window_length
                    if end > T:
                        break
                    x_windows.append(data[:, start:end])
                    delta = labels[end - 1, :] - labels[start, :]
                    y_windows.append(delta)
                y_t = torch.tensor(y_windows, dtype=torch.float32)

            all_x.extend([torch.from_numpy(xw) for xw in x_windows])
            all_y.append(y_t)
            all_meta.extend([{"trajectory_id": os.path.basename(traj_dir), "timestamp_start": float(i), "imu_ids": picked_imu_ids} for i in range(len(x_windows))])

        if not all_x:
            raise RuntimeError(
                "No windows could be generated. Ensure the dataset contains IMU channels "
                "(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z) and a distance/altitude reference."
            )

        self.X = all_x
        self.y = torch.cat(all_y, dim=0)
        self.meta = all_meta

    def __len__(self) -> int:
        return len(self.X)

    def _apply_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment_cfg.get("enabled", False):
            return x
        noise_std = float(self.augment_cfg.get("noise_std", 0.0))
        bias_drift_std = float(self.augment_cfg.get("bias_drift_std", 0.0))
        axis_scale_std = float(self.augment_cfg.get("axis_scale_std", 0.0))
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        if bias_drift_std > 0:
            drift = torch.randn(x.shape[0], 1) * bias_drift_std
            x = x + drift
        if axis_scale_std > 0:
            scale = 1.0 + torch.randn(x.shape[0], 1) * axis_scale_std
            x = x * scale
        return x

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()  # (C, L)
        y = self.y[idx]
        x = self._apply_augment(x)
        meta = self.meta[idx]
        return x, y, meta

