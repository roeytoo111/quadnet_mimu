"""
PyTorch Dataset and DataLoader for QuadNet MIMU dataset.
Supports flexible data loading from CSV files with multiple IMUs.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import glob
from scipy.interpolate import interp1d
from utils import augment_data, compute_distance_delta, compute_altitude_delta, Normalizer


def resolve_dataset_path(data_dir: Union[str, Path], dataset_type: str) -> Path:
    """
    Resolve the concrete dataset directory that contains trajectory folders.

    Handles different capitalizations and cases where the user passes either the
    dataset root (e.g. '../dataset_repo') or the subdirectory directly
    (e.g. '../dataset_repo/Horizontal').
    """
    base_path = Path(data_dir)

    if not base_path.exists():
        raise ValueError(f"[ERROR] Dataset root directory does not exist: {base_path}")

    dataset_type_lower = dataset_type.lower() if dataset_type else ''

    def has_trajectory_dirs(path: Path) -> bool:
        try:
            return any(child.is_dir() and child.name.startswith('path_') for child in path.iterdir())
        except FileNotFoundError:
            return False

    # If the provided path already contains trajectory folders, return it directly
    if has_trajectory_dirs(base_path):
        return base_path

    # If the provided path name already matches the dataset type (e.g. Horizontal)
    if dataset_type_lower in base_path.name.lower() and has_trajectory_dirs(base_path):
        return base_path

    candidate_names = [
        dataset_type,
        dataset_type_lower,
        dataset_type_lower.capitalize() if dataset_type_lower else '',
        dataset_type_lower.upper() if dataset_type_lower else '',
        dataset_type_lower.replace("line", "Line") if dataset_type_lower else '',
    ]

    for name in candidate_names:
        if not name:
            continue
        candidate_path = base_path / name
        if candidate_path.exists() and candidate_path.is_dir() and has_trajectory_dirs(candidate_path):
            return candidate_path

    # Fallback: try partial match within the base directory
    for sub in base_path.iterdir():
        if sub.is_dir() and dataset_type_lower in sub.name.lower() and has_trajectory_dirs(sub):
            return sub

    raise ValueError(
        f"[ERROR] Dataset directory not found under {base_path} for dataset_type='{dataset_type}'"
    )


class QuadNetDataset(Dataset):
    """
    Dataset for QuadNet MIMU regression.
    
    Returns windows of IMU data and corresponding delta distance/altitude labels.
    """
    
    def __init__(self,
                 data_dir: str,
                 trajectory_ids: List[str],
                 window_length: int = 120,
                 stride: int = 120,
                 target: str = 'distance',  # 'distance', 'altitude', or 'both'
                 n_imus: int = 4,
                 imu_ids: Optional[List[int]] = None,
                 mode: str = 'rda',  # 'rda' or 'ara'
                 normalize: bool = True,
                 normalizer: Optional[Normalizer] = None,
                 augment: bool = False,
                 augment_params: Optional[Dict] = None,
                 sampling_rate: float = 120.0,  # Hz
                 gt_sampling_rate: float = 10.0,  # Hz
                 dataset_type: str = 'horizontal'):  # 'horizontal', 'vertical', 'straightline'
        """
        Args:
            data_dir: Root directory containing dataset (e.g., Horizontal/, Vertical/)
            trajectory_ids: List of trajectory IDs to include (e.g., ['path_1', 'path_2'])
            window_length: Length of time window in samples (default 120 for 1 second at 120Hz)
            stride: Stride for sliding windows (default 120 for non-overlapping)
            target: Regression target ('distance', 'altitude', or 'both')
            n_imus: Number of IMUs to use
            imu_ids: Specific IMU IDs to use (if None, uses first n_imus)
            mode: 'rda' (Raw Data Average) or 'ara' (After Regression Average)
            normalize: Whether to normalize data
            normalizer: Pre-fitted normalizer (if None, will compute from data)
            augment: Whether to apply data augmentation
            augment_params: Parameters for augmentation
            sampling_rate: IMU sampling rate in Hz
            gt_sampling_rate: Ground truth sampling rate in Hz
            dataset_type: Type of dataset ('horizontal', 'vertical', 'straightline')
        """
        self.base_dir = Path(data_dir)
        self.trajectory_ids = trajectory_ids
        self.window_length = window_length
        self.stride = stride
        self.target = target
        self.n_imus = n_imus
        self.imu_ids = imu_ids if imu_ids is not None else list(range(1, n_imus + 1))
        self.mode = mode
        self.normalize = normalize
        self.normalizer = normalizer
        self.augment = augment
        self.augment_params = augment_params or {}
        self.sampling_rate = sampling_rate
        self.gt_sampling_rate = gt_sampling_rate
        self.dataset_type = dataset_type
        self.data_dir = resolve_dataset_path(self.base_dir, self.dataset_type)
        
        # Load all trajectories and create windows
        self.windows = []
        self._load_trajectories()
        
        # Fit normalizer if not provided
        if self.normalize and self.normalizer is None:
            self._fit_normalizer()
    
    def _load_trajectories(self):
        """Load all trajectories and create sliding windows."""
        for traj_id in self.trajectory_ids:
            traj_path = self.data_dir / traj_id



            # Load ground truth
            gt_file = traj_path / 'GT.csv'
            if not gt_file.exists():
                print(f"Warning: GT.csv not found for {traj_id}, skipping...")
                continue
            
            gt_data = pd.read_csv(gt_file)
            
            # Load IMU data
            imu_data_list = []
            available_imu_ids = []
            
            for imu_id in self.imu_ids:
                imu_file = traj_path / f'IMU_{imu_id}.csv'
                if imu_file.exists():
                    imu_data = pd.read_csv(imu_file)
                    imu_data_list.append(imu_data)
                    available_imu_ids.append(imu_id)
                else:
                    print(f"Warning: IMU_{imu_id}.csv not found for {traj_id}")
            
            if len(imu_data_list) == 0:
                print(f"Warning: No IMU data found for {traj_id}, skipping...")
                continue
            
            # Update n_imus to actual available IMUs
            actual_n_imus = len(imu_data_list)
            
            # Align and resample IMU data to common timebase
            imu_data_aligned = self._align_imu_data(imu_data_list)
            
            # Resample GT to IMU timebase for window-based labels
            gt_aligned = self._align_gt_to_imu(gt_data, imu_data_aligned)
            
            # Create sliding windows
            self._create_windows(imu_data_aligned, gt_aligned, traj_id, actual_n_imus)
    
    def _align_imu_data(self, imu_data_list: List[pd.DataFrame]) -> np.ndarray:
        """
        Align multiple IMU data to common timebase and extract 6 channels.
        
        Returns:
            Array of shape (n_samples, n_imus, 6) with [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
        """
        # Find common time range
        time_min = max([df['time'].min() for df in imu_data_list])
        time_max = min([df['time'].max() for df in imu_data_list])
        
        # Create common timebase (assuming uniform sampling, use first IMU's rate)
        dt = 1.0 / self.sampling_rate
        time_common = np.arange(time_min, time_max, dt)
        
        # Interpolate each IMU to common timebase
        imu_channels = []
        for imu_df in imu_data_list:
            # Extract 6 channels: Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z
            acc_xyz = imu_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
            gyr_xyz = imu_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
            imu_6ch = np.concatenate([acc_xyz, gyr_xyz], axis=1)  # (n, 6)
            time_imu = imu_df['time'].values
            
            # Interpolate to common timebase
            if len(time_common) > len(time_imu):
                # Upsample
                f = interp1d(time_imu, imu_6ch, axis=0, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
                imu_interp = f(time_common)
            else:
                # Downsample or use as-is
                f = interp1d(time_imu, imu_6ch, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
                imu_interp = f(time_common)
            
            imu_channels.append(imu_interp)
        
        # Stack: (n_samples, n_imus, 6)
        return np.stack(imu_channels, axis=1)
    
    def _align_gt_to_imu(self, gt_data: pd.DataFrame, 
                        imu_data_aligned: np.ndarray) -> pd.DataFrame:
        """Resample GT data to IMU timebase."""
        n_samples = imu_data_aligned.shape[0]
        time_min = gt_data['time'].min()
        dt = 1.0 / self.sampling_rate
        time_common = np.arange(time_min, time_min + n_samples * dt, dt)[:n_samples]
        
        # Interpolate GT columns
        gt_interp = pd.DataFrame()
        gt_interp['time'] = time_common
        
        for col in ['distance(meters)', 'height_above_takeoff(meters)', 
                   'North', 'East', 'Down']:
            if col in gt_data.columns:
                f = interp1d(gt_data['time'].values, gt_data[col].values,
                           kind='linear', bounds_error=False, fill_value='extrapolate')
                gt_interp[col] = f(time_common)
        
        return gt_interp
    
    def _create_windows(self, imu_data: np.ndarray, gt_data: pd.DataFrame,
                       traj_id: str, n_imus: int):
        """
        Create sliding windows from aligned data.
        
        Args:
            imu_data: Array of shape (n_samples, n_imus, 6)
            gt_data: DataFrame with GT columns
            traj_id: Trajectory ID
            n_imus: Number of IMUs
        """
        n_samples = imu_data.shape[0]
        
        for start_idx in range(0, n_samples - self.window_length + 1, self.stride):
            end_idx = start_idx + self.window_length
            
            # Extract IMU window: (window_length, n_imus, 6)
            imu_window = imu_data[start_idx:end_idx, :, :]
            
            # Extract GT window for labels
            gt_window = gt_data.iloc[start_idx:end_idx]
            
            # Compute labels
            if self.target == 'distance':
                # Compute horizontal distance change
                if 'North' in gt_window.columns and 'East' in gt_window.columns:
                    positions = gt_window[['North', 'East']].values
                    label = compute_distance_delta(positions)
                elif 'distance(meters)' in gt_window.columns:
                    # Use distance column directly
                    distances = gt_window['distance(meters)'].values
                    label = distances[-1] - distances[0] if len(distances) > 1 else 0.0
                else:
                    label = 0.0
                labels = np.array([label])
                
            elif self.target == 'altitude':
                # Compute altitude change
                if 'height_above_takeoff(meters)' in gt_window.columns:
                    heights = gt_window['height_above_takeoff(meters)'].values
                    label = compute_altitude_delta(heights)
                elif 'Down' in gt_window.columns:
                    # Down is negative of altitude
                    down_values = gt_window['Down'].values
                    label = -(down_values[-1] - down_values[0]) if len(down_values) > 1 else 0.0
                else:
                    label = 0.0
                labels = np.array([label])
                
            elif self.target == 'both':
                # Both distance and altitude
                distance_label = 0.0
                altitude_label = 0.0
                
                if 'North' in gt_window.columns and 'East' in gt_window.columns:
                    positions = gt_window[['North', 'East']].values
                    distance_label = compute_distance_delta(positions)
                elif 'distance(meters)' in gt_window.columns:
                    distances = gt_window['distance(meters)'].values
                    distance_label = distances[-1] - distances[0] if len(distances) > 1 else 0.0
                
                if 'height_above_takeoff(meters)' in gt_window.columns:
                    heights = gt_window['height_above_takeoff(meters)'].values
                    altitude_label = compute_altitude_delta(heights)
                elif 'Down' in gt_window.columns:
                    down_values = gt_window['Down'].values
                    altitude_label = -(down_values[-1] - down_values[0]) if len(down_values) > 1 else 0.0
                
                labels = np.array([distance_label, altitude_label])
            else:
                raise ValueError(f"Unknown target: {self.target}")
            
            # Reshape for model input
            # For RDA: will be averaged to (6, window_length)
            # For ARA: will be (6*n_imus, window_length) or kept as (n_imus, 6, window_length)
            if self.mode == 'rda':
                # Average across IMUs: (window_length, n_imus, 6) -> (window_length, 6)
                imu_window_avg = np.mean(imu_window, axis=1)  # (window_length, 6)
                # Transpose to (6, window_length) for model
                imu_window_processed = imu_window_avg.T
            else:  # ARA mode
                # Flatten IMUs: (window_length, n_imus, 6) -> (window_length, 6*n_imus)
                imu_window_flat = imu_window.reshape(self.window_length, -1)  # (window_length, 6*n_imus)
                # Transpose to (6*n_imus, window_length)
                imu_window_processed = imu_window_flat.T
            
            # Store window
            window_data = {
                'data': imu_window_processed.astype(np.float32),
                'label': labels.astype(np.float32),
                'trajectory_id': traj_id,
                'start_idx': start_idx,
                'timestamp_start': gt_window['time'].iloc[0] if 'time' in gt_window.columns else start_idx / self.sampling_rate,
                'n_imus': n_imus,
                'imu_ids': self.imu_ids[:n_imus]
            }
            self.windows.append(window_data)
    
    def _fit_normalizer(self):
        """Fit normalizer on all windows."""
        all_data = np.stack([w['data'] for w in self.windows])  # (N, C, L)
        self.normalizer = Normalizer(per_imu=(not self.mode == 'rda'), 
                                    n_imus=self.n_imus if self.mode == 'ara' else 1)
        self.normalizer.fit(all_data)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single window."""
        window = self.windows[idx]
        data = window['data'].copy()  # (C, L)
        label = window['label'].copy()
        
        # Apply augmentation if training
        if self.augment:
            data = augment_data(data, **self.augment_params)
        
        # Normalize
        if self.normalize and self.normalizer is not None:
            # Reshape to (1, C, L) for normalizer
            data = self.normalizer.transform(data[np.newaxis, :, :])[0]
        
        # Convert to torch tensors
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(label).float()
        
        return {
            'data': data_tensor,
            'label': label_tensor,
            'trajectory_id': window['trajectory_id'],
            'timestamp_start': window['timestamp_start'],
            'imu_ids': window['imu_ids']
        }


def get_trajectory_ids(data_dir: str, dataset_type: str = 'horizontal') -> List[str]:
    """
    Get list of available trajectory IDs from dataset directory (case-insensitive).

    This version automatically detects the correct subfolder
    (e.g. "Horizontal", "Vertical", "StraightLine") even if the
    dataset_type argument uses a different case.
    """
    data_path = resolve_dataset_path(data_dir, dataset_type)
    dataset_type_lower = dataset_type.lower() if dataset_type else ''

    # Find all trajectory directories (path_1, path_2, etc.)
    trajectory_dirs = sorted([
        d.name for d in data_path.iterdir()
        if d.is_dir() and d.name.startswith('path_')
    ])

    if len(trajectory_dirs) == 0:
        raise ValueError(f"[ERROR] No trajectory folders found under: {data_path}")

    print(f"[INFO] Using dataset path: {data_path} ({len(trajectory_dirs)} trajectories found)")
    return trajectory_dirs


def create_data_splits(data_dir: str,
                      dataset_type: str = 'Horizontal',
                      test_trajectory_id: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Create train/val/test splits for the QuadNet MIMU dataset.

    Automatically handles case-insensitive dataset names like:
    'Horizontal', 'Vertical', or 'StraightLine'.

    Default test trajectories:
      - Horizontal → path_4
      - Vertical → path_9
      - StraightLine → path_4
    """
    dataset_type_lower = dataset_type.lower()
    trajectory_ids = get_trajectory_ids(data_dir, dataset_type)

    if dataset_type_lower == 'horizontal':
        # D1/D3: use path_4 as test set
        test_trajectory_id = test_trajectory_id or 'path_4'
    elif dataset_type_lower == 'vertical':
        # D2/D4: use path_9 as test set
        test_trajectory_id = test_trajectory_id or 'path_9'
    elif dataset_type_lower == 'straightline':
        # Default for straight line: path_4
        test_trajectory_id = test_trajectory_id or 'path_4'
    else:
        # Generic fallback
        n_test = max(1, int(0.2 * len(trajectory_ids)))
        test_ids = trajectory_ids[-n_test:]
        train_val_ids = trajectory_ids[:-n_test]
        n_train = int(0.8 * len(train_val_ids))
        return {
            'train': train_val_ids[:n_train],
            'val': train_val_ids[n_train:],
            'test': test_ids
        }

    # Make sure test trajectory exists
    if test_trajectory_id not in trajectory_ids:
        raise ValueError(f"[ERROR] Test trajectory '{test_trajectory_id}' not found in dataset.")

    # Split remaining into train/val
    test_ids = [test_trajectory_id]
    train_val_ids = [tid for tid in trajectory_ids if tid != test_trajectory_id]
    n_train = int(0.8 * len(train_val_ids))
    train_ids = train_val_ids[:n_train]
    val_ids = train_val_ids[n_train:]

    print(f"[INFO] Dataset type: {dataset_type} | Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }



def create_dataloader(data_dir: str,
                     trajectory_ids: List[str],
                     window_length: int = 120,
                     stride: int = 120,
                     target: str = 'distance',
                     n_imus: int = 4,
                     imu_ids: Optional[List[int]] = None,
                     mode: str = 'rda',
                     batch_size: int = 64,
                     shuffle: bool = True,
                     normalize: bool = True,
                     normalizer: Optional[Normalizer] = None,
                     augment: bool = False,
                     augment_params: Optional[Dict] = None,
                     num_workers: int = 4,
                     **kwargs) -> Tuple[DataLoader, Optional[Normalizer]]:
    """
    Create DataLoader for QuadNet dataset.
    
    Returns:
        DataLoader and normalizer (if fitted)
    """
    dataset = QuadNetDataset(
        data_dir=data_dir,
        trajectory_ids=trajectory_ids,
        window_length=window_length,
        stride=stride,
        target=target,
        n_imus=n_imus,
        imu_ids=imu_ids,
        mode=mode,
        normalize=normalize,
        normalizer=normalizer,
        augment=augment,
        augment_params=augment_params,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, dataset.normalizer

