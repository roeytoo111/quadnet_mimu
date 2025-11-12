"""
Utility functions for QuadNet MIMU: normalization, augmentation, seeding, etc.
"""

import numpy as np
import torch
import random
import json
import os
from typing import Dict, Tuple, Optional, List
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Normalizer:
    """Normalize data using mean and std statistics."""
    
    def __init__(self, mean: Optional[np.ndarray] = None, 
                 std: Optional[np.ndarray] = None,
                 per_imu: bool = False,
                 n_imus: int = 1):
        """
        Args:
            mean: Mean values for normalization (shape: (C,) or (n_imus, 6))
            std: Std values for normalization (shape: (C,) or (n_imus, 6))
            per_imu: If True, normalize per IMU separately
            n_imus: Number of IMUs
        """
        self.mean = mean
        self.std = std
        self.per_imu = per_imu
        self.n_imus = n_imus
        
    def fit(self, data: np.ndarray):
        """
        Compute normalization statistics from data.
        
        Args:
            data: Array of shape (N, C, L) where N is number of samples,
                  C is channels (6*n_imus), L is sequence length
        """
        if self.per_imu:
            # Reshape to (N, n_imus, 6, L)
            n_samples, n_channels, seq_len = data.shape
            n_imus = n_channels // 6
            data_reshaped = data.reshape(n_samples, n_imus, 6, seq_len)
            # Compute mean/std per IMU and per channel
            self.mean = np.mean(data_reshaped, axis=(0, 3), keepdims=False)  # (n_imus, 6)
            self.std = np.std(data_reshaped, axis=(0, 3), keepdims=False)  # (n_imus, 6)
            # Avoid division by zero
            self.std = np.where(self.std < 1e-8, 1.0, self.std)
            self.n_imus = n_imus
        else:
            # Global normalization
            self.mean = np.mean(data, axis=(0, 2), keepdims=False)  # (C,)
            self.std = np.std(data, axis=(0, 2), keepdims=False)  # (C,)
            self.std = np.where(self.std < 1e-8, 1.0, self.std)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using computed statistics."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if self.per_imu:
            n_samples, n_channels, seq_len = data.shape
            n_imus = n_channels // 6
            data_reshaped = data.reshape(n_samples, n_imus, 6, seq_len)
            # Normalize per IMU
            data_norm = (data_reshaped - self.mean[:, :, np.newaxis]) / self.std[:, :, np.newaxis]
            return data_norm.reshape(n_samples, n_channels, seq_len)
        else:
            return (data - self.mean[:, np.newaxis]) / self.std[:, np.newaxis]
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if self.per_imu:
            n_samples, n_channels, seq_len = data.shape
            n_imus = n_channels // 6
            data_reshaped = data.reshape(n_samples, n_imus, 6, seq_len)
            data_denorm = data_reshaped * self.std[:, :, np.newaxis] + self.mean[:, :, np.newaxis]
            return data_denorm.reshape(n_samples, n_channels, seq_len)
        else:
            return data * self.std[:, np.newaxis] + self.mean[:, np.newaxis]
    
    def save(self, path: str):
        """Save normalizer to disk."""
        save_dict = {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'per_imu': self.per_imu,
            'n_imus': self.n_imus
        }
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Normalizer':
        """Load normalizer from disk."""
        with open(path, 'r') as f:
            save_dict = json.load(f)
        
        mean = np.array(save_dict['mean']) if save_dict['mean'] is not None else None
        std = np.array(save_dict['std']) if save_dict['std'] is not None else None
        
        return cls(mean=mean, std=std, 
                  per_imu=save_dict['per_imu'],
                  n_imus=save_dict['n_imus'])


def augment_data(data: np.ndarray, 
                 noise_std: float = 0.01,
                 bias_drift: float = 0.001,
                 scale_factor: float = 0.02,
                 seed: Optional[int] = None) -> np.ndarray:
    """
    Apply data augmentation: Gaussian noise, bias drift, axis scaling.
    
    Args:
        data: Input data of shape (C, L)
        noise_std: Standard deviation of additive Gaussian noise
        bias_drift: Magnitude of bias drift
        scale_factor: Magnitude of random axis scaling
        seed: Random seed for reproducibility
    
    Returns:
        Augmented data of same shape
    """
    if seed is not None:
        np.random.seed(seed)
    
    augmented = data.copy()
    n_channels, seq_len = data.shape
    
    # Additive Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, data.shape)
        augmented += noise
    
    # Bias drift (linear drift over time)
    if bias_drift > 0:
        drift = np.linspace(0, bias_drift, seq_len)
        drift = drift[np.newaxis, :] * np.random.normal(0, 1, (n_channels, 1))
        augmented += drift
    
    # Random axis scaling
    if scale_factor > 0:
        scale = 1.0 + np.random.normal(0, scale_factor, (n_channels, 1))
        augmented *= scale
    
    return augmented


def compute_distance_delta(positions: np.ndarray) -> float:
    """
    Compute horizontal distance change from position sequence.
    
    Args:
        positions: Array of shape (N, 3) with [North, East, Down] or (N, 2) with [North, East]
    
    Returns:
        Horizontal distance change (meters)
    """
    if positions.shape[1] == 3:
        # Use only North and East components
        positions_2d = positions[:, :2]
    else:
        positions_2d = positions
    
    if len(positions_2d) < 2:
        return 0.0
    
    # Compute cumulative distance
    diffs = np.diff(positions_2d, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)


def compute_altitude_delta(heights: np.ndarray) -> float:
    """
    Compute altitude change from height sequence.
    
    Args:
        heights: Array of shape (N,) with height values
    
    Returns:
        Altitude change (meters)
    """
    if len(heights) < 2:
        return 0.0
    
    return heights[-1] - heights[0]


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute RMSE."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute MAE."""
    return np.mean(np.abs(predictions - targets))


def get_device(device: Optional[str] = None) -> torch.device:
    """Get PyTorch device (CUDA if available, else CPU)."""
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   metrics: Dict,
                   config: Dict,
                   normalizer: Optional[Normalizer],
                   checkpoint_path: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config,
    }
    
    if normalizer is not None:
        # Save normalizer separately
        normalizer_path = checkpoint_path.replace('.pth', '_normalizer.json')
        normalizer.save(normalizer_path)
        checkpoint['normalizer_path'] = normalizer_path
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str,
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

