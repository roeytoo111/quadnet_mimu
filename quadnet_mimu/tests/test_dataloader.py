"""
Unit tests for data loaders.
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import QuadNetDataset, create_dataloader, create_data_splits
from utils import Normalizer


class TestDataLoader(unittest.TestCase):
    """Test data loader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Assume dataset is in ../dataset_repo
        self.data_dir = Path(__file__).parent.parent.parent / 'dataset_repo'
        self.trajectory_ids = ['path_1', 'path_2']
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        if not (self.data_dir / 'Horizontal').exists():
            self.skipTest("Dataset not found")
        
        dataset = QuadNetDataset(
            data_dir=str(self.data_dir / 'Horizontal'),
            trajectory_ids=self.trajectory_ids,
            window_length=120,
            stride=120,
            target='distance',
            n_imus=4,
            mode='rda',
            normalize=False,
            augment=False
        )
        
        self.assertGreater(len(dataset), 0)
    
    def test_dataset_shapes(self):
        """Test dataset output shapes."""
        if not (self.data_dir / 'Horizontal').exists():
            self.skipTest("Dataset not found")
        
        dataset = QuadNetDataset(
            data_dir=str(self.data_dir / 'Horizontal'),
            trajectory_ids=self.trajectory_ids,
            window_length=120,
            stride=120,
            target='distance',
            n_imus=4,
            mode='rda',
            normalize=False,
            augment=False
        )
        
        if len(dataset) == 0:
            self.skipTest("No data loaded")
        
        sample = dataset[0]
        
        # Check data shape
        self.assertEqual(sample['data'].shape[0], 6)  # 6 channels for RDA
        self.assertEqual(sample['data'].shape[1], 120)  # window length
        
        # Check label shape
        self.assertEqual(sample['label'].shape[0], 1)  # single target
    
    def test_normalizer(self):
        """Test normalizer functionality."""
        # Create dummy data
        n_samples = 100
        n_channels = 6
        seq_len = 120
        data = np.random.randn(n_samples, n_channels, seq_len)
        
        # Fit normalizer
        normalizer = Normalizer(per_imu=False, n_imus=1)
        normalizer.fit(data)
        
        # Transform
        data_norm = normalizer.transform(data)
        
        # Check that normalized data has zero mean and unit std
        mean_norm = np.mean(data_norm, axis=(0, 2))
        std_norm = np.std(data_norm, axis=(0, 2))
        
        np.testing.assert_allclose(mean_norm, 0, atol=1e-6)
        np.testing.assert_allclose(std_norm, 1, atol=1e-6)
    
    def test_dataloader(self):
        """Test DataLoader creation."""
        if not (self.data_dir / 'Horizontal').exists():
            self.skipTest("Dataset not found")
        
        dataloader, normalizer = create_dataloader(
            data_dir=str(self.data_dir / 'Horizontal'),
            trajectory_ids=self.trajectory_ids,
            window_length=120,
            stride=120,
            target='distance',
            n_imus=4,
            mode='rda',
            batch_size=2,
            shuffle=False,
            normalize=True,
            augment=False
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Check batch shapes
        self.assertEqual(batch['data'].shape[0], 2)  # batch size
        self.assertEqual(batch['data'].shape[1], 6)  # channels
        self.assertEqual(batch['data'].shape[2], 120)  # window length
        self.assertEqual(batch['label'].shape[0], 2)  # batch size


if __name__ == '__main__':
    unittest.main()

