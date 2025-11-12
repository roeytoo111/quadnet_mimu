"""
Unit tests for model forward pass.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import QuadNet, QuadNetRDA, QuadNetARA, create_model


class TestModelForward(unittest.TestCase):
    """Test model forward pass."""
    
    def test_quadnet_forward(self):
        """Test QuadNet forward pass."""
        batch_size = 4
        in_channels = 6
        window_length = 120
        out_dim = 1
        
        model = QuadNet(
            in_channels=in_channels,
            window_length=window_length,
            out_dim=out_dim
        )
        
        # Create dummy input
        x = torch.randn(batch_size, in_channels, window_length)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_dim))
    
    def test_quadnet_rda_forward(self):
        """Test QuadNetRDA forward pass."""
        batch_size = 4
        n_imus = 4
        window_length = 120
        out_dim = 1
        
        model = QuadNetRDA(
            window_length=window_length,
            out_dim=out_dim,
            n_imus=n_imus
        )
        
        # Create dummy input: (batch, 6*n_imus, L)
        x = torch.randn(batch_size, 6 * n_imus, window_length)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_dim))
    
    def test_quadnet_ara_forward(self):
        """Test QuadNetARA forward pass."""
        batch_size = 4
        n_imus = 4
        window_length = 120
        out_dim = 1
        
        model = QuadNetARA(
            window_length=window_length,
            out_dim=out_dim,
            n_imus=n_imus,
            shared_weights=True
        )
        
        # Create dummy input: (batch, 6*n_imus, L)
        x = torch.randn(batch_size, 6 * n_imus, window_length)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_dim))
    
    def test_create_model_rda(self):
        """Test model factory for RDA."""
        model = create_model(
            mode='rda',
            n_imus=4,
            window_length=120,
            out_dim=1
        )
        
        self.assertIsInstance(model, QuadNetRDA)
        
        # Test forward
        x = torch.randn(2, 24, 120)  # (batch, 6*4, L)
        output = model(x)
        self.assertEqual(output.shape, (2, 1))
    
    def test_create_model_ara(self):
        """Test model factory for ARA."""
        model = create_model(
            mode='ara',
            n_imus=4,
            window_length=120,
            out_dim=1,
            shared_weights=True
        )
        
        self.assertIsInstance(model, QuadNetARA)
        
        # Test forward
        x = torch.randn(2, 24, 120)  # (batch, 6*4, L)
        output = model(x)
        self.assertEqual(output.shape, (2, 1))
    
    def test_model_summary(self):
        """Test model summary method."""
        model = QuadNet(
            in_channels=6,
            window_length=120,
            out_dim=1
        )
        
        # Should not raise an error
        model.summary()


if __name__ == '__main__':
    unittest.main()

