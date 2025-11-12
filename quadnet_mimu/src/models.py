"""
QuadNet model implementation with RDA and ARA multi-IMU strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import numpy as np


class QuadNet(nn.Module):
    """
    QuadNet 1D-CNN + FC regression model.
    
    Architecture:
    - 7 convolutional 1D layers with BatchNorm and ReLU
    - Fully connected layers for regression
    - Output: scalar (distance) or 2D (distance, altitude)
    """
    
    def __init__(self,
                 in_channels: int = 6,
                 window_length: int = 120,
                 out_dim: int = 1,
                 conv_channels: List[int] = None,
                 conv_kernels: List[int] = None,
                 fc_dims: List[int] = None,
                 dropout: float = 0.5,
                 use_batch_norm: bool = True,
                 pool_sizes: List[int] = None):
        """
        Args:
            in_channels: Number of input channels (6 for single IMU, 6*n for multi-IMU in ARA)
            window_length: Length of time window
            out_dim: Output dimension (1 for distance/altitude, 2 for both)
            conv_channels: List of output channels for each conv layer
            conv_kernels: List of kernel sizes for each conv layer
            fc_dims: List of dimensions for FC layers (before output)
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            pool_sizes: List of pooling sizes (None = no pooling, 2 = pool after that layer)
        """
        super(QuadNet, self).__init__()
        
        # Default architecture (configurable)
        if conv_channels is None:
            conv_channels = [64, 128, 128, 256, 256, 512, 512]
        if conv_kernels is None:
            conv_kernels = [5, 5, 3, 3, 3, 3, 3]
        if fc_dims is None:
            fc_dims = [256, 128]
        if pool_sizes is None:
            pool_sizes = [0, 2, 0, 2, 0, 0, 0]  # Pool after layer 1 and 3
        
        self.in_channels = in_channels
        self.window_length = window_length
        self.out_dim = out_dim
        self.conv_channels = conv_channels
        self.use_batch_norm = use_batch_norm
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels
        current_length = window_length
        
        for i, (out_ch, kernel_size) in enumerate(zip(conv_channels, conv_kernels)):
            # Padding to maintain length (for kernel_size=3, padding=1; for kernel_size=5, padding=2)
            padding = kernel_size // 2
            
            conv = nn.Conv1d(current_channels, out_ch, kernel_size, padding=padding)
            self.conv_layers.append(conv)
            
            if use_batch_norm:
                bn = nn.BatchNorm1d(out_ch)
                self.conv_layers.append(bn)
            
            # ReLU
            self.conv_layers.append(nn.ReLU(inplace=True))
            
            # Pooling if specified
            if i < len(pool_sizes) and pool_sizes[i] > 1:
                pool = nn.MaxPool1d(kernel_size=pool_sizes[i])
                self.conv_layers.append(pool)
                current_length = current_length // pool_sizes[i]
            
            current_channels = out_ch
        
        # Adaptive average pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_input_dim = conv_channels[-1]
        
        for fc_dim in fc_dims:
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_dim))
            self.fc_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                self.fc_layers.append(nn.Dropout(dropout))
            fc_input_dim = fc_dim
        
        # Output layer
        self.output_layer = nn.Linear(fc_input_dim, out_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    # to preventing vanishing/exploding gradients, slow or unstable training
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, in_channels, window_length)
        
        Returns:
            Output tensor of shape (batch, out_dim)
        """
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        
        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        # Output
        x = self.output_layer(x)
        
        return x
    
    def summary(self):
        """Print model summary with parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"QuadNet Model Summary:")
        print(f"  Input channels: {self.in_channels}")
        print(f"  Window length: {self.window_length}")
        print(f"  Output dimension: {self.out_dim}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"\nArchitecture:")
        print(f"  Conv channels: {self.conv_channels}")
        print(f"  FC dims: {[self.fc_layers[i*3].out_features if i*3 < len(self.fc_layers) else 'N/A' for i in range(len(self.fc_layers)//3 + 1)]}")
        
        # Print layer details
        print(f"\nLayer details:")
        sample_input = torch.randn(1, self.in_channels, self.window_length)
        x = sample_input
        layer_idx = 0
        for i, layer in enumerate(self.conv_layers):
            if isinstance(layer, (nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d)):
                x_old = x
                x = layer(x)
                if isinstance(layer, nn.Conv1d):
                    print(f"  Conv{layer_idx}: {x_old.shape} -> {x.shape}")
                    layer_idx += 1
                elif isinstance(layer, nn.MaxPool1d):
                    print(f"  Pool: {x_old.shape} -> {x.shape}")


class QuadNetRDA(nn.Module):
    """
    QuadNet with Raw Data Average (RDA) strategy.
    
    Averages IMU signals across multiple IMUs before feeding to network.
    """
    
    def __init__(self,
                 window_length: int = 120,
                 out_dim: int = 1,
                 n_imus: int = 4,
                 weighted_avg: bool = False,
                 imu_weights: Optional[torch.Tensor] = None,
                 **quadnet_kwargs):
        """
        Args:
            window_length: Length of time window
            out_dim: Output dimension
            n_imus: Number of IMUs
            weighted_avg: Whether to use weighted averaging
            imu_weights: Weights for each IMU (if None, uses uniform weights)
            **quadnet_kwargs: Additional arguments for QuadNet
        """
        super(QuadNetRDA, self).__init__()
        
        # Single QuadNet with 6 input channels (after averaging)
        self.quadnet = QuadNet(in_channels=6, 
                              window_length=window_length,
                              out_dim=out_dim,
                              **quadnet_kwargs)
        
        self.n_imus = n_imus
        self.weighted_avg = weighted_avg
        
        if weighted_avg:
            if imu_weights is None:
                # Uniform weights
                self.register_buffer('imu_weights', torch.ones(n_imus) / n_imus)
            else:
                self.register_buffer('imu_weights', imu_weights / imu_weights.sum())
        else:
            self.register_buffer('imu_weights', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with RDA.
        
        Args:
            x: Input tensor of shape (batch, 6*n_imus, window_length)
               or (batch, n_imus, 6, window_length)
        
        Returns:
            Output tensor of shape (batch, out_dim)
        """
        batch_size = x.shape[0]
        
        # Handle already-averaged input: (batch, 6, L)
        if x.dim() == 3 and x.shape[1] == 6:
            x_avg = x
        else:
            # Reshape if needed: (batch, 6*n_imus, L) -> (batch, n_imus, 6, L)
            if x.shape[1] == 6 * self.n_imus:
                x = x.view(batch_size, self.n_imus, 6, x.shape[2])
            elif x.shape[1] == self.n_imus and x.shape[2] == 6:
                x = x  # Already (batch, n_imus, 6, L)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
            
            # Average across IMU dimension
            if self.weighted_avg and self.imu_weights is not None:
                # Weighted average: (batch, n_imus, 6, L) * (n_imus, 1, 1) -> (batch, 6, L)
                weights = self.imu_weights.view(1, self.n_imus, 1, 1)
                x_avg = (x * weights).sum(dim=1)
            else:
                # Simple arithmetic mean
                x_avg = x.mean(dim=1)  # (batch, 6, L)
        
        # Forward through QuadNet
        return self.quadnet(x_avg)


class QuadNetARA(nn.Module):
    """
    QuadNet with After Regression Average (ARA) strategy.
    
    Each IMU feeds through its own network, then outputs are averaged.
    """
    
    def __init__(self,
                 window_length: int = 120,
                 out_dim: int = 1,
                 n_imus: int = 4,
                 shared_weights: bool = True,
                 **quadnet_kwargs):
        """
        Args:
            window_length: Length of time window
            out_dim: Output dimension
            n_imus: Number of IMUs
            shared_weights: If True, use a single network for all IMUs (more efficient)
                           If False, use separate networks for each IMU
            **quadnet_kwargs: Additional arguments for QuadNet
        """
        super(QuadNetARA, self).__init__()
        
        self.n_imus = n_imus
        self.shared_weights = shared_weights
        
        if shared_weights:
            # Single network applied to each IMU
            self.quadnet = QuadNet(in_channels=6,
                                  window_length=window_length,
                                  out_dim=out_dim,
                                  **quadnet_kwargs)
        else:
            # Separate network for each IMU
            self.quadnets = nn.ModuleList([
                QuadNet(in_channels=6,
                       window_length=window_length,
                       out_dim=out_dim,
                       **quadnet_kwargs)
                for _ in range(n_imus)
            ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ARA.
        
        Args:
            x: Input tensor of shape (batch, 6*n_imus, window_length)
               or (batch, n_imus, 6, window_length)
        
        Returns:
            Output tensor of shape (batch, out_dim)
        """
        batch_size = x.shape[0]
        
        # Reshape if needed: (batch, 6*n_imus, L) -> (batch, n_imus, 6, L)
        if x.shape[1] == 6 * self.n_imus:
            x = x.view(batch_size, self.n_imus, 6, x.shape[2])
        elif x.shape[1] != self.n_imus or x.shape[2] != 6:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Process each IMU
        if self.shared_weights:
            # Apply same network to each IMU
            outputs = []
            for i in range(self.n_imus):
                imu_data = x[:, i, :, :]  # (batch, 6, L)
                output = self.quadnet(imu_data)  # (batch, out_dim)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)  # (n_imus, batch, out_dim)
        else:
            # Apply separate network to each IMU
            outputs = []
            for i, quadnet in enumerate(self.quadnets):
                imu_data = x[:, i, :, :]  # (batch, 6, L)
                output = quadnet(imu_data)  # (batch, out_dim)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)  # (n_imus, batch, out_dim)
        
        # Average across IMUs
        output_avg = outputs.mean(dim=0)  # (batch, out_dim)
        
        return output_avg


def create_model(mode: str = 'rda',
                n_imus: int = 4,
                window_length: int = 120,
                out_dim: int = 1,
                shared_weights: bool = True,
                **kwargs) -> nn.Module:
    """
    Factory function to create QuadNet model with specified mode.
    
    Args:
        mode: 'rda' or 'ara'
        n_imus: Number of IMUs
        window_length: Window length
        out_dim: Output dimension
        shared_weights: For ARA, whether to use shared weights
        **kwargs: Additional arguments for QuadNet
    
    Returns:
        QuadNet model (RDA or ARA)
    """
    if mode.lower() == 'rda':
        return QuadNetRDA(
            window_length=window_length,
            out_dim=out_dim,
            n_imus=n_imus,
            **kwargs
        )
    elif mode.lower() == 'ara':
        return QuadNetARA(
            window_length=window_length,
            out_dim=out_dim,
            n_imus=n_imus,
            shared_weights=shared_weights,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'rda' or 'ara'.")

