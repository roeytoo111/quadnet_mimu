#!/usr/bin/env python3
"""
Example usage of QuadNet MIMU codebase.

This script demonstrates how to:
1. Load and inspect the dataset
2. Create a model
3. Train a model
4. Evaluate a model
5. View results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.datasets import create_dataloader, create_data_splits, get_trajectory_ids
from src.models import create_model
from src.utils import set_seed, get_device
import torch
import numpy as np

def example_1_load_dataset():
    """Example 1: Load and inspect dataset"""
    print("=" * 50)
    print("Example 1: Load Dataset")
    print("=" * 50)
    
    data_dir = "../dataset_repo"
    dataset_type = 'horizontal'
    
    # Get available trajectories
    trajectory_ids = get_trajectory_ids(data_dir, dataset_type)
    print(f"Available trajectories: {len(trajectory_ids)}")
    print(f"First 5 trajectories: {trajectory_ids[:5]}")
    
    # Create data splits
    splits = create_data_splits(data_dir, dataset_type=dataset_type)
    print(f"\nData splits:")
    print(f"  Train: {len(splits['train'])} trajectories")
    print(f"  Val: {len(splits['val'])} trajectories")
    print(f"  Test: {len(splits['test'])} trajectories")
    
    # Create a data loader
    train_loader, normalizer = create_dataloader(
        data_dir=data_dir,
        trajectory_ids=splits['train'][:2],  # Use first 2 trajectories for demo
        window_length=120,
        stride=120,
        target='distance',
        n_imus=4,
        mode='rda',
        batch_size=2,
        shuffle=False,
        normalize=True,
        augment=False,
        dataset_type=dataset_type
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Data: {batch['data'].shape}")  # (batch, channels, window_length)
    print(f"  Labels: {batch['label'].shape}")  # (batch, out_dim)
    print(f"  Trajectory IDs: {batch['trajectory_id']}")


def example_2_create_model():
    """Example 2: Create and inspect model"""
    print("\n" + "=" * 50)
    print("Example 2: Create Model")
    print("=" * 50)
    
    # Create RDA model
    model_rda = create_model(
        mode='rda',
        n_imus=4,
        window_length=120,
        out_dim=1
    )
    
    print("RDA Model:")
    # model_rda.summary()
    
    # Create ARA model
    model_ara = create_model(
        mode='ara',
        n_imus=4,
        window_length=120,
        out_dim=1,
        shared_weights=True
    )
    
    print("\nARA Model:")
    # model_ara.summary()
    
    # Test forward pass
    device = get_device()
    model_rda = model_rda.to(device)
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 24, 120).to(device)  # (batch, 6*4, window_length)
    
    # Forward pass
    with torch.no_grad():
        output = model_rda(x)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")


def example_3_train_model():
    """Example 3: Train a model (simplified)"""
    print("\n" + "=" * 50)
    print("Example 3: Train Model")
    print("=" * 50)
    
    print("To train a model, run:")
    print("  python src/train.py --config configs/default.yaml --mode rda --n_imus 4 --split D1 --data_dir ../dataset_repo")
    print("\nThis will:")
    print("  1. Load dataset from ../dataset_repo")
    print("  2. Create train/val/test splits")
    print("  3. Train QuadNet model")
    print("  4. Save checkpoint to results/checkpoints/")
    print("  5. Log metrics to TensorBoard")


def example_4_evaluate_model():
    """Example 4: Evaluate a model"""
    print("\n" + "=" * 50)
    print("Example 4: Evaluate Model")
    print("=" * 50)
    
    print("To evaluate a model, run:")
    print("  python src/eval.py --checkpoint results/checkpoints/D1_rda_4im_best.pth --split D1 --mode rda --n_imus 4 --data_dir ../dataset_repo")
    print("\nThis will:")
    print("  1. Load model from checkpoint")
    print("  2. Evaluate on test trajectories")
    print("  3. Compute metrics (RMSE, MAE, max error, std error)")
    print("  4. Save results to results/D1_rda_4im_eval.csv")


def example_5_view_results():
    """Example 5: View results"""
    print("\n" + "=" * 50)
    print("Example 5: View Results")
    print("=" * 50)
    
    print("Results are saved to:")
    print("  1. Checkpoints: results/checkpoints/")
    print("  2. Metrics: results/*_metrics.json")
    print("  3. Evaluation CSV: results/*_eval.csv")
    print("  4. TensorBoard logs: results/logs/")
    print("\nTo view TensorBoard:")
    print("  tensorboard --logdir results/logs")
    print("\nTo view metrics:")
    print("  cat results/D1_rda_4im_metrics.json")


def main():
    """Run all examples"""
    print("QuadNet MIMU Example Usage")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Run examples
    try:
        example_1_load_dataset()
    except Exception as e:
        print(f"Error in Example 1: {e}")
        print("Make sure dataset is available at ../dataset_repo")
    
    example_2_create_model()
    example_3_train_model()
    example_4_evaluate_model()
    example_5_view_results()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)
    print("\nFor more details, see README.md")


if __name__ == '__main__':
    main()



