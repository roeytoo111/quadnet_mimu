"""
Training script for QuadNet MIMU model.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import os
import sys
from typing import Optional, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import create_dataloader, create_data_splits
from models import create_model
from utils import (set_seed, get_device, save_checkpoint, 
                  Normalizer, compute_distance_delta, compute_altitude_delta)


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute RMSE."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute MAE."""
    return np.mean(np.abs(predictions - targets))


def train_epoch(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               device: torch.device,
               epoch: int,
               writer: Optional[SummaryWriter] = None) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        data = batch['data'].to(device)  # (batch, C, L)
        labels = batch['label'].to(device)  # (batch, out_dim)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)  # (batch, out_dim)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        all_predictions.append(outputs.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Aggregate metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    avg_loss = total_loss / len(dataloader)
    rmse = compute_rmse(all_predictions, all_targets)
    mae = compute_mae(all_predictions, all_targets)
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/RMSE', rmse, epoch)
        writer.add_scalar('Train/MAE', mae, epoch)
    
    return {
        'loss': avg_loss,
        'rmse': rmse,
        'mae': mae
    }


def validate_epoch(model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  epoch: int,
                  writer: Optional[SummaryWriter] = None) -> Dict:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            data = batch['data'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Metrics
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Aggregate metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    avg_loss = total_loss / len(dataloader)
    rmse = compute_rmse(all_predictions, all_targets)
    mae = compute_mae(all_predictions, all_targets)
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/RMSE', rmse, epoch)
        writer.add_scalar('Val/MAE', mae, epoch)
    
    return {
        'loss': avg_loss,
        'rmse': rmse,
        'mae': mae
    }


def main():
    parser = argparse.ArgumentParser(description='Train QuadNet MIMU model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--mode', type=str, choices=['rda', 'ara'], default='rda',
                       help='Multi-IMU strategy: rda or ara')
    parser.add_argument('--n_imus', type=int, default=4,
                       help='Number of IMUs to use')
    parser.add_argument('--split', type=str, default='D1',
                       help='Dataset split (D1, D2, D3, D4)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Dataset directory (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.device is not None:
        config['training']['device'] = args.device
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = get_device(config['training'].get('device', None))
    print(f"Using device: {device}")
    
    # Dataset setup
    data_dir = config['data']['data_dir']
    dataset_type = 'horizontal' if args.split in ['D1', 'D3'] else 'vertical'
    
    # Create data splits
    splits = create_data_splits(data_dir, dataset_type=dataset_type)
    
    # Determine test trajectory based on split
    if args.split == 'D1':
        test_trajectory = 'path_4'  # From guide.txt
    elif args.split == 'D2':
        test_trajectory = 'path_9'  # Typical convention
    elif args.split == 'D3':
        test_trajectory = 'path_4'  # Same as D1
    elif args.split == 'D4':
        test_trajectory = 'path_9'  # Same as D2
    else:
        test_trajectory = splits['test'][0]
    
    train_ids = splits['train']
    val_ids = splits['val']
    test_ids = [test_trajectory] if test_trajectory in splits['test'] else splits['test']
    
    print(f"Train trajectories: {len(train_ids)}")
    print(f"Val trajectories: {len(val_ids)}")
    print(f"Test trajectories: {len(test_ids)}")
    
    # Create data loaders
    train_loader, normalizer = create_dataloader(
        data_dir=data_dir,
        trajectory_ids=train_ids,
        window_length=config['data']['window_length'],
        stride=config['data'].get('stride', config['data']['window_length']),
        target=config['data']['target'],
        n_imus=args.n_imus,
        mode=args.mode,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        normalize=True,
        augment=config['training'].get('augment', False),
        augment_params=config['training'].get('augment_params', {}),
        dataset_type=dataset_type,
        sampling_rate=config['data'].get('sampling_rate', 120.0),
        gt_sampling_rate=config['data'].get('gt_sampling_rate', 10.0)
    )
    
    val_loader, _ = create_dataloader(
        data_dir=data_dir,
        trajectory_ids=val_ids,
        window_length=config['data']['window_length'],
        stride=config['data'].get('stride', config['data']['window_length']),
        target=config['data']['target'],
        n_imus=args.n_imus,
        mode=args.mode,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        normalize=True,
        normalizer=normalizer,
        augment=False,
        dataset_type=dataset_type,
        sampling_rate=config['data'].get('sampling_rate', 120.0),
        gt_sampling_rate=config['data'].get('gt_sampling_rate', 10.0)
    )
    
    # Create model
    model = create_model(
        mode=args.mode,
        n_imus=args.n_imus,
        window_length=config['data']['window_length'],
        out_dim=1 if config['data']['target'] != 'both' else 2,
        shared_weights=config['model'].get('ara_shared_weights', True),
        **config['model'].get('quadnet_kwargs', {})
    )
    
    model = model.to(device)
    if hasattr(model, 'summary') and callable(getattr(model, 'summary')):
        model.summary()
    else:
        print(model)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    # Scheduler
    scheduler = None
    if config['training'].get('scheduler', None) == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    elif config['training'].get('scheduler', None) == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_rmse = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_rmse = checkpoint.get('metrics', {}).get('val_rmse', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Create output directories
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # TensorBoard writer
    exp_name = f"{args.split}_{args.mode}_{args.n_imus}im"
    writer = SummaryWriter(log_dir=str(logs_dir / exp_name))
    
    # Training loop
    epochs = config['training']['epochs']
    early_stop_patience = config['training'].get('early_stop_patience', 20)
    patience_counter = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Experiment: {exp_name}")
    
    for epoch in range(start_epoch, epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                   device, epoch, writer)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, 
                                    epoch, writer)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['rmse'])
            else:
                scheduler.step()
        
        # Print metrics
        print(f"Epoch {epoch}: "
              f"Train Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        
        # Save best model
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            patience_counter = 0
            
            checkpoint_path = checkpoints_dir / f"{exp_name}_best.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['loss'],
                metrics={'val_rmse': val_metrics['rmse'], 
                        'val_mae': val_metrics['mae'],
                        'train_rmse': train_metrics['rmse']},
                config=config,
                normalizer=normalizer,
                checkpoint_path=str(checkpoint_path)
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save final model
    final_checkpoint_path = checkpoints_dir / f"{exp_name}_final.pth"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs - 1,
        loss=val_metrics['loss'],
        metrics={'val_rmse': val_metrics['rmse'], 
                'val_mae': val_metrics['mae']},
        config=config,
        normalizer=normalizer,
        checkpoint_path=str(final_checkpoint_path)
    )
    
    writer.close()
    print(f"\nTraining completed. Best val RMSE: {best_val_rmse:.4f}")
    print(f"Checkpoints saved to: {checkpoints_dir}")


if __name__ == '__main__':
    main()

