"""
Evaluation script for QuadNet MIMU model.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
from tqdm import tqdm
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import create_dataloader, create_data_splits
from models import create_model
from utils import get_device, load_checkpoint, Normalizer, compute_rmse, compute_mae


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Compute evaluation metrics."""
    rmse = compute_rmse(predictions, targets)
    mae = compute_mae(predictions, targets)
    max_error = np.max(np.abs(predictions - targets))
    std_error = np.std(predictions - targets)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'std_error': std_error
    }


def evaluate_trajectory(model: nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: torch.device,
                       target: str = 'distance') -> Dict:
    """Evaluate model on a single trajectory."""
    model.eval()
    all_predictions = []
    all_targets = []
    trajectory_ids = []
    timestamps = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            data = batch['data'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(data)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            trajectory_ids.extend(batch['trajectory_id'])
            timestamps.extend(batch['timestamp_start'].cpu().numpy())
    
    # Aggregate predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    if target == 'both':
        # Separate metrics for distance and altitude
        metrics = {
            'distance': compute_metrics(all_predictions[:, 0], all_targets[:, 0]),
            'altitude': compute_metrics(all_predictions[:, 1], all_targets[:, 1])
        }
    else:
        metrics = compute_metrics(all_predictions, all_targets)
    
    return {
        'metrics': metrics,
        'predictions': all_predictions,
        'targets': all_targets,
        'trajectory_ids': trajectory_ids,
        'timestamps': timestamps
    }


def evaluate_all_trajectories(model: nn.Module,
                             data_dir: str,
                             trajectory_ids: List[str],
                             config: Dict,
                             mode: str,
                             n_imus: int,
                             device: torch.device,
                             normalizer: Optional[Normalizer],
                             dataset_type: str = 'horizontal') -> Dict:
    """Evaluate model on all trajectories."""
    all_results = {}
    
    for traj_id in trajectory_ids:
        print(f"\nEvaluating trajectory: {traj_id}")
        
        # Create dataloader for this trajectory
        dataloader, _ = create_dataloader(
            data_dir=data_dir,
            trajectory_ids=[traj_id],
            window_length=config['data']['window_length'],
            stride=config['data'].get('stride', config['data']['window_length']),
            target=config['data']['target'],
            n_imus=n_imus,
            mode=mode,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            normalize=True,
            normalizer=normalizer,
            augment=False,
            dataset_type=dataset_type,
            sampling_rate=config['data'].get('sampling_rate', 120.0),
            gt_sampling_rate=config['data'].get('gt_sampling_rate', 10.0)
        )
        
        # Evaluate
        results = evaluate_trajectory(model, dataloader, device, config['data']['target'])
        all_results[traj_id] = results
    
    return all_results


def aggregate_metrics(all_results: Dict, target: str = 'distance') -> Dict:
    """Aggregate metrics across all trajectories."""
    if target == 'both':
        distance_metrics = []
        altitude_metrics = []
        
        for traj_id, results in all_results.items():
            distance_metrics.append(results['metrics']['distance'])
            altitude_metrics.append(results['metrics']['altitude'])
        
        # Average metrics
        avg_distance = {
            'rmse': np.mean([m['rmse'] for m in distance_metrics]),
            'mae': np.mean([m['mae'] for m in distance_metrics]),
            'max_error': np.mean([m['max_error'] for m in distance_metrics]),
            'std_error': np.mean([m['std_error'] for m in distance_metrics])
        }
        
        avg_altitude = {
            'rmse': np.mean([m['rmse'] for m in altitude_metrics]),
            'mae': np.mean([m['mae'] for m in altitude_metrics]),
            'max_error': np.mean([m['max_error'] for m in altitude_metrics]),
            'std_error': np.mean([m['std_error'] for m in altitude_metrics])
        }
        
        return {
            'distance': avg_distance,
            'altitude': avg_altitude
        }
    else:
        metrics_list = [results['metrics'] for results in all_results.values()]
        
        return {
            'rmse': np.mean([m['rmse'] for m in metrics_list]),
            'mae': np.mean([m['mae'] for m in metrics_list]),
            'max_error': np.mean([m['max_error'] for m in metrics_list]),
            'std_error': np.mean([m['std_error'] for m in metrics_list])
        }


def save_results_csv(all_results: Dict, 
                    output_path: str,
                    target: str = 'distance'):
    """Save evaluation results to CSV."""
    rows = []
    
    for traj_id, results in all_results.items():
        if target == 'both':
            row = {
                'trajectory_id': traj_id,
                'distance_rmse': results['metrics']['distance']['rmse'],
                'distance_mae': results['metrics']['distance']['mae'],
                'distance_max_error': results['metrics']['distance']['max_error'],
                'distance_std_error': results['metrics']['distance']['std_error'],
                'altitude_rmse': results['metrics']['altitude']['rmse'],
                'altitude_mae': results['metrics']['altitude']['mae'],
                'altitude_max_error': results['metrics']['altitude']['max_error'],
                'altitude_std_error': results['metrics']['altitude']['std_error']
            }
        else:
            row = {
                'trajectory_id': traj_id,
                'rmse': results['metrics']['rmse'],
                'mae': results['metrics']['mae'],
                'max_error': results['metrics']['max_error'],
                'std_error': results['metrics']['std_error']
            }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate QuadNet MIMU model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='D1',
                       help='Dataset split (D1, D2, D3, D4)')
    parser.add_argument('--mode', type=str, choices=['rda', 'ara'], default='rda',
                       help='Multi-IMU strategy: rda or ara')
    parser.add_argument('--n_imus', type=int, default=4,
                       help='Number of IMUs to use')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Dataset directory (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--trajectories', type=str, nargs='+', default=None,
                       help='Specific trajectories to evaluate (if None, evaluates test set)')
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, None)
    config = checkpoint['config']
    
    # Override config with command line arguments
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.device is not None:
        config['training']['device'] = args.device
    
    # Device
    device = get_device(config['training'].get('device', None))
    print(f"Using device: {device}")
    
    # Dataset setup
    data_dir = config['data']['data_dir']
    dataset_type = 'horizontal' if args.split in ['D1', 'D3'] else 'vertical'
    
    # Create data splits
    splits = create_data_splits(data_dir, dataset_type=dataset_type)
    
    # Determine test trajectories
    if args.trajectories is not None:
        test_ids = args.trajectories
    else:
        if args.split == 'D1':
            test_trajectory = 'path_4'
        elif args.split == 'D2':
            test_trajectory = 'path_9'
        elif args.split == 'D3':
            test_trajectory = 'path_4'
        elif args.split == 'D4':
            test_trajectory = 'path_9'
        else:
            test_trajectory = splits['test'][0]
        
        test_ids = [test_trajectory] if test_trajectory in splits['test'] else splits['test']
    
    print(f"Evaluating on trajectories: {test_ids}")
    
    # Load normalizer
    normalizer = None
    if 'normalizer_path' in checkpoint:
        normalizer = Normalizer.load(checkpoint['normalizer_path'])
    elif Path(args.checkpoint).parent.parent / 'checkpoints' / f"{args.split}_{args.mode}_{args.n_imus}im_best_normalizer.json":
        normalizer_path = Path(args.checkpoint).parent.parent / 'checkpoints' / f"{args.split}_{args.mode}_{args.n_imus}im_best_normalizer.json"
        if normalizer_path.exists():
            normalizer = Normalizer.load(str(normalizer_path))
    
    # Create model
    model = create_model(
        mode=args.mode,
        n_imus=args.n_imus,
        window_length=config['data']['window_length'],
        out_dim=1 if config['data']['target'] != 'both' else 2,
        shared_weights=config['model'].get('ara_shared_weights', True),
        **config['model'].get('quadnet_kwargs', {})
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    # Evaluate
    all_results = evaluate_all_trajectories(
        model=model,
        data_dir=data_dir,
        trajectory_ids=test_ids,
        config=config,
        mode=args.mode,
        n_imus=args.n_imus,
        device=device,
        normalizer=normalizer,
        dataset_type=dataset_type
    )
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_results, config['data']['target'])
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    
    if config['data']['target'] == 'both':
        print("\nDistance Metrics:")
        for metric, value in aggregated['distance'].items():
            print(f"  {metric}: {value:.4f}")
        print("\nAltitude Metrics:")
        for metric, value in aggregated['altitude'].items():
            print(f"  {metric}: {value:.4f}")
    else:
        for metric, value in aggregated.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exp_name = f"{args.split}_{args.mode}_{args.n_imus}im"
    results_csv = output_dir / f"{exp_name}_eval.csv"
    save_results_csv(all_results, str(results_csv), config['data']['target'])
    
    # Save aggregated metrics
    metrics_json = output_dir / f"{exp_name}_metrics.json"
    # Helper serializer to convert numpy/torch types to native Python types
    def _json_serializer(obj):
        try:
            import numpy as _np
            import torch as _torch
        except Exception:
            _np = None
            _torch = None

        # numpy scalar
        if _np is not None and isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        # numpy arrays
        if _np is not None and isinstance(obj, _np.ndarray):
            return obj.tolist()
        # torch tensors
        if _torch is not None and isinstance(obj, _torch.Tensor):
            return obj.cpu().numpy().tolist()

        # Fallback to string representation
        return str(obj)

    with open(metrics_json, 'w') as f:
        json.dump(aggregated, f, indent=2, default=_json_serializer)
    print(f"\nMetrics saved to {metrics_json}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()

