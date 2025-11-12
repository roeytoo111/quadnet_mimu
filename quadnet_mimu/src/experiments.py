"""
Experiment script to run full sweeps reproducing paper tables.
"""

import argparse
import yaml
import subprocess
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List
import itertools


def run_experiment(config_path: str,
                  mode: str,
                  n_imus: int,
                  split: str,
                  data_dir: str,
                  output_dir: str = 'results') -> str:
    """Run a single experiment and return checkpoint path."""
    print(f"\nRunning experiment: {split} {mode} {n_imus}IMU")
    
    # Train
    train_cmd = [
        'python', 'src/train.py',
        '--config', config_path,
        '--mode', mode,
        '--n_imus', str(n_imus),
        '--split', split,
        '--data_dir', data_dir
    ]
    
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Training failed: {result.stderr}")
        return None
    
    # Get checkpoint path
    checkpoint_dir = Path(output_dir) / 'checkpoints'
    exp_name = f"{split}_{mode}_{n_imus}im"
    checkpoint_path = checkpoint_dir / f"{exp_name}_best.pth"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    return str(checkpoint_path)


def run_evaluation(checkpoint_path: str,
                  split: str,
                  mode: str,
                  n_imus: int,
                  data_dir: str,
                  output_dir: str = 'results'):
    """Run evaluation on a checkpoint."""
    print(f"Evaluating: {checkpoint_path}")
    
    eval_cmd = [
        'python', 'src/eval.py',
        '--checkpoint', checkpoint_path,
        '--split', split,
        '--mode', mode,
        '--n_imus', str(n_imus),
        '--data_dir', data_dir,
        '--output_dir', output_dir
    ]
    
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr}")
        return None
    
    return result.stdout


def run_full_sweep(config_path: str,
                   data_dir: str,
                   splits: List[str] = ['D1', 'D2', 'D3', 'D4'],
                   modes: List[str] = ['rda', 'ara'],
                   n_imus_list: List[int] = [1, 2, 3, 4],
                   output_dir: str = 'results'):
    """Run full experiment sweep."""
    results = []
    
    for split in splits:
        for mode in modes:
            for n_imus in n_imus_list:
                # Train
                checkpoint_path = run_experiment(
                    config_path=config_path,
                    mode=mode,
                    n_imus=n_imus,
                    split=split,
                    data_dir=data_dir,
                    output_dir=output_dir
                )
                
                if checkpoint_path is None:
                    continue
                
                # Evaluate
                eval_output = run_evaluation(
                    checkpoint_path=checkpoint_path,
                    split=split,
                    mode=mode,
                    n_imus=n_imus,
                    data_dir=data_dir,
                    output_dir=output_dir
                )
                
                # Load results
                exp_name = f"{split}_{mode}_{n_imus}im"
                results_csv = Path(output_dir) / f"{exp_name}_eval.csv"
                metrics_json = Path(output_dir) / f"{exp_name}_metrics.json"
                
                if results_csv.exists():
                    df = pd.read_csv(results_csv)
                    # Aggregate across trajectories
                    row = {
                        'split': split,
                        'mode': mode,
                        'n_imus': n_imus,
                        'rmse': df['rmse'].mean() if 'rmse' in df.columns else None,
                        'mae': df['mae'].mean() if 'mae' in df.columns else None,
                        'max_error': df['max_error'].mean() if 'max_error' in df.columns else None,
                        'std_error': df['std_error'].mean() if 'std_error' in df.columns else None
                    }
                    results.append(row)
    
    # Save aggregated results
    results_df = pd.DataFrame(results)
    results_csv = Path(output_dir) / 'full_sweep_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nFull sweep results saved to {results_csv}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Run full experiment sweep')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--splits', type=str, nargs='+', default=['D1', 'D2', 'D3', 'D4'],
                       help='Dataset splits to run')
    parser.add_argument('--modes', type=str, nargs='+', default=['rda', 'ara'],
                       help='Modes to run (rda, ara)')
    parser.add_argument('--n_imus', type=int, nargs='+', default=[1, 2, 3, 4],
                       help='Number of IMUs to test')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run full sweep
    results_df = run_full_sweep(
        config_path=args.config,
        data_dir=args.data_dir,
        splits=args.splits,
        modes=args.modes,
        n_imus_list=args.n_imus,
        output_dir=args.output_dir
    )
    
    print("\nExperiment sweep completed!")
    print(results_df)


if __name__ == '__main__':
    main()

