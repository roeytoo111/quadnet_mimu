"""
Visualization utilities for QuadNet MIMU.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns


def plot_predictions_vs_ground_truth(predictions: np.ndarray,
                                    targets: np.ndarray,
                                    trajectory_id: str,
                                    save_path: Optional[str] = None):
    """Plot predictions vs ground truth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if predictions.ndim == 1:
        # Single target
        ax.scatter(targets, predictions, alpha=0.5)
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
               'r--', label='Perfect prediction')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title(f'Predictions vs Ground Truth - {trajectory_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Multiple targets (distance and altitude)
        ax.scatter(targets[:, 0], predictions[:, 0], alpha=0.5, label='Distance')
        if predictions.shape[1] > 1:
            ax.scatter(targets[:, 1], predictions[:, 1], alpha=0.5, label='Altitude')
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
               'r--', label='Perfect prediction')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title(f'Predictions vs Ground Truth - {trajectory_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_trajectory_errors(predictions: np.ndarray,
                          targets: np.ndarray,
                          timestamps: np.ndarray,
                          trajectory_id: str,
                          save_path: Optional[str] = None):
    """Plot errors over trajectory."""
    errors = predictions - targets
    
    if errors.ndim == 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, errors, label='Error')
        ax.axhline(y=0, color='r', linestyle='--', label='Zero error')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error')
        ax.set_title(f'Prediction Errors Over Time - {trajectory_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        axes[0].plot(timestamps, errors[:, 0], label='Distance Error')
        axes[0].axhline(y=0, color='r', linestyle='--', label='Zero error')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Error')
        axes[0].set_title(f'Distance Prediction Errors - {trajectory_id}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if errors.shape[1] > 1:
            axes[1].plot(timestamps, errors[:, 1], label='Altitude Error')
            axes[1].axhline(y=0, color='r', linestyle='--', label='Zero error')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Error')
            axes[1].set_title(f'Altitude Prediction Errors - {trajectory_id}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_error_histogram(errors: np.ndarray,
                        trajectory_id: str,
                        save_path: Optional[str] = None):
    """Plot error histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if errors.ndim == 1:
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution - {trajectory_id}')
        ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.hist(errors[:, 0], bins=50, alpha=0.7, label='Distance', edgecolor='black')
        if errors.shape[1] > 1:
            ax.hist(errors[:, 1], bins=50, alpha=0.7, label='Altitude', edgecolor='black')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution - {trajectory_id}')
        ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_metrics_comparison(results_df: pd.DataFrame,
                           metric: str = 'rmse',
                           save_path: Optional[str] = None):
    """Plot metrics comparison across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot table for easier plotting
    if 'n_imus' in results_df.columns and 'mode' in results_df.columns:
        pivot = results_df.pivot_table(values=metric, index='n_imus', columns='mode', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax)
        ax.set_xlabel('Number of IMUs')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison: RDA vs ARA')
        ax.legend(title='Mode')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=0)
    else:
        ax.bar(range(len(results_df)), results_df[metric])
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Experiment')
        ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def visualize_evaluation_results(results: Dict,
                                output_dir: str,
                                trajectory_id: str):
    """Create all visualization plots for evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    predictions = results['predictions']
    targets = results['targets']
    timestamps = np.array(results['timestamps'])
    
    # Predictions vs Ground Truth
    plot_path = output_path / f"{trajectory_id}_predictions_vs_gt.png"
    plot_predictions_vs_ground_truth(predictions, targets, trajectory_id, str(plot_path))
    
    # Errors over time
    plot_path = output_path / f"{trajectory_id}_errors_over_time.png"
    plot_trajectory_errors(predictions, targets, timestamps, trajectory_id, str(plot_path))
    
    # Error histogram
    errors = predictions - targets
    plot_path = output_path / f"{trajectory_id}_error_histogram.png"
    plot_error_histogram(errors, trajectory_id, str(plot_path))

