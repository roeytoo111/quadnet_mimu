# QuadNet MIMU: Quadrotor Dead Reckoning with Multiple Inertial Sensors

This repository implements the QuadNet MIMU approach from the paper "Quadrotor Dead Reckoning with Multiple Inertial Sensors" (D. Hurwitz & I. Klein). It provides a complete PyTorch codebase for training and evaluating deep learning models that predict distance and altitude changes from multiple IMU sensor data.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Running Training](#running-training)
- [Running Evaluation](#running-evaluation)
- [Running Experiments](#running-experiments)
- [Viewing Results](#viewing-results)
- [Configuration](#configuration)
- [Project Files Description](#project-files-description)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

QuadNet MIMU uses deep learning (1D-CNN + Fully Connected layers) to predict horizontal distance and altitude changes from IMU sensor data. The implementation supports two multi-IMU strategies:

- **RDA (Raw Data Average)**: Averages IMU signals before feeding to the network
- **ARA (After Regression Average)**: Each IMU feeds through its own network, then outputs are averaged

The model predicts delta distance and/or delta altitude for 1-second windows (120 samples at 120Hz sampling rate).

## Repository Structure

```
quadnet_mimu/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── datasets.py              # Data loading and preprocessing
│   ├── models.py                # QuadNet model definitions (RDA/ARA)
│   ├── train.py                 # Training script
│   ├── eval.py                  # Evaluation script
│   ├── utils.py                 # Utility functions (normalization, etc.)
│   ├── experiments.py           # Experiment runner for full sweeps
│   └── visualize.py             # Visualization utilities
├── configs/                     # Configuration files
│   └── default.yaml             # Default training configuration
├── tests/                       # Unit tests
│   ├── test_dataloader.py       # Data loader tests
│   └── test_model_forward.py    # Model forward pass tests
├── notebooks/                   # Jupyter notebooks
│   └── data_inspection.ipynb    # Dataset inspection notebook
├── data/                        # Dataset helpers (empty, for user data)
├── results/                     # Training results (created during training)
│   ├── checkpoints/             # Model checkpoints
│   ├── logs/                    # TensorBoard logs
│   └── *.csv                    # Evaluation results
├── requirements.txt             # Python dependencies
├── run_reproduce.sh            # Script to reproduce paper results
└── README.md                   # This file
```

## Installation

### 1. Clone the Repository

```bash
cd /home/roey/quadrotor-net
git clone <repository-url>  # If applicable
cd quadnet_mimu
```

### 2. Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

Run unit tests to verify installation:

```bash
python -m pytest tests/ -v
```

## Dataset Setup

### 1. Clone the Dataset Repository

The dataset repository should be cloned to the parent directory:

```bash
cd /home/roey/quadrotor-net
git clone https://github.com/ansfl/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors.git dataset_repo
```

### 2. Verify Dataset Structure

The dataset should have the following structure:

```
dataset_repo/
├── Horizontal/
│   ├── path_1/
│   │   ├── GT.csv              # Ground truth (distance, altitude, positions)
│   │   ├── IMU_1.csv           # IMU 1 data (accel, gyro)
│   │   ├── IMU_2.csv
│   │   ├── IMU_3.csv
│   │   └── IMU_4.csv
│   ├── path_2/
│   └── ...
├── Vertical/
│   └── ...
└── StraightLine/
    └── ...
```

### 3. Inspect Dataset (Optional)

Run the data inspection notebook to understand the dataset:

```bash
jupyter notebook notebooks/data_inspection.ipynb
```

## Quick Start

### 1. Train a Single Model

Train a model with RDA strategy, 4 IMUs, on D1 split:

```bash
python3 src/train.py \
    --config configs/default.yaml \
    --mode rda \
    --n_imus 4 \
    --split D1 \
    --data_dir ../dataset_repo
```

### 2. Evaluate the Model

Evaluate the trained model:

```bash
python3 src/eval.py \
    --checkpoint results/checkpoints/D1_rda_4im_best.pth \
    --split D1 \
    --mode rda \
    --n_imus 4 \
    --data_dir ../dataset_repo
```

### 3. View Results

Results are saved to `results/` directory:
- CSV files with metrics: `results/D1_rda_4im_eval.csv`
- JSON file with aggregated metrics: `results/D1_rda_4im_metrics.json`
- TensorBoard logs: `results/logs/D1_rda_4im/`

View TensorBoard:

```bash
tensorboard --logdir results/logs
```

## Running Training

### Basic Training Command

```bash
python3 src/train.py \
    --config configs/default.yaml \
    --mode <rda|ara> \
    --n_imus <1|2|3|4> \
    --split <D1|D2|D3|D4> \
    --data_dir <path_to_dataset>
```

### Arguments

- `--config`: Path to YAML configuration file (default: `configs/default.yaml`)
- `--mode`: Multi-IMU strategy (`rda` or `ara`)
- `--n_imus`: Number of IMUs to use (1, 2, 3, or 4)
- `--split`: Dataset split (`D1`, `D2`, `D3`, or `D4`)
  - `D1`, `D3`: Horizontal datasets (test trajectory: path_4)
  - `D2`, `D4`: Vertical datasets (test trajectory: path_9)
- `--data_dir`: Path to dataset repository (default: `../dataset_repo`)
- `--device`: Device to use (`cuda` or `cpu`, default: auto-detect)
- `--seed`: Random seed (default: 42)
- `--resume`: Path to checkpoint to resume training from (optional)

### Examples

**Train RDA model with 4 IMUs on D1:**

```bash
python3 src/train.py \
    --config configs/default.yaml \
    --mode rda \
    --n_imus 4 \
    --split D1 \
    --data_dir ../dataset_repo
```

**Train ARA model with 2 IMUs on D2:**

```bash
python src/train.py \
    --config configs/default.yaml \
    --mode ara \
    --n_imus 2 \
    --split D2 \
    --data_dir ../dataset_repo
```

**Resume training from checkpoint:**

```bash
python src/train.py \
    --config configs/default.yaml \
    --mode rda \
    --n_imus 4 \
    --split D1 \
    --data_dir ../dataset_repo \
    --resume results/checkpoints/D1_rda_4im_best.pth
```

### Training Output

During training, you'll see:
- Per-epoch training and validation metrics (loss, RMSE, MAE)
- Best model checkpoints saved to `results/checkpoints/`
- TensorBoard logs in `results/logs/`
- Normalizer saved with checkpoint

Checkpoints are saved as:
- `{split}_{mode}_{n_imus}im_best.pth`: Best model (lowest validation RMSE)
- `{split}_{mode}_{n_imus}im_final.pth`: Final model after training

## Running Evaluation

### Basic Evaluation Command

```bash
python src/eval.py \
    --checkpoint <path_to_checkpoint> \
    --split <D1|D2|D3|D4> \
    --mode <rda|ara> \
    --n_imus <1|2|3|4> \
    --data_dir <path_to_dataset>
```

### Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--split`: Dataset split (`D1`, `D2`, `D3`, or `D4`)
- `--mode`: Multi-IMU strategy (`rda` or `ara`)
- `--n_imus`: Number of IMUs used (must match checkpoint)
- `--data_dir`: Path to dataset repository
- `--output_dir`: Output directory for results (default: `results`)
- `--trajectories`: Specific trajectories to evaluate (optional, defaults to test set)
- `--device`: Device to use (`cuda` or `cpu`)

### Examples

**Evaluate on test set:**

```bash
python src/eval.py \
    --checkpoint results/checkpoints/D1_rda_4im_best.pth \
    --split D1 \
    --mode rda \
    --n_imus 4 \
    --data_dir ../dataset_repo
```

**Evaluate on specific trajectories:**

```bash
python src/eval.py \
    --checkpoint results/checkpoints/D1_rda_4im_best.pth \
    --split D1 \
    --mode rda \
    --n_imus 4 \
    --data_dir ../dataset_repo \
    --trajectories path_4 path_5
```

### Evaluation Output

Evaluation produces:
- **CSV file**: `results/{split}_{mode}_{n_imus}im_eval.csv` with per-trajectory metrics
- **JSON file**: `results/{split}_{mode}_{n_imus}im_metrics.json` with aggregated metrics
- Console output with RMSE, MAE, max error, and std error

## Running Experiments

### Reproduce Paper Results

Run the full experiment sweep to reproduce paper tables:

```bash
./run_reproduce.sh
```

Or manually run the experiments script:

```bash
python src/experiments.py \
    --config configs/default.yaml \
    --data_dir ../dataset_repo \
    --splits D1 D2 D3 D4 \
    --modes rda ara \
    --n_imus 1 2 3 4
```

This will:
1. Train models for all combinations of splits, modes, and IMU counts
2. Evaluate each model
3. Save results to CSV files
4. Generate aggregated results table

### Custom Experiment Sweep

Run experiments for specific configurations:

```bash
python3 src/experiments.py \
    --config configs/default.yaml \
    --data_dir ../dataset_repo \
    --splits D1 D3 \
    --modes rda \
    --n_imus 2 4 \
    --output_dir results/custom_experiments
```

## Viewing Results

### 1. TensorBoard

View training curves (loss, RMSE, MAE):

```bash
tensorboard --logdir results/logs
```

Then open http://localhost:6006 in your browser.

### 2. Evaluation CSV Files

View per-trajectory metrics:

```bash
cat results/D1_rda_4im_eval.csv
```

Or in Python:

```python
import pandas as pd
df = pd.read_csv('results/D1_rda_4im_eval.csv')
print(df)
```

### 3. Aggregated Metrics JSON

View aggregated metrics:

```bash
cat results/D1_rda_4im_metrics.json
```

Or in Python:

```python
import json
with open('results/D1_rda_4im_metrics.json', 'r') as f:
    metrics = json.load(f)
print(json.dumps(metrics, indent=2))
```

### 4. Visualization (Optional)

Create visualization plots:

```python
from src.visualize import visualize_evaluation_results
import pickle

# Load results (you may need to modify eval.py to save full results)
# visualize_evaluation_results(results, 'results/plots', 'path_4')
```

### 5. Compare Results

Compare RDA vs ARA:

```python
import pandas as pd

rda_results = pd.read_csv('results/D1_rda_4im_eval.csv')
ara_results = pd.read_csv('results/D1_ara_4im_eval.csv')

print("RDA RMSE:", rda_results['rmse'].mean())
print("ARA RMSE:", ara_results['rmse'].mean())
```

## Configuration

### Configuration File (`configs/default.yaml`)

The configuration file controls all hyperparameters:

```yaml
data:
  data_dir: "../dataset_repo"
  window_length: 120          # Window length in samples (1 second at 120Hz)
  stride: 120                 # Stride for sliding windows
  target: "distance"          # "distance", "altitude", or "both"
  sampling_rate: 120.0        # IMU sampling rate (Hz)
  gt_sampling_rate: 10.0      # Ground truth sampling rate (Hz)

model:
  conv_channels: [64, 128, 128, 256, 256, 512, 512]  # Conv layer channels
  conv_kernels: [5, 5, 3, 3, 3, 3, 3]               # Kernel sizes
  fc_dims: [256, 128]                                # FC layer dimensions
  dropout: 0.5
  use_batch_norm: true
  ara_shared_weights: true    # For ARA: use shared weights

training:
  batch_size: 64
  epochs: 200
  learning_rate: 0.001
  scheduler: "ReduceLROnPlateau"
  early_stop_patience: 20
  augment: false
```

### Custom Configuration

Create a custom config file:

```yaml
# configs/my_config.yaml
data:
  window_length: 120
  target: "both"  # Predict both distance and altitude

model:
  dropout: 0.3
  fc_dims: [512, 256, 128]

training:
  batch_size: 32
  learning_rate: 0.0005
```

Use it:

```bash
python src/train.py --config configs/my_config.yaml --mode rda --n_imus 4 --split D1
```

## Project Files Description

### Source Files (`src/`)

#### `src/datasets.py`
- **Purpose**: Data loading and preprocessing
- **Key Classes**:
  - `QuadNetDataset`: PyTorch Dataset for IMU data windows
  - Functions: `create_dataloader()`, `create_data_splits()`, `get_trajectory_ids()`
- **Features**:
  - Loads CSV files (GT.csv, IMU_*.csv)
  - Creates sliding windows (default: 120 samples, 1 second)
  - Aligns multiple IMUs to common timebase
  - Computes delta distance/altitude labels
  - Supports RDA and ARA data formats
  - Normalization and augmentation

#### `src/models.py`
- **Purpose**: Model definitions
- **Key Classes**:
  - `QuadNet`: Base 1D-CNN + FC regression model
  - `QuadNetRDA`: RDA wrapper (averages IMUs before network)
  - `QuadNetARA`: ARA wrapper (separate networks per IMU, average outputs)
  - Function: `create_model()`: Factory function to create models
- **Features**:
  - Configurable architecture (conv channels, kernels, FC dimensions)
  - Supports single-target (distance or altitude) and multi-target (both)
  - Shared or separate weights for ARA mode
  - Kaiming weight initialization
  - Model summary method

#### `src/train.py`
- **Purpose**: Training script
- **Features**:
  - Training and validation loops
  - Early stopping based on validation RMSE
  - Learning rate scheduling (ReduceLROnPlateau, CosineAnnealing)
  - Checkpointing (best and final models)
  - TensorBoard logging
  - Configurable augmentations
  - Mixed precision training (optional)

#### `src/eval.py`
- **Purpose**: Evaluation script
- **Features**:
  - Evaluates model on test trajectories
  - Computes metrics (RMSE, MAE, max error, std error)
  - Saves results to CSV and JSON
  - Supports per-trajectory and aggregated metrics
  - Handles single-target and multi-target models

#### `src/utils.py`
- **Purpose**: Utility functions
- **Key Functions**:
  - `set_seed()`: Set random seed for reproducibility
  - `Normalizer`: Normalize data (global or per-IMU)
  - `augment_data()`: Data augmentation (noise, bias drift, scaling)
  - `compute_distance_delta()`, `compute_altitude_delta()`: Compute labels
  - `get_device()`: Get PyTorch device
  - `save_checkpoint()`, `load_checkpoint()`: Model checkpointing
  - `compute_rmse()`, `compute_mae()`: Metrics

#### `src/experiments.py`
- **Purpose**: Experiment runner for full sweeps
- **Features**:
  - Runs training and evaluation for all combinations
  - Aggregates results across experiments
  - Saves results to CSV tables
  - Reproduces paper tables (D1-D4, RDA/ARA, 1-4 IMUs)

#### `src/visualize.py`
- **Purpose**: Visualization utilities
- **Functions**:
  - `plot_predictions_vs_ground_truth()`: Scatter plot of predictions vs GT
  - `plot_trajectory_errors()`: Error over time
  - `plot_error_histogram()`: Error distribution
  - `plot_metrics_comparison()`: Compare metrics across experiments
  - `visualize_evaluation_results()`: Create all plots for a trajectory

### Configuration Files (`configs/`)

#### `configs/default.yaml`
- Default hyperparameters for training
- Model architecture settings
- Data preprocessing options
- Training settings (batch size, learning rate, etc.)

### Test Files (`tests/`)

#### `tests/test_dataloader.py`
- Tests data loader functionality
- Verifies dataset shapes and formats
- Tests normalizer
- Tests DataLoader creation

#### `tests/test_model_forward.py`
- Tests model forward pass
- Verifies output shapes
- Tests RDA and ARA modes
- Tests model factory function

### Scripts

#### `run_reproduce.sh`
- Bash script to reproduce paper results
- Runs training and evaluation for all combinations
- Aggregates results
- Usage: `./run_reproduce.sh`

### Notebooks (`notebooks/`)

#### `notebooks/data_inspection.ipynb`
- Inspects dataset structure
- Visualizes sample data
- Shows IMU signals and ground truth
- Explores data distributions

## Examples

### Example 1: Train and Evaluate Single Model

```bash
# Train
python src/train.py \
    --config configs/default.yaml \
    --mode rda \
    --n_imus 4 \
    --split D1 \
    --data_dir ../dataset_repo

# Evaluate
python src/eval.py \
    --checkpoint results/checkpoints/D1_rda_4im_best.pth \
    --split D1 \
    --mode rda \
    --n_imus 4 \
    --data_dir ../dataset_repo

# View results
cat results/D1_rda_4im_metrics.json
```

### Example 2: Compare RDA vs ARA

```bash
# Train RDA
python src/train.py --config configs/default.yaml --mode rda --n_imus 4 --split D1 --data_dir ../dataset_repo

# Train ARA
python src/train.py --config configs/default.yaml --mode ara --n_imus 4 --split D1 --data_dir ../dataset_repo

# Evaluate both
python src/eval.py --checkpoint results/checkpoints/D1_rda_4im_best.pth --split D1 --mode rda --n_imus 4 --data_dir ../dataset_repo
python src/eval.py --checkpoint results/checkpoints/D1_ara_4im_best.pth --split D1 --mode ara --n_imus 4 --data_dir ../dataset_repo

# Compare
python -c "
import pandas as pd
rda = pd.read_csv('results/D1_rda_4im_eval.csv')
ara = pd.read_csv('results/D1_ara_4im_eval.csv')
print('RDA RMSE:', rda['rmse'].mean())
print('ARA RMSE:', ara['rmse'].mean())
"
```

### Example 3: Train with Custom Configuration

```bash
# Create custom config
cat > configs/custom.yaml << EOF
data:
  data_dir: "../dataset_repo"
  window_length: 120
  target: "both"

model:
  dropout: 0.3
  fc_dims: [512, 256, 128]

training:
  batch_size: 32
  learning_rate: 0.0005
  epochs: 100
EOF

# Train with custom config
python src/train.py \
    --config configs/custom.yaml \
    --mode rda \
    --n_imus 4 \
    --split D1 \
    --data_dir ../dataset_repo
```

## Troubleshooting

### Common Issues

1. **Dataset not found**
   - Ensure dataset repository is cloned to `../dataset_repo`
   - Check that `--data_dir` points to correct path
   - Verify dataset structure (Horizontal/, Vertical/ directories)

2. **CUDA out of memory**
   - Reduce batch size in config: `training.batch_size: 32`
   - Reduce window length: `data.window_length: 60`
   - Use CPU: `--device cpu`

3. **No trajectories found**
   - Check that trajectory IDs match dataset (path_1, path_2, etc.)
   - Verify CSV files exist in trajectory directories
   - Check dataset type (horizontal vs vertical)

4. **Checkpoint not found**
   - Verify checkpoint path is correct
   - Check that training completed successfully
   - Ensure checkpoint file exists: `ls results/checkpoints/`

5. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
   - Verify virtual environment is activated

### Getting Help

- Check TensorBoard logs for training curves
- Review evaluation CSV files for per-trajectory metrics
- Inspect dataset with `notebooks/data_inspection.ipynb`
- Run unit tests: `python -m pytest tests/ -v`

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{hurwitz2023quadrotor,
  title={Quadrotor Dead Reckoning with Multiple Inertial Sensors},
  author={Hurwitz, Dror and Klein, Itzik},
  booktitle={2023 DGON Inertial Sensors and Systems (ISS)},
  pages={1--18},
  year={2023},
  organization={IEEE}
}
```

## License

This implementation is provided for research purposes. Please refer to the original dataset repository for dataset licensing information.

## Acknowledgments

- Original paper: D. Hurwitz & I. Klein, "Quadrotor Dead Reckoning with Multiple Inertial Sensors"
- Dataset: https://github.com/ansfl/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors



