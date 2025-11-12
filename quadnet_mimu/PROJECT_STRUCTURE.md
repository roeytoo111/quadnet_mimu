# Project Structure Explanation

## Overview

This project implements QuadNet MIMU for quadrotor dead reckoning using multiple IMU sensors. The codebase is organized into logical modules for data loading, model definition, training, evaluation, and experimentation.

## Directory Structure

```
quadnet_mimu/
├── src/                    # Main source code
├── configs/                # Configuration files
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks for data exploration
├── data/                   # Dataset helpers (empty by default)
├── results/                # Output directory (created during execution)
├── requirements.txt        # Python dependencies
├── run_reproduce.sh       # Script to reproduce paper results
├── README.md              # Main documentation
├── QUICKSTART.md          # Quick start guide
└── PROJECT_STRUCTURE.md   # This file
```

## Source Files (`src/`)

### `src/datasets.py`
**Purpose**: Data loading and preprocessing

**What it does**:
- Reads CSV files from the dataset (GT.csv, IMU_*.csv)
- Creates sliding windows of IMU data (default: 120 samples = 1 second)
- Aligns multiple IMUs to a common timebase
- Computes labels (delta distance, delta altitude) from ground truth
- Handles normalization and data augmentation
- Supports both RDA and ARA data formats

**Key Classes**:
- `QuadNetDataset`: PyTorch Dataset class that returns (data, label, metadata) tuples

**Key Functions**:
- `create_dataloader()`: Creates PyTorch DataLoader
- `create_data_splits()`: Splits trajectories into train/val/test sets
- `get_trajectory_ids()`: Lists available trajectories

### `src/models.py`
**Purpose**: Neural network model definitions

**What it does**:
- Defines QuadNet architecture (1D-CNN + Fully Connected layers)
- Implements RDA (Raw Data Average) strategy
- Implements ARA (After Regression Average) strategy
- Provides model factory function

**Key Classes**:
- `QuadNet`: Base model with 7 convolutional layers + FC layers
- `QuadNetRDA`: Wrapper that averages IMU data before network
- `QuadNetARA`: Wrapper that processes each IMU separately, then averages outputs

**Key Functions**:
- `create_model()`: Factory function to create models based on mode

### `src/train.py`
**Purpose**: Training script

**What it does**:
- Loads configuration from YAML file
- Creates data loaders for training and validation
- Initializes model, optimizer, and loss function
- Runs training loop with validation
- Implements early stopping
- Saves checkpoints (best and final models)
- Logs metrics to TensorBoard

**Key Features**:
- Early stopping based on validation RMSE
- Learning rate scheduling
- Checkpointing
- TensorBoard logging
- Configurable augmentations

### `src/eval.py`
**Purpose**: Evaluation script

**What it does**:
- Loads trained model from checkpoint
- Evaluates on test trajectories
- Computes metrics (RMSE, MAE, max error, std error)
- Saves results to CSV and JSON files

**Key Features**:
- Per-trajectory metrics
- Aggregated metrics
- Supports single-target and multi-target models

### `src/utils.py`
**Purpose**: Utility functions

**What it does**:
- Provides normalization utilities
- Data augmentation functions
- Random seed setting for reproducibility
- Checkpoint saving/loading
- Metric computation (RMSE, MAE)
- Device management (CPU/GPU)

**Key Classes**:
- `Normalizer`: Normalizes data using mean/std statistics

**Key Functions**:
- `set_seed()`: Sets random seed for reproducibility
- `augment_data()`: Applies data augmentation
- `compute_distance_delta()`, `compute_altitude_delta()`: Compute labels
- `save_checkpoint()`, `load_checkpoint()`: Model checkpointing

### `src/experiments.py`
**Purpose**: Experiment runner for full sweeps

**What it does**:
- Runs training and evaluation for all combinations of:
  - Dataset splits (D1, D2, D3, D4)
  - Modes (RDA, ARA)
  - Number of IMUs (1, 2, 3, 4)
- Aggregates results across experiments
- Saves results to CSV tables

### `src/visualize.py`
**Purpose**: Visualization utilities

**What it does**:
- Creates plots for predictions vs ground truth
- Plots errors over time
- Creates error histograms
- Compares metrics across experiments

## Configuration Files (`configs/`)

### `configs/default.yaml`
**Purpose**: Default configuration for training

**What it contains**:
- Data settings (window length, stride, target, sampling rates)
- Model architecture (conv channels, kernels, FC dimensions, dropout)
- Training settings (batch size, learning rate, epochs, scheduler)
- Augmentation parameters

## Test Files (`tests/`)

### `tests/test_dataloader.py`
**Purpose**: Tests data loading functionality

**What it tests**:
- Dataset creation
- Data shapes
- Normalizer functionality
- DataLoader creation

### `tests/test_model_forward.py`
**Purpose**: Tests model forward pass

**What it tests**:
- QuadNet forward pass
- QuadNetRDA forward pass
- QuadNetARA forward pass
- Model factory function
- Output shapes

## Notebooks (`notebooks/`)

### `notebooks/data_inspection.ipynb`
**Purpose**: Dataset inspection and exploration

**What it does**:
- Loads and displays sample data
- Visualizes IMU signals
- Shows ground truth trajectories
- Explores data distributions
- Computes example labels

## Scripts

### `run_reproduce.sh`
**Purpose**: Reproduce paper results

**What it does**:
- Runs training for all combinations (D1-D4, RDA/ARA, 1-4 IMUs)
- Evaluates all models
- Aggregates results
- Saves results to CSV

## Output Directory (`results/`)

### `results/checkpoints/`
**Purpose**: Saved model checkpoints

**Files**:
- `{split}_{mode}_{n_imus}im_best.pth`: Best model (lowest validation RMSE)
- `{split}_{mode}_{n_imus}im_final.pth`: Final model after training
- `{split}_{mode}_{n_imus}im_best_normalizer.json`: Normalizer parameters

### `results/logs/`
**Purpose**: TensorBoard logs

**Files**:
- Training and validation metrics (loss, RMSE, MAE)
- Learning rate schedules
- Model graphs

### `results/*.csv`
**Purpose**: Evaluation results

**Files**:
- `{split}_{mode}_{n_imus}im_eval.csv`: Per-trajectory metrics
- `aggregated_results.csv`: Aggregated results across experiments

### `results/*.json`
**Purpose**: Aggregated metrics

**Files**:
- `{split}_{mode}_{n_imus}im_metrics.json`: Aggregated metrics for a single experiment

## Data Flow

### Training Flow
1. `train.py` loads config from YAML
2. `datasets.py` creates data loaders from CSV files
3. `models.py` creates model based on mode (RDA/ARA)
4. Training loop:
   - Load batch from data loader
   - Forward pass through model
   - Compute loss
   - Backward pass and optimizer step
   - Log metrics to TensorBoard
5. Save checkpoint when validation RMSE improves

### Evaluation Flow
1. `eval.py` loads checkpoint
2. `datasets.py` creates test data loader
3. `models.py` loads model weights
4. Evaluation loop:
   - Load batch from test data loader
   - Forward pass through model
   - Compare predictions to ground truth
   - Compute metrics
5. Save results to CSV and JSON

## Key Concepts

### RDA (Raw Data Average)
- Averages IMU signals across all IMUs before feeding to network
- Input shape: (batch, 6, window_length) after averaging
- Single network processes averaged data

### ARA (After Regression Average)
- Each IMU feeds through its own network (or shared network)
- Outputs from all IMUs are averaged
- Input shape: (batch, 6*n_imus, window_length) or (batch, n_imus, 6, window_length)

### Data Windows
- Default: 120 samples = 1 second at 120Hz
- Sliding windows with configurable stride
- Labels: delta distance and/or delta altitude over window

### Normalization
- Per-channel normalization using training set statistics
- Option for per-IMU normalization (separate statistics per IMU)
- Normalizer saved with checkpoint for inference

## Usage Patterns

### Single Experiment
```bash
# Train
python src/train.py --config configs/default.yaml --mode rda --n_imus 4 --split D1

# Evaluate
python src/eval.py --checkpoint results/checkpoints/D1_rda_4im_best.pth --split D1 --mode rda --n_imus 4
```

### Full Experiment Sweep
```bash
./run_reproduce.sh
```

### Custom Configuration
```bash
# Create custom config
cp configs/default.yaml configs/my_config.yaml
# Edit my_config.yaml

# Train with custom config
python src/train.py --config configs/my_config.yaml --mode rda --n_imus 4 --split D1
```

## Extension Points

### Adding New Models
- Modify `src/models.py` to add new model architectures
- Update `create_model()` factory function

### Adding New Datasets
- Modify `src/datasets.py` to support new data formats
- Update `QuadNetDataset` class to handle new file formats

### Adding New Metrics
- Add metric computation functions to `src/utils.py`
- Update `src/eval.py` to compute and save new metrics

### Adding New Visualizations
- Add visualization functions to `src/visualize.py`
- Update evaluation script to generate visualizations



