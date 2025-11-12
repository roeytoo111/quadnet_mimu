# Quick Start Guide

This guide provides a quick overview of how to get started with QuadNet MIMU.

## Prerequisites

- Python 3.9+
- PyTorch 1.12+
- Dataset repository cloned to `../dataset_repo`

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Run

### 1. Train a Model

```bash
python src/train.py \
    --config configs/default.yaml \
    --mode rda \
    --n_imus 4 \
    --split D1 \
    --data_dir ../dataset_repo
```

### 2. Evaluate the Model

```bash
python src/eval.py \
    --checkpoint results/checkpoints/D1_rda_4im_best.pth \
    --split D1 \
    --mode rda \
    --n_imus 4 \
    --data_dir ../dataset_repo
```

### 3. View Results

```bash
# View metrics
cat results/D1_rda_4im_metrics.json

# View TensorBoard
tensorboard --logdir results/logs
```

## What Each Command Does

### Training (`src/train.py`)
- Loads dataset from `../dataset_repo`
- Creates train/val/test splits
- Trains QuadNet model with specified mode (RDA/ARA) and number of IMUs
- Saves best model checkpoint to `results/checkpoints/`
- Logs training metrics to TensorBoard

### Evaluation (`src/eval.py`)
- Loads trained model from checkpoint
- Evaluates on test trajectories
- Computes metrics (RMSE, MAE, max error, std error)
- Saves results to CSV and JSON files

## Results Location

- **Checkpoints**: `results/checkpoints/`
- **Metrics**: `results/*_metrics.json`
- **Evaluation CSV**: `results/*_eval.csv`
- **TensorBoard logs**: `results/logs/`

## Next Steps

- See `README.md` for detailed documentation
- Run `./run_reproduce.sh` to reproduce paper results
- Check `notebooks/data_inspection.ipynb` to explore the dataset


