# QuadNet MIMU Project Summary

## What This Project Does

This project implements a deep learning system for quadrotor dead reckoning using multiple IMU (Inertial Measurement Unit) sensors. It predicts horizontal distance and altitude changes from IMU sensor data using a 1D-CNN neural network.

## Key Features

1. **Multi-IMU Support**: Handles 1-4 IMU sensors with two strategies:
   - **RDA (Raw Data Average)**: Averages IMU signals before processing
   - **ARA (After Regression Average)**: Processes each IMU separately, then averages outputs

2. **Flexible Data Loading**: 
   - Loads CSV files from the dataset
   - Creates sliding windows of IMU data
   - Handles multiple IMUs with different sampling rates
   - Supports normalization and augmentation

3. **Complete Training Pipeline**:
   - Training and validation loops
   - Early stopping
   - Learning rate scheduling
   - Checkpointing
   - TensorBoard logging

4. **Evaluation Tools**:
   - Computes metrics (RMSE, MAE, max error, std error)
   - Per-trajectory and aggregated metrics
   - Saves results to CSV and JSON

5. **Reproducibility**:
   - Configuration files for hyperparameters
   - Random seed setting
   - Checkpoint saving/loading
   - Experiment scripts to reproduce paper results

## Project Files Overview

### Core Implementation
- **`src/datasets.py`**: Data loading and preprocessing
- **`src/models.py`**: Neural network model definitions
- **`src/train.py`**: Training script
- **`src/eval.py`**: Evaluation script
- **`src/utils.py`**: Utility functions
- **`src/experiments.py`**: Experiment runner
- **`src/visualize.py`**: Visualization utilities

### Configuration
- **`configs/default.yaml`**: Default hyperparameters and settings

### Tests
- **`tests/test_dataloader.py`**: Data loader tests
- **`tests/test_model_forward.py`**: Model forward pass tests

### Documentation
- **`README.md`**: Main documentation
- **`QUICKSTART.md`**: Quick start guide
- **`PROJECT_STRUCTURE.md`**: Detailed project structure explanation
- **`SUMMARY.md`**: This file

### Scripts
- **`run_reproduce.sh`**: Reproduce paper results
- **`EXAMPLE_USAGE.py`**: Example usage script

### Notebooks
- **`notebooks/data_inspection.ipynb`**: Dataset inspection notebook

## How to Use

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
```bash
# Clone dataset repository to parent directory
cd ..
git clone https://github.com/ansfl/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors.git dataset_repo
cd quadnet_mimu
```

### 3. Train a Model
```bash
python src/train.py \
    --config configs/default.yaml \
    --mode rda \
    --n_imus 4 \
    --split D1 \
    --data_dir ../dataset_repo
```

### 4. Evaluate the Model
```bash
python src/eval.py \
    --checkpoint results/checkpoints/D1_rda_4im_best.pth \
    --split D1 \
    --mode rda \
    --n_imus 4 \
    --data_dir ../dataset_repo
```

### 5. View Results
```bash
# View metrics
cat results/D1_rda_4im_metrics.json

# View TensorBoard
tensorboard --logdir results/logs
```

## Results Location

- **Checkpoints**: `results/checkpoints/`
- **Metrics**: `results/*_metrics.json`
- **Evaluation CSV**: `results/*_eval.csv`
- **TensorBoard logs**: `results/logs/`

## Dataset Structure

The dataset should be organized as:
```
dataset_repo/
├── Horizontal/
│   ├── path_1/
│   │   ├── GT.csv          # Ground truth
│   │   ├── IMU_1.csv       # IMU 1 data
│   │   ├── IMU_2.csv       # IMU 2 data
│   │   ├── IMU_3.csv       # IMU 3 data
│   │   └── IMU_4.csv       # IMU 4 data
│   ├── path_2/
│   └── ...
├── Vertical/
│   └── ...
└── StraightLine/
    └── ...
```

## Model Architecture

The QuadNet model consists of:
1. **7 Convolutional Layers**: 1D-CNN with BatchNorm and ReLU
2. **Adaptive Average Pooling**: Reduces temporal dimension to 1
3. **Fully Connected Layers**: 2-3 FC layers with dropout
4. **Output Layer**: Regression output (1 for distance/altitude, 2 for both)

## Training Process

1. **Data Loading**: Load CSV files and create sliding windows
2. **Normalization**: Normalize data using training set statistics
3. **Training Loop**: 
   - Forward pass through model
   - Compute loss (MSE)
   - Backward pass and optimizer step
   - Validate on validation set
4. **Early Stopping**: Stop if validation RMSE doesn't improve
5. **Checkpointing**: Save best model based on validation RMSE

## Evaluation Process

1. **Load Model**: Load trained model from checkpoint
2. **Load Test Data**: Create test data loader
3. **Evaluate**: Run model on test trajectories
4. **Compute Metrics**: Calculate RMSE, MAE, max error, std error
5. **Save Results**: Save metrics to CSV and JSON files

## Key Parameters

### Data Parameters
- **window_length**: 120 samples (1 second at 120Hz)
- **stride**: 120 samples (non-overlapping windows)
- **target**: "distance", "altitude", or "both"
- **n_imus**: Number of IMUs (1-4)

### Model Parameters
- **mode**: "rda" or "ara"
- **conv_channels**: [64, 128, 128, 256, 256, 512, 512]
- **fc_dims**: [256, 128]
- **dropout**: 0.5

### Training Parameters
- **batch_size**: 64
- **learning_rate**: 0.001
- **epochs**: 200
- **early_stop_patience**: 20

## Common Workflows

### Single Experiment
1. Train model: `python src/train.py ...`
2. Evaluate model: `python src/eval.py ...`
3. View results: `cat results/*_metrics.json`

### Compare RDA vs ARA
1. Train RDA: `python src/train.py --mode rda ...`
2. Train ARA: `python src/train.py --mode ara ...`
3. Evaluate both: `python src/eval.py ...`
4. Compare results: Compare CSV files

### Reproduce Paper Results
1. Run full sweep: `./run_reproduce.sh`
2. View aggregated results: `cat results/aggregated_results.csv`

## Troubleshooting

### Dataset Not Found
- Ensure dataset is cloned to `../dataset_repo`
- Check `--data_dir` argument

### CUDA Out of Memory
- Reduce batch size in config
- Reduce window length
- Use CPU: `--device cpu`

### Import Errors
- Install dependencies: `pip install -r requirements.txt`
- Check Python path

## Next Steps

1. **Run Example**: `python EXAMPLE_USAGE.py`
2. **Inspect Dataset**: `jupyter notebook notebooks/data_inspection.ipynb`
3. **Train Model**: Follow quick start guide
4. **View Results**: Check TensorBoard and CSV files
5. **Experiment**: Try different configurations

## Documentation

- **README.md**: Complete documentation
- **QUICKSTART.md**: Quick start guide
- **PROJECT_STRUCTURE.md**: Detailed file descriptions
- **SUMMARY.md**: This file

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

This implementation is provided for research purposes.



