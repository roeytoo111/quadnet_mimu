QuadNet MIMU — Quadrotor Dead Reckoning with Multiple Inertial Sensors
========================================================================

This repository implements the core components to reproduce the QuadNet MIMU approach from “Quadrotor Dead Reckoning with Multiple Inertial Sensors” (Hurwitz & Klein).

What’s included
---------------
- Data loading with flexible format support (CSV/NPY/NPZ/MAT) and windowing
- QuadNet model (1D-CNN + FC) and multi-IMU strategies:
  - RDA: Raw Data Average
  - ARA: After Regression Average (shared or separate weights)
- Training with TensorBoard logging, checkpointing and early stopping
- Evaluation with RMSE/MAE/Max/STD and CSV outputs
- Default config for reproducibility

Quick start
-----------
1) Clone the dataset repository (once):
```bash
git clone https://github.com/ansfl/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors.git data/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors
```

2) Create a Python environment and install requirements:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r quadnet_mimu/requirements.txt
```

3) Train (RDA, 4 IMUs, D1 split). Add `--mock` to validate the pipeline without data:
```bash
python -m quadnet_mimu.src.train --config quadnet_mimu/configs/default.yaml --mode rda --n_imus 4 --split D1
```

4) Evaluate a checkpoint:
```bash
python -m quadnet_mimu.src.eval --config quadnet_mimu/configs/default.yaml --checkpoint checkpoints/best_D1_rda_4im.pt --split D1 --mode rda --n_imus 4
```

Configuration
-------------
Edit `quadnet_mimu/configs/default.yaml` to adjust:
- data: dataset_root, split (D1–D4), target (distance/altitude/both), window_length, stride, n_imus
- model: mode (rda/ara), ara_shared_weights, dropout
- training: batch_size, epochs, lr, early_stopping, mixed_precision
- evaluation: results_dir

Notes on data and splits
------------------------
The loader attempts to discover IMU channels with names like `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`. Distance/altitude references are discovered via common keys: `distance/dist/s` and `altitude/alt/z/height`. If the dataset format differs, add custom mapping in `quadnet_mimu/src/datasets.py`.

The official D1–D4 splits are not explicitly embedded here; you can approximate by setting `data.split` and preparing per-split folders, or extend the loader to read `configs/splits.json` if you create one.

Troubleshooting
---------------
- If you see “No data files found”, verify the dataset clone path matches `data.dataset_root` in the config.
- To validate pipeline end-to-end without data, run with `--mock` on train; evaluation expects real data.

Reproduce script
----------------
Run a quick end-to-end (with mock data) to verify installation:
```bash
bash run_reproduce.sh
```

