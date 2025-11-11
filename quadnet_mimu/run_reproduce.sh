#!/usr/bin/env bash
set -euo pipefail

# Clone dataset if not present (optional)
if [ ! -d "data/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors" ]; then
  mkdir -p data
  git clone --depth 1 https://github.com/ansfl/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors.git data/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors || true
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r quadnet_mimu/requirements.txt

# Quick mock training to validate pipeline
python -m quadnet_mimu.src.train --config quadnet_mimu/configs/default.yaml --mode rda --n_imus 4 --split D1 --mock

echo "Done."

