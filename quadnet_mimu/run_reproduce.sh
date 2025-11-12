#!/bin/bash
# Script to reproduce paper results

set -e

# Configuration
DATA_DIR="../dataset_repo"
CONFIG="configs/default.yaml"
OUTPUT_DIR="results"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting reproduction of paper results..."
echo "Data directory: $DATA_DIR"
echo "Config: $CONFIG"
echo "Output directory: $OUTPUT_DIR"

# Run experiments for D1-D4 splits
SPLITS=("D1" "D2" "D3" "D4")
MODES=("rda" "ara")
N_IMUS=(1 2 3 4)

# Run training and evaluation for each combination
for SPLIT in "${SPLITS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for N_IMU in "${N_IMUS[@]}"; do
            echo ""
            echo "=========================================="
            echo "Running: $SPLIT $MODE $N_IMU IMU"
            echo "=========================================="
            
            # Train
            python src/train.py \
                --config $CONFIG \
                --mode $MODE \
                --n_imus $N_IMU \
                --split $SPLIT \
                --data_dir $DATA_DIR
            
            # Evaluate
            CHECKPOINT="$OUTPUT_DIR/checkpoints/${SPLIT}_${MODE}_${N_IMU}im_best.pth"
            if [ -f "$CHECKPOINT" ]; then
                python src/eval.py \
                    --checkpoint $CHECKPOINT \
                    --split $SPLIT \
                    --mode $MODE \
                    --n_imus $N_IMU \
                    --data_dir $DATA_DIR \
                    --output_dir $OUTPUT_DIR
            else
                echo "Warning: Checkpoint not found: $CHECKPOINT"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Aggregate results
python -c "
import pandas as pd
import glob
from pathlib import Path

results_files = glob.glob('results/*_eval.csv')
if results_files:
    dfs = []
    for f in results_files:
        df = pd.read_csv(f)
        # Extract experiment info from filename
        name = Path(f).stem.replace('_eval', '')
        parts = name.split('_')
        if len(parts) >= 3:
            df['split'] = parts[0]
            df['mode'] = parts[1]
            df['n_imus'] = int(parts[2].replace('im', ''))
        dfs.append(df)
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv('results/aggregated_results.csv', index=False)
        print('Aggregated results saved to results/aggregated_results.csv')
"

