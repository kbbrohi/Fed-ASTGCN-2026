#!/bin/bash


# Run the experiment
cd /path/to/repo
source venv/bin/activate

python -u run_baselines.py --config ../configurations/PEMS03_astgcn.conf --dataset PEMS03 --models autofed --seeds 42 123 456 789 2024 --epochs 100 --device cuda:0 2>&1 | tee training_pems03_autofed-baseline.log

echo ""
echo "Finished training_pems03_autofed"