#!/bin/bash


cd /path/to/repo
source venv/bin/activate

mkdir -p logs
mkdir -p saved_results/fedastgcn_results


echo "========== Experiment: PEMS03 alpha=0.1, 10 clients =========="
python -u fed_astgcn.py \
    --config configurations/PEMS03_astgcn.conf \
    --num_clients 10 \
    --rounds 100 \
    --data_split noniid \
    --alpha 0.1 \
    --seeds 42 123 456 789 2024 \
    --aggregation attention \
    --fedprox_mu 0.01 \
    --use_graph_reg True \
    --local_epochs 5 \
    --server_momentum 0 \
    --use_adaptive True \
    --dropout 0.1 \
    --node_emb_dim 16 \
    2>&1 | tee logs/pems03_v2_alpha01.log

echo ""
echo "========================================="
echo "PEMS03 Alpha 0.1 Experiments Complete!"
echo "========================================="