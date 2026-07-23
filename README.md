# Fed-ASTGCN

Source code for **Fed-ASTGCN**, a federated learning framework for spatio-temporal
traffic forecasting with component-specific aggregation on the ASTGCN backbone.

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

Experiments use the public **PEMS03 / PEMS04 / PEMS07 / PEMS08** traffic-flow
benchmarks. Download them from the standard sources (e.g. the ASTGCN / STSGCN
repositories) and place each dataset's raw files under `data/<NAME>/`, e.g.:

```
data/PEMS08/PEMS08.npz
data/PEMS08/PEMS08_adj.npy
```

Then build the model-ready inputs and adjacency:

```bash
python generate_adj_npy.py          # build adjacency .npy from the edge list
python prepareData.py --config configurations/PEMS08_astgcn.conf
# prepare_pems03_ids.py / prepare_pems07_ids.py / prepare_pems08_ids.py
# regenerate the sensor-id ordering for those datasets if needed
```

Data paths are read from each `configurations/<NAME>_astgcn.conf`.

## Running

```bash
python -u fed_astgcn.py \
    --config configurations/PEMS08_astgcn.conf \
    --num_clients 10 --rounds 100 \
    --data_split noniid --alpha 0.1 \
    --seeds 42 123 456 789 2024 \
    --aggregation attention --fedprox_mu 0.01 \
    --use_graph_reg True --local_epochs 5 \
    --use_adaptive True --dropout 0.1 --node_emb_dim 16
```

See `bash-files/bash-sample-fedastgcn.sh` for a full example. Per-round logs are
written to `logs/` and results to `saved_results/`.

## Repository layout

```
fed_astgcn.py            # main federated training / evaluation
model/                   # ASTGCN backbone and attention layers
lib/                     # metrics and utilities
configurations/          # per-dataset .conf files
prepareData.py, prepare_*_ids.py, generate_adj_npy.py   # data preparation
bash-files/              # sample launch scripts
```

## Baselines

Baseline methods are the standard federated / spatio-temporal models cited in the
paper; please refer to their original repositories for their implementations.

## Citation

If you use this code, please cite our paper:



