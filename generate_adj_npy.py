"""
Generate adj.npy files for all PEMS datasets from their CSV edge lists.
Skips datasets that already have adj.npy.

Usage:
    python generate_adj_npy.py
"""
import numpy as np
import csv
import os
import configparser

DATASETS = [
    'configurations/PEMS03_astgcn.conf',
    'configurations/PEMS04_astgcn.conf',
    'configurations/PEMS07_astgcn.conf',
    'configurations/PEMS08_astgcn.conf',
]

for conf_path in DATASETS:
    config = configparser.ConfigParser()
    config.read(conf_path)

    num_vertices = int(config['Data']['num_of_vertices'])
    dataset_name = config['Data']['dataset_name']
    data_dir = os.path.dirname(config['Data']['graph_signal_matrix_filename'])

    npy_path = os.path.join(data_dir, f'{dataset_name}_adj.npy')

    if os.path.exists(npy_path):
        adj = np.load(npy_path)
        print(f'[SKIP] {npy_path} already exists (shape={adj.shape}, edges={int(adj.sum()//2)})')
        continue

    # Find CSV file
    csv_candidates = [
        os.path.join(data_dir, f'{dataset_name}.csv'),
        os.path.join(data_dir, 'distance.csv'),
    ]
    csv_path = None
    for c in csv_candidates:
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        print(f'[ERROR] No CSV found for {dataset_name} in {data_dir}')
        continue

    # Read all edges and collect unique IDs
    edges = []
    ids = set()
    with open(csv_path, 'r') as f:
        f.readline()  # skip header
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            i, j = int(row[0]), int(row[1])
            ids.add(i)
            ids.add(j)
            edges.append((i, j))

    max_id = max(ids)

    # If IDs exceed num_vertices, we need a mapping (e.g. PEMS03 has large sensor IDs)
    if max_id >= num_vertices:
        id_dict = {v: idx for idx, v in enumerate(sorted(ids))}
        print(f'[{dataset_name}] Large IDs detected (max={max_id}), mapping {len(id_dict)} IDs to 0-{len(id_dict)-1}')
    else:
        id_dict = None

    # Build adjacency matrix
    A = np.zeros((num_vertices, num_vertices), dtype=np.float32)
    for i, j in edges:
        ri = id_dict[i] if id_dict else i
        rj = id_dict[j] if id_dict else j
        if ri < num_vertices and rj < num_vertices:
            A[ri][rj] = 1.0
            A[rj][ri] = 1.0

    np.save(npy_path, A)
    edge_count = int(A.sum() // 2)
    print(f'[DONE] {npy_path} — shape={A.shape}, edges={edge_count}')

print('\nAll done.')
