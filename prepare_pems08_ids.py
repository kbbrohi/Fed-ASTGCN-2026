#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare PEMS08 ID mapping file
Extracts unique sensor IDs from PEMS08.csv and creates 0-indexed mapping
"""

import csv
import numpy as np

# Read PEMS08.csv and extract unique sensor IDs
csv_file = 'data/PEMS08/PEMS08.csv'
unique_ids = set()

with open(csv_file, 'r') as f:
    f.readline()  # Skip header
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 3:
            continue
        from_id = int(row[0])
        to_id = int(row[1])
        unique_ids.add(from_id)
        unique_ids.add(to_id)

# Sort IDs to ensure consistent mapping
sorted_ids = sorted(list(unique_ids))

print(f"Found {len(sorted_ids)} unique sensor IDs")
print(f"ID range: {min(sorted_ids)} to {max(sorted_ids)}")

# Expected number of sensors for PEMS08
expected_sensors = 170

if len(sorted_ids) != expected_sensors:
    print(f"WARNING: Found {len(sorted_ids)} IDs but config expects {expected_sensors}")
    print(f"Proceeding with {len(sorted_ids)} sensors")

# Check if already 0-indexed and consecutive
if sorted_ids == list(range(len(sorted_ids))):
    print("\n✓ IDs are already 0-indexed and consecutive!")
    print("  No ID mapping needed - can use CSV directly")
else:
    print("\n⚠ IDs are not consecutive - creating mapping file")

# Save ID mapping file (one ID per line, in order)
id_file = 'data/PEMS08/PEMS08_node_ids.txt'
with open(id_file, 'w') as f:
    for sensor_id in sorted_ids:
        f.write(f"{sensor_id}\n")

print(f"\nID mapping file saved to: {id_file}")
print(f"First 10 IDs: {sorted_ids[:10]}")
print(f"Last 10 IDs: {sorted_ids[-10:]}")

# Also create adjacency matrix directly as numpy array and save as .npy
A = np.zeros((len(sorted_ids), len(sorted_ids)), dtype=np.float32)
distance_A = np.zeros((len(sorted_ids), len(sorted_ids)), dtype=np.float32)

# Create ID to index mapping
id_to_idx = {sensor_id: idx for idx, sensor_id in enumerate(sorted_ids)}

# Read CSV again and build adjacency matrix
with open(csv_file, 'r') as f:
    f.readline()  # Skip header
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 3:
            continue
        from_id = int(row[0])
        to_id = int(row[1])
        distance = float(row[2])

        i = id_to_idx[from_id]
        j = id_to_idx[to_id]

        A[i, j] = 1
        distance_A[i, j] = distance

# Save adjacency matrices
np.save('data/PEMS08/PEMS08_adj.npy', A)
np.save('data/PEMS08/PEMS08_distance.npy', distance_A)

print(f"\nAdjacency matrices saved:")
print(f"  - data/PEMS08/PEMS08_adj.npy")
print(f"  - data/PEMS08/PEMS08_distance.npy")
print(f"\nAdjacency matrix shape: {A.shape}")
print(f"Number of edges: {int(A.sum())}")

print("\n" + "="*60)
print("DONE! Now update PEMS08_astgcn.conf to use:")
print(f"  num_of_vertices = {len(sorted_ids)}")
print("="*60)