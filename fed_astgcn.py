#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fed-ASTGCN: Component-Specific Aggregation for Federated Spatial-Temporal Traffic Forecasting

Main Contributions:
1. Component-Specific Aggregation:
   * Spatial attention + adaptive graph params: Performance-weighted (softmax-temperature, τ=2.0)
   * Temporal + other params: Data-proportional FedAvg
2. Adaptive Graph Learning: Node embeddings learn spatial dependencies
   beyond the fixed adjacency matrix (blended with identity via learnable α_blend)
3. Gated Dilated Temporal Convolution: Multi-scale temporal modeling (dilations [1,2,4], RF=7)
   with multi-layer prediction head (GELU + weight normalization)

Baselines: FedGODE, FedGTP, AutoFed, FedDis (federated); MTGNN, ASTGCN, PDFormer, STGODE, STGCN (centralized)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import time
import argparse
import configparser
import random
import json
import os
import math
from scipy import stats

# Import ASTGCN modules
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, scaled_Laplacian, cheb_polynomial
from model.ASTGCN_r import make_model


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FederatedASTGCN:
    """
    Federated ASTGCN with Component-Specific Aggregation and Adaptive Graph Learning
    """

    def __init__(self, config_file, num_clients=5, data_split='iid', alpha=0.5,
                 aggregation='attention', fedprox_mu=0.01, use_graph_reg=True,
                 local_epochs=5, server_momentum=0,
                 dropout=0.1, use_adaptive=True, node_emb_dim=16,
                 loss_type='huber', warmup_rounds=0, personalization_epochs=0,
                 dump_clients_round=None, current_seed=None):
        """
        Args:
            config_file: Configuration file path
            num_clients: Number of federated clients
            data_split: 'iid' or 'noniid' data partitioning
            alpha: Dirichlet concentration parameter for Non-IID
            aggregation: 'attention' (component-specific) or 'standard' (FedAvg)
            fedprox_mu: FedProx regularization strength (0 to disable)
            use_graph_reg: Use graph Laplacian smoothness regularization
            local_epochs: Number of local training epochs per round
            server_momentum: Server-side momentum (FedAvgM). 0 to disable.
            dropout: Dropout rate in ASTGCN blocks (0 to disable)
            use_adaptive: Use adaptive graph learning with node embeddings
            node_emb_dim: Dimension of node embeddings for adaptive adjacency
            loss_type: 'huber', 'mae', or 'mse'
            warmup_rounds: Number of linear LR warmup rounds
            personalization_epochs: Local fine-tuning epochs after aggregation (0 to disable)
        """
        self.config_file = config_file
        self.num_clients = num_clients
        self.data_split = data_split
        self.alpha = alpha
        self.aggregation = aggregation
        self.fedprox_mu = fedprox_mu
        self.use_graph_reg = use_graph_reg
        self.local_epochs = local_epochs
        self.server_momentum = server_momentum
        self.dropout = dropout
        self.use_adaptive = use_adaptive
        self.node_emb_dim = node_emb_dim
        self.loss_type = loss_type
        self.warmup_rounds = warmup_rounds
        self.personalization_epochs = personalization_epochs
        # Rounds at which to save each client's pre-aggregation weights.
        self.dump_clients_round = set(dump_clients_round) if dump_clients_round else set()
        self.current_seed = current_seed

        # Graph reg scheduling: ramp from 0.01 to 0.2 over training
        self.graph_reg_weight = 0.01

        # Communication cost tracking
        self.total_comm_rounds = 0
        self.total_params_communicated = 0
        self.model_size_mb = 0

        # Load configuration
        config = configparser.ConfigParser()
        config.read(config_file)

        # Extract parameters
        self.batch_size = config.getint('Training', 'batch_size')
        self.epochs = config.getint('Training', 'epochs')
        self.lr = config.getfloat('Training', 'learning_rate')
        self.num_of_vertices = config.getint('Data', 'num_of_vertices')
        self.points_per_hour = config.getint('Data', 'points_per_hour')
        self.num_for_predict = config.getint('Data', 'num_for_predict')
        self.len_input = config.getint('Data', 'len_input')
        self.dataset_name = config.get('Data', 'dataset_name')
        self.checkpoint_path = f'fed_astgcn_best_{self.dataset_name}_c{num_clients}_e{local_epochs}_mu{fedprox_mu}_gr{use_graph_reg}_ada{use_adaptive}.pkl'

        # Model parameters
        self.nb_block = config.getint('Training', 'nb_block')
        self.K = config.getint('Training', 'K')
        self.nb_chev_filter = config.getint('Training', 'nb_chev_filter')
        self.nb_time_filter = config.getint('Training', 'nb_time_filter')
        self.in_channels = config.getint('Training', 'in_channels')

        # Temporal parameters
        self.num_of_hours = config.getint('Training', 'num_of_hours')
        self.num_of_days = config.getint('Training', 'num_of_days')
        self.num_of_weeks = config.getint('Training', 'num_of_weeks')

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load adjacency matrix
        adj_filename = config['Data']['adj_filename']
        self.adj_mx, _ = get_adjacency_matrix(adj_filename, self.num_of_vertices)

        # Precompute normalized Laplacian for graph regularization
        if self.use_graph_reg:
            self.laplacian = self._compute_normalized_laplacian(self.adj_mx)
            self.laplacian_tensor = torch.from_numpy(self.laplacian).float().to(self.device)

        # Load dataset
        self.load_data()

        # Split data for clients
        if self.data_split == 'iid':
            print(f"Using I.I.D data split")
            self.split_clients_iid()
        elif self.data_split == 'noniid':
            print(f"Using Non-I.I.D data split (alpha={self.alpha})")
            self.split_clients_noniid()
        else:
            raise ValueError(f"Unknown data_split: {self.data_split}")

        # Store client data sizes for proper FedAvg weighting
        self.client_data_sizes = [len(self.client_datasets[i].dataset) for i in range(num_clients)]
        self.total_data_size = sum(self.client_data_sizes)

        print(f"\nClient data distribution:")
        for i, size in enumerate(self.client_data_sizes):
            weight = size / self.total_data_size
            print(f"  Client {i}: {size} samples, weight={weight:.4f}")

        print(f"\nOptimization settings:")
        print(f"  FedProx mu: {self.fedprox_mu}")
        print(f"  Graph regularization: {self.use_graph_reg}")
        print(f"  Local epochs: {self.local_epochs}")
        print(f"  Server momentum (FedAvgM): {self.server_momentum}")
        print(f"  Dropout: {self.dropout}")
        print(f"  Adaptive graph: {self.use_adaptive} (emb_dim={self.node_emb_dim})")
        print(f"  Loss: {self.loss_type}")
        print(f"  Warmup rounds: {self.warmup_rounds}")
        print(f"  Personalization epochs: {self.personalization_epochs}")

    def _compute_normalized_laplacian(self, adj_mx):
        """Compute symmetric normalized Laplacian: L = I - D_hat^(-1/2) A_hat D_hat^(-1/2)
        where A_hat = A + I (self-loops ensure isolated nodes have degree 1,
        so they contribute 0 to smoothness loss instead of being wrongly penalized)."""
        adj = adj_mx.copy()
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_inv_sqrt = np.diag(d_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
        L = np.eye(adj.shape[0]) - norm_adj
        return L.astype(np.float32)

    def load_data(self):
        """Load and prepare dataset"""
        config = configparser.ConfigParser()
        config.read(self.config_file)

        (self.train_loader, self.train_target_tensor,
         self.val_loader, self.val_target_tensor,
         self.test_loader, self.test_target_tensor,
         self.mean, self.std) = load_graphdata_channel1(
            config['Data']['graph_signal_matrix_filename'],
            self.num_of_hours,
            self.num_of_days,
            self.num_of_weeks,
            self.device,
            self.batch_size
        )

    def split_clients_iid(self):
        """Split training data across clients (I.I.D)"""
        train_dataset = self.train_loader.dataset
        n_samples = len(train_dataset)

        indices = np.random.permutation(n_samples)
        samples_per_client = n_samples // self.num_clients

        self.client_datasets = []
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else n_samples
            client_indices = indices[start_idx:end_idx]

            client_dataset = Subset(train_dataset, client_indices)
            client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
            self.client_datasets.append(client_loader)

            print(f"Client {i}: {len(client_indices)} samples")

    def split_clients_noniid(self):
        """Split training data across clients (Non-I.I.D using Dirichlet)"""
        train_dataset = self.train_loader.dataset
        n_samples = len(train_dataset)
        min_samples = 100

        proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
        samples_per_client = (proportions * n_samples).astype(int)

        while np.min(samples_per_client) < min_samples:
            max_idx = np.argmax(samples_per_client)
            min_idx = np.argmin(samples_per_client)
            transfer = min(samples_per_client[max_idx] - min_samples,
                          min_samples - samples_per_client[min_idx])
            if transfer <= 0:
                break
            samples_per_client[max_idx] -= transfer
            samples_per_client[min_idx] += transfer

        samples_per_client[-1] = n_samples - samples_per_client[:-1].sum()
        indices = np.random.permutation(n_samples)

        self.client_datasets = []
        start_idx = 0

        for i in range(self.num_clients):
            end_idx = start_idx + samples_per_client[i]
            client_indices = indices[start_idx:end_idx]

            client_dataset = Subset(train_dataset, client_indices)
            client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
            self.client_datasets.append(client_loader)

            percentage = (len(client_indices) / n_samples) * 100
            print(f"Client {i}: {len(client_indices)} samples ({percentage:.1f}%)")

            start_idx = end_idx

        sample_counts = [len(self.client_datasets[i].dataset) for i in range(self.num_clients)]
        print(f"\nNon-IID Statistics (alpha={self.alpha}):")
        print(f"  Min samples: {np.min(sample_counts)}")
        print(f"  Max samples: {np.max(sample_counts)}")
        print(f"  Heterogeneity ratio: {np.max(sample_counts)/np.min(sample_counts):.2f}x")

    def create_model(self):
        """Initialize ASTGCN model"""
        num_components = (self.num_of_hours > 0) + (self.num_of_days > 0) + (self.num_of_weeks > 0)
        time_strides = max(1, num_components)

        model = make_model(
            self.device,
            self.nb_block,
            self.in_channels,
            self.K,
            self.nb_chev_filter,
            self.nb_time_filter,
            time_strides,
            self.adj_mx,
            self.num_for_predict,
            self.len_input,
            self.num_of_vertices,
            dropout=self.dropout,
            use_adaptive=self.use_adaptive,
            node_emb_dim=self.node_emb_dim
        )

        if self.model_size_mb == 0:
            total_params = sum(p.numel() for p in model.parameters())
            self.model_size_mb = total_params * 4 / (1024 ** 2)
            print(f"Model parameters: {total_params:,}, Size: {self.model_size_mb:.2f} MB")

        return model

    def track_communication(self, round_idx):
        """Track communication cost"""
        self.total_comm_rounds += 1
        self.total_params_communicated += self.model_size_mb * self.num_clients * 2

    def compute_graph_reg_loss(self, predictions, targets):
        """
        Compute graph Laplacian smoothness loss on prediction RESIDUALS (Rayleigh quotient).
        L_smooth = sum(r * L r) / sum(r * r)   (bounded in [0, 2])
        """
        residuals = predictions - targets
        B, N, T = residuals.shape
        r = residuals.permute(0, 2, 1).reshape(-1, N)
        Lr = torch.mm(r, self.laplacian_tensor)
        numerator = (r * Lr).sum()
        denominator = (r * r).sum().clamp(min=1e-8)
        smooth_loss = numerator / denominator
        return smooth_loss

    def _get_criterion(self):
        """Get loss function based on loss_type"""
        if self.loss_type == 'mae':
            return nn.L1Loss()
        elif self.loss_type == 'mse':
            return nn.MSELoss()
        else:  # huber
            return nn.SmoothL1Loss()

    def client_update(self, model, client_loader, global_weights, current_lr):
        """
        Train model on client data with:
        - Configurable loss function
        - FedProx regularization
        - Graph Laplacian smoothness loss
        - Gradient clipping
        - Weight decay
        """
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-5)
        criterion = self._get_criterion()

        # Store global weights for FedProx
        if self.fedprox_mu > 0:
            global_params = {name: global_weights[name].clone().detach()
                           for name in dict(model.named_parameters())
                           if name in global_weights}

        for epoch in range(self.local_epochs):
            task_losses = []
            for batch_idx, (inputs, labels) in enumerate(client_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)

                task_loss = criterion(outputs, labels)
                loss = task_loss

                if self.use_graph_reg:
                    graph_loss = self.compute_graph_reg_loss(outputs, labels)
                    loss = loss + self.graph_reg_weight * graph_loss

                if self.fedprox_mu > 0:
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        if name in global_params:
                            proximal_term += ((param - global_params[name]) ** 2).sum()
                    loss = loss + (self.fedprox_mu / 2) * proximal_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                task_losses.append(task_loss.item())

        return model.state_dict(), np.mean(task_losses)

    def personalize(self, model, rounds_lr):
        """
        Post-aggregation personalization: fine-tune global model on ALL training data
        with a small learning rate for a few epochs. This adapts the global model
        to the local data distribution before evaluation.
        """
        if self.personalization_epochs <= 0:
            return

        model.train()
        finetune_lr = rounds_lr * 0.1  # 10x smaller LR for fine-tuning
        optimizer = optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=1e-5)
        criterion = self._get_criterion()

        for epoch in range(self.personalization_epochs):
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

    def fedavg_aggregate(self, client_weights, client_performance=None):
        """
        Component-Specific Aggregation (Main Contribution)

        Spatial attention + adaptive graph params: Performance-weighted
        Temporal + other params: Data-proportional FedAvg
        """
        avg_weights = {}
        num_clients = len(client_weights)

        # Data-proportional weights
        data_weights = np.array(self.client_data_sizes) / self.total_data_size

        if self.aggregation == 'standard':
            for key in client_weights[0].keys():
                weighted_sum = sum(
                    data_weights[i] * client_weights[i][key].float()
                    for i in range(num_clients)
                )
                avg_weights[key] = weighted_sum

        else:  # 'attention' / 'attention_all' / 'attention_other'
            if client_performance is not None:
                perf_array = np.array(client_performance)

                temperature = 2.0
                perf_weights = np.exp(-(perf_array - perf_array.min()) / temperature)
                perf_weights = perf_weights / perf_weights.sum()
                max_ratio = 5.0
                min_allowed = perf_weights.max() / max_ratio
                perf_weights = np.maximum(perf_weights, min_allowed)
                perf_weights = perf_weights / perf_weights.sum()

                print(f"  Client train loss: {[f'{x:.4f}' for x in client_performance]}")
                print(f"  Perf weights:      {[f'{x:.4f}' for x in perf_weights]}")
            else:
                perf_weights = data_weights

            # Spatial attention params + adaptive graph params (node embeddings, adaptive_theta)
            spatial_params = ['W1', 'W2', 'W3', 'bs', 'Vs', 'node_emb', 'adaptive_theta', 'alpha_blend']

            spatial_count = 0
            other_count = 0

            for key in client_weights[0].keys():
                is_spatial = any(sp in key for sp in spatial_params)

                # Choose performance vs data weighting for this key.
                if self.aggregation == 'attention_all':
                    use_perf = True
                elif self.aggregation == 'attention_other':
                    use_perf = not is_spatial
                else:  # 'attention'
                    use_perf = is_spatial

                key_weights = perf_weights if use_perf else data_weights
                avg_weights[key] = sum(
                    key_weights[i] * client_weights[i][key].float()
                    for i in range(num_clients)
                )
                if is_spatial:
                    spatial_count += 1
                else:
                    other_count += 1

            if not hasattr(self, '_logged_params'):
                if self.aggregation == 'attention_all':
                    perf_grp, data_grp = 'all', 'none'
                elif self.aggregation == 'attention_other':
                    perf_grp, data_grp = 'other', 'spatial+adaptive'
                else:
                    perf_grp, data_grp = 'spatial+adaptive', 'other'
                print(f"  Param groups [{self.aggregation}]: {spatial_count} spatial+adaptive, "
                      f"{other_count} other | perf-weighted={perf_grp}, data-weighted={data_grp}")
                self._logged_params = True

        return avg_weights

    def evaluate(self, model):
        """Evaluate model on validation set"""
        model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                target = target.to(self.device)

                output = model(inputs)
                predictions.append(output.cpu().numpy())
                labels.append(target.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)

        mae = np.mean(np.abs(predictions - labels))
        rmse = np.sqrt(np.mean((predictions - labels) ** 2))

        mask = labels > 10.0
        if mask.sum() > 0:
            mape = np.mean(np.abs((predictions[mask] - labels[mask]) / labels[mask])) * 100
        else:
            mape = 0.0

        return mae, rmse, mape

    def test(self, model):
        """Test model on test set"""
        model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                target = target.to(self.device)

                output = model(inputs)
                predictions.append(output.cpu().numpy())
                labels.append(target.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)

        mae = np.mean(np.abs(predictions - labels))
        rmse = np.sqrt(np.mean((predictions - labels) ** 2))

        mask = labels > 10.0
        if mask.sum() > 0:
            mape = np.mean(np.abs((predictions[mask] - labels[mask]) / labels[mask])) * 100
        else:
            mape = 0.0

        return mae, rmse, mape

    def train(self, rounds=100):
        """Federated training with warmup + cosine LR schedule + personalization"""
        global_model = self.create_model()
        global_model = global_model.to(self.device)

        best_val_mae = float('inf')
        best_round = 0

        # Server-side momentum buffer (FedAvgM)
        momentum_buffer = {} if self.server_momentum > 0 else None

        # Learning rate schedule
        initial_lr = self.lr
        min_lr = initial_lr * 0.01

        for round_idx in range(rounds):
            round_start = time.time()
            print(f"\n=== Round {round_idx}/{rounds} ===")

            # LR with warmup + cosine annealing
            if round_idx < self.warmup_rounds:
                # Linear warmup
                current_lr = initial_lr * (round_idx + 1) / self.warmup_rounds
            else:
                # Cosine annealing after warmup
                progress = (round_idx - self.warmup_rounds) / max(1, rounds - self.warmup_rounds)
                current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))

            # Schedule graph reg weight: ramp 0.01 -> 0.2 over training
            if self.use_graph_reg:
                progress = round_idx / max(1, rounds - 1)
                self.graph_reg_weight = 0.01 + 0.19 * progress

            if round_idx % 10 == 0:
                print(f"  Learning rate: {current_lr:.6f}")

            client_weights = []
            client_losses = []

            global_weights_dict = global_model.state_dict()

            for client_id in range(self.num_clients):
                print(f"Training client {client_id}...", end=" ")

                client_model = copy.deepcopy(global_model)

                weights, train_loss = self.client_update(
                    client_model,
                    self.client_datasets[client_id],
                    global_weights_dict,
                    current_lr
                )

                client_losses.append(train_loss)
                print(f"loss={train_loss:.4f}")

                client_weights.append(weights)

            # Save each client's pre-aggregation weights at the requested rounds.
            if round_idx in self.dump_clients_round:
                seed_tag = self.current_seed if self.current_seed is not None else 'NA'
                dump_dir = os.path.join(
                    'client_dumps',
                    f'{self.dataset_name}_alpha{self.alpha}_seed{seed_tag}',
                    f'round{round_idx}')
                os.makedirs(dump_dir, exist_ok=True)
                for cid, w in enumerate(client_weights):
                    cpu_w = {k: v.detach().cpu() for k, v in w.items()}
                    torch.save(cpu_w, os.path.join(dump_dir, f'client_{cid}.pkl'))
                print(f"  [dump] saved {self.num_clients} client state_dicts to {dump_dir}")
                if round_idx == max(self.dump_clients_round):
                    print(f"  [dump] last dump round reached; stopping training early.")
                    break

            # Aggregate
            global_weights = self.fedavg_aggregate(client_weights, client_losses)

            # Apply server-side momentum (FedAvgM)
            if self.server_momentum > 0:
                for key in global_weights:
                    delta = global_weights[key] - global_weights_dict[key]
                    if key not in momentum_buffer:
                        momentum_buffer[key] = delta.clone()
                    else:
                        momentum_buffer[key] = self.server_momentum * momentum_buffer[key] + delta
                    global_weights[key] = global_weights_dict[key] + momentum_buffer[key]

            global_model.load_state_dict(global_weights)

            # Post-aggregation personalization
            self.personalize(global_model, current_lr)

            self.track_communication(round_idx)

            # Evaluate global model
            val_mae, val_rmse, val_mape = self.evaluate(global_model)

            round_end = time.time()
            print(f"Round {round_idx} - Time: {round_end - round_start:.2f}s")
            print(f"  Valid MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}%")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_round = round_idx
                torch.save(global_model.state_dict(), self.checkpoint_path)
                print(f"  *** New best MAE: {best_val_mae:.4f} ***")

        print(f"\n=== Training Complete ===")
        print(f"Best validation MAE: {best_val_mae:.4f} (round {best_round})")

        print(f"\n=== Final Test Evaluation ===")
        global_model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))
        test_mae, test_rmse, test_mape = self.test(global_model)
        print(f"Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")

        print(f"\n=== Communication Cost ===")
        print(f"Model size: {self.model_size_mb:.2f} MB")
        print(f"Total rounds: {self.total_comm_rounds}")
        print(f"Total data communicated: {self.total_params_communicated:.2f} MB")

        return best_val_mae, test_mae, test_rmse, test_mape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configurations/PEMS04_astgcn.conf')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--data_split', type=str, default='noniid', choices=['iid', 'noniid'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--aggregation', type=str, default='attention',
                        choices=['attention', 'standard', 'attention_all', 'attention_other'],
                        help='attention=component-specific, standard=FedAvg, '
                             'attention_all/attention_other=grouping variants')
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                        help='FedProx regularization strength (0 to disable)')
    parser.add_argument('--use_graph_reg', type=lambda x: x.lower() == 'true', default=True,
                        help='Use graph Laplacian smoothness regularization')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Number of local training epochs per round')
    parser.add_argument('--server_momentum', type=float, default=0,
                        help='Server-side momentum (FedAvgM). 0 to disable.')
    # New arguments
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in ASTGCN blocks (0 to disable)')
    parser.add_argument('--use_adaptive', type=lambda x: x.lower() == 'true', default=True,
                        help='Use adaptive graph learning with node embeddings')
    parser.add_argument('--node_emb_dim', type=int, default=16,
                        help='Dimension of node embeddings for adaptive adjacency')
    parser.add_argument('--loss_type', type=str, default='huber', choices=['huber', 'mae', 'mse'],
                        help='Loss function: huber (SmoothL1), mae (L1), mse')
    parser.add_argument('--warmup_rounds', type=int, default=0,
                        help='Number of linear LR warmup rounds')
    parser.add_argument('--personalization_epochs', type=int, default=0,
                        help='Post-aggregation fine-tuning epochs (0 to disable)')
    parser.add_argument('--dump_clients_round', type=int, nargs='+', default=None,
                        help='Rounds at which to dump each client pre-aggregation '
                             'state_dict; training stops after the last one. '
                             'E.g. --dump_clients_round 10 30 50')
    args = parser.parse_args()

    dataset_name = os.path.basename(args.config).split('_')[0].lower()
    os.makedirs('saved_results/fedastgcn_results', exist_ok=True)

    print("="*60)
    print("FED-ASTGCN: Adaptive Federated Spatial-Temporal Forecasting")
    print("="*60)
    print(f"  Component-specific aggregation: {args.aggregation}")
    print(f"  FedProx mu: {args.fedprox_mu}")
    print(f"  Graph regularization: {args.use_graph_reg}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Cosine LR + warmup: {args.warmup_rounds} rounds")
    print(f"  Gradient clipping: 3.0, Weight decay: 1e-5")
    print(f"  Loss: {args.loss_type}")
    print(f"  Server momentum: {args.server_momentum}")
    print(f"  Adaptive graph: {args.use_adaptive} (emb_dim={args.node_emb_dim})")
    print(f"  Dropout: {args.dropout}")
    print(f"  Personalization epochs: {args.personalization_epochs}")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Clients: {args.num_clients}, Rounds: {args.rounds}")
    print(f"Data: {args.data_split.upper()}, Alpha: {args.alpha}")

    if args.seeds is not None:
        seed_list = args.seeds
        actual_num_runs = len(seed_list)
        print(f"Seeds: {seed_list}")
    elif args.num_runs > 1:
        seed_list = [args.seed + i for i in range(args.num_runs)]
        actual_num_runs = args.num_runs
    else:
        seed_list = None
        actual_num_runs = 1
        print(f"Seed: {args.seed}")
    print("="*60)

    results_filename = f"saved_results/fedastgcn_results/results_{args.aggregation}_{args.data_split}"
    if args.data_split == 'noniid':
        results_filename += f"_alpha{args.alpha}"
    if actual_num_runs > 1:
        results_filename += f"_{actual_num_runs}runs_{dataset_name}.json"
    else:
        results_filename += f"_seed{args.seed}_{dataset_name}.json"

    if actual_num_runs > 1:
        all_results = []

        for run_idx, current_seed in enumerate(seed_list):
            print(f"\n{'='*60}")
            print(f"RUN {run_idx + 1}/{actual_num_runs} (Seed: {current_seed})")
            print(f"{'='*60}")

            set_seed(current_seed)

            fed_trainer = FederatedASTGCN(
                args.config,
                num_clients=args.num_clients,
                data_split=args.data_split,
                alpha=args.alpha,
                aggregation=args.aggregation,
                fedprox_mu=args.fedprox_mu,
                use_graph_reg=args.use_graph_reg,
                local_epochs=args.local_epochs,
                server_momentum=args.server_momentum,
                dropout=args.dropout,
                use_adaptive=args.use_adaptive,
                node_emb_dim=args.node_emb_dim,
                loss_type=args.loss_type,
                warmup_rounds=args.warmup_rounds,
                personalization_epochs=args.personalization_epochs,
                dump_clients_round=args.dump_clients_round,
                current_seed=current_seed
            )
            best_val_mae, test_mae, test_rmse, test_mape = fed_trainer.train(rounds=args.rounds)
            all_results.append({
                'seed': int(current_seed),
                'val_mae': float(best_val_mae),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_mape': float(test_mape),
                'comm_cost_mb': float(fed_trainer.total_params_communicated),
                'model_size_mb': float(fed_trainer.model_size_mb)
            })

        val_maes = [r['val_mae'] for r in all_results]
        test_maes = [r['test_mae'] for r in all_results]
        test_rmses = [r['test_rmse'] for r in all_results]
        test_mapes = [r['test_mape'] for r in all_results]

        n = len(test_maes)
        test_mae_ci = stats.t.interval(0.95, n-1, loc=np.mean(test_maes), scale=stats.sem(test_maes))

        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        print(f"Test MAE:  {np.mean(test_maes):.4f} +/- {np.std(test_maes):.4f}")
        print(f"Test RMSE: {np.mean(test_rmses):.4f} +/- {np.std(test_rmses):.4f}")
        print(f"Test MAPE: {np.mean(test_mapes):.2f}% +/- {np.std(test_mapes):.2f}%")
        print(f"95% CI:    [{test_mae_ci[0]:.4f}, {test_mae_ci[1]:.4f}]")
        print("="*60)

        fedagat_targets = {'pems03': 15.00, 'pems04': 19.22, 'pems07': 20.43, 'pems08': 15.66}
        if dataset_name in fedagat_targets:
            target = fedagat_targets[dataset_name]
            our_mae = np.mean(test_maes)
            print(f"\nFedAGAT {dataset_name.upper()}: {target:.2f}")
            print(f"Our Result: {our_mae:.2f} +/- {np.std(test_maes):.2f}")
            if our_mae < target:
                print(">>> BEATS FedAGAT! <<<")
            else:
                print(f"Gap: +{our_mae - target:.2f} MAE ({(our_mae-target)/target*100:.1f}% worse)")

        results_data = {
            'config': args.config,
            'num_clients': args.num_clients,
            'rounds': args.rounds,
            'data_split': args.data_split,
            'alpha': args.alpha if args.data_split == 'noniid' else None,
            'aggregation': args.aggregation,
            'fedprox_mu': args.fedprox_mu,
            'use_graph_reg': args.use_graph_reg,
            'local_epochs': args.local_epochs,
            'server_momentum': args.server_momentum,
            'dropout': args.dropout,
            'use_adaptive': args.use_adaptive,
            'node_emb_dim': args.node_emb_dim,
            'loss_type': args.loss_type,
            'warmup_rounds': args.warmup_rounds,
            'personalization_epochs': args.personalization_epochs,
            'num_runs': actual_num_runs,
            'seeds_used': seed_list,
            'summary': {
                'test_mae_mean': float(np.mean(test_maes)),
                'test_mae_std': float(np.std(test_maes)),
                'test_rmse_mean': float(np.mean(test_rmses)),
                'test_rmse_std': float(np.std(test_rmses)),
                'test_mape_mean': float(np.mean(test_mapes)),
                'test_mape_std': float(np.std(test_mapes)),
            },
            'individual_runs': all_results
        }
        with open(results_filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {results_filename}")

    else:
        set_seed(args.seed)

        fed_trainer = FederatedASTGCN(
            args.config,
            num_clients=args.num_clients,
            data_split=args.data_split,
            alpha=args.alpha,
            aggregation=args.aggregation,
            fedprox_mu=args.fedprox_mu,
            use_graph_reg=args.use_graph_reg,
            local_epochs=args.local_epochs,
            server_momentum=args.server_momentum,
            dropout=args.dropout,
            use_adaptive=args.use_adaptive,
            node_emb_dim=args.node_emb_dim,
            loss_type=args.loss_type,
            warmup_rounds=args.warmup_rounds,
            personalization_epochs=args.personalization_epochs,
            dump_clients_round=args.dump_clients_round,
            current_seed=args.seed
        )
        best_val_mae, test_mae, test_rmse, test_mape = fed_trainer.train(rounds=args.rounds)

        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")
        print("="*60)

        results_data = {
            'config': args.config,
            'num_clients': args.num_clients,
            'rounds': args.rounds,
            'data_split': args.data_split,
            'alpha': args.alpha if args.data_split == 'noniid' else None,
            'aggregation': args.aggregation,
            'fedprox_mu': args.fedprox_mu,
            'use_graph_reg': args.use_graph_reg,
            'local_epochs': args.local_epochs,
            'server_momentum': args.server_momentum,
            'dropout': args.dropout,
            'use_adaptive': args.use_adaptive,
            'node_emb_dim': args.node_emb_dim,
            'loss_type': args.loss_type,
            'warmup_rounds': args.warmup_rounds,
            'personalization_epochs': args.personalization_epochs,
            'seed': args.seed,
            'results': {
                'val_mae': float(best_val_mae),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_mape': float(test_mape),
            }
        }
        with open(results_filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {results_filename}")
























