# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from lib.utils import scaled_Laplacian, cheb_polynomial


class GatedDilatedTimeConv(nn.Module):
    """
    Gated dilated temporal convolution (inspired by FedAGAT's TimeBlock).
    3 layers with dilations [1, 2, 4], kernel_size=2, receptive field=7.
    Each layer: tanh(conv) * sigmoid(gate) + (1-sigmoid(gate)) * residual
    Drop-in replacement for Conv2d(F_in, F_out, (1, 3)).
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, num_layers=3):
        super(GatedDilatedTimeConv, self).__init__()
        self.num_layers = num_layers
        dilations = [2 ** i for i in range(num_layers)]

        # Skip connection from input
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.main_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            d = dilations[i]
            c_in = in_channels if i == 0 else out_channels
            # Causal padding: pad left side only
            self.main_convs.append(nn.Conv2d(c_in, out_channels, (1, kernel_size), dilation=(1, d)))
            self.gate_convs.append(nn.Conv2d(c_in, out_channels, (1, kernel_size), dilation=(1, d)))
            self.residual_convs.append(nn.Conv2d(c_in, out_channels, kernel_size=1) if c_in != out_channels else nn.Identity())
            self.norms.append(nn.BatchNorm2d(out_channels))

        self._dilations = dilations
        self._kernel_size = kernel_size

    def forward(self, x):
        """
        :param x: (B, F, N, T)
        :return: (B, F_out, N, T)
        """
        skip = self.skip_conv(x)
        out = x

        for i in range(self.num_layers):
            d = self._dilations[i]
            pad = (self._kernel_size - 1) * d
            # Causal pad: only left side in temporal dimension
            padded = F.pad(out, (pad, 0))  # pad last dim (T) on left

            main = torch.tanh(self.main_convs[i](padded))
            gate = torch.sigmoid(self.gate_convs[i](padded))
            res = self.residual_convs[i](out)

            out = gate * main + (1 - gate) * res
            out = self.norms[i](out)

        return F.relu(out + skip)


class PredictionHead(nn.Module):
    """
    Multi-layer prediction head with GELU activation and weight normalization.
    Replaces single Conv2d for better multi-step forecasting.
    """

    def __init__(self, in_features, hidden_features, num_for_predict):
        super(PredictionHead, self).__init__()
        self.net = nn.Sequential(
            weight_norm(nn.Linear(in_features, hidden_features)),
            nn.GELU(),
            weight_norm(nn.Linear(hidden_features, hidden_features)),
            nn.GELU(),
            weight_norm(nn.Linear(hidden_features, num_for_predict)),
        )

    def forward(self, x):
        """
        :param x: (B, N, F)
        :return: (B, N, T_out)
        """
        return self.net(x)


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in)

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps,
                 dropout=0.0, use_adaptive=False):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = GatedDilatedTimeConv(nb_chev_filter, nb_time_filter)
        self.time_stride = time_strides
        if time_strides > 1:
            self.time_pool = nn.AvgPool2d(kernel_size=(1, time_strides), stride=(1, time_strides))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

        # Adaptive graph convolution: learns spatial dependencies beyond fixed adjacency
        self.use_adaptive = use_adaptive
        if use_adaptive:
            self.adaptive_theta = nn.Parameter(
                torch.FloatTensor(in_channels, nb_chev_filter).to(DEVICE))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, adaptive_adj=None):
        '''
        :param x: (batch_size, N, F_in, T)
        :param adaptive_adj: (N, N) learned adjacency or None
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)

        # Adaptive graph convolution path (learns what fixed adjacency misses)
        if self.use_adaptive and adaptive_adj is not None:
            adaptive_outputs = []
            for t in range(num_of_timesteps):
                graph_signal = x[:, :, :, t]  # (B, N, F_in)
                # Propagate through learned adjacency
                agg = torch.matmul(adaptive_adj, graph_signal)  # (B, N, F_in)
                out = torch.matmul(agg, self.adaptive_theta)  # (B, N, nb_chev_filter)
                adaptive_outputs.append(out.unsqueeze(-1))
            adaptive_gcn = F.relu(torch.cat(adaptive_outputs, dim=-1))  # (B, N, F_out, T)
            spatial_gcn = spatial_gcn + adaptive_gcn

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,F,N,T)
        if self.time_stride > 1:
            time_conv_output = self.time_pool(time_conv_output)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        if self.dropout is not None:
            x_residual = self.dropout(x_residual)

        return x_residual


class ASTGCN_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices,
                 dropout=0.0, use_adaptive=False, node_emb_dim=10):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(ASTGCN_submodule, self).__init__()

        # Node embeddings for adaptive adjacency learning
        self.use_adaptive = use_adaptive
        if use_adaptive:
            self.node_emb1 = nn.Parameter(torch.randn(num_of_vertices, node_emb_dim).to(DEVICE))
            self.node_emb2 = nn.Parameter(torch.randn(num_of_vertices, node_emb_dim).to(DEVICE))
            # Learnable blend: alpha * learned_adj + (1-alpha) * identity
            self.alpha_blend = nn.Parameter(torch.tensor(0.3).to(DEVICE))

        self.BlockList = nn.ModuleList([ASTGCN_block(
            DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
            cheb_polynomials, num_of_vertices, len_input,
            dropout=dropout, use_adaptive=use_adaptive)])

        self.BlockList.extend([ASTGCN_block(
            DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
            cheb_polynomials, num_of_vertices, len_input//time_strides,
            dropout=dropout, use_adaptive=use_adaptive)
            for _ in range(nb_block-1)])

        self.prediction_head = PredictionHead(nb_time_filter, 256, num_for_predict)

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        # Compute adaptive adjacency from node embeddings
        adaptive_adj = None
        if self.use_adaptive:
            learned_adj = F.softmax(F.relu(torch.mm(self.node_emb1, self.node_emb2.T)), dim=1)
            # Blend learned adjacency with identity (preserves self-information)
            alpha = torch.sigmoid(self.alpha_blend)
            identity = torch.eye(learned_adj.size(0), device=learned_adj.device)
            adaptive_adj = alpha * learned_adj + (1 - alpha) * identity

        for block in self.BlockList:
            x = block(x, adaptive_adj)

        # x: (B, N, F, T) -> take last timestep -> (B, N, F) -> prediction head -> (B, N, T_out)
        output = self.prediction_head(x[:, :, :, -1])

        return output


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
               time_strides, adj_mx, num_for_predict, len_input, num_of_vertices,
               dropout=0.0, use_adaptive=False, node_emb_dim=10):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = ASTGCN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                              time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices,
                              dropout=dropout, use_adaptive=use_adaptive, node_emb_dim=node_emb_dim)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
