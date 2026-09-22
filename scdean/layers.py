"""
Building-block layers used by the scDEAN dual encoder.

Contains:
    GCNlayer                 - thin wrapper around torch_geometric GCNConv
    MLP                      - dropout + linear + LeakyReLU stack (FNN branch)
    StudentskernalAdjacency  - Student's t similarity on the latent space (Eq. 11)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNlayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        return self.gcn(x, edge_index, edge_weight=edge_weight)


# MLP Layer definition
class MLP(nn.Module):
    def __init__(self, layers_size, final_activation=False, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        for i in range(1, len(layers_size)):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[i - 1], layers_size[i]))
            if i == len(layers_size) - 1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Adjacency computation using Student's t-kernel
class StudentskernalAdjacency:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def compute_adjacency(self, x):
        # Compute pairwise distances
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        dist = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)
        dist = torch.clamp(dist, min=1e-8)  # Ensure non-negative distances

        # Apply Student's t-kernel
        q = 1.0 / (1.0 + (dist / self.sigma))
        q = q ** ((self.sigma + 1.0) / 2.0)
        adj = q / (q.sum(dim=1, keepdim=True) + 1e-8)

        # Symmetrization
        adj = (adj + adj.T) / 2.0

        eps = 1e-8
        adj = torch.clamp(adj, eps, 1 - eps)
        return adj
