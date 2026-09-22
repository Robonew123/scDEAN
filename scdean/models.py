"""
scDEAN model definitions.

Contains:
    GCNLatentRepresentation  - Representation Learning module (dual encoder + decoder)
    DeepClustering           - Deep Clustering module (classifier + learnable centroids)

Fusion
------
`merge_with_softmax_gating` and the `softmax_gate1..3` submodules implement the
softmax-gated fusion of Section 2.2.2 / Fig. 3 of the paper:

    (d1, d2) = Softmax( concat(E_fe, E_ge) @ W_fus )      (Eq. 9)
    E        = d1 * E_fe + d2 * E_ge                      (Eq. 10)

with W_fus in R^{2d x 2}, i.e. 4d parameters per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GCNlayer, MLP, StudentskernalAdjacency


# Main GCN model with batching support
class GCNLatentRepresentation(nn.Module):
    def __init__(self, in_fea, hidden1, hidden2, latent, samplesize, use_zinb=False):
        super(GCNLatentRepresentation, self).__init__()
        self.use_zinb = use_zinb
        # Encoder layers
        self.gcn1 = GCNlayer(in_fea, hidden1)
        self.mlp1 = MLP([in_fea, hidden1], final_activation=True)
        self.gcn2 = GCNlayer(hidden1, hidden2)
        self.mlp2 = MLP([hidden1, hidden2], final_activation=True)
        self.gcn3 = GCNlayer(hidden2, latent)
        self.mlp3 = MLP([hidden2, latent], final_activation=True)

        # Softmax-gated fusion layers for each merge point (Eq. 9)
        self.softmax_gate1 = nn.Sequential(nn.Linear(hidden1 * 2, 2), nn.Softmax(dim=1))
        self.softmax_gate2 = nn.Sequential(nn.Linear(hidden2 * 2, 2), nn.Softmax(dim=1))
        self.softmax_gate3 = nn.Sequential(nn.Linear(latent * 2, 2), nn.Softmax(dim=1))

        # Decoder layers
        self.mlp4 = nn.Sequential(nn.Linear(latent, hidden2), nn.LeakyReLU(0.1))
        self.mlp5 = nn.Sequential(nn.Linear(hidden2, hidden1), nn.LeakyReLU(0.1))
        self.mlp6 = nn.Sequential(nn.Linear(hidden1, in_fea), nn.LeakyReLU(0.1))

        self.mean_layer = nn.Linear(in_fea, in_fea)
        self.disp_layer = nn.Linear(in_fea, in_fea)

        # Dropout-probability head, only instantiated for the ZINB ablation
        if use_zinb:
            self.drop_layer = nn.Linear(in_fea, in_fea)

        # Normalization layers (declared for checkpoint compatibility; not used in forward)
        self.batch_norm_input = nn.BatchNorm1d(samplesize)
        self.batch_norm_hidden1 = nn.BatchNorm1d(hidden1)
        self.batch_norm_hidden2 = nn.BatchNorm1d(hidden2)
        self.batch_norm_latent = nn.BatchNorm1d(latent)
        self.batch_norm_adj = nn.BatchNorm1d(samplesize)

        # Layer normalizations for feature-wise normalization (not used in forward)
        self.layer_norm1 = nn.LayerNorm(hidden1)
        self.layer_norm2 = nn.LayerNorm(hidden2)
        self.layer_norm_latent = nn.LayerNorm(latent)

        # Pre-instantiate adjacency kernel - avoids object creation on every forward call
        self._student_adj = StudentskernalAdjacency(sigma=1.0)

    def merge_with_softmax_gating(self, x1, x2, gate_layer):
        """Softmax-gated fusion of the GCN and FNN branches (Eqs. 9-10)."""
        combined = torch.cat([x1, x2], dim=1)
        weights = gate_layer(combined)
        return weights[:, 0:1] * x1 + weights[:, 1:2] * x2

    def forward(self, x, adj, size_factor_matrix, batch_indices=None):
        device = x.device

        # If batch_indices is provided, we're in batch mode
        if batch_indices is not None:
            # Extract the relevant parts of adj for the batch
            # For a batch, we need to work with the submatrices of adj and size_factor_matrix
            batch_adj = adj[batch_indices][:, batch_indices]
            batch_size_factor = size_factor_matrix[batch_indices][:, batch_indices]

            # Convert batch adjacency matrix to sparse format
            edge_index, edge_weight = self.dense_to_sparse(batch_adj)
        else:
            # Full dataset mode
            batch_adj = adj
            batch_size_factor = size_factor_matrix

            # Convert adjacency matrix to sparse format
            edge_index, edge_weight = self.dense_to_sparse(adj)

        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        # GCN path
        xg = F.leaky_relu(self.gcn1(x, edge_index, edge_weight=edge_weight))
        xa = self.mlp1(x)
        x = self.merge_with_softmax_gating(xg, xa, self.softmax_gate1)

        xg = F.leaky_relu(self.gcn2(x, edge_index, edge_weight=edge_weight))
        xa = self.mlp2(x)
        x = self.merge_with_softmax_gating(xg, xa, self.softmax_gate2)

        xg = self.gcn3(x, edge_index, edge_weight=edge_weight)
        xa = self.mlp3(x)
        latent_rep = self.merge_with_softmax_gating(xg, xa, self.softmax_gate3)

        # Decoder
        x = self.mlp4(latent_rep)
        x_last = self.mlp5(x)
        x_pred = self.mlp6(x_last)

        # Compute mean with numerical stability (softplus used instead of exp)
        mean = torch.matmul(batch_size_factor, F.softplus(self.mean_layer(x_pred)))

        # Compute dispersion
        disp = torch.exp(self.disp_layer(x_pred))

        # adj calculation from latent layer (use cached instance - no per-call allocation)
        adj_latent = self._student_adj.compute_adjacency(latent_rep)

        return x_pred, latent_rep, adj_latent, mean, disp

    def compute_dropout(self, x_pred):
        """Zero-inflation probability pi for the ZINB ablation."""
        if not self.use_zinb:
            raise RuntimeError("Model was built with use_zinb=False; no drop_layer exists.")
        return torch.sigmoid(self.drop_layer(x_pred))

    def dense_to_sparse(self, adj):
        """Convert a dense adjacency matrix to sparse format (edge_index, edge_weight)"""
        # Find the indices of non-zero elements
        indices = torch.nonzero(adj, as_tuple=True)
        edge_index = torch.stack(indices)

        # Get the corresponding values
        edge_weight = adj[indices]

        return edge_index, edge_weight

    def encode(self, x, adj, batch_indices=None):
        """Extract only the encoder part for getting latent representations"""
        device = x.device

        # If batch_indices is provided, we're in batch mode
        if batch_indices is not None:
            # Extract the relevant parts of adj for the batch
            batch_adj = adj[batch_indices][:, batch_indices]

            # Convert batch adjacency matrix to sparse format
            edge_index, edge_weight = self.dense_to_sparse(batch_adj)
        else:
            # Full dataset mode
            batch_adj = adj

            # Convert adjacency matrix to sparse format
            edge_index, edge_weight = self.dense_to_sparse(adj)

        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        # Encoder path
        xg = F.leaky_relu(self.gcn1(x, edge_index, edge_weight))
        xa = self.mlp1(x)

        x = self.merge_with_softmax_gating(xg, xa, self.softmax_gate1)

        xg = F.leaky_relu(self.gcn2(x, edge_index, edge_weight))
        xa = self.mlp2(x)

        x = self.merge_with_softmax_gating(xg, xa, self.softmax_gate2)

        xg = self.gcn3(x, edge_index, edge_weight)
        xa = self.mlp3(x)

        latent_rep = self.merge_with_softmax_gating(xg, xa, self.softmax_gate3)

        return latent_rep


class DeepClustering(nn.Module):
    """
    Deep Clustering module (Section 2.2.3).

    `forward` returns the raw classifier logits; the softmax of Eq. (22) is
    applied inside torch.nn.CrossEntropyLoss during fine-tuning, so the
    objective is exactly Eq. (23).
    """

    def __init__(self, in_fea, out_fea, initial_centroids=None):
        super(DeepClustering, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_fea, in_fea), nn.LeakyReLU(0.1), nn.Linear(in_fea, out_fea))
        self.softmax = nn.Softmax(dim=1)
        self.n_clusters = out_fea

        if initial_centroids is not None:
            self.centroids = nn.Parameter(torch.tensor(initial_centroids, dtype=torch.float32, requires_grad=True))
        else:
            self.centroids = nn.Parameter(torch.randn(out_fea, in_fea, requires_grad=True))

    def forward(self, latent):
        last_layer = self.mlp(latent)
        cluster_probs = last_layer
        return cluster_probs

    def get_centroids(self):
        return self.centroids

    def update_centroids(self, latent_rep, membership, epoch):
        """
        Optional FCM-style hard centroid reassignment.

        Disabled by default (`--update_centroids` off). All results reported in
        the paper use purely gradient-based centroid refinement.
        """
        with torch.no_grad():
            m = 1.5
            membership_p = membership ** m
            mt = membership_p.contiguous().T
            new_centroids = torch.matmul(mt, latent_rep) / (membership_p.sum(dim=0, keepdim=True).t() + 1e-8)

            if epoch % 20 == 0 and epoch != 0 and epoch < 65:
                print('updating to new centroid')
                self.centroids.data = new_centroids
