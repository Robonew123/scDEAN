"""
Loss functions for the scDEAN Representation Learning module.

    autoencoder_loss              - L_rec, MSE reconstruction        (Eq. 15)
    adjacency_reconstruction_loss - L_bce, graph structure BCE       (Eq. 12)
    graph_regularization_loss     - L_reg, graph Laplacian smoothing (Eq. 13)
    nb_loss                       - L_NB, negative binomial NLL      (Eq. 19)
    zinb_loss                     - zero-inflated NB NLL (ablation only)

All terms are averaged over their elements; the constant rescaling relative to
the summed form written in the paper is absorbed into the weights
alpha / beta / gamma / lambda.
"""

import torch
import torch.nn.functional as F


def autoencoder_loss(x, x_pred):
    """Compute the autoencoder reconstruction loss"""
    return F.mse_loss(x_pred, x)


def adjacency_reconstruction_loss(adj_true, adj_pred):
    """Compute the adjacency matrix reconstruction loss"""
    eps = 1e-7
    adj_true = torch.clamp(adj_true, 0, 1)
    adj_pred = torch.clamp(adj_pred, eps, 1 - eps)
    loss = F.binary_cross_entropy(adj_pred, adj_true, reduction='mean')

    # Masking the diagonal was tested and decreased performance on the Baron
    # dataset, so self-similarities are retained in the objective.

    return loss


def graph_regularization_loss(latent, edge_index, edge_weight):
    """Compute the graph regularization loss"""
    if edge_index.shape[1] == 0:  # No edges present
        return torch.tensor(0.0, requires_grad=True, device=latent.device)
    row, col = edge_index[0], edge_index[1]
    # Vectorized - numerically identical to the original per-edge F.mse_loss loop:
    # F.mse_loss(a, b) = mean((a-b)^2) over elements, then sum over edges / n_edges
    diff = latent[row] - latent[col]                    # (E, latent_dim)
    per_edge_mse = (diff * diff).mean(dim=1)            # (E,)  matches F.mse_loss reduction='mean'
    loss = (edge_weight * per_edge_mse).sum() / edge_index.shape[1]
    return loss


def nb_loss(X, mu, theta, eps=1e-10, reduction='mean'):
    """Compute the Negative Binomial (NB) loss (negative log-likelihood)."""
    assert X.shape == mu.shape == theta.shape, "All inputs must have the same shape"

    # Clip mu and theta to avoid log(0) and division by zero
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)

    # Log-gamma functions
    term1 = torch.lgamma(X + theta) - torch.lgamma(theta) - torch.lgamma(X + 1)

    # Logarithms of probabilities for negative binomial
    term2 = theta * torch.log(theta / ((mu + theta) + eps))
    term3 = X * torch.log(mu / ((mu + theta) + eps))

    # Negative log-likelihood for the NB distribution
    nb_nll = -(term1 + term2 + term3)

    if reduction == 'sum':
        return torch.sum(nb_nll)
    elif reduction == 'mean':
        return torch.mean(nb_nll)
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")


def zinb_loss(X, mu, theta, pi, eps=1e-10, reduction='mean'):
    """
    Zero-inflated Negative Binomial (ZINB) negative log-likelihood.

    Used only for the reconstruction-objective ablation of Section 3.3.1; the
    default scDEAN configuration uses `nb_loss`. `pi` is the zero-inflation
    probability produced by GCNLatentRepresentation.compute_dropout().
    """
    assert X.shape == mu.shape == theta.shape == pi.shape, "All inputs must have the same shape"

    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)

    # Negative binomial negative log-likelihood
    t1 = torch.lgamma(theta + eps) + torch.lgamma(X + 1.0) - torch.lgamma(X + theta + eps)
    t2 = (theta + X) * torch.log(1.0 + (mu / (theta + eps))) \
        + (X * (torch.log(theta + eps) - torch.log(mu + eps)))
    nb = t1 + t2

    # Mix in the zero-inflation component
    zinb = -torch.log(pi + ((1.0 - pi) * torch.exp(-nb)) + eps)

    if reduction == 'sum':
        return torch.sum(zinb)
    elif reduction == 'mean':
        return torch.mean(zinb)
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")
