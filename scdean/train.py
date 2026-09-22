"""
Training routines for the two scDEAN phases.

    train_gcn_with_batching       - Phase 1, Representation Learning  (Algorithm 2)
    fine_tune_model               - Phase 2, Deep Clustering          (Algorithm 3)
    extract_latent_representations- inference pass returning Z

Decoupling
----------
`fine_tune_model` freezes every parameter of the Representation Learning module
and puts it in eval() mode, so only the clustering head and the centroids are
optimised (Section 2.2.3). The reconstruction / adjacency / NB tensors are still
produced by the forward pass but do not contribute to the clustering objective.
"""

import torch

from .losses import (autoencoder_loss, adjacency_reconstruction_loss,
                     graph_regularization_loss, nb_loss, zinb_loss)


def extract_latent_representations(gcn_model, dataloader, device, adj_gpu=None):
    gcn_model.eval()
    all_latent = []
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            batch_indices = batch['indices'].to(device)
            x = batch['x'].to(device)
            adj = adj_gpu if adj_gpu is not None else batch['adj'].to(device)

            latent = gcn_model.encode(x, adj, batch_indices)
            all_latent.append(latent)        # stay on device - no .cpu() round-trip
            all_indices.append(batch_indices)

    all_latent = torch.cat(all_latent, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    sorted_indices = torch.argsort(all_indices)
    sorted_latent = all_latent[sorted_indices]

    return sorted_latent  # already on device


def train_gcn_with_batching(model, dataloader, optimizer, alpha, beta, gamma, lambda_param,
                            adj_gpu=None, sf_gpu=None, impu_loss_type='nb'):
    model.train()
    total_loss = 0
    device = next(model.parameters()).device

    for batch in dataloader:
        batch_indices = batch['indices'].to(device)
        x = batch['x'].to(device)
        raw = batch['raw'].to(device)
        # Use pre-cached GPU tensors to skip per-batch N x N transfer
        adj = adj_gpu if adj_gpu is not None else batch['adj'].to(device)
        size_factor_matrix = sf_gpu if sf_gpu is not None else batch['size_factor_matrix'].to(device)

        optimizer.zero_grad()
        x_pred, latent, adj_pred, mean, disp = model(x, adj, size_factor_matrix, batch_indices)

        batch_adj = adj[batch_indices][:, batch_indices]
        # edge tensors inherit device from batch_adj - no .to() needed
        edge_index, edge_weight = model.dense_to_sparse(batch_adj)

        recon_loss = autoencoder_loss(x, x_pred)
        bce_loss = adjacency_reconstruction_loss(batch_adj, adj_pred)
        reg_loss = graph_regularization_loss(latent, edge_index, edge_weight)
        if impu_loss_type == 'zinb':
            drop = model.compute_dropout(x_pred)
            impu_loss = zinb_loss(raw, mean, disp, drop)
        else:
            impu_loss = nb_loss(raw, mean, disp)

        batch_loss = alpha * recon_loss + beta * bce_loss + gamma * reg_loss + lambda_param * impu_loss

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    return total_loss / len(dataloader)


def fine_tune_model(
    gcn_model, clust_model, dataloader, optimizer, scheduler, clust_criterion, k,
    alpha, beta, gamma, lambda_param, clustering_weight, epoch, fuzzifier_m, do_update_centroids, debug=False,
    adj_gpu=None, sf_gpu=None
):
    for param in gcn_model.parameters():
        param.requires_grad = False

    gcn_model.eval()
    clust_model.train()

    total_loss = 0
    device = next(gcn_model.parameters()).device

    all_latent_reps = []
    all_memberships = []
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        batch_count += 1
        batch_indices = batch['indices'].to(device)
        x = batch['x'].to(device)
        raw = batch['raw'].to(device)
        # Use pre-cached GPU tensors to skip per-batch N x N transfer
        adj = adj_gpu if adj_gpu is not None else batch['adj'].to(device)
        size_factor_matrix = sf_gpu if sf_gpu is not None else batch['size_factor_matrix'].to(device)

        optimizer.zero_grad()
        x_pred, latent_rep, adj_pred, mean, disp = gcn_model(x, adj, size_factor_matrix, batch_indices)

        batch_adj = adj[batch_indices][:, batch_indices]
        # edge tensors inherit device from batch_adj - no .to() needed
        edge_index, edge_weight = gcn_model.dense_to_sparse(batch_adj)

        softmax_out = clust_model(latent_rep)

        # latent_rep is already float32 on device; detach() suffices
        latent_tensor = latent_rep.detach()

        centroids = clust_model.get_centroids()
        distances = torch.cdist(latent_tensor, centroids, p=2)
        distances = torch.clamp(distances, min=1e-10)

        # Fuzzy membership, Eq. (21)
        m = fuzzifier_m
        power = -2 / (m - 1)
        distances_powered = distances ** power
        membership = distances_powered / torch.sum(distances_powered, dim=1, keepdim=True)

        global_mean = torch.mean(centroids, dim=0, keepdim=True)
        clust_mean = torch.mean(membership, dim=0)
        centroid_distances = torch.mean(torch.norm(centroids - global_mean, dim=1) ** 2)

        # WCSS / BCSS, Eqs. (25)-(26)
        wcss_loss = torch.mean(membership * (distances ** 2))
        bcss_loss = torch.sum(clust_mean * centroid_distances)

        all_latent_reps.append(latent_tensor)
        all_memberships.append(membership)

        # Distance-ratio regulariser, Eq. (24)
        dist_loss = wcss_loss / (bcss_loss + 1e-10)
        kl_loss = clust_criterion(softmax_out, membership.detach())

        # Total clustering objective, Eq. (27)
        batch_loss = clustering_weight * (dist_loss + kl_loss)

        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += batch_loss.item()

    full_latent_rep = torch.cat(all_latent_reps, dim=0)
    full_membership = torch.cat(all_memberships, dim=0)

    if do_update_centroids:
        clust_model.update_centroids(full_latent_rep, full_membership, epoch)

    return total_loss / batch_count, latent_tensor, membership, full_latent_rep, full_membership

