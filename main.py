#!/usr/bin/env python3
"""
scDEAN end-to-end pipeline.

    Phase 0  Graph Construction        - UMAP fuzzy simplicial set
    Phase 1  Representation Learning   - dual encoder, 200 epochs
    Phase 2  Deep Clustering           - frozen encoder, 100 epochs

Example
-------
    python main.py --dataset_name Camp --data_type .h5 \
                   --base_path ./data --batch_size 256
"""

import os
import time
import argparse

import numpy as np
import anndata as ad
import torch
import umap
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

import resource  # peak CPU RAM tracking (Linux)

from scdean.data import load_dataset, load_data, GCNDataset, collate_fn
from scdean.models import GCNLatentRepresentation, DeepClustering
from scdean.train import (train_gcn_with_batching, fine_tune_model,
                          extract_latent_representations)
from scdean.metrics import clustering_accuracy
from scdean.utils import set_seed, load_model_state, save_model_state


def main():

    args = get_args()
    set_seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine Base Path
    base_path = args.base_path
    if base_path is None:
        base_path = "./data"

    # Load and Preprocess Data
    data, true_label, n_cluster = load_dataset(base_path, args.dataset_name, args.data_type)
    adata = ad.AnnData(data)
    print(f"Loaded adata shape: {adata.shape}")

    ndata, rawdata = load_data(adata, n_top_genes=args.n_top_genes)

    # UMAP graph processing
    umap_model = umap.UMAP(n_neighbors=args.k_neighbors, min_dist=args.min_dist,
                           metric='correlation', random_state=args.seed)
    umap_model.fit(ndata)
    dense_graph = umap_model.graph_.todense()
    adj1 = torch.tensor(dense_graph, dtype=torch.float32)

    size_factors = np.array(adata.obs['size_factors'].values)
    size_factor_matrix = torch.diag(torch.tensor(size_factors, dtype=torch.float32))

    # Dataset & Dataloader
    x_tensor = torch.tensor(ndata, dtype=torch.float)
    raw_tensor = torch.tensor(rawdata, dtype=torch.float).clone()

    # Dataset keeps CPU tensors (preserves original shuffle/ordering behaviour)
    dataset = GCNDataset(x_tensor, raw_tensor, adj1, size_factor_matrix)

    # Pre-cache the shared N x N matrices on GPU once - avoids re-transferring them every batch
    adj_gpu = adj1.to(device)
    sf_gpu = size_factor_matrix.to(device)

    # Define Worker Init Fn for Dataloader reproduction
    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
        worker_init_fn=worker_init_fn
    )

    in_features = x_tensor.shape[1]
    num_samples = x_tensor.shape[0]

    # ====================================================
    # 1. START PERFORMANCE TRACKING
    # ====================================================
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Start VRAM tracker at 0
    start_time_total = time.time()
    # ====================================================

    # Initialize GCN
    gcn_model = GCNLatentRepresentation(in_features, args.hidden1, args.hidden2,
                                        args.latent_dim, num_samples,
                                        use_zinb=(args.recon_loss == 'zinb')).to(device)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=args.lr_pretrain)

    model_path = os.path.join(args.model_dir, f'gcn_model_final_f{args.dataset_name}.pt')

    if os.path.exists(model_path):
        print("Loading existing model...")
        gcn_model = load_model_state(gcn_model, model_path)
    else:
        print("No existing model found. Training a new one...")
        start_time = time.time()
        for epoch in range(args.epochs):
            loss = train_gcn_with_batching(gcn_model, dataloader, optimizer,
                                           alpha=args.alpha, beta=args.beta,
                                           gamma=args.gamma, lambda_param=args.lambda_param,
                                           adj_gpu=adj_gpu, sf_gpu=sf_gpu,
                                           impu_loss_type=args.recon_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

        if args.save_pt:
            save_model_state(gcn_model, model_path)

    # KMeans Initialization
    print("Extracting latent representations & initializing k-means...")
    latent_representations = extract_latent_representations(gcn_model, dataloader, device,
                                                            adj_gpu=adj_gpu)

    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', n_init=200, max_iter=1000,
                    tol=0.0001, random_state=args.seed, algorithm='lloyd')
    kmeans.fit(latent_representations.cpu().numpy())
    initial_centroids = kmeans.cluster_centers_

    # Deep Clustering Fine-Tuning Setup
    clust_model = DeepClustering(args.latent_dim, n_cluster, initial_centroids=initial_centroids).to(device)
    clust_optimizer = torch.optim.Adam(list(clust_model.parameters()), lr=args.lr_finetune, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(clust_optimizer, step_size=30, gamma=0.5)
    clust_criterion = torch.nn.CrossEntropyLoss()

    print("Fine-tuning clustering model...")
    for epoch in range(args.fine_tune_epochs):
        loss, latent_rep, membership, full_latent, full_mem = fine_tune_model(
            gcn_model, clust_model, dataloader, clust_optimizer, scheduler, clust_criterion,
            n_cluster, alpha=args.alpha, beta=args.beta, gamma=args.gamma, lambda_param=args.lambda_param,
            clustering_weight=args.clustering_weight, epoch=epoch, fuzzifier_m=args.m,
            do_update_centroids=args.update_centroids, adj_gpu=adj_gpu, sf_gpu=sf_gpu
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.fine_tune_epochs}, Loss: {loss:.6f}")

    # Final Evaluation
    gcn_model.eval()
    clust_model.eval()

    with torch.no_grad():
        final_latent = extract_latent_representations(gcn_model, dataloader, device,
                                                      adj_gpu=adj_gpu)
        cluster_probs = clust_model(final_latent)
        _, predicted_clusters = torch.max(cluster_probs, dim=1)
        predicted_labels = predicted_clusters.cpu().numpy()

    # ====================================================
    # 2. END PERFORMANCE TRACKING & PRINT
    # ====================================================
    total_runtime = time.time() - start_time_total

    # Label-based Metrics
    acc = clustering_accuracy(true_label, predicted_labels)
    nmi = normalized_mutual_info_score(true_label, predicted_labels)
    ari = adjusted_rand_score(true_label, predicted_labels)
    sil_score = silhouette_score(final_latent.cpu().numpy(), predicted_labels)

    print(f"\nFinal Clustering Performance:")
    print(f"Accuracy (ACC): {acc:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Silhouette Score (SIL): {sil_score:.4f}")

    unique_clusters, counts = np.unique(predicted_labels, return_counts=True)
    print(f"\nPredicted Unique clusters: {unique_clusters}")
    print(f"Predicted Counts: {counts}")

    unique_clusters_true, counts_true = np.unique(true_label, return_counts=True)
    print(f"True Unique clusters: {unique_clusters_true}")
    print(f"True Counts: {counts_true}")

    # Extract Peak GPU VRAM
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Extract Peak CPU RAM (ru_maxrss is usually in KB on Linux)
    peak_cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"\n=======================================")
    print(f"--- HARDWARE EFFICIENCY BENCHMARK ---")
    print(f"=======================================")
    print(f"Total Pipeline Runtime : {total_runtime:.2f} s")
    print(f"Peak GPU VRAM (MB)     : {peak_vram_mb:.2f}")
    print(f"Peak CPU RAM (MB)      : {peak_cpu_mb:.2f}")
    print(f"=======================================")


def get_args():
    parser = argparse.ArgumentParser(description="scDEAN Training and Fine-Tuning Script")

    # Data arguments
    parser.add_argument('--dataset_name', type=str, default='Camp', help="Name of the dataset")
    parser.add_argument('--data_type', type=str, default='.h5', choices=['.csv', '.h5'], help="Data format (.csv or .h5)")
    parser.add_argument('--base_path', type=str, default=None, help="Base directory. Defaults to ./data")
    parser.add_argument('--n_top_genes', type=int, default=300, help="Number of highly variable genes")

    # Graph Construction (UMAP) Hyperparameters
    parser.add_argument('--k_neighbors', type=int, default=15, help="Number of neighbors (k) for UMAP graph construction")
    parser.add_argument('--min_dist', type=float, default=0.1, help="Minimum distance parameter for UMAP")

    # Model Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=32, help="Latent space dimension")
    parser.add_argument('--m', type=float, default=1.3, help="Fuzzifier value (m) for FCM membership")
    parser.add_argument('--hidden1', type=int, default=128, help="First hidden layer size")
    parser.add_argument('--hidden2', type=int, default=64, help="Second hidden layer size")

    # Ablation & Loss Weights
    parser.add_argument('--alpha', type=float, default=0.25, help="Weight for reconstruction loss")
    parser.add_argument('--beta', type=float, default=0.25, help="Weight for adjacency reconstruction loss")
    parser.add_argument('--gamma', type=float, default=0.25, help="Weight for graph regularization loss")
    parser.add_argument('--lambda_param', type=float, default=0.25, help="Weight for NB loss")
    parser.add_argument('--clustering_weight', type=float, default=1.0, help="Weight for clustering loss")
    parser.add_argument('--recon_loss', type=str, default='nb', choices=['nb', 'zinb'],
                        help="Count-based reconstruction regulariser. Use --lambda_param 0 for MSE only.")

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=200, help="Number of pre-training epochs")
    parser.add_argument('--fine_tune_epochs', type=int, default=100, help="Number of fine-tuning epochs")
    parser.add_argument('--lr_pretrain', type=float, default=0.001, help="Learning rate for pre-training")
    parser.add_argument('--lr_finetune', type=float, default=0.005, help="Learning rate for fine-tuning")

    # Storage and execution
    parser.add_argument('--save_pt', action='store_true', help="Flag to save model parameters as .pt files")
    parser.add_argument('--update_centroids', action='store_true', help="Enable updating centroids dynamically during fine-tuning (default: off)")
    parser.add_argument('--model_dir', type=str, default='./', help="Directory to save/load models")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for multiple initialization robustness tests")

    return parser.parse_args()


if __name__ == "__main__":
    main()

