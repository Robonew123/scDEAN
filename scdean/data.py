"""
Data loading, preprocessing and batching utilities for scDEAN.

Contains:
    load_dataset             - read a .h5 or .csv dataset from disk
    load_data                - preprocessing (filtering, normalisation, HVG selection)
    GCNDataset               - Dataset returning per-cell features + full N x N matrices
    GCNFineTuningDataset     - Dataset variant carrying labels / latent representations
    collate_fn               - collate that keeps the square matrices unbatched
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset

import h5py


def load_dataset(base_path, dataset_name, ext):
    if ext == '.h5':
        file_path = os.path.join(base_path, dataset_name + '.h5')
        with h5py.File(file_path, 'r') as f:
            data = pd.DataFrame(f['X'][:])
            true_label = f['obs/Group'][:]
            unique_values, true_label = np.unique(true_label, return_inverse=True)
            n_cluster = len(unique_values)
    else:
        data_path = os.path.join(base_path, dataset_name, 'data.csv')
        label_path = os.path.join(base_path, dataset_name, 'label.csv')
        data = pd.read_csv(data_path, index_col=0)
        true_label = np.genfromtxt(label_path, delimiter=',', skip_header=1, dtype=str)[:, 1:].reshape(-1)
        unique_values, true_label = np.unique(true_label, return_inverse=True)
        n_cluster = len(unique_values)

    return data, true_label, n_cluster


def load_data(adata, n_top_genes=300):
    adata.X = np.nan_to_num(adata.X)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    adata.raw = adata.copy()

    sc.pp.normalize_total(adata)
    adata.obs['n_counts'] = adata.raw.X.sum(axis=1).flatten()
    adata.obs['size_factors'] = adata.obs['n_counts'] / np.median(adata.obs['n_counts'])
    sc.pp.log1p(adata)

    gene_stds = np.std(adata.X, axis=0)
    top_genes_idx = np.argsort(gene_stds)[::-1][:n_top_genes]

    highly_variable = np.zeros(adata.var.shape[0], dtype=bool)
    highly_variable[top_genes_idx] = True

    adata.var['highly_variable'] = highly_variable
    adata.raw.var['highly_variable'] = highly_variable

    adata = adata[:, adata.var['highly_variable']]
    rawData = adata.raw[:, adata.raw.var['highly_variable']].X

    return adata.X, rawData


# Define a custom dataset class for our data
class GCNDataset(Dataset):
    def __init__(self, x, raw, adj, size_factor_matrix):
        """
        Custom dataset for GCN training with batching

        Args:
            x: Feature matrix
            adj: Adjacency matrix (square)
            size_factor_matrix: Size factor matrix (square)
        """
        self.x = x
        self.raw = raw
        self.adj = adj
        self.size_factor_matrix = size_factor_matrix
        self.num_samples = x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # For batching, we need to return the full matrices but with a batch indicator
        # Since adj and size_factor_matrix are square matrices, we can't simply slice them
        # We'll handle the batching in the collate function
        return {
            'idx': idx,
            'x': self.x[idx],
            'raw': self.raw[idx],  # Raw data for reconstruction
            'adj': self.adj,  # Full adjacency matrix
            'size_factor_matrix': self.size_factor_matrix  # Full size factor matrix
        }


# Custom dataset for fine-tuning with labels
class GCNFineTuningDataset(Dataset):
    def __init__(self, x, adj, size_factor_matrix, labels=None, latent_representations=None):
        """
        Custom dataset for GCN fine-tuning with batching

        Args:
            x: Feature matrix
            adj: Adjacency matrix (square)
            size_factor_matrix: Size factor matrix (square)
            labels: Class labels for supervised fine-tuning
            latent_representations: Pre-computed latent representations (optional)
        """
        self.x = x
        self.adj = adj
        self.size_factor_matrix = size_factor_matrix
        self.labels = labels
        self.latent_representations = latent_representations
        self.num_samples = x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = {
            'idx': idx,
            'x': self.x[idx],
            'adj': self.adj,
            'size_factor_matrix': self.size_factor_matrix
        }

        if self.labels is not None:
            item['label'] = self.labels[idx]

        if self.latent_representations is not None:
            item['latent'] = self.latent_representations[idx]

        return item


# Custom collate function to handle batching with square matrices
def collate_fn(batch):
    indices = [item['idx'] for item in batch]
    x = torch.stack([item['x'] for item in batch])
    raw = torch.stack([item['raw'] for item in batch])

    # All items have the same adj and size_factor_matrix (full matrices)
    adj = batch[0]['adj']
    size_factor_matrix = batch[0]['size_factor_matrix']

    # Create batch masks for the square matrices
    batch_size = len(indices)
    batch_indices = torch.tensor(indices)

    result = {
        'indices': batch_indices,
        'x': x,
        'raw': raw,
        'adj': adj,
        'size_factor_matrix': size_factor_matrix
    }

    # Add labels if they exist
    if 'label' in batch[0]:
        labels = torch.stack([item['label'] for item in batch])
        result['labels'] = labels

    # Add latent representations if they exist
    if 'latent' in batch[0]:
        latent = torch.stack([item['latent'] for item in batch])
        result['latent'] = latent

    return result
