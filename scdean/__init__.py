"""
scDEAN: A Lightweight Adaptive Fusion Model for Clustering scRNA-seq Data.

Public API
----------
data    : load_dataset, load_data, GCNDataset, GCNFineTuningDataset, collate_fn
layers  : GCNlayer, MLP, StudentskernalAdjacency
models  : GCNLatentRepresentation, DeepClustering
losses  : autoencoder_loss, adjacency_reconstruction_loss,
          graph_regularization_loss, nb_loss
train   : train_gcn_with_batching, fine_tune_model,
          extract_latent_representations
metrics : clustering_accuracy
utils   : set_seed, save_model_state, load_model_state
"""

from .data import (load_dataset, load_data, GCNDataset, GCNFineTuningDataset,
                   collate_fn)
from .layers import GCNlayer, MLP, StudentskernalAdjacency
from .models import GCNLatentRepresentation, DeepClustering
from .losses import (autoencoder_loss, adjacency_reconstruction_loss,
                     graph_regularization_loss, nb_loss, zinb_loss)
from .train import (train_gcn_with_batching, fine_tune_model,
                    extract_latent_representations)
from .metrics import clustering_accuracy
from .utils import set_seed, save_model_state, load_model_state

__version__ = "1.0.0"

__all__ = [
    "load_dataset", "load_data", "GCNDataset", "GCNFineTuningDataset", "collate_fn",
    "GCNlayer", "MLP", "StudentskernalAdjacency",
    "GCNLatentRepresentation", "DeepClustering",
    "autoencoder_loss", "adjacency_reconstruction_loss",
    "graph_regularization_loss", "nb_loss", "zinb_loss",
    "train_gcn_with_batching", "fine_tune_model", "extract_latent_representations",
    "clustering_accuracy",
    "set_seed", "save_model_state", "load_model_state",
]
