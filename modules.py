"""
Backwards-compatibility shim.

The original single-file `modules.py` has been split into the `scdean` package
(see scdean/data.py, layers.py, models.py, losses.py, train.py, utils.py).

This module re-exports every public name under its original spelling so that
existing scripts and notebooks written against the old layout - for example

    from modules import GCNDataset, collate_fn, GCNLatentRepresentation

- continue to work without modification.

Note on removed code
--------------------
The old `modules.py` also contained a second `fine_tune_model` that referenced
an undefined `zinb_loss_val` (a leftover from the ZINB variant, which was
commented out). It raised NameError if called and was superseded by the
implementation in `scdean/train.py`, so it is not re-exported here.
"""

from scdean.data import (load_dataset, load_data, GCNDataset,
                         GCNFineTuningDataset, collate_fn)
from scdean.layers import GCNlayer, MLP, StudentskernalAdjacency
from scdean.models import GCNLatentRepresentation, DeepClustering
from scdean.losses import (autoencoder_loss, adjacency_reconstruction_loss,
                           graph_regularization_loss, nb_loss, zinb_loss)
from scdean.train import (train_gcn_with_batching, fine_tune_model,
                          extract_latent_representations)
from scdean.metrics import clustering_accuracy
from scdean.utils import set_seed, save_model_state, load_model_state

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
