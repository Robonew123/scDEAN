"""
Reproducibility and checkpoint helpers.

    set_seed          - seed python / numpy / torch and enable deterministic kernels
    save_model_state  - torch.save wrapper
    load_model_state  - torch.load wrapper that falls back to a fresh model
"""

import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required for scatter_add / atomics used inside GCNConv to be deterministic on CUDA
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_model_state(model, path):
    """
    Save a model's state dictionary to a file

    Args:
        model: PyTorch model to save
        path: Path where to save the model state
    """
    try:
        torch.save(model.state_dict(), path)
        print(f"Model state saved to {path}")
    except Exception as e:
        print(f"Error saving model state: {str(e)}")


def load_model_state(model, path):
    """
    Load a saved state dictionary into a model

    Args:
        model: PyTorch model to load state into
        path: Path to the saved state dictionary

    Returns:
        The model with loaded state, or the original model if no saved state exists
    """
    if not os.path.exists(path):
        print(f"No saved model found at {path}. Starting with a fresh model.")
        return model

    try:
        model.load_state_dict(torch.load(path))
        print(f"Model state loaded from {path}")
        return model
    except Exception as e:
        print(f"Error loading model state: {str(e)}")
        print("Starting with a fresh model.")
        return model
