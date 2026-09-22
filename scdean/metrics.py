"""
Evaluation metrics.

    clustering_accuracy - ACC after optimal label matching (Hungarian algorithm)

NMI, ARI and the Silhouette Index are taken directly from scikit-learn in
main.py; only ACC requires a custom implementation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size
