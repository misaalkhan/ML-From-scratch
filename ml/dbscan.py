"""DBSCAN density-based clustering (Ester et al. 1996). Data: iris projected onto its
two principal components (via ml.pca) and standardized; reports clusters and noise count."""

import numpy as np

from ml.datasets import load_iris
from ml.metrics import purity
from ml.pca import pca
from ml.preprocessing import standardize


def _pairwise(X):
    d2 = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * X @ X.T
    return np.sqrt(np.maximum(d2, 0))


def dbscan(X, eps, min_samples):
    """Returns integer labels; -1 marks noise points."""
    D = _pairwise(X)
    n = len(X)
    labels = np.full(n, -1)
    visited = np.zeros(n, bool)
    c = -1
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = np.where(D[i] <= eps)[0]
        if len(neigh) < min_samples:
            continue  # provisional noise; may become a border point later
        c += 1
        labels[i] = c
        queue = list(neigh)
        qi = 0
        while qi < len(queue):
            j = queue[qi]
            qi += 1
            if not visited[j]:
                visited[j] = True
                nj = np.where(D[j] <= eps)[0]
                if len(nj) >= min_samples:
                    queue.extend(nj.tolist())
            if labels[j] == -1:
                labels[j] = c
    return labels


if __name__ == "__main__":
    X, y = load_iris()
    Xs, _, _ = standardize(X)
    scores, _, _, _ = pca(Xs, n_components=2)
    Z, _, _ = standardize(scores)

    labels = dbscan(Z, eps=0.4, min_samples=4)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    core = labels != -1
    pur = purity(labels[core], y[core])
    # per-cluster purity; DBSCAN isolates setosa cleanly but merges the overlapping species
    best_cluster = max(np.unique(labels[core]),
                       key=lambda c: np.bincount(y[labels == c]).max() / (labels == c).sum())
    setosa_purity = np.bincount(y[labels == best_cluster]).max() / (labels == best_cluster).sum()
    print(f"iris 2 PCs  eps 0.4 min_samples 4  clusters {n_clusters}  noise {n_noise}"
          f"  purity(non-noise) {pur:.3f}  purest cluster {setosa_purity:.3f}")

    assert n_clusters == 2 and n_noise < 25 and setosa_purity == 1.0
    print("dbscan ok")
