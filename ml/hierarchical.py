"""Agglomerative hierarchical clustering with single, complete, and average linkage,
cut to k clusters. Data: a balanced wine subset; reports purity per linkage."""

import numpy as np

from ml.datasets import load_wine
from ml.metrics import purity
from ml.preprocessing import standardize


def _pairwise(X):
    d2 = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * X @ X.T
    return np.sqrt(np.maximum(d2, 0))


def agglomerative(X, k, linkage="average"):
    """Naive O(n^3) merging; adequate for the small subset used here."""
    D = _pairwise(X)
    clusters = [[i] for i in range(len(X))]
    reduce = {"single": np.min, "complete": np.max, "average": np.mean}[linkage]
    while len(clusters) > k:
        best, pair = np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                dist = reduce(D[np.ix_(clusters[a], clusters[b])])
                if dist < best:
                    best, pair = dist, (a, b)
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = np.empty(len(X), int)
    for c, members in enumerate(clusters):
        labels[members] = c
    return labels


if __name__ == "__main__":
    X, y = load_wine()
    idx = np.concatenate([np.where(y == c)[0][:30] for c in np.unique(y)])  # 30 per class
    Xs, _, _ = standardize(X[idx])
    ys = y[idx]

    purities = {}
    for link in ("single", "complete", "average"):
        labels = agglomerative(Xs, k=3, linkage=link)
        purities[link] = purity(labels, ys)
        print(f"{link:<9} linkage  k=3  purity {purities[link]:.3f}")

    assert purities["complete"] > 0.80  # complete linkage resists the chaining that hurts single/average here
    print("hierarchical ok")
