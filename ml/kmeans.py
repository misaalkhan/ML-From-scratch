"""Lloyd's k-means with k-means++ seeding (Arthur & Vassilvitskii 2007) and
best-of-n_init restarts. k chosen by silhouette on standardized wine; cluster quality
scored against true labels by purity and the adjusted Rand index (from ml.metrics)."""

import numpy as np

from ml.datasets import load_wine
from ml.metrics import adjusted_rand_index, purity, silhouette_score
from ml.preprocessing import standardize


def _pairwise_sq(X, C):
    return (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None] - 2 * X @ C.T


def kmeanspp_init(X, k, rng):
    n = len(X)
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        c = X[rng.choice(n, p=d2 / d2.sum())]  # sample proportional to D^2
        centers.append(c)
        d2 = np.minimum(d2, ((X - c) ** 2).sum(1))
    return np.array(centers)


def kmeans(X, k, n_init=10, max_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = kmeanspp_init(X, k, rng)
        for _ in range(max_iter):
            labels = _pairwise_sq(X, C).argmin(1)
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(np.maximum(_pairwise_sq(X, C).min(1), 0).sum())
        if best is None or inertia < best[2]:
            best = (C, labels, inertia)
    return best


if __name__ == "__main__":
    X, y = load_wine()
    Xs, _, _ = standardize(X)
    print("silhouette by k (standardized wine):")
    sils = {}
    for k in range(2, 7):
        _, labels, _ = kmeans(Xs, k, n_init=10, seed=0)
        sils[k] = silhouette_score(Xs, labels)
        print(f"  k={k}  silhouette {sils[k]:.3f}")
    k_star = max(sils, key=sils.get)

    C, labels, inertia = kmeans(Xs, k_star, n_init=10, seed=0)
    pur = purity(labels, y)
    ari = adjusted_rand_index(y, labels)
    print(f"chosen k={k_star}  inertia {inertia:.1f}  purity {pur:.3f}  adjusted Rand {ari:.3f}")

    assert k_star in (2, 3)
    _, labels3, _ = kmeans(Xs, 3, n_init=10, seed=0)
    assert purity(labels3, y) > 0.9 and adjusted_rand_index(y, labels3) > 0.8
    print("kmeans ok")
