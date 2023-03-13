"""Exact t-SNE (van der Maaten & Hinton 2008): per-point perplexity calibration for the
high-dimensional affinities, a Student-t kernel in 2-D, and momentum gradient descent.
Data: 500 MNIST digits (PCA-reduced to 30). Quality is a k-NN label purity in the embedding."""

import numpy as np

from ml.datasets import load_mnist


def _joint_p(X, perplexity, tol=1e-5, max_iter=50):
    # binary-search a per-point Gaussian bandwidth to hit the target perplexity
    n = len(X)
    D = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * X @ X.T
    P = np.zeros((n, n))
    logU = np.log(perplexity)
    for i in range(n):
        beta, lo, hi = 1.0, -np.inf, np.inf
        Di = np.delete(D[i], i)
        for _ in range(max_iter):
            Pi = np.exp(-Di * beta)
            sumP = Pi.sum() + 1e-12
            H = np.log(sumP) + beta * (Di * Pi).sum() / sumP  # Shannon entropy of P_i
            if abs(H - logU) < tol:
                break
            if H > logU:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        P[i, np.arange(n) != i] = Pi / sumP
    P = (P + P.T) / (2 * n)  # symmetrize into a joint distribution
    return np.maximum(P, 1e-12)


def tsne(X, dim=2, perplexity=30.0, n_iter=1000, lr=200.0, seed=0):
    rng = np.random.default_rng(seed)
    P = _joint_p(X, perplexity) * 4.0  # early exaggeration
    Y = rng.normal(0, 1e-4, (len(X), dim))
    velocity = np.zeros_like(Y)
    for it in range(n_iter):
        d2 = (Y ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * Y @ Y.T
        num = 1.0 / (1.0 + d2)  # Student-t kernel, one degree of freedom
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        L = (P - Q) * num
        grad = 4.0 * ((np.diag(L.sum(1)) - L) @ Y)
        momentum = 0.5 if it < 250 else 0.8
        velocity = momentum * velocity - lr * grad
        Y += velocity
        Y -= Y.mean(0)
        if it == 100:
            P /= 4.0  # stop exaggeration
    return Y


def knn_purity(Y, labels, k=10):
    D = (Y ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * Y @ Y.T
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    return float(np.mean(labels[nn] == labels[:, None]))


if __name__ == "__main__":
    N, PERP = 500, 30.0
    X, y = load_mnist(N)
    X = X / 255.0
    Xc = X - X.mean(0)
    # SVD is used here only as a 784 -> 30 denoising tool; PCA from first principles is in ml.pca
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ Vt[:30].T
    Y = tsne(Xp, perplexity=PERP, n_iter=1000, seed=0)
    purity = knn_purity(Y, y, k=10)
    print(f"MNIST {N} digits (subsample)  perplexity {PERP:.0f}  10-NN embedding purity {purity:.3f}  (chance ~0.10)")

    assert purity > 0.6
    print("tsne ok")
