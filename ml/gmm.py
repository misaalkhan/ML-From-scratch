"""Gaussian mixture model fitted by expectation-maximisation with full covariances
(Bishop PRML sec. 9.2). The data log-likelihood is monotone non-decreasing across EM
iterations, which the demo asserts. Data: iris."""

import numpy as np

from ml.datasets import load_iris
from ml.kmeans import kmeans
from ml.metrics import purity
from ml.preprocessing import standardize


def _covariance(Xj, reg, d):
    diff = Xj - Xj.mean(0)
    return diff.T @ diff / len(Xj) + reg * np.eye(d)


def _log_gaussian(X, mu, cov):
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    diff = np.linalg.solve(L, (X - mu).T).T  # whiten with the Cholesky factor
    log_det = 2 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + log_det + (diff ** 2).sum(1))


def fit(X, k, max_iter=100, tol=1e-6, reg=1e-6, seed=0):
    n, d = X.shape
    _, labels, _ = kmeans(X, k, n_init=5, seed=seed)  # warm start EM from k-means
    mu = np.array([X[labels == j].mean(0) for j in range(k)])
    cov = np.array([_covariance(X[labels == j], reg, d) for j in range(k)])
    pi = np.bincount(labels, minlength=k) / n
    lls = []
    for _ in range(max_iter):
        logp = np.array([np.log(pi[j]) + _log_gaussian(X, mu[j], cov[j]) for j in range(k)]).T
        m = logp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logp - m).sum(1))  # log-sum-exp per point
        lls.append(float(lse.sum()))
        R = np.exp(logp - lse[:, None])  # responsibilities
        Nk = R.sum(0)
        mu = (R.T @ X) / Nk[:, None]
        for j in range(k):
            diff = X - mu[j]
            cov[j] = (R[:, j, None] * diff).T @ diff / Nk[j] + reg * np.eye(d)
        pi = Nk / n
        if len(lls) > 1 and lls[-1] - lls[-2] < tol:
            break
    return pi, mu, cov, R.argmax(1), lls


if __name__ == "__main__":
    X, y = load_iris()
    Xs, _, _ = standardize(X)
    pi, mu, cov, labels, lls = fit(Xs, k=3, seed=0)
    print(f"iris GMM k=3  iterations {len(lls)}  final log-likelihood {lls[-1]:.2f}  purity {purity(labels, y):.3f}")
    print(f"log-likelihood path: {' '.join(f'{v:.1f}' for v in lls[:6])} ...")

    diffs = np.diff(lls)
    assert (diffs >= -1e-6).all()  # EM never decreases the log-likelihood
    assert purity(labels, y) > 0.9
    print("gmm ok")
