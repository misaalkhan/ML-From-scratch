"""k-nearest neighbours for classification (iris, wine) and regression (housing),
with optional inverse-distance weighting. Brute-force Euclidean search on scaled data."""

import numpy as np

from ml.datasets import load_housing, load_iris, load_wine
from ml.metrics import accuracy, mse, r2
from ml.preprocessing import standardize, train_test_split


def _pairwise(A, B):
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, clipped for numerical safety
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))


def classify(Xtr, ytr, Xte, k=5, weighted=False):
    D = _pairwise(Xte, Xtr)
    nn = np.argsort(D, axis=1)[:, :k]
    out = np.empty(len(Xte), int)
    for i, row in enumerate(nn):
        w = 1.0 / (D[i, row] + 1e-12) if weighted else None
        out[i] = np.bincount(ytr[row], weights=w, minlength=ytr.max() + 1).argmax()
    return out


def regress(Xtr, ytr, Xte, k=5, weighted=False):
    D = _pairwise(Xte, Xtr)
    nn = np.argsort(D, axis=1)[:, :k]
    out = np.empty(len(Xte))
    for i, row in enumerate(nn):
        if weighted:
            w = 1.0 / (D[i, row] + 1e-12)
            out[i] = (w * ytr[row]).sum() / w.sum()
        else:
            out[i] = ytr[row].mean()
    return out


if __name__ == "__main__":
    K = 5
    acc_by = {}
    for name, (X, y) in [("iris", load_iris()), ("wine", load_wine())]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
        Xtr_s, mu, sigma = standardize(Xtr)
        Xte_s, _, _ = standardize(Xte, mu, sigma)
        acc = accuracy(yte, classify(Xtr_s, ytr, Xte_s, K))
        accw = accuracy(yte, classify(Xtr_s, ytr, Xte_s, K, weighted=True))
        acc_by[name] = acc
        print(f"classify {name:<5} k={K}  accuracy {acc:.3f}  (distance-weighted {accw:.3f})")

    X, y = load_housing()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=0)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)
    pu = regress(Xtr_s, ytr, Xte_s, K)
    pw = regress(Xtr_s, ytr, Xte_s, K, weighted=True)
    print(f"regress  housing k={K}  RMSE {mse(yte, pu) ** 0.5:.3f} R2 {r2(yte, pu):.3f}"
          f"  (weighted RMSE {mse(yte, pw) ** 0.5:.3f} R2 {r2(yte, pw):.3f})")

    assert acc_by["iris"] > 0.9 and acc_by["wine"] > 0.9
    assert r2(yte, pw) > 0.6
    print("knn ok")
