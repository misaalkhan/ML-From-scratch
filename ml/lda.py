"""Fisher's linear discriminant analysis: the directions maximising between- over
within-class scatter (ESL sec. 4.3). Solves the generalized eigenproblem Sb w = lambda Sw w
by symmetric whitening with the from-scratch eigensolver (ml.linalg.symmetric_eig).
Projects wine 13 -> 2, then classifies by nearest centroid in the discriminant space."""

import numpy as np

from ml.datasets import load_wine
from ml.linalg import symmetric_eig
from ml.metrics import accuracy
from ml.preprocessing import standardize, train_test_split


def fit(X, y, n_components):
    classes = np.unique(y)
    d = X.shape[1]
    mean = X.mean(0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(0)
        Sw += (Xc - mc).T @ (Xc - mc)
        diff = (mc - mean)[:, None]
        Sb += len(Xc) * diff @ diff.T
    # whiten with Sw^{-1/2} so the generalized problem becomes a symmetric one
    wv, U = symmetric_eig(Sw)
    Sw_inv_half = U @ np.diag(np.maximum(wv, 1e-12) ** -0.5) @ U.T
    _, evec = symmetric_eig(Sw_inv_half @ Sb @ Sw_inv_half)
    return Sw_inv_half @ evec[:, :n_components]  # map discriminant directions back


def transform(X, W):
    return X @ W


if __name__ == "__main__":
    X, y = load_wine()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)

    W = fit(Xtr_s, ytr, n_components=2)
    Ztr, Zte = transform(Xtr_s, W), transform(Xte_s, W)
    centroids = np.array([Ztr[ytr == c].mean(0) for c in np.unique(ytr)])
    d2 = ((Zte[:, None] - centroids[None]) ** 2).sum(2)
    pred = d2.argmin(1)
    acc = accuracy(yte, pred)
    print(f"wine 13 -> 2  nearest-centroid accuracy in LDA space {acc:.3f}")

    assert acc > 0.94
    print("lda ok")
