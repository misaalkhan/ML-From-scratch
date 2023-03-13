"""Two SVM solvers: Pegasos primal sub-gradient descent for the linear case
(Shalev-Shwartz et al. 2007) on banknote, and simplified SMO for the RBF-kernel
dual (Platt 1998; CS229 notes) on sonar."""

import numpy as np

from ml.datasets import load_banknote, load_sonar
from ml.metrics import accuracy
from ml.preprocessing import standardize, train_test_split


def pegasos(X, y, lam=1e-4, epochs=20, seed=0):
    """y in {-1, +1}; bias folded in as a constant feature. Returns weight vector."""
    rng = np.random.default_rng(seed)
    Xb = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(Xb.shape[1])
    t = 0
    for _ in range(epochs):
        for i in rng.permutation(len(y)):
            t += 1
            eta = 1.0 / (lam * t)
            if y[i] * (Xb[i] @ w) < 1:
                w = (1 - eta * lam) * w + eta * y[i] * Xb[i]
            else:
                w = (1 - eta * lam) * w
    return w


def pegasos_predict(X, w):
    return np.sign(np.column_stack([X, np.ones(len(X))]) @ w)


def rbf_kernel(A, B, gamma):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0))


def smo(K, y, C=1.0, tol=1e-3, max_passes=10, seed=0):
    """Simplified SMO on the dual with precomputed kernel K. Returns (alpha, b)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    passes = 0
    while passes < max_passes:
        changed = 0
        for i in range(n):
            Ei = (alpha * y) @ K[i] + b - y[i]
            if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
                j = rng.integers(n - 1)
                j += j >= i  # uniform over indices != i
                Ej = (alpha * y) @ K[j] + b - y[j]
                ai, aj = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L, H = max(0, aj - ai), min(C, C + aj - ai)
                else:
                    L, H = max(0, ai + aj - C), min(C, ai + aj)
                if L == H:
                    continue
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                alpha[j] = np.clip(aj - y[j] * (Ei - Ej) / eta, L, H)
                if abs(alpha[j] - aj) < 1e-5:
                    continue
                alpha[i] = ai + y[i] * y[j] * (aj - alpha[j])
                b1 = b - Ei - y[i] * (alpha[i] - ai) * K[i, i] - y[j] * (alpha[j] - aj) * K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - ai) * K[i, j] - y[j] * (alpha[j] - aj) * K[j, j]
                b = b1 if 0 < alpha[i] < C else b2 if 0 < alpha[j] < C else (b1 + b2) / 2
                changed += 1
        passes = passes + 1 if changed == 0 else 0
    return alpha, b


if __name__ == "__main__":
    X, y = load_banknote()
    y = np.where(y == 1, 1, -1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)
    w = pegasos(Xtr_s, ytr, lam=1e-3, epochs=50)
    acc_lin = accuracy(yte, pegasos_predict(Xte_s, w))
    print(f"Pegasos linear   banknote (n={len(ytr)})   test accuracy {acc_lin:.3f}")

    X, y = load_sonar()
    y = np.where(y == 1, 1, -1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, seed=0, stratify=True)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)
    gamma = 1.0 / Xtr_s.shape[1]
    K = rbf_kernel(Xtr_s, Xtr_s, gamma)
    alpha, b = smo(K, ytr, C=1.0, max_passes=10)
    Kte = rbf_kernel(Xte_s, Xtr_s, gamma)
    pred = np.sign(Kte @ (alpha * ytr) + b)
    acc_rbf = accuracy(yte, pred)
    print(f"SMO RBF kernel   sonar (n={len(ytr)})    test accuracy {acc_rbf:.3f}  support vectors {(alpha > 1e-6).sum()}")

    assert acc_lin > 0.98
    assert acc_rbf > 0.8
    print("svm ok")
