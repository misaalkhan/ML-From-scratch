"""Penalized least squares: ridge in closed form, lasso and elastic net by
cyclic coordinate descent (Friedman et al. 2010). Data: housing + degree-2 features."""

import numpy as np

from ml.datasets import load_housing
from ml.metrics import mse, r2
from ml.preprocessing import polynomial_features, standardize, train_test_split


def ridge(X, y, lam):
    """Closed form (X'X + lam I)^-1 X'y on centered data; intercept unpenalized."""
    Xc, yc = X - X.mean(0), y - y.mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, y.mean() - X.mean(0) @ w


def _soft_threshold(z, g):
    return np.sign(z) * max(abs(z) - g, 0.0)


def coordinate_descent(X, y, lam, alpha=1.0, iters=500, tol=1e-6):
    """Elastic net: (1/2n)||y-Xw||^2 + lam(alpha|w|_1 + (1-alpha)/2 |w|^2).
    alpha=1 is lasso; 0<alpha<1 mixes an L2 term. Assumes standardized X."""
    n, d = X.shape
    Xc, yc = X - X.mean(0), y - y.mean()
    col_ss = (Xc ** 2).sum(0) / n
    w = np.zeros(d)
    for _ in range(iters):
        w_old = w.copy()
        for j in range(d):
            rho = Xc[:, j] @ (yc - Xc @ w + Xc[:, j] * w[j]) / n
            w[j] = _soft_threshold(rho, lam * alpha) / (col_ss[j] + lam * (1 - alpha))
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w, y.mean() - X.mean(0) @ w


if __name__ == "__main__":
    X, y = load_housing()
    X = polynomial_features(X, degree=2)  # 13 -> 104 features to expose sparsity
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=0)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)

    w, b = ridge(Xtr_s, ytr, lam=10.0)
    pr = Xte_s @ w + b
    print(f"ridge    lam 10.0   test R2 {r2(yte, pr):.3f}  nonzero {np.sum(np.abs(w) > 1e-6)}/{len(w)}")

    print("lasso    lambda sweep (nonzero coefficients, test R2):")
    counts = []
    for lam in [0.01, 0.1, 0.5, 2.0]:
        w, b = coordinate_descent(Xtr_s, ytr, lam, alpha=1.0)
        nz = int(np.sum(np.abs(w) > 1e-6))
        counts.append(nz)
        print(f"         lam {lam:<5} nonzero {nz:>3}/{len(w)}   R2 {r2(yte, Xte_s @ w + b):.3f}")

    w_en, b_en = coordinate_descent(Xtr_s, ytr, lam=0.1, alpha=0.5)
    nz_en = int(np.sum(np.abs(w_en) > 1e-6))
    print(f"elastic  lam 0.1 a 0.5  nonzero {nz_en}/{len(w_en)}  R2 {r2(yte, Xte_s @ w_en + b_en):.3f}")

    assert r2(yte, pr) > 0.7
    assert counts[0] > counts[-1] and counts[-1] < len(w)  # heavier L1 -> sparser
    print("ridge_lasso ok")
