"""Rosenblatt's perceptron with the pocket variant (Gallant 1990), which retains
the best-on-training weights for non-separable data. Data: banknote authentication."""

import numpy as np

from ml.datasets import load_banknote
from ml.metrics import accuracy
from ml.preprocessing import standardize, train_test_split


def _errors(X, y, w, b):
    return int(np.sum(np.sign(X @ w + b) != y))


def train(X, y, epochs=50, pocket=True, seed=0):
    """y in {-1, +1}. Returns pocket weights (fewest training errors seen)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    best_w, best_b, best_err = w.copy(), b, _errors(X, y, w, b)
    for _ in range(epochs):
        for i in rng.permutation(n):
            if y[i] * (X[i] @ w + b) <= 0:
                w += y[i] * X[i]
                b += y[i]
                if pocket:
                    err = _errors(X, y, w, b)
                    if err < best_err:
                        best_w, best_b, best_err = w.copy(), b, err
    return (best_w, best_b) if pocket else (w, b)


def predict(X, w, b):
    return np.sign(X @ w + b)


if __name__ == "__main__":
    X, y = load_banknote()
    y = np.where(y == 1, 1, -1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)

    w, b = train(Xtr_s, ytr, epochs=50, pocket=True)
    tr = accuracy(ytr, predict(Xtr_s, w, b))
    te = accuracy(yte, predict(Xte_s, w, b))
    print(f"pocket perceptron  train accuracy {tr:.3f}  test accuracy {te:.3f}")

    w0, b0 = train(Xtr_s, ytr, epochs=1, pocket=False)
    print(f"vanilla (1 epoch)  train accuracy {accuracy(ytr, predict(Xtr_s, w0, b0)):.3f}")

    assert te > 0.95
    print("perceptron ok")
