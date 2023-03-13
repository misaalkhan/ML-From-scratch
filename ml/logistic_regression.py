"""L2-regularized logistic regression by gradient descent: binary (breast cancer)
and multinomial softmax (wine). Bishop PRML sec. 4.3."""

import numpy as np

from ml.datasets import load_breast_cancer, load_wine
from ml.metrics import accuracy, roc_auc
from ml.preprocessing import one_hot, standardize, train_test_split


def _add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_binary(X, y, lam=1e-2, lr=0.5, epochs=3000):
    Xb = _add_bias(X)
    n = len(y)
    w = np.zeros(Xb.shape[1])
    pen = np.ones_like(w)
    pen[0] = 0.0  # do not regularize the intercept
    for _ in range(epochs):
        grad = Xb.T @ (_sigmoid(Xb @ w) - y) / n + lam * pen * w
        w -= lr * grad
    return w


def predict_proba(X, w):
    return _sigmoid(_add_bias(X) @ w)


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_softmax(X, y, k, lam=1e-2, lr=0.5, epochs=3000):
    Xb = _add_bias(X)
    n = len(y)
    Y = one_hot(y, k)
    W = np.zeros((Xb.shape[1], k))
    pen = np.ones((Xb.shape[1], 1))
    pen[0] = 0.0
    for _ in range(epochs):
        grad = Xb.T @ (_softmax(Xb @ W) - Y) / n + lam * pen * W
        W -= lr * grad
    return W


def predict_softmax(X, W):
    return np.argmax(_add_bias(X) @ W, axis=1)


if __name__ == "__main__":
    X, y = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)
    w = fit_binary(Xtr_s, ytr)
    proba = predict_proba(Xte_s, w)
    acc = accuracy(yte, (proba >= 0.5).astype(int))
    auc = roc_auc(yte, proba)
    print(f"binary  breast cancer  test accuracy {acc:.3f}  AUC {auc:.3f}")

    Xw, yw = load_wine()
    Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.25, seed=0, stratify=True)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)
    W = fit_softmax(Xtr_s, ytr, k=3)
    acc_w = accuracy(yte, predict_softmax(Xte_s, W))
    print(f"softmax wine           test accuracy {acc_w:.3f}")

    assert acc > 0.95 and auc > 0.98
    assert acc_w > 0.94
    print("logistic_regression ok")
