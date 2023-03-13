"""Gradient boosting with regression-tree base learners (Friedman 2001): least-squares
boosting (housing) and binary logistic-loss boosting fit to negative gradients (breast cancer)."""

import numpy as np

from ml.datasets import load_breast_cancer, load_housing
from ml.decision_tree import DecisionTree
from ml.metrics import accuracy, mse, r2, roc_auc
from ml.preprocessing import train_test_split


def fit_regression(X, y, n_estimators=100, lr=0.1, max_depth=3, seed=0):
    F0 = float(y.mean())
    F = np.full(len(y), F0)
    trees = []
    for _ in range(n_estimators):
        tree = DecisionTree("regression", max_depth=max_depth, min_samples_split=10, seed=seed).fit(X, y - F)
        F += lr * tree.predict(X)
        trees.append(tree)
    return F0, trees


def predict_regression(model, X, lr=0.1):
    F0, trees = model
    return F0 + lr * sum(t.predict(X) for t in trees)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_binary(X, y, n_estimators=100, lr=0.1, max_depth=3, seed=0):
    p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    F0 = float(np.log(p / (1 - p)))
    F = np.full(len(y), F0)
    trees = []
    for _ in range(n_estimators):
        residual = y - _sigmoid(F)  # negative gradient of the logistic loss
        tree = DecisionTree("regression", max_depth=max_depth, min_samples_split=10, seed=seed).fit(X, residual)
        F += lr * tree.predict(X)
        trees.append(tree)
    return F0, trees


def decision_function(model, X, lr=0.1):
    F0, trees = model
    return F0 + lr * sum(t.predict(X) for t in trees)


if __name__ == "__main__":
    LR = 0.1
    X, y = load_housing()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=0)
    model = fit_regression(Xtr, ytr, n_estimators=200, lr=LR)
    pred = predict_regression(model, Xte, LR)
    rmse, r2_reg = mse(yte, pred) ** 0.5, r2(yte, pred)
    print(f"regression     housing        200 trees  RMSE {rmse:.3f}  R2 {r2_reg:.3f}")

    X, y = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
    model = fit_binary(Xtr, ytr, n_estimators=100, lr=LR)
    score = decision_function(model, Xte, LR)
    acc = accuracy(yte, (score > 0).astype(int))
    auc = roc_auc(yte, _sigmoid(score))
    print(f"classification breast cancer   100 trees  accuracy {acc:.3f}  AUC {auc:.3f}")

    assert r2_reg > 0.8
    assert acc > 0.95 and auc > 0.98
    print("gradient_boosting ok")
