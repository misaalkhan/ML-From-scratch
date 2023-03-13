"""Ordinary least squares by the normal equation and by batch gradient descent.
Data: Boston housing; reports RMSE and R^2 for both solvers (ESL sec. 3.2)."""

import numpy as np

from ml.datasets import load_housing
from ml.metrics import mse, r2
from ml.preprocessing import standardize, train_test_split


def _add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def fit_normal_equation(X, y):
    Xb = _add_bias(X)
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)  # (X'X) w = X'y


def fit_gradient_descent(X, y, lr=0.1, epochs=5000):
    Xb = _add_bias(X)
    n = len(y)
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        w -= lr * (Xb.T @ (Xb @ w - y)) / n  # gradient of (1/2n)||Xw - y||^2
    return w


def predict(X, w):
    return _add_bias(X) @ w


if __name__ == "__main__":
    LR, EPOCHS = 0.1, 5000
    X, y = load_housing()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=0)

    w_ne = fit_normal_equation(Xtr, ytr)
    p_ne = predict(Xte, w_ne)

    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sigma)
    w_gd = fit_gradient_descent(Xtr_s, ytr, LR, EPOCHS)
    p_gd = predict(Xte_s, w_gd)

    rmse_ne, rmse_gd = mse(yte, p_ne) ** 0.5, mse(yte, p_gd) ** 0.5
    print(f"normal equation   test RMSE {rmse_ne:.3f}  R2 {r2(yte, p_ne):.3f}")
    print(f"gradient descent  test RMSE {rmse_gd:.3f}  R2 {r2(yte, p_gd):.3f}")

    assert r2(yte, p_ne) > 0.65 and r2(yte, p_gd) > 0.65
    assert abs(rmse_ne - rmse_gd) < 0.05  # both converge to the OLS optimum
    print("linear_regression ok")
