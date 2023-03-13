"""Biased matrix factorization trained by SGD (Koren et al. 2009):
r_ui ~ mu + b_u + b_i + p_u . q_i. Data: MovieLens 100k, held-out RMSE."""

import numpy as np

from ml.datasets import load_movielens


def fit(train, n_users, n_items, f=20, epochs=20, lr=0.005, reg=0.02, seed=0):
    rng = np.random.default_rng(seed)
    mu = train[:, 2].mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    P = rng.normal(0, 0.1, (n_users, f))
    Q = rng.normal(0, 0.1, (n_items, f))
    for _ in range(epochs):
        for idx in rng.permutation(len(train)):
            u, i, r = train[idx, 0], train[idx, 1], train[idx, 2]
            err = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            Pu = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * Pu - reg * Q[i])
    return mu, bu, bi, P, Q


def predict(model, pairs):
    mu, bu, bi, P, Q = model
    u, i = pairs[:, 0], pairs[:, 1]
    return mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1)


def rmse(model, data):
    return float(np.sqrt(np.mean((data[:, 2] - predict(model, data)) ** 2)))


if __name__ == "__main__":
    data = load_movielens()
    data[:, 0] -= 1  # to 0-based user/item indices
    data[:, 1] -= 1
    n_users, n_items = data[:, 0].max() + 1, data[:, 1].max() + 1

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(data))
    cut = int(0.8 * len(data))
    train, test = data[perm[:cut]], data[perm[cut:]]

    model = fit(train, n_users, n_items, f=20, epochs=20, lr=0.005, reg=0.05)
    tr, te = rmse(model, train), rmse(model, test)
    print(f"MovieLens 100k  f=20 epochs=20  train RMSE {tr:.3f}  test RMSE {te:.3f}")

    assert te <= 1.00
    print("recommender ok")
