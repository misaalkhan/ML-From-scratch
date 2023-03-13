"""Feature scaling, encoding, feature expansion, and index-level data splitting.
Scalers return fitted statistics so the same transform applies to held-out data."""

from itertools import combinations_with_replacement

import numpy as np

from ml.datasets import load_iris


def standardize(X, mu=None, sigma=None):
    """Zero mean, unit variance. Pass fitted (mu, sigma) to transform new data."""
    if mu is None:
        mu = X.mean(0)
    if sigma is None:
        sigma = X.std(0)
        sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def minmax_scale(X, lo=None, hi=None):
    if lo is None:
        lo = X.min(0)
    if hi is None:
        hi = X.max(0)
    span = np.where(hi == lo, 1.0, hi - lo)
    return (X - lo) / span, lo, hi


def one_hot(y, k=None):
    y = np.asarray(y, int)
    k = int(y.max()) + 1 if k is None else k
    M = np.zeros((len(y), k))
    M[np.arange(len(y)), y] = 1.0
    return M


def polynomial_features(X, degree=2, include_bias=False):
    """All monomials up to `degree`, ordered by degree then feature index (as sklearn)."""
    n, d = X.shape
    cols = [np.ones(n)] if include_bias else []
    for p in range(1, degree + 1):
        for idx in combinations_with_replacement(range(d), p):
            cols.append(np.prod(X[:, idx], axis=1))
    return np.column_stack(cols)


def train_test_split(X, y, test_size=0.25, seed=0, stratify=False):
    rng = np.random.default_rng(seed)
    n = len(y)
    if stratify:
        test = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            test.extend(idx[: int(round(test_size * len(idx)))].tolist())
        test = np.array(sorted(test))
    else:
        idx = rng.permutation(n)
        test = np.sort(idx[: int(round(test_size * n))])
    mask = np.zeros(n, bool)
    mask[test] = True
    return X[~mask], X[mask], y[~mask], y[mask]


def kfold_indices(n, k=5, seed=0, stratify=None):
    """Return k (train_idx, test_idx) pairs; pass labels as `stratify` for balanced folds."""
    rng = np.random.default_rng(seed)
    fold = np.empty(n, int)
    if stratify is not None:
        for c in np.unique(stratify):
            idx = np.where(stratify == c)[0]
            rng.shuffle(idx)
            fold[idx] = np.arange(len(idx)) % k  # round-robin keeps class counts even
    else:
        perm = rng.permutation(n)
        fold[perm] = np.arange(n) % k
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


if __name__ == "__main__":
    X, y = load_iris()
    Xs, mu, sigma = standardize(X)
    print(f"standardize   mean {np.abs(Xs.mean(0)).max():.1e}  std {Xs.std(0).mean():.4f}")

    Xm, _, _ = minmax_scale(X)
    print(f"minmax        min {Xm.min():.3f}  max {Xm.max():.3f}")

    H = one_hot(y)
    print(f"one_hot       shape {H.shape}  row sums unique {np.unique(H.sum(1))}")

    P = polynomial_features(X, degree=2)
    print(f"poly deg2     {X.shape[1]} -> {P.shape[1]} features")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=True)
    print(f"split strat   train {Xtr.shape[0]}  test {Xte.shape[0]}  test balance {np.bincount(yte).tolist()}")

    folds = kfold_indices(len(y), k=5, stratify=y)
    sizes = [len(te) for _, te in folds]
    print(f"kfold         sizes {sizes}  fold0 balance {np.bincount(y[folds[0][1]]).tolist()}")

    assert np.abs(Xs.mean(0)).max() < 1e-12 and abs(Xs.std(0).mean() - 1) < 1e-12
    assert 0.0 <= Xm.min() and Xm.max() <= 1.0
    assert np.array_equal(np.unique(H.sum(1)), [1.0]) and H.shape == (150, 3)
    assert P.shape[1] == 14  # 4 linear + 4 squares + 6 cross terms
    assert Xtr.shape[0] + Xte.shape[0] == 150 and np.bincount(yte).tolist() == [10, 10, 10]
    assert sum(sizes) == 150 and np.bincount(y[folds[0][1]]).tolist() == [10, 10, 10]
    print("preprocessing ok")
