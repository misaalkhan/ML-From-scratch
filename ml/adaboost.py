"""AdaBoost with decision stumps, binary SAMME (Freund & Schapire 1997; Zhu et al.
2009), which reduces to AdaBoost.M1 for two classes. Data: banknote authentication."""

import numpy as np

from ml.datasets import load_banknote
from ml.metrics import accuracy
from ml.preprocessing import train_test_split


def fit_stump(X, y, w):
    """Weighted-error-minimising stump. Returns (feature, threshold, polarity, error)."""
    n, d = X.shape
    best = (0, 0.0, 1, np.inf)
    for f in range(d):
        order = np.argsort(X[:, f], kind="mergesort")
        xs, ws, ys = X[order, f], w[order], y[order]
        pos_left = np.cumsum(ws * (ys == 1))
        neg_left = np.cumsum(ws * (ys == -1))
        tot_pos, tot_neg = pos_left[-1], neg_left[-1]
        err_pos = pos_left + (tot_neg - neg_left)  # predict -1 left, +1 right
        err_neg = neg_left + (tot_pos - pos_left)  # opposite polarity
        for err, pol in ((err_pos, 1), (err_neg, -1)):
            e = err.copy()
            e[:-1] = np.where(xs[:-1] != xs[1:], e[:-1], np.inf)
            e[-1] = np.inf
            i = int(e.argmin())
            if e[i] < best[3]:
                best = (f, 0.5 * (xs[i] + xs[i + 1]), pol, float(e[i]))
    return best


def _stump_predict(X, f, thr, pol):
    return np.where(X[:, f] > thr, pol, -pol)


def fit(X, y, T=50):
    """y in {-1, +1}. Returns list of (feature, threshold, polarity, alpha)."""
    n = len(y)
    w = np.full(n, 1.0 / n)
    model = []
    for _ in range(T):
        f, thr, pol, err = fit_stump(X, y, w)
        err = min(max(err, 1e-10), 1 - 1e-10)
        alpha = 0.5 * np.log((1 - err) / err)
        pred = _stump_predict(X, f, thr, pol)
        w *= np.exp(-alpha * y * pred)
        w /= w.sum()
        model.append((f, thr, pol, alpha))
    return model


def predict(model, X):
    agg = sum(a * _stump_predict(X, f, thr, pol) for f, thr, pol, a in model)
    return np.sign(agg)


if __name__ == "__main__":
    X, y = load_banknote()
    y = np.where(y == 1, 1, -1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)

    for T in (5, 20, 50):
        model = fit(Xtr, ytr, T=T)
        print(f"T={T:<3} stumps  train {accuracy(ytr, predict(model, Xtr)):.3f}"
              f"  test {accuracy(yte, predict(model, Xte)):.3f}")

    final = fit(Xtr, ytr, T=50)
    acc = accuracy(yte, predict(final, Xte))
    assert acc > 0.97
    print("adaboost ok")
