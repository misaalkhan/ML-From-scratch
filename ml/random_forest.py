"""Random forest classifier (Breiman 2001): bagged CART trees on bootstrap samples
with a sqrt-sized random feature subset per split, plus an out-of-bag accuracy estimate."""

import numpy as np

from ml.datasets import load_breast_cancer, load_sonar
from ml.decision_tree import DecisionTree
from ml.metrics import accuracy
from ml.preprocessing import train_test_split


def fit(X, y, n_trees=100, max_depth=None, max_features="sqrt", min_samples_split=2, seed=0):
    rng = np.random.default_rng(seed)
    n, k = len(y), int(y.max()) + 1
    trees = []
    oob_votes = np.zeros((n, k))
    for s in rng.integers(0, 2 ** 31, n_trees):
        boot = rng.integers(0, n, n)
        t = DecisionTree("classification", "gini", max_depth, min_samples_split,
                         max_features, seed=int(s)).fit(X[boot], y[boot])
        trees.append(t)
        oob = np.ones(n, bool)
        oob[np.unique(boot)] = False  # samples never drawn for this tree
        if oob.any():
            oob_votes[oob] += t.predict_proba(X[oob])
    voted = oob_votes.sum(1) > 0
    oob_acc = accuracy(y[voted], oob_votes[voted].argmax(1))
    return trees, oob_acc


def predict(trees, X):
    proba = sum(t.predict_proba(X) for t in trees) / len(trees)
    return proba.argmax(1)


if __name__ == "__main__":
    N_TREES = 100
    for name, (X, y) in [("breast_cancer", load_breast_cancer()), ("sonar", load_sonar())]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
        trees, oob = fit(Xtr, ytr, n_trees=N_TREES, seed=0)
        acc = accuracy(yte, predict(trees, Xte))
        print(f"{name:<14} {N_TREES} trees  test accuracy {acc:.3f}  OOB accuracy {oob:.3f}")
        if name == "breast_cancer":
            bc_acc, bc_oob = acc, oob

    assert bc_acc > 0.95 and bc_oob > 0.93
    print("random_forest ok")
